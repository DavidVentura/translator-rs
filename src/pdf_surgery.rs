//! Content-stream surgery: walk a page's operators and drop every text-show
//! whose origin lies inside a translated source fragment, sampling the dropped
//! Tjs' style/geometry on the way through so the writeback can mimic the
//! producer.

use std::collections::{HashMap, HashSet};

use lopdf::content::{Content, Operation};
use lopdf::{Dictionary, Document, Object, ObjectId};

use crate::pdf_content::{
    ContentState, FontAdvanceMap, FontStyleFlags, Matrix, PageGeometry, UserRect, font_flags,
    is_text_show_operator, matrix_from_operands,
};
use crate::pdf_write::{BlockGeometry, BlockTypography, PdfWriteError, SampledBlockStyle};

/// Minimum vertical separation floor (in PDF points) between two Tj samples
/// for them to be treated as belonging to distinct lines. The effective
/// threshold scales with sampled font size so TeX superscripts/subscripts
/// stay attached to their main text baseline.
const DISTINCT_BASELINE_PT_FLOOR: f32 = 1.0;
const DISTINCT_BASELINE_FONT_FRACTION: f32 = 0.55;

#[derive(Debug, Clone)]
pub(crate) struct CapturedTextShow {
    pub(crate) operation: Operation,
    pub(crate) font_resource: Vec<u8>,
    pub(crate) font_size: f32,
    pub(crate) fill_rgb: (f32, f32, f32),
    pub(crate) char_spacing: f32,
    pub(crate) word_spacing: f32,
    pub(crate) horizontal_scaling: f32,
    pub(crate) combined: Matrix,
    pub(crate) origin: (f32, f32),
}

/// Walk the page's decoded content stream, drop every text-show operator
/// whose origin lies inside any of `removal_rects`, and write the result
/// back. Non-text operators (paths, images, shading) are left untouched.
/// Returns the CTM that's still active at the end of the content stream
/// so the appended translated-text stream can match the producer's local
/// coordinate convention.
pub(crate) fn rewrite_page_content(
    doc: &mut Document,
    page_id: ObjectId,
    removal_rects: &[Vec<UserRect>],
    capture_text: &[bool],
    geom: PageGeometry,
) -> Result<(Matrix, Vec<SampledBlockStyle>, Vec<Vec<CapturedTextShow>>), PdfWriteError> {
    let content = doc.get_and_decode_page_content(page_id)?;
    let resources = ResourceContext::from_page(doc, page_id);
    let (filtered, final_ctm, raw_samples, captured) = filter_text_ops(
        doc,
        content.operations,
        removal_rects,
        capture_text,
        &resources,
        ContentState::new(),
        &mut HashSet::new(),
    )?;
    let new_bytes = Content {
        operations: filtered,
    }
    .encode()?;
    let block_styles = resolve_block_styles(doc, page_id, &raw_samples, removal_rects, geom);
    doc.change_page_content(page_id, new_bytes)?;
    Ok((final_ctm, block_styles, captured))
}

#[derive(Debug, Clone, Default)]
struct ResourceContext {
    font_advances: FontAdvanceMap,
    xobjects: HashMap<Vec<u8>, ObjectId>,
}

impl ResourceContext {
    fn from_page(doc: &Document, page_id: ObjectId) -> Self {
        let mut context = Self {
            font_advances: FontAdvanceMap::from_page(doc, page_id),
            xobjects: HashMap::new(),
        };
        if let Ok((resource_dict, resource_ids)) = doc.get_page_resources(page_id) {
            if let Some(resources) = resource_dict {
                context.collect_xobjects(doc, resources);
            }
            for resource_id in resource_ids {
                if let Ok(resources) = doc.get_dictionary(resource_id) {
                    context.collect_xobjects(doc, resources);
                }
            }
        }
        context
    }

    fn from_resources(doc: &Document, resources: &Dictionary) -> Self {
        let mut context = Self {
            font_advances: FontAdvanceMap::from_resources(doc, resources),
            xobjects: HashMap::new(),
        };
        context.collect_xobjects(doc, resources);
        context
    }

    fn collect_xobjects(&mut self, doc: &Document, resources: &Dictionary) {
        let xobjects = match resources.get(b"XObject") {
            Ok(Object::Reference(id)) => doc.get_object(*id).and_then(Object::as_dict).ok(),
            Ok(Object::Dictionary(dict)) => Some(dict),
            _ => None,
        };
        let Some(xobjects) = xobjects else {
            return;
        };
        for (name, value) in xobjects.iter() {
            if self.xobjects.contains_key(name) {
                continue;
            }
            if let Ok(id) = value.as_reference() {
                self.xobjects.insert(name.clone(), id);
            }
        }
    }
}

/// For each removal rect, pick the dominant raw style sample (most common
/// font + the median fill colour) and resolve its font resource to a
/// `SampledBlockStyle` with bold/italic flags.
fn resolve_block_styles(
    doc: &Document,
    page_id: ObjectId,
    raw: &[Vec<RawStyleSample>],
    removal_rects: &[Vec<UserRect>],
    geom: PageGeometry,
) -> Vec<SampledBlockStyle> {
    let fonts = doc.get_page_fonts(page_id).ok();
    raw.iter()
        .enumerate()
        .map(|(block_idx, samples)| {
            // Visual leftmost edge in user space, lifted from the block's
            // mupdf-derived bbox. We use this to override anchor.x — surgery's
            // Tj-origin tracking advances by *text-matrix translation*, so for
            // PDFs that emit text via `TJ` arrays with leading negative
            // spacing (`[-3889 (Alg.) ...]`) the Tj origin sits at the
            // gutter position (where the matrix was set), not where the
            // glyphs actually land. mupdf's per-char x is the truth here.
            let block_left_x = removal_rects
                .get(block_idx)
                .and_then(|rects| rects.iter().map(|r| r.x0).reduce(f32::min))
                .filter(|x| x.is_finite());
            if samples.is_empty() {
                return SampledBlockStyle::default();
            }

            // Dominant font resource = mode of font_resource values.
            let mut font_counts: HashMap<Vec<u8>, usize> = HashMap::new();
            for sample in samples {
                if let Some(name) = &sample.font_resource {
                    *font_counts.entry(name.clone()).or_default() += 1;
                }
            }
            let dominant_font = font_counts
                .into_iter()
                .max_by_key(|(_, count)| *count)
                .map(|(name, _)| name);

            // Mean fill color across samples.
            let n = samples.len() as f32;
            let mut r = 0.0f32;
            let mut g = 0.0f32;
            let mut b = 0.0f32;
            for s in samples {
                r += s.fill_rgb.0;
                g += s.fill_rgb.1;
                b += s.fill_rgb.2;
            }
            let fill_rgb = (r / n, g / n, b / n);

            let flags = dominant_font
                .as_deref()
                .and_then(|res_name| {
                    let fonts = fonts.as_ref()?;
                    let font_dict = fonts.get(res_name)?;
                    Some(font_flags(doc, font_dict))
                })
                .unwrap_or_default();
            let FontStyleFlags {
                bold,
                italic,
                monospace,
            } = flags;

            // Median font size across the rect's samples. Median (not mean)
            // is robust against the occasional outlier Tj that doesn't
            // belong to the main run of text in this block.
            let mut sizes: Vec<f32> = samples.iter().map(|s| s.font_size).collect();
            sizes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let font_size = if sizes.is_empty() {
                None
            } else {
                Some(sizes[sizes.len() / 2])
            };

            // Anchor = the visually top-left baseline among the sampled Tjs.
            // We sort by visual y first (smallest visual_y = visually
            // topmost line), then by visual x (leftmost on that line). Then
            // we override the x with the bbox left edge (mupdf's per-char x)
            // — surgery's Tj-origin tracking can land further right than the
            // first visible glyph when the producer uses `TJ` arrays with
            // leading negative-spacing.
            let anchor = samples
                .iter()
                .min_by(|a, b| {
                    let (ax, ay) = geom.to_display(a.origin);
                    let (bx, by) = geom.to_display(b.origin);
                    ay.partial_cmp(&by)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(ax.partial_cmp(&bx).unwrap_or(std::cmp::Ordering::Equal))
                })
                .map(|s| {
                    let x = block_left_x.unwrap_or(s.origin.0);
                    (x, s.origin.1)
                });

            let line_anchors = original_line_anchors(samples, geom, None);
            // Per-line x override from the block's source_rects: surgery's
            // Tj-origin x can land further right than the first visible
            // glyph (TJ arrays with leading negative spacing land at the
            // gutter, not the glyph). The fix used to be a single global
            // `block_left_x` override, but that conflated lines that
            // genuinely start at different x positions — e.g. a multi-line
            // prose block whose FIRST line continues an inline formula
            // (starts at x=473) and whose later lines wrap to the left
            // margin (x=88). Pushing every line to the global min would
            // shove the "and we commit" continuation back to x=88 where it
            // overlaps the formula it's supposed to follow.
            let block_rects = removal_rects
                .get(block_idx)
                .map(|v| v.as_slice())
                .unwrap_or(&[]);
            let line_anchors: Vec<(f32, f32)> = line_anchors
                .into_iter()
                .map(|(sample_x, y)| {
                    let line_left_x = block_rects
                        .iter()
                        .filter(|r| y >= r.y0 - 1.0 && y <= r.y1 + 1.0)
                        .map(|r| r.x0)
                        .fold(f32::INFINITY, f32::min);
                    let new_x = if line_left_x.is_finite() {
                        line_left_x
                    } else {
                        sample_x
                    };
                    (new_x, y)
                })
                .collect();
            let original_line_count = line_anchors.len().max(1);

            // Pick the orientation of the first sample. Producers almost
            // always use one orientation per block; if it's mixed, the first
            // is as good as any and the others would visually clash anyway.
            let text_orientation = samples
                .first()
                .map(|s| s.text_orientation)
                .unwrap_or_else(Matrix::identity);

            SampledBlockStyle {
                typography: BlockTypography {
                    flags: FontStyleFlags {
                        bold,
                        italic,
                        monospace,
                    },
                    fill_rgb,
                    font_size,
                },
                geometry: BlockGeometry {
                    anchor,
                    original_line_count,
                    text_orientation,
                    line_anchors,
                },
            }
        })
        .collect()
}

fn original_line_anchors(
    samples: &[RawStyleSample],
    geom: PageGeometry,
    override_x: Option<f32>,
) -> Vec<(f32, f32)> {
    let baseline_threshold = distinct_baseline_threshold(samples);
    let mut positioned: Vec<(f32, f32, (f32, f32))> = samples
        .iter()
        .map(|s| {
            let (vx, vy) = geom.to_display(s.origin);
            (vy, vx, s.origin)
        })
        .collect();
    positioned.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    });

    let mut anchors = Vec::new();
    let mut current_y: Option<f32> = None;
    let mut best_x = f32::INFINITY;
    let mut best_origin = (0.0, 0.0);
    let push_anchor = |anchors: &mut Vec<(f32, f32)>, origin: (f32, f32)| match override_x {
        Some(x) => anchors.push((x, origin.1)),
        None => anchors.push(origin),
    };
    for (vy, vx, origin) in positioned {
        if current_y.is_some_and(|y| (vy - y).abs() >= baseline_threshold) {
            push_anchor(&mut anchors, best_origin);
            current_y = Some(vy);
            best_x = vx;
            best_origin = origin;
            continue;
        }
        if current_y.is_none() {
            current_y = Some(vy);
        }
        if vx < best_x {
            best_x = vx;
            best_origin = origin;
        }
    }
    if current_y.is_some() {
        push_anchor(&mut anchors, best_origin);
    }

    // Post-merge near-duplicate anchors. The greedy bucketing above compares
    // each sample's vy against the bucket's *start* y, which means a tightly
    // packed run of samples at vy = (start, start+T-ε, start+T+ε, …) splits
    // at the second cell even when the latest accepted sample was within the
    // threshold. In TeX papers that's how subscripts/superscripts orphan
    // themselves into their own anchor (a subscript baseline 3pt below the
    // main baseline misses the bucket, becomes its own anchor, and the
    // overlay then renders two wrap-lines stacked on top of each other at
    // ≈3pt apart). Walk the resulting anchors and merge any two whose
    // y-gap is smaller than half the dominant inter-line gap — that floor
    // catches sub/super orphans without collapsing genuine tight math
    // line-spacing into one anchor.
    if anchors.len() >= 3 {
        let mut gaps: Vec<f32> = anchors
            .windows(2)
            .map(|w| (w[0].1 - w[1].1).abs())
            .collect();
        gaps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let typical_gap = gaps[gaps.len() * 3 / 4];
        let merge_threshold = (typical_gap * 0.5).max(baseline_threshold);
        let mut merged: Vec<(f32, f32)> = Vec::with_capacity(anchors.len());
        merged.push(anchors[0]);
        for next in anchors.iter().skip(1) {
            let last = merged.last().copied().expect("merged seeded above");
            if (last.1 - next.1).abs() < merge_threshold {
                continue;
            }
            merged.push(*next);
        }
        anchors = merged;
    }
    anchors
}

fn distinct_baseline_threshold(samples: &[RawStyleSample]) -> f32 {
    let mut sizes: Vec<f32> = samples
        .iter()
        .map(|s| s.font_size)
        .filter(|size| size.is_finite() && *size > 0.0)
        .collect();
    if sizes.is_empty() {
        return DISTINCT_BASELINE_PT_FLOOR;
    }
    sizes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    (sizes[sizes.len() / 2] * DISTINCT_BASELINE_FONT_FRACTION).max(DISTINCT_BASELINE_PT_FLOOR)
}

/// One observation of original-text style for a single removal rect.
#[derive(Debug, Clone)]
struct RawStyleSample {
    font_resource: Option<Vec<u8>>,
    fill_rgb: (f32, f32, f32),
    font_size: f32,
    /// User-space origin of the Tj — i.e. the baseline-leading-edge of the
    /// original text, after CTM has been applied.
    origin: (f32, f32),
    /// Linear part of the *combined* `text_matrix × CTM` at this Tj (with
    /// the translation column zeroed) — captures whatever rotation/flip the
    /// producer applied. Used to keep our emitted glyphs in the same
    /// orientation. Pure translation originals come out as identity, Y-flip
    /// producers as `(1, 0, 0, -1)`, /Rotate-compensating producers as
    /// `(0, 1, -1, 0)`.
    text_orientation: Matrix,
}

/// Track text/graphics state across `ops` and drop any text-show op whose
/// current origin lies inside a removal rect. Returns the filtered op list,
/// the CTM still active after the (balanced) graphics-state stack has
/// unwound, and per-rect samples of the dropped Tjs' font + fill colour so
/// the appended translation can mimic the producer's style.
fn filter_text_ops(
    doc: &mut Document,
    ops: Vec<Operation>,
    removal_rects: &[Vec<UserRect>],
    capture_text: &[bool],
    resources: &ResourceContext,
    mut state: ContentState,
    xobject_stack: &mut HashSet<ObjectId>,
) -> Result<
    (
        Vec<Operation>,
        Matrix,
        Vec<Vec<RawStyleSample>>,
        Vec<Vec<CapturedTextShow>>,
    ),
    PdfWriteError,
> {
    let mut out = Vec::with_capacity(ops.len());
    let mut samples: Vec<Vec<RawStyleSample>> = vec![Vec::new(); removal_rects.len()];
    let mut captured: Vec<Vec<CapturedTextShow>> = vec![Vec::new(); removal_rects.len()];

    for op in ops {
        if !is_text_show_operator(&op.operator) {
            if op.operator == "Do" {
                rewrite_form_xobject(
                    doc,
                    &op,
                    &state,
                    resources,
                    removal_rects,
                    capture_text,
                    &mut samples,
                    &mut captured,
                    xobject_stack,
                )?;
            }
            state.apply_non_show_op(&op);
            out.push(op);
            continue;
        }
        let snapshot = state.process_text_show(&op, &resources.font_advances);
        // Pick the rect that BEST matches this Tj's origin, not just the
        // first one that "touches" it. Adjacent fragments often have
        // bboxes that overlap by a fraction of a point at glyph
        // boundaries — a TeX big paren (CMEX10, e.g. x in [467, 472])
        // sits flush against a following comma (x=471), and surgery's
        // first-match-wins iteration would attribute the comma's Tj to
        // the paren's block. Prefer the rect whose left edge is closest
        // to (and not past) the Tj's own origin x — that's the rect the
        // glyph actually belongs to.
        let dropped_rect = removal_rects
            .iter()
            .enumerate()
            .filter_map(|(i, rects)| {
                rects
                    .iter()
                    .filter(|r| text_show_touches_source_rect(snapshot, **r))
                    .map(|r| (i, *r))
                    .next()
            })
            .min_by(|(_, a), (_, b)| {
                let dist_a = (snapshot.origin.0 - a.x0).abs();
                let dist_b = (snapshot.origin.0 - b.x0).abs();
                dist_a
                    .partial_cmp(&dist_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i);
        let Some(rect_index) = dropped_rect else {
            out.push(op);
            continue;
        };
        // Combined `text_matrix × CTM` carries both the producer's glyph
        // orientation *and* whatever scale the CTM applied (a
        // `.75 0 0 .75 ... cm` pre-scale is common). Split: store the
        // orientation as a unit-length rotation/flip, and bake the scale into
        // `font_size` so the value we record is the effective user-space size.
        let combined = snapshot.combined;
        let x_scale = (combined.a * combined.a + combined.b * combined.b).sqrt();
        let y_scale = (combined.c * combined.c + combined.d * combined.d).sqrt();
        let safe_x = if x_scale > 1e-6 { x_scale } else { 1.0 };
        let safe_y = if y_scale > 1e-6 { y_scale } else { 1.0 };
        samples[rect_index].push(RawStyleSample {
            font_resource: state.font_resource().clone(),
            fill_rgb: state.fill_rgb(),
            font_size: state.font_size() * safe_x,
            origin: snapshot.origin,
            text_orientation: Matrix {
                a: combined.a / safe_x,
                b: combined.b / safe_x,
                c: combined.c / safe_y,
                d: combined.d / safe_y,
                e: 0.0,
                f: 0.0,
            },
        });
        if capture_text.get(rect_index).copied().unwrap_or(false)
            && let Some(font_resource) = state.font_resource().clone()
        {
            captured[rect_index].push(CapturedTextShow {
                operation: normalize_text_show_operation(&op),
                font_resource,
                font_size: state.font_size(),
                fill_rgb: state.fill_rgb(),
                char_spacing: state.char_spacing(),
                word_spacing: state.word_spacing(),
                horizontal_scaling: state.horizontal_scaling(),
                combined,
                origin: snapshot.origin,
            });
        }
        // Drop the glyphs but emit a cursor-advance-only TJ in their place
        // so subsequent text-show operators in the same BT/ET still draw at
        // the correct x. Without this, dropping a Tj `(hello) Tj` removes
        // both the glyphs and the cursor advance the next Tj relied on —
        // and any *surviving* later Tj on the same line ends up at the
        // wrong x.
        //
        // `[N] TJ` with a single number advances the text-matrix cursor by
        // `-N * font_size * horizontal_scaling / 1000` user-space units
        // and draws nothing. We solve for N from the dropped op's advance.
        let font_size = state.font_size();
        let scaling = state.horizontal_scaling();
        if snapshot.advance.abs() > f32::EPSILON
            && font_size.abs() > f32::EPSILON
            && scaling.abs() > f32::EPSILON
        {
            let tj_units = -snapshot.advance * 1000.0 / (font_size * scaling);
            out.push(Operation::new(
                "TJ",
                vec![Object::Array(vec![Object::Real(tj_units)])],
            ));
        }
    }

    Ok((out, state.current_ctm(), samples, captured))
}

fn rewrite_form_xobject(
    doc: &mut Document,
    op: &Operation,
    state: &ContentState,
    resources: &ResourceContext,
    removal_rects: &[Vec<UserRect>],
    capture_text: &[bool],
    samples: &mut [Vec<RawStyleSample>],
    captured: &mut [Vec<CapturedTextShow>],
    xobject_stack: &mut HashSet<ObjectId>,
) -> Result<(), PdfWriteError> {
    let Some(Object::Name(name)) = op.operands.first() else {
        return Ok(());
    };
    let Some(xobject_id) = resources.xobjects.get(name).copied() else {
        return Ok(());
    };
    if !xobject_stack.insert(xobject_id) {
        return Ok(());
    }

    let object = doc.get_object(xobject_id)?.clone();
    let Object::Stream(mut stream) = object else {
        xobject_stack.remove(&xobject_id);
        return Ok(());
    };
    let is_form = stream
        .dict
        .get(b"Subtype")
        .and_then(Object::as_name)
        .is_ok_and(|subtype| subtype == b"Form");
    if !is_form {
        xobject_stack.remove(&xobject_id);
        return Ok(());
    }

    let form_resources = form_resource_context(doc, &stream.dict);
    let form_matrix = stream
        .dict
        .get(b"Matrix")
        .ok()
        .and_then(|obj| obj.as_array().ok())
        .and_then(|arr| matrix_from_operands(arr))
        .unwrap_or_else(Matrix::identity);
    let form_state = ContentState::with_ctm(form_matrix.mul(state.current_ctm()));

    let decoded = stream.get_plain_content()?;
    let content = Content::decode(&decoded)?;
    let (filtered, _, nested_samples, nested_captured) = filter_text_ops(
        doc,
        content.operations,
        removal_rects,
        capture_text,
        &form_resources,
        form_state,
        xobject_stack,
    )?;
    for (dst, src) in samples.iter_mut().zip(nested_samples) {
        dst.extend(src);
    }
    for (dst, src) in captured.iter_mut().zip(nested_captured) {
        dst.extend(src);
    }
    stream.set_plain_content(
        Content {
            operations: filtered,
        }
        .encode()?,
    );
    *doc.get_object_mut(xobject_id)? = Object::Stream(stream);
    xobject_stack.remove(&xobject_id);
    Ok(())
}

fn normalize_text_show_operation(op: &Operation) -> Operation {
    match op.operator.as_str() {
        "Tj" | "TJ" => op.clone(),
        "'" => {
            let text = op
                .operands
                .last()
                .cloned()
                .unwrap_or(Object::String(Vec::new(), lopdf::StringFormat::Literal));
            Operation::new("Tj", vec![text])
        }
        "\"" => {
            let text = op
                .operands
                .get(2)
                .cloned()
                .unwrap_or(Object::String(Vec::new(), lopdf::StringFormat::Literal));
            Operation::new("Tj", vec![text])
        }
        _ => op.clone(),
    }
}

fn form_resource_context(doc: &Document, dict: &Dictionary) -> ResourceContext {
    match dict.get(b"Resources") {
        Ok(Object::Reference(id)) => doc
            .get_dictionary(*id)
            .map(|resources| ResourceContext::from_resources(doc, resources))
            .unwrap_or_default(),
        Ok(Object::Dictionary(resources)) => ResourceContext::from_resources(doc, resources),
        _ => ResourceContext::default(),
    }
}

fn text_show_touches_source_rect(
    snapshot: crate::pdf_content::ShowSnapshot,
    rect: UserRect,
) -> bool {
    if rect.contains(snapshot.origin.0, snapshot.origin.1) {
        return true;
    }
    if snapshot.advance.abs() <= f32::EPSILON {
        return false;
    }
    let start = snapshot.combined.transform_point(0.0, 0.0);
    let end = snapshot.combined.transform_point(snapshot.advance, 0.0);
    segment_intersects_rect(start, end, rect)
}

fn segment_intersects_rect(start: (f32, f32), end: (f32, f32), rect: UserRect) -> bool {
    if rect.contains(start.0, start.1) || rect.contains(end.0, end.1) {
        return true;
    }
    let edges = [
        ((rect.x0, rect.y0), (rect.x1, rect.y0)),
        ((rect.x1, rect.y0), (rect.x1, rect.y1)),
        ((rect.x1, rect.y1), (rect.x0, rect.y1)),
        ((rect.x0, rect.y1), (rect.x0, rect.y0)),
    ];
    edges
        .iter()
        .any(|(edge_start, edge_end)| segments_intersect(start, end, *edge_start, *edge_end))
}

fn segments_intersect(a0: (f32, f32), a1: (f32, f32), b0: (f32, f32), b1: (f32, f32)) -> bool {
    const EPS: f32 = 1e-4;
    let d1 = cross(a0, a1, b0);
    let d2 = cross(a0, a1, b1);
    let d3 = cross(b0, b1, a0);
    let d4 = cross(b0, b1, a1);

    if d1.abs() <= EPS && point_on_segment(b0, a0, a1) {
        return true;
    }
    if d2.abs() <= EPS && point_on_segment(b1, a0, a1) {
        return true;
    }
    if d3.abs() <= EPS && point_on_segment(a0, b0, b1) {
        return true;
    }
    if d4.abs() <= EPS && point_on_segment(a1, b0, b1) {
        return true;
    }

    (d1 > 0.0) != (d2 > 0.0) && (d3 > 0.0) != (d4 > 0.0)
}

fn cross(a: (f32, f32), b: (f32, f32), c: (f32, f32)) -> f32 {
    (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)
}

fn point_on_segment(p: (f32, f32), a: (f32, f32), b: (f32, f32)) -> bool {
    const EPS: f32 = 1e-4;
    p.0 >= a.0.min(b.0) - EPS
        && p.0 <= a.0.max(b.0) + EPS
        && p.1 >= a.1.min(b.1) - EPS
        && p.1 <= a.1.max(b.1) + EPS
}

#[cfg(test)]
mod tests {
    use super::*;
    use lopdf::Object;

    fn run_filter(
        ops: Vec<Operation>,
        rects: &[Vec<UserRect>],
    ) -> (
        Vec<Operation>,
        Matrix,
        Vec<Vec<RawStyleSample>>,
        Vec<Vec<CapturedTextShow>>,
    ) {
        filter_text_ops(
            &mut Document::with_version("1.5"),
            ops,
            rects,
            &vec![false; rects.len()],
            &ResourceContext::default(),
            ContentState::new(),
            &mut HashSet::new(),
        )
        .unwrap()
    }

    fn run_filter_with_capture(
        ops: Vec<Operation>,
        rects: &[Vec<UserRect>],
        capture_text: &[bool],
    ) -> (
        Vec<Operation>,
        Matrix,
        Vec<Vec<RawStyleSample>>,
        Vec<Vec<CapturedTextShow>>,
    ) {
        filter_text_ops(
            &mut Document::with_version("1.5"),
            ops,
            rects,
            capture_text,
            &ResourceContext::default(),
            ContentState::new(),
            &mut HashSet::new(),
        )
        .unwrap()
    }

    #[test]
    fn matrix_mul_identity() {
        let i = Matrix::identity();
        let t = Matrix::translate(10.0, 20.0);
        let r = t.mul(i);
        assert!((r.e - 10.0).abs() < 1e-5);
        assert!((r.f - 20.0).abs() < 1e-5);
    }

    #[test]
    fn filter_drops_text_show_inside_rect() {
        let ops = vec![
            Operation::new("BT", vec![]),
            Operation::new("Tf", vec![Object::Name(b"F1".to_vec()), Object::Real(12.0)]),
            Operation::new("Td", vec![Object::Real(100.0), Object::Real(700.0)]),
            Operation::new(
                "Tj",
                vec![Object::String(b"hi".to_vec(), lopdf::StringFormat::Literal)],
            ),
            Operation::new("ET", vec![]),
        ];

        let rect = UserRect {
            x0: 50.0,
            y0: 650.0,
            x1: 200.0,
            y1: 750.0,
        };
        let (filtered, _, _, _) = run_filter(ops.clone(), &[vec![rect]]);
        // The Tj is replaced by a glyphless `[N] TJ` that preserves the
        // cursor advance for any subsequent text-show on the same line.
        assert!(filtered.iter().all(|o| o.operator != "Tj"));
        let tj_advance = filtered
            .iter()
            .find(|o| o.operator == "TJ")
            .expect("dropped Tj is replaced by an advance-only TJ");
        let array = tj_advance
            .operands
            .first()
            .and_then(|o| o.as_array().ok())
            .expect("TJ has an array operand");
        // Single negative number, no strings — the cursor moves but
        // nothing is drawn.
        assert_eq!(array.len(), 1);
        let value = match &array[0] {
            Object::Real(r) => Some(*r),
            Object::Integer(i) => Some(*i as f32),
            _ => None,
        };
        assert!(
            value.is_some_and(|v| v < 0.0),
            "advance-only TJ should be a single negative number"
        );
    }

    #[test]
    fn filter_captures_dropped_opaque_text_show() {
        let ops = vec![
            Operation::new("BT", vec![]),
            Operation::new("Tf", vec![Object::Name(b"F1".to_vec()), Object::Real(12.0)]),
            Operation::new("Td", vec![Object::Real(100.0), Object::Real(700.0)]),
            Operation::new(
                "Tj",
                vec![Object::String(
                    b"math".to_vec(),
                    lopdf::StringFormat::Literal,
                )],
            ),
            Operation::new("ET", vec![]),
        ];
        let rect = UserRect {
            x0: 50.0,
            y0: 650.0,
            x1: 200.0,
            y1: 750.0,
        };

        let (_, _, _, captured) = run_filter_with_capture(ops, &[vec![rect]], &[true]);

        assert_eq!(captured.len(), 1);
        assert_eq!(captured[0].len(), 1);
        assert_eq!(captured[0][0].font_resource, b"F1".to_vec());
        assert_eq!(captured[0][0].font_size, 12.0);
        assert_eq!(captured[0][0].operation.operator, "Tj");
    }

    #[test]
    fn filter_keeps_text_show_outside_rect() {
        let ops = vec![
            Operation::new("BT", vec![]),
            Operation::new("Tf", vec![Object::Name(b"F1".to_vec()), Object::Real(12.0)]),
            Operation::new("Td", vec![Object::Real(100.0), Object::Real(700.0)]),
            Operation::new(
                "Tj",
                vec![Object::String(b"hi".to_vec(), lopdf::StringFormat::Literal)],
            ),
            Operation::new("ET", vec![]),
        ];

        let rect = UserRect {
            x0: 0.0,
            y0: 0.0,
            x1: 50.0,
            y1: 50.0,
        };
        let (filtered, _, _, _) = run_filter(ops, &[vec![rect]]);
        assert!(filtered.iter().any(|o| o.operator == "Tj"));
    }

    #[test]
    fn filter_drops_text_show_when_baseline_crosses_rect() {
        let ops = vec![
            Operation::new("BT", vec![]),
            Operation::new("Tf", vec![Object::Name(b"F1".to_vec()), Object::Real(10.0)]),
            Operation::new("Td", vec![Object::Real(100.0), Object::Real(700.0)]),
            Operation::new(
                "Tj",
                vec![Object::String(
                    b"hello".to_vec(),
                    lopdf::StringFormat::Literal,
                )],
            ),
            Operation::new("ET", vec![]),
        ];

        let rect = UserRect {
            x0: 110.0,
            y0: 699.0,
            x1: 115.0,
            y1: 701.0,
        };
        let (filtered, _, _, _) = run_filter(ops, &[vec![rect]]);
        assert!(filtered.iter().all(|op| op.operator != "Tj"));
    }

    #[test]
    fn filter_advances_between_consecutive_text_shows() {
        let ops = vec![
            Operation::new("BT", vec![]),
            Operation::new("Tf", vec![Object::Name(b"F1".to_vec()), Object::Real(10.0)]),
            Operation::new("Td", vec![Object::Real(100.0), Object::Real(700.0)]),
            Operation::new(
                "Tj",
                vec![Object::String(
                    b"hello".to_vec(),
                    lopdf::StringFormat::Literal,
                )],
            ),
            Operation::new(
                "Tj",
                vec![Object::String(b"!".to_vec(), lopdf::StringFormat::Literal)],
            ),
            Operation::new("ET", vec![]),
        ];
        let rect = UserRect {
            x0: 127.0,
            y0: 690.0,
            x1: 129.0,
            y1: 710.0,
        };
        let (filtered, _, _, _) = run_filter(ops, &[vec![rect]]);
        assert_eq!(filtered.iter().filter(|op| op.operator == "Tj").count(), 1);
    }

    #[test]
    fn line_anchors_group_superscripts_with_main_baseline() {
        let geom = PageGeometry {
            user_w: 612.0,
            user_h: 792.0,
            user_x_min: 0.0,
            user_y_min: 0.0,
            rotate: 0,
        };
        let samples = vec![
            RawStyleSample {
                font_resource: None,
                fill_rgb: (0.0, 0.0, 0.0),
                font_size: 10.0,
                origin: (282.0, 714.0),
                text_orientation: Matrix::identity(),
            },
            RawStyleSample {
                font_resource: None,
                fill_rgb: (0.0, 0.0, 0.0),
                font_size: 10.0,
                origin: (72.0, 710.0),
                text_orientation: Matrix::identity(),
            },
            RawStyleSample {
                font_resource: None,
                fill_rgb: (0.0, 0.0, 0.0),
                font_size: 10.0,
                origin: (72.0, 698.0),
                text_orientation: Matrix::identity(),
            },
        ];

        let anchors = original_line_anchors(&samples, geom, None);
        assert_eq!(anchors, vec![(72.0, 710.0), (72.0, 698.0)]);
    }
}
