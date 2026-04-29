//! Content-stream surgery: walk a page's operators and drop every text-show
//! whose origin lies inside a translated-block bbox, sampling the dropped
//! Tjs' style/geometry on the way through so the writeback can mimic the
//! producer.

use std::collections::HashMap;

use lopdf::content::{Content, Operation};
use lopdf::{Document, ObjectId};

use crate::pdf_content::{
    ContentState, FontAdvanceMap, FontStyleFlags, Matrix, PageGeometry, UserRect, font_flags,
    is_text_show_operator,
};
use crate::pdf_write::{BlockGeometry, BlockTypography, PdfWriteError, SampledBlockStyle};

/// Minimum vertical separation (in PDF points) between two Tj samples for them
/// to be treated as belonging to distinct lines when reconstructing original
/// line anchors. Anything closer is the same baseline + sub-point jitter.
const DISTINCT_BASELINE_PT: f32 = 1.0;

/// Walk the page's decoded content stream, drop every text-show operator
/// whose origin lies inside any of `removal_rects`, and write the result
/// back. Non-text operators (paths, images, shading) are left untouched.
/// Returns the CTM that's still active at the end of the content stream
/// so the appended translated-text stream can match the producer's local
/// coordinate convention.
pub(crate) fn rewrite_page_content(
    doc: &mut Document,
    page_id: ObjectId,
    removal_rects: &[UserRect],
    geom: PageGeometry,
) -> Result<(Matrix, Vec<SampledBlockStyle>), PdfWriteError> {
    let content = doc.get_and_decode_page_content(page_id)?;
    let font_advances = FontAdvanceMap::from_page(doc, page_id);
    let (filtered, final_ctm, raw_samples) =
        filter_text_ops(content.operations, removal_rects, &font_advances);
    let new_bytes = Content {
        operations: filtered,
    }
    .encode()?;
    let block_styles = resolve_block_styles(doc, page_id, &raw_samples, geom);
    doc.change_page_content(page_id, new_bytes)?;
    Ok((final_ctm, block_styles))
}

/// For each removal rect, pick the dominant raw style sample (most common
/// font + the median fill colour) and resolve its font resource to a
/// `SampledBlockStyle` with bold/italic flags.
fn resolve_block_styles(
    doc: &Document,
    page_id: ObjectId,
    raw: &[Vec<RawStyleSample>],
    geom: PageGeometry,
) -> Vec<SampledBlockStyle> {
    let fonts = doc.get_page_fonts(page_id).ok();
    raw.iter()
        .map(|samples| {
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
            // topmost line), then by visual x (leftmost on that line).
            let anchor = samples
                .iter()
                .min_by(|a, b| {
                    let (ax, ay) = geom.to_display(a.origin);
                    let (bx, by) = geom.to_display(b.origin);
                    ay.partial_cmp(&by)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(ax.partial_cmp(&bx).unwrap_or(std::cmp::Ordering::Equal))
                })
                .map(|s| s.origin);

            let line_anchors = original_line_anchors(samples, geom);
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

fn original_line_anchors(samples: &[RawStyleSample], geom: PageGeometry) -> Vec<(f32, f32)> {
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
    for (vy, vx, origin) in positioned {
        if current_y.is_some_and(|y| (vy - y).abs() >= DISTINCT_BASELINE_PT) {
            anchors.push(best_origin);
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
        anchors.push(best_origin);
    }
    anchors
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
    ops: Vec<Operation>,
    removal_rects: &[UserRect],
    font_advances: &FontAdvanceMap,
) -> (Vec<Operation>, Matrix, Vec<Vec<RawStyleSample>>) {
    let mut state = ContentState::new();
    let mut out = Vec::with_capacity(ops.len());
    let mut samples: Vec<Vec<RawStyleSample>> = vec![Vec::new(); removal_rects.len()];

    for op in ops {
        if !is_text_show_operator(&op.operator) {
            state.apply_non_show_op(&op);
            out.push(op);
            continue;
        }
        let snapshot = state.process_text_show(&op, font_advances);
        let dropped_rect = removal_rects
            .iter()
            .position(|r| r.contains(snapshot.origin.0, snapshot.origin.1));
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
        // Drop the op (don't push), keeping the cursor advance that
        // process_text_show already applied so subsequent show ops in the
        // same BT/ET block see their true origin.
    }

    (out, state.current_ctm(), samples)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lopdf::Object;

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
        let (filtered, _, _) = filter_text_ops(ops.clone(), &[rect], &FontAdvanceMap::default());
        // BT, Tf, Td, ET — the Tj should have been dropped.
        assert_eq!(filtered.len(), 4);
        assert!(filtered.iter().all(|o| o.operator != "Tj"));
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
        let (filtered, _, _) = filter_text_ops(ops, &[rect], &FontAdvanceMap::default());
        assert!(filtered.iter().any(|o| o.operator == "Tj"));
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
            x0: 124.0,
            y0: 690.0,
            x1: 130.0,
            y1: 710.0,
        };
        let (filtered, _, _) = filter_text_ops(ops, &[rect], &FontAdvanceMap::default());
        assert_eq!(filtered.iter().filter(|op| op.operator == "Tj").count(), 1);
    }
}
