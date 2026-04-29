//! Extract text + bounding boxes from a PDF via mupdf's stext API, and
//! enrich each line with intra-line style spans probed from the source PDF
//! (bold / italic / monospace flags per character) so translation preserves
//! styled words inside otherwise-regular paragraphs.

use mupdf::text_page::TextBlockType;
use mupdf::{Document, Page, TextPageFlags};

use crate::ocr::Rect;
use crate::pdf::{PageDims, PdfError};
use crate::pdf_content::FontStyleFlags;
use crate::pdf_style_probe::{PageStyles, TjSample, probe_pages};
use crate::styled::{StyledFragment, TextStyle};

#[derive(Debug, Clone)]
pub struct PageTextFragments {
    pub page_index: usize,
    pub page: PageDims,
    pub fragments: Vec<StyledFragment>,
}

const STEXT_FLAGS: TextPageFlags = TextPageFlags::from_bits_truncate(
    TextPageFlags::PRESERVE_LIGATURES.bits()
        | TextPageFlags::PRESERVE_WHITESPACE.bits()
        | TextPageFlags::DEHYPHENATE.bits()
        | TextPageFlags::ACCURATE_BBOXES.bits(),
);

/// Vertical slack (in PDF points) when associating Tj samples with a stext
/// line. Wide enough to catch the baseline of an ascender-heavy line, narrow
/// enough that adjacent lines (e.g. a 9pt heading just above a 12pt body)
/// don't bleed their style into one another.
const LINE_TJ_VERTICAL_TOLERANCE_PT: f32 = 2.0;

/// Horizontal slack (in PDF points) when matching a char to the latest Tj at
/// or before its display-x. MuPDF reports glyph origins at the glyph ink edge,
/// while PDF text-show origins are baseline advances; coloured words with a
/// left side bearing can otherwise start a few characters late.
const TJ_X_MATCH_TOLERANCE_PT: f32 = 2.0;
const TJ_INTERVAL_MATCH_SLACK_EM: f32 = 0.45;

pub fn extract_text(pdf_bytes: &[u8]) -> Result<Vec<PageTextFragments>, PdfError> {
    let document = Document::from_bytes(pdf_bytes, "application/pdf")?;
    let page_count = document.page_count()?;

    // Best-effort style probe: if lopdf can't parse the file we still emit
    // unsplit fragments rather than failing the whole extraction.
    let styles_per_page = probe_pages(pdf_bytes).ok();

    let mut pages = Vec::with_capacity(page_count as usize);
    for i in 0..page_count {
        let page = document.load_page(i)?;
        let style_samples = styles_per_page.as_ref().and_then(|v| v.get(i as usize));
        pages.push(extract_page(&page, i as usize, style_samples)?);
    }
    Ok(pages)
}

fn extract_page(
    page: &Page,
    page_index: usize,
    page_styles: Option<&PageStyles>,
) -> Result<PageTextFragments, PdfError> {
    let bounds = page.bounds()?;
    let dims = PageDims {
        width_pts: bounds.x1 - bounds.x0,
        height_pts: bounds.y1 - bounds.y0,
    };

    let stext = page.to_text_page(STEXT_FLAGS)?;
    let mut fragments = Vec::new();

    for (block_index, block) in stext
        .blocks()
        .filter(|b| matches!(b.r#type(), TextBlockType::Text))
        .enumerate()
    {
        let translation_group = block_index as u32;
        for line in block.lines() {
            let line_bbox = line.bounds();
            let line_rect = rect_from_mupdf(line_bbox);

            // Collect chars + their display-coord origins. Drop NBSPs and
            // invisible chars to avoid polluting the visible text.
            let mut chars: Vec<(char, f32)> = Vec::new(); // (char, display_x)
            for ch in line.chars() {
                if let Some(c) = ch.char() {
                    chars.push((c, ch.origin().x));
                }
            }

            if chars.iter().all(|(c, _)| c.is_whitespace()) {
                continue;
            }
            let line_text: String = chars.iter().map(|(c, _)| *c).collect();
            if should_skip_pdf_line(&line_text, line_rect, dims) {
                continue;
            }

            // Split the line into runs of consecutive chars that share style
            // flags. With no probe, the whole line is one run with style: None.
            let runs = split_line_by_style(&chars, line_rect, page_styles).unwrap_or_else(|| {
                vec![LineRun {
                    text: chars.iter().map(|(c, _)| *c).collect(),
                    style: None,
                    bbox: line_rect,
                }]
            });

            for run in runs {
                let trimmed = run.text.trim();
                if trimmed.is_empty() {
                    continue;
                }
                fragments.push(StyledFragment {
                    text: trimmed.to_string(),
                    bounding_box: run.bbox,
                    style: run.style,
                    layout_group: 0,
                    translation_group,
                    cluster_group: translation_group,
                });
            }
        }
    }

    Ok(PageTextFragments {
        page_index,
        page: dims,
        fragments,
    })
}

#[derive(Debug)]
struct LineRun {
    text: String,
    style: Option<TextStyle>,
    bbox: Rect,
}

/// Split a line's chars into runs by style flag transitions, using the
/// per-Tj samples from the lopdf probe. Returns `None` if no probe samples
/// fall on this line (caller falls back to a single style-less run).
fn split_line_by_style(
    chars: &[(char, f32)],
    line_rect: Rect,
    page_styles: Option<&PageStyles>,
) -> Option<Vec<LineRun>> {
    let page_styles = page_styles?;
    let line_top = line_rect.top as f32;
    let line_bottom = line_rect.bottom as f32;

    // Tjs whose display-converted origin lies inside this line's bbox (the
    // baseline is near the bottom for plain text and not far above the top
    // for ascender-heavy lines). Using `[line_top, line_bottom] ± 2pt`
    // instead of distance-from-mid keeps adjacent lines from polluting one
    // another — a 9pt heading sitting just above a 12pt body line was
    // pulling its bold Tjs into the body and producing spurious mid-word
    // bold splits.
    let mut on_line: Vec<LineTjSample<'_>> = page_styles
        .samples
        .iter()
        .filter_map(|s| {
            let (dx, dy) = page_styles.to_display(s.origin);
            let (end_dx, end_dy) = page_styles.to_display(s.end_origin);
            if dy >= line_top - LINE_TJ_VERTICAL_TOLERANCE_PT
                && dy <= line_bottom + LINE_TJ_VERTICAL_TOLERANCE_PT
                || end_dy >= line_top - LINE_TJ_VERTICAL_TOLERANCE_PT
                    && end_dy <= line_bottom + LINE_TJ_VERTICAL_TOLERANCE_PT
            {
                Some(LineTjSample {
                    start_x: dx.min(end_dx),
                    end_x: dx.max(end_dx),
                    slack: (s.font_size * s.xy_scale.0 * TJ_INTERVAL_MATCH_SLACK_EM)
                        .max(TJ_X_MATCH_TOLERANCE_PT),
                    sample: s,
                })
            } else {
                None
            }
        })
        .collect();
    on_line.sort_by(|a, b| {
        a.start_x
            .partial_cmp(&b.start_x)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    if on_line.is_empty() {
        return None;
    }

    // Prefer the Tj/TJ run interval containing this char. Near boundaries,
    // choose the latest run whose start is within slack so colour resets after
    // short styled words win over the previous run. Fall back to latest-before
    // for producers where our advance estimate is too rough.
    fn style_at(on_line: &[LineTjSample<'_>], x: f32) -> PdfCharStyle {
        let mut last: Option<&TjSample> = None;
        let mut interval_match: Option<&TjSample> = None;
        for entry in on_line {
            if entry.start_x <= x + entry.slack && x <= entry.end_x + entry.slack {
                interval_match = Some(entry.sample);
            }
            if entry.start_x <= x + TJ_X_MATCH_TOLERANCE_PT {
                last = Some(entry.sample);
            } else {
                break;
            }
        }
        let sample = interval_match.or(last).unwrap_or(on_line[0].sample);
        PdfCharStyle {
            flags: sample.flags,
            fill_argb: rgb_to_argb(sample.fill_rgb),
        }
    }

    let mut runs: Vec<LineRun> = Vec::new();
    let mut run_text = String::new();
    let mut run_start_x: Option<f32> = None;
    let mut run_end_x: f32 = line_rect.left as f32;
    let mut run_style: Option<PdfCharStyle> = None;

    for (c, x) in chars {
        let style = style_at(&on_line, *x);
        if Some(style) != run_style && !run_text.is_empty() {
            let end_x = if run_text.chars().last().is_some_and(char::is_whitespace) {
                run_end_x
            } else {
                *x
            };
            runs.push(finish_run(
                &run_text,
                run_style,
                run_start_x,
                end_x,
                line_rect,
            ));
            run_text.clear();
            run_start_x = None;
        }
        if run_start_x.is_none() {
            run_start_x = Some(*x);
        }
        run_end_x = run_end_x.max(*x);
        run_style = Some(style);
        run_text.push(*c);
    }
    if !run_text.is_empty() {
        runs.push(finish_run(
            &run_text,
            run_style,
            run_start_x,
            line_rect.right as f32,
            line_rect,
        ));
    }

    // If the whole line collapsed to one style, return a single run with
    // that style — no point fragmenting a uniform line.
    Some(runs)
}

struct LineTjSample<'a> {
    start_x: f32,
    end_x: f32,
    slack: f32,
    sample: &'a TjSample,
}

fn finish_run(
    text: &str,
    pdf_style: Option<PdfCharStyle>,
    start_x: Option<f32>,
    end_x: f32,
    line_rect: Rect,
) -> LineRun {
    // TextStyle has no `monospace` field — that gets re-derived from the
    // font name on the write side.
    let style = pdf_style.map(|s| TextStyle {
        text_color: Some(s.fill_argb),
        bg_color: None,
        text_size: None,
        bold: s.flags.bold,
        italic: s.flags.italic,
        underline: false,
        strikethrough: false,
    });
    let bbox = Rect {
        left: start_x.unwrap_or(line_rect.left as f32) as u32,
        top: line_rect.top,
        right: end_x.ceil() as u32,
        bottom: line_rect.bottom,
    };
    LineRun {
        text: text.to_string(),
        style,
        bbox,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PdfCharStyle {
    flags: FontStyleFlags,
    fill_argb: u32,
}

fn rgb_to_argb(rgb: (f32, f32, f32)) -> u32 {
    let channel = |v: f32| -> u32 { (v.clamp(0.0, 1.0) * 255.0).round() as u32 };
    0xFF00_0000 | (channel(rgb.0) << 16) | (channel(rgb.1) << 8) | channel(rgb.2)
}

fn should_skip_pdf_line(text: &str, rect: Rect, page: PageDims) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return true;
    }

    let words = trimmed.split_whitespace().count();
    let letters = trimmed.chars().filter(|c| c.is_alphabetic()).count();
    let symbols = trimmed
        .chars()
        .filter(|c| {
            matches!(
                c,
                '=' | '<'
                    | '>'
                    | '{'
                    | '}'
                    | '['
                    | ']'
                    | '∑'
                    | 'Σ'
                    | 'σ'
                    | '≤'
                    | '≥'
                    | '→'
                    | '←'
                    | '↔'
            )
        })
        .count();
    let underscores = trimmed.matches('_').count();
    let centered = {
        let center = (rect.left + rect.right) as f32 * 0.5;
        (center - page.width_pts * 0.5).abs() < page.width_pts * 0.2
    };

    // Display equations/code signatures in papers should stay verbatim. They
    // are usually centered, symbol-heavy, or underscore-heavy, and contain
    // little ordinary prose.
    (centered && symbols > 0 && words <= 8)
        || (symbols >= 2 && letters < 24)
        || (underscores >= 2 && words <= 8)
}

/// mupdf::Rect (top-left origin, points, f32) → our Rect (top-left origin, u32).
fn rect_from_mupdf(r: mupdf::Rect) -> Rect {
    Rect {
        left: r.x0.max(0.0).floor() as u32,
        top: r.y0.max(0.0).floor() as u32,
        right: r.x1.max(0.0).ceil() as u32,
        bottom: r.y1.max(0.0).ceil() as u32,
    }
}
