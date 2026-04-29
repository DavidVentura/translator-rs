//! Extract text + bounding boxes from a PDF via mupdf's stext API. mupdf
//! reports per-character colour, bold flag and font on every emitted glyph,
//! which is enough to split a line into runs of consecutive same-style
//! characters in a single pass — no second content-stream parser needed.

use std::collections::HashMap;

use mupdf::text_page::TextBlockType;
use mupdf::{Document, Page, TextCharFlags, TextPageFlags};

use crate::ocr::Rect;
use crate::pdf::{PageDims, PdfError};
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

pub fn extract_text(pdf_bytes: &[u8]) -> Result<Vec<PageTextFragments>, PdfError> {
    let document = Document::from_bytes(pdf_bytes, "application/pdf")?;
    let page_count = document.page_count()?;
    let mut pages = Vec::with_capacity(page_count as usize);
    for i in 0..page_count {
        let page = document.load_page(i)?;
        pages.push(extract_page(&page, i as usize)?);
    }
    Ok(pages)
}

fn extract_page(page: &Page, page_index: usize) -> Result<PageTextFragments, PdfError> {
    let bounds = page.bounds()?;
    let dims = PageDims {
        width_pts: bounds.x1 - bounds.x0,
        height_pts: bounds.y1 - bounds.y0,
    };

    let stext = page.to_text_page(STEXT_FLAGS)?;
    let mut fragments = Vec::new();
    // Cache italic lookups per font name. fz_font_name is stable per font and
    // calling fz_font_is_italic per char is wasteful when a paragraph reuses
    // the same handful of fonts.
    let mut italic_by_font: HashMap<String, bool> = HashMap::new();

    for (block_index, block) in stext
        .blocks()
        .filter(|b| matches!(b.r#type(), TextBlockType::Text))
        .enumerate()
    {
        let translation_group = block_index as u32;
        for line in block.lines() {
            let line_rect = rect_from_mupdf(line.bounds());

            let mut typed_chars: Vec<TypedChar> = Vec::new();
            for ch in line.chars() {
                let Some(c) = ch.char() else {
                    continue;
                };
                let italic = ch
                    .font()
                    .map(|f| {
                        *italic_by_font
                            .entry(f.name().to_string())
                            .or_insert_with(|| f.is_italic())
                    })
                    .unwrap_or(false);
                typed_chars.push(TypedChar {
                    c,
                    x: ch.origin().x,
                    style: PdfCharStyle {
                        bold: ch.flags().contains(TextCharFlags::BOLD),
                        italic,
                        fill_argb: ch.argb(),
                    },
                });
            }

            if typed_chars.iter().all(|tc| tc.c.is_whitespace()) {
                continue;
            }
            let line_text: String = typed_chars.iter().map(|tc| tc.c).collect();
            if should_skip_pdf_line(&line_text, line_rect, dims) {
                continue;
            }

            for run in split_line_by_style(&typed_chars, line_rect) {
                if run.text.trim().is_empty() {
                    continue;
                }
                fragments.push(StyledFragment {
                    text: run.text,
                    bounding_box: run.bbox,
                    style: Some(run.style.into()),
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PdfCharStyle {
    bold: bool,
    italic: bool,
    fill_argb: u32,
}

impl From<PdfCharStyle> for TextStyle {
    fn from(s: PdfCharStyle) -> Self {
        TextStyle {
            text_color: Some(s.fill_argb),
            bg_color: None,
            text_size: None,
            bold: s.bold,
            italic: s.italic,
            underline: false,
            strikethrough: false,
        }
    }
}

struct TypedChar {
    c: char,
    x: f32,
    style: PdfCharStyle,
}

#[derive(Debug)]
struct LineRun {
    text: String,
    style: PdfCharStyle,
    bbox: Rect,
}

/// Group consecutive chars into runs of identical style. Whitespace is
/// treated as style-neutral: it inherits whichever run it falls into and
/// never triggers a transition. That way the trailing space of "bold word "
/// stays inside the bold run instead of starting a new one — and downstream
/// concatenation in `build_block` sees the space and joins fragments
/// correctly.
fn split_line_by_style(chars: &[TypedChar], line_rect: Rect) -> Vec<LineRun> {
    let mut runs: Vec<LineRun> = Vec::new();
    let mut run_text = String::new();
    let mut run_start_x: Option<f32> = None;
    let mut run_end_x: f32 = line_rect.left as f32;
    let mut run_style: Option<PdfCharStyle> = None;

    for tc in chars {
        let is_ws = tc.c.is_whitespace();
        let style_changed = !is_ws && run_style.is_some_and(|s| s != tc.style);
        if style_changed && !run_text.is_empty() {
            runs.push(finish_run(
                &run_text,
                run_style.expect("run_style set when run_text non-empty"),
                run_start_x,
                tc.x,
                line_rect,
            ));
            run_text.clear();
            run_start_x = None;
        }
        if run_start_x.is_none() {
            run_start_x = Some(tc.x);
        }
        run_end_x = run_end_x.max(tc.x);
        if !is_ws {
            run_style = Some(tc.style);
        }
        run_text.push(tc.c);
    }
    if !run_text.is_empty() {
        let style = run_style.unwrap_or(PdfCharStyle {
            bold: false,
            italic: false,
            fill_argb: 0xFF00_0000,
        });
        runs.push(finish_run(
            &run_text,
            style,
            run_start_x,
            line_rect.right as f32,
            line_rect,
        ));
    }
    runs
}

fn finish_run(
    text: &str,
    style: PdfCharStyle,
    start_x: Option<f32>,
    end_x: f32,
    line_rect: Rect,
) -> LineRun {
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

fn should_skip_pdf_line(text: &str, rect: Rect, page: PageDims) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return true;
    }

    // Veto markers — these signal "this is code or prose with logic, not a
    // display equation". A `;` terminator, a dotted identifier like
    // `p.view`, or two-plus runs of 4+ alphabetic characters (real words
    // like `then`, `votes`, `view`) all rule out the symbol/letter-density
    // shortcuts below. Real display equations don't have any of these.
    if trimmed.contains(';')
        || has_dotted_identifier(trimmed)
        || count_long_alpha_runs(trimmed, 4) >= 2
    {
        return false;
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
    // Truly-centered display equations sit at page center *and* have
    // roughly balanced left/right padding. Lines that are merely indented
    // (algorithm bodies, code blocks) often sit in the central 40% of the
    // page width but are anchored to the left margin — `is_centered` here
    // rejects those by demanding both small center-offset AND symmetric
    // padding. Without this, a 4-word indented body like
    // `"vote ← Alg. 6.CreateVote(...)"` would trigger the skip below and
    // disappear from the translated output.
    let centered = {
        let line_center = (rect.left + rect.right) as f32 * 0.5;
        let center_offset = (line_center - page.width_pts * 0.5).abs();
        let left_pad = rect.left as f32;
        let right_pad = (page.width_pts - rect.right as f32).max(0.0);
        let pad_imbalance = (left_pad - right_pad).abs();
        let avg_pad = (left_pad + right_pad) * 0.5;
        center_offset < page.width_pts * 0.05 && (avg_pad <= 0.0 || pad_imbalance < avg_pad * 0.3)
    };

    // Display equations/code signatures in papers should stay verbatim. They
    // are usually centered, symbol-heavy, or underscore-heavy, and contain
    // little ordinary prose.
    (centered && symbols > 0 && words <= 8)
        || (symbols >= 2 && letters < 24)
        || (underscores >= 2 && words <= 8)
}

/// True if `text` contains a `<lower>+ . <lower>+` sequence — i.e. a field
/// access like `p.view` or `local_tip.block`. Used to flag code-like lines
/// before they hit the equation-skip heuristics.
fn has_dotted_identifier(text: &str) -> bool {
    let chars: Vec<char> = text.chars().collect();
    for i in 1..chars.len().saturating_sub(1) {
        if chars[i] != '.' {
            continue;
        }
        let prev_is_lower = chars[i - 1].is_ascii_lowercase();
        let next_is_lower = chars[i + 1].is_ascii_lowercase();
        if prev_is_lower && next_is_lower {
            return true;
        }
    }
    false
}

/// Count maximal runs of `min_len` or more consecutive alphabetic chars.
/// `count_long_alpha_runs("if |votes[proposal id]| then", 4)` returns 3
/// (`votes`, `proposal`, `then`). A real word in code or prose; equations
/// rarely have any.
fn count_long_alpha_runs(text: &str, min_len: usize) -> usize {
    let mut count = 0;
    let mut run = 0usize;
    for c in text.chars() {
        if c.is_alphabetic() {
            run += 1;
        } else {
            if run >= min_len {
                count += 1;
            }
            run = 0;
        }
    }
    if run >= min_len {
        count += 1;
    }
    count
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
