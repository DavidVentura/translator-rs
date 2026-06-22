use image::{Rgb, Rgba};
use translator_core::BackgroundMode;
use translator_core::ocr::{
    BoldRange, OrientedRect, OverlayColors, OverlayLayoutHints, OverlayLayoutMode,
    PreparedImageOverlay, PreparedTextBlock, PreparedTextLine, RasterImageMut, ReadingOrder,
    TextBlock, argb, channel_b, channel_g, channel_r, clamp_rect,
};

use crate::color_matting::{BG_BLOCK, background_field, dilate, fill_radius, still_fg_argb};

fn overlay_layout_hints(block: &TextBlock, reading_order: ReadingOrder) -> OverlayLayoutHints {
    let layout_mode = match reading_order {
        ReadingOrder::LeftToRight => OverlayLayoutMode::PerLine,
        ReadingOrder::TopToBottomRightToLeft => OverlayLayoutMode::VerticalBlockRect,
    };
    let suggested_font_size_px = if block.lines.is_empty() {
        match reading_order {
            ReadingOrder::LeftToRight => block.bounds().height() as f32,
            ReadingOrder::TopToBottomRightToLeft => block.bounds().width() as f32,
        }
    } else {
        // For per-line layout, use the oriented box's height (perpendicular to reading
        // direction) so tilted lines aren't sized off their inflated AABB height. For block-
        // rect (top-to-bottom) layout the box is axis-aligned, so the oriented and AABB widths
        // are equal.
        let total = block
            .lines
            .iter()
            .map(|line| match reading_order {
                ReadingOrder::LeftToRight => line.oriented_box.height,
                ReadingOrder::TopToBottomRightToLeft => line.bounding_box.width() as f32,
            })
            .sum::<f32>();
        total / block.lines.len() as f32
    };
    OverlayLayoutHints {
        layout_mode,
        suggested_font_size_px,
    }
}

fn erase_text_region(
    image: &mut RasterImageMut,
    oriented: OrientedRect,
    background_mode: BackgroundMode,
    ink_mask: Option<&[bool]>,
) -> OverlayColors {
    // The oriented rect carries the same DB-unclip + DET_BOX_BORDER inflation the AABB path
    // applies (see `oriented_rect_from_contour`), so it reliably covers ascenders/descenders
    // without spilling sideways the way an AABB does for tilted lines.
    match background_mode {
        BackgroundMode::WhiteOnBlack => {
            let colors = OverlayColors {
                background_argb: argb(0, 0, 0),
                foreground_argb: argb(255, 255, 255),
            };
            image.fill_oriented_rect(oriented, colors.background_argb);
            colors
        }
        BackgroundMode::BlackOnWhite => {
            let colors = OverlayColors {
                background_argb: argb(255, 255, 255),
                foreground_argb: argb(0, 0, 0),
            };
            image.fill_oriented_rect(oriented, colors.background_argb);
            colors
        }
        BackgroundMode::AutoDetect => {
            // No algorithmic colour guess. When the ink model matted the line we
            // erase only its ink pixels (the page texture/gradient shows through
            // untouched) and colour the translated text with the real ink-median
            // foreground, the same derivation the live overlay uses. Lines the
            // model couldn't matte fall back to a flat white-on-black pill.
            if let Some(mask) = ink_mask {
                if let Some(fg) = matte_erase_oriented(image, oriented, mask) {
                    return OverlayColors {
                        background_argb: argb(0, 0, 0),
                        foreground_argb: fg,
                    };
                }
            }
            let colors = OverlayColors {
                background_argb: argb(0, 0, 0),
                foreground_argb: argb(255, 255, 255),
            };
            image.fill_oriented_rect(oriented, colors.background_argb);
            colors
        }
    }
}

/// Erase the ink pixels the model matte marked inside `oriented`'s AABB and
/// replace them with a reconstructed background field, leaving everything else
/// untouched. `ink_mask` is the full-image union matte (`y * width + x`).
/// Returns the median colour of the erased ink pixels as an opaque ARGB — the
/// real ink colour, for the translated text's foreground (`None` when too few
/// ink pixels to estimate). This is the same derivation
/// [`translator_core::color_matting::mat_detections`] uses for the live overlay's
/// `fg_argb`, so still and live colour text identically.
fn matte_erase_oriented(
    image: &mut RasterImageMut,
    oriented: OrientedRect,
    ink_mask: &[bool],
) -> Option<u32> {
    let aabb = clamp_rect(oriented.to_aabb(), image.width, image.height)?;
    let aw = aabb.right - aabb.left;
    let ah = aabb.bottom - aabb.top;
    if aw == 0 || ah == 0 {
        return None;
    }
    let w = image.width;
    let mut pixels = vec![Rgba([0u8; 4]); (aw * ah) as usize];
    let mut sub = vec![false; (aw * ah) as usize];
    {
        let view = image.as_image();
        for ly in 0..ah {
            for lx in 0..aw {
                let (gx, gy) = (aabb.left + lx, aabb.top + ly);
                let c = view.pixel_argb(gx, gy);
                let i = (ly * aw + lx) as usize;
                pixels[i] = Rgba([channel_r(c), channel_g(c), channel_b(c), 255]);
                sub[i] = ink_mask[(gy * w + gx) as usize];
            }
        }
    }

    // Foreground colour via the shared still-path derivation (between-stroke/margin background
    // + fg_from_samples), so still and live colour text identically and the decision is
    // reproducible outside the erase.
    let fg = still_fg_argb(&pixels, &sub, aw, ah);

    // Grow the fill set by a height-proportional radius so the original ink's
    // anti-aliased rim is replaced too (the matte edge sits just inside it).
    let sub = dilate(&sub, aw, ah, fill_radius(oriented.height));
    let bg: Vec<Rgb<u8>> = background_field(&pixels, &sub, aw, ah, BG_BLOCK);

    for ly in 0..ah {
        for lx in 0..aw {
            let i = (ly * aw + lx) as usize;
            if !sub[i] {
                continue;
            }
            let b = bg[i];
            let bytes = argb(b[0], b[1], b[2]).to_ne_bytes();
            let idx = (((aabb.top + ly) * w + (aabb.left + lx)) * 4) as usize;
            image.rgba[idx..idx + 4].copy_from_slice(&bytes);
        }
    }
    fg
}

pub fn prepare_overlay_image(
    rgba_bytes: &[u8],
    width: u32,
    height: u32,
    blocks: &[TextBlock],
    translated_blocks: &[String],
    block_bold_ranges: &[Vec<BoldRange>],
    background_mode: BackgroundMode,
    reading_order: ReadingOrder,
    ink_mask: Option<&[bool]>,
) -> Result<PreparedImageOverlay, String> {
    let mut image = RasterImageMut::new(rgba_bytes, width, height)?;
    let mut prepared_blocks = Vec::with_capacity(blocks.len());

    for (index, (block, translated_text)) in blocks.iter().zip(translated_blocks.iter()).enumerate()
    {
        let block_bounds = block.bounds();
        let layout_hints = overlay_layout_hints(block, reading_order);
        // Per-word bold carried through translation (byte ranges into the translated text).
        let bold_ranges = block_bold_ranges.get(index).cloned().unwrap_or_default();
        match reading_order {
            ReadingOrder::LeftToRight => {
                let mut prepared_lines = Vec::with_capacity(block.lines.len());
                let mut block_background = argb(255, 255, 255);
                let mut block_foreground = argb(0, 0, 0);
                for (index, line) in block.lines.iter().enumerate() {
                    let colors =
                        erase_text_region(&mut image, line.oriented_box, background_mode, ink_mask);
                    if index == 0 {
                        block_background = colors.background_argb;
                        block_foreground = colors.foreground_argb;
                    }
                    prepared_lines.push(PreparedTextLine {
                        text: line.text.clone(),
                        bounding_box: line.bounding_box,
                        oriented_box: line.oriented_box,
                        word_rects: line.word_rects.clone(),
                        background_argb: colors.background_argb,
                        foreground_argb: colors.foreground_argb,
                    });
                }
                prepared_blocks.push(PreparedTextBlock {
                    source_text: block.source_text(),
                    translated_text: translated_text.clone(),
                    bounding_box: block_bounds,
                    lines: prepared_lines,
                    layout_hints,
                    background_argb: block_background,
                    foreground_argb: block_foreground,
                    bold_ranges: bold_ranges.clone(),
                });
            }
            ReadingOrder::TopToBottomRightToLeft => {
                // Block-rect (CJK vertical) layout: the per-block region is the union of
                // possibly differently-rotated lines, so rotation doesn't carry up. Erase the
                // block AABB unrotated.
                let colors = erase_text_region(
                    &mut image,
                    OrientedRect::axis_aligned(block_bounds),
                    background_mode,
                    ink_mask,
                );
                let prepared_lines = block
                    .lines
                    .iter()
                    .map(|line| PreparedTextLine {
                        text: line.text.clone(),
                        bounding_box: line.bounding_box,
                        oriented_box: line.oriented_box,
                        word_rects: line.word_rects.clone(),
                        background_argb: colors.background_argb,
                        foreground_argb: colors.foreground_argb,
                    })
                    .collect();
                prepared_blocks.push(PreparedTextBlock {
                    source_text: block.source_text(),
                    translated_text: translated_text.clone(),
                    bounding_box: block_bounds,
                    lines: prepared_lines,
                    layout_hints,
                    background_argb: colors.background_argb,
                    foreground_argb: colors.foreground_argb,
                    bold_ranges,
                });
            }
        }
    }

    Ok(PreparedImageOverlay {
        rgba_bytes: image.rgba,
        width,
        height,
        extracted_text: blocks
            .iter()
            .map(TextBlock::source_text)
            .collect::<Vec<_>>()
            .join("\n"),
        translated_text: translated_blocks.join("\n"),
        blocks: prepared_blocks,
        source_words: Vec::new(),
        translated_words: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use translator_core::BackgroundMode;
    use translator_core::ocr::{OverlayLayoutMode, ReadingOrder, Rect, TextBlock, TextLine};

    #[test]
    fn prepare_overlay_image_erases_left_to_right_lines_without_touching_gap() {
        let width = 8;
        let height = 8;
        let gap_color = 0xFF12_34_56u32;
        let mut rgba = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            let color = if y == 3 { gap_color } else { 0xFF00_0000 };
            for _ in 0..width {
                rgba.extend_from_slice(&color.to_ne_bytes());
            }
        }

        let top_rect = Rect {
            left: 1,
            top: 1,
            right: 7,
            bottom: 3,
        };
        let bottom_rect = Rect {
            left: 1,
            top: 4,
            right: 7,
            bottom: 6,
        };
        let blocks = vec![TextBlock {
            lines: vec![
                TextLine {
                    text: "top".to_string(),
                    bounding_box: top_rect,
                    oriented_box: translator_core::ocr::OrientedRect::axis_aligned(top_rect),
                    tight_box: translator_core::ocr::OrientedRect::axis_aligned(top_rect),
                    word_rects: vec![top_rect],
                    bold_ranges: Vec::new(),
                },
                TextLine {
                    text: "bottom".to_string(),
                    bounding_box: bottom_rect,
                    oriented_box: translator_core::ocr::OrientedRect::axis_aligned(bottom_rect),
                    tight_box: translator_core::ocr::OrientedRect::axis_aligned(bottom_rect),
                    word_rects: vec![bottom_rect],
                    bold_ranges: Vec::new(),
                },
            ],
        }];
        let translated = vec!["translated text".to_string()];

        let prepared = prepare_overlay_image(
            &rgba,
            width,
            height,
            &blocks,
            &translated,
            &[],
            BackgroundMode::BlackOnWhite,
            ReadingOrder::LeftToRight,
            None,
        )
        .expect("overlay should prepare");

        let gap_index = ((3 * width + 2) * 4) as usize;
        let gap_pixel = u32::from_ne_bytes(
            prepared.rgba_bytes[gap_index..gap_index + 4]
                .try_into()
                .expect("gap pixel"),
        );
        assert_eq!(gap_pixel, gap_color);

        let erased_index = ((1 * width + 2) * 4) as usize;
        let erased_pixel = u32::from_ne_bytes(
            prepared.rgba_bytes[erased_index..erased_index + 4]
                .try_into()
                .expect("erased pixel"),
        );
        assert_eq!(erased_pixel, 0xFFFF_FFFF);
        assert_eq!(prepared.blocks[0].lines.len(), 2);
        assert_eq!(prepared.blocks[0].lines[0].foreground_argb, 0xFF00_0000);
        assert_eq!(
            prepared.blocks[0].layout_hints.layout_mode,
            OverlayLayoutMode::PerLine
        );
        assert_eq!(prepared.blocks[0].layout_hints.suggested_font_size_px, 2.0);
    }

    #[test]
    fn block_bold_ranges_pass_through_to_prepared() {
        let rgba = vec![0xFFu8; 8 * 8 * 4];
        let rect = Rect {
            left: 1,
            top: 1,
            right: 7,
            bottom: 6,
        };
        let make = || TextBlock {
            lines: vec![TextLine {
                text: "x".to_string(),
                bounding_box: rect,
                oriented_box: translator_core::ocr::OrientedRect::axis_aligned(rect),
                tight_box: translator_core::ocr::OrientedRect::axis_aligned(rect),
                word_rects: vec![rect],
                bold_ranges: Vec::new(),
            }],
        };
        let blocks = vec![make(), make()];
        let translated = vec!["a".to_string(), "b".to_string()];
        let block_bold_ranges = vec![
            vec![translator_core::ocr::BoldRange { start: 0, end: 1 }],
            Vec::new(),
        ];
        let prepared = prepare_overlay_image(
            &rgba,
            8,
            8,
            &blocks,
            &translated,
            &block_bold_ranges,
            BackgroundMode::BlackOnWhite,
            ReadingOrder::LeftToRight,
            None,
        )
        .expect("overlay should prepare");
        assert_eq!(
            prepared.blocks[0].bold_ranges,
            vec![translator_core::ocr::BoldRange { start: 0, end: 1 }],
            "bold range carried to the prepared block"
        );
        assert!(
            prepared.blocks[1].bold_ranges.is_empty(),
            "regular block has no bold ranges"
        );
    }

    #[test]
    fn autodetect_without_matte_falls_back_to_white_on_black() {
        // No ink matte: AutoDetect no longer samples the image — it flat-fills the
        // line black and colours the translated text white.
        let width = 8;
        let height = 8;
        let mut rgba = Vec::with_capacity((width * height * 4) as usize);
        for _ in 0..(width * height) {
            rgba.extend_from_slice(&0xFFFF_FFFFu32.to_ne_bytes());
        }
        let rect = Rect {
            left: 1,
            top: 1,
            right: 7,
            bottom: 6,
        };
        let blocks = vec![TextBlock {
            lines: vec![TextLine {
                text: "hi".to_string(),
                bounding_box: rect,
                oriented_box: translator_core::ocr::OrientedRect::axis_aligned(rect),
                tight_box: translator_core::ocr::OrientedRect::axis_aligned(rect),
                word_rects: vec![rect],
                bold_ranges: Vec::new(),
            }],
        }];

        let prepared = prepare_overlay_image(
            &rgba,
            width,
            height,
            &blocks,
            &["x".to_string()],
            &[],
            BackgroundMode::AutoDetect,
            ReadingOrder::LeftToRight,
            None,
        )
        .expect("overlay should prepare");

        let idx = ((2 * width + 3) * 4) as usize;
        let erased = u32::from_ne_bytes(prepared.rgba_bytes[idx..idx + 4].try_into().unwrap());
        assert_eq!(erased, 0xFF00_0000);
        assert_eq!(prepared.blocks[0].lines[0].foreground_argb, 0xFFFF_FFFF);
    }
}
