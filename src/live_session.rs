//! Cross-platform live-translate session state.
//!
//! Holds the orchestration state that's shared between the Android
//! `LivePlanarTracker` (uniffi-exposed) and the `surface_sim`
//! desktop binary. The goal is **one source of truth** for the
//! pipeline: tracking → detection → surface-map update → recognition
//! → translation → overlay update. Without this, every new feature
//! (re-OCR triggers, pixel stripes, etc.) has to be implemented
//! twice and the two implementations drift.
//!
//! This module starts small: it owns the persistent `SurfaceMap`
//! and is grown in subsequent phases to encompass the engine,
//! overlay store, matting cache, and the orchestration methods
//! themselves. See FUTURE_SURFACE_MAP.md for the bigger plan.

use std::sync::Mutex;

use crate::ocr::OrientedRect;
use crate::surface_map::{AddResult, SurfaceLineId, SurfaceLineObservation, SurfaceMap};

/// One rasterized overlay item resident across composite calls. The
/// caller hashes the source content (strips + texts + language) and
/// only re-rasterizes items whose hash changed, so dense pages with
/// stable content stay cheap to render.
#[derive(Clone)]
pub struct OverlayItem {
    pub id: u64,
    /// RGBA bitmap in canonical (surface) coords.
    pub bitmap: Vec<u8>,
    pub width: u32,
    pub height: u32,
    /// Where the bitmap's top-left sits in surface coords. The
    /// compositor warps from this origin through the per-frame H.
    pub surface_origin_x: f32,
    pub surface_origin_y: f32,
    /// Hash of (strips + display text + language). Used to skip
    /// re-raster when content is unchanged across acquires.
    pub content_hash: u64,
}

/// Lifetime-bound state shared between platform wrappers (Android
/// `LivePlanarTracker`, desktop `surface_sim`). One instance per
/// active session; cleared on reset (tap-to-focus, language change).
pub struct LiveSession {
    /// Per-physical-line storage across acquires. The `same_line`
    /// predicate's geometric matching makes this safe across
    /// scene changes — stale entries from a different scene stay
    /// inert (no false merges) until `clear()` is called.
    pub surface_map: Mutex<SurfaceMap>,
    /// Resident rasterized overlays, keyed by block id. Populated
    /// by `upsert_overlay_item` and consumed by the compositor.
    pub overlay_items: Mutex<Vec<OverlayItem>>,
}

impl LiveSession {
    pub fn new() -> Self {
        Self {
            surface_map: Mutex::new(SurfaceMap::new()),
            overlay_items: Mutex::new(Vec::new()),
        }
    }

    /// Drop all session state. Caller invokes on tap-to-focus,
    /// language change, or any other coarse-grained reset signal.
    pub fn clear(&self) {
        if let Ok(mut map) = self.surface_map.lock() {
            map.clear();
        }
        if let Ok(mut items) = self.overlay_items.lock() {
            items.clear();
        }
    }

    pub fn clear_overlays(&self) {
        if let Ok(mut items) = self.overlay_items.lock() {
            items.clear();
        }
    }

    /// Feed a batch of detections (in surface coords) into the
    /// surface map and return per-detection outcomes the caller
    /// uses to (a) decide which detections need recognition, and
    /// (b) push rec results back via [`Self::ingest_rec`].
    ///
    /// `source_language` is used as the default for newly-created
    /// lines; existing lines keep their previously-recorded
    /// language unless the observation carries a non-empty value.
    ///
    /// Replaces the open-coded `for entry in entries: map.add_or_merge`
    /// loops in `uniffi_catalog::run_acquire_pipeline` and the
    /// simulator's `run_detection_cycle`.
    pub fn observe_detections(
        &self,
        detections: &[OrientedRect],
        source_language: &str,
    ) -> Vec<DetectionOutcome> {
        let mut out = Vec::with_capacity(detections.len());
        let mut map = match self.surface_map.lock() {
            Ok(m) => m,
            Err(_) => {
                return detections
                    .iter()
                    .map(|_| DetectionOutcome::poisoned())
                    .collect();
            }
        };
        for d in detections {
            let obs = SurfaceLineObservation {
                bbox: d.clone(),
                source_text: String::new(),
                translated_text: String::new(),
                source_language: source_language.to_string(),
            };
            let res = map.add_or_merge(obs);
            let needs_rec = res.needs_rec();
            let line_id = res.id();
            let kind = AddResultKind::from(&res);
            let cached_source_text = if !needs_rec {
                map.get(line_id)
                    .map(|l| l.source_text.clone())
                    .unwrap_or_default()
            } else {
                String::new()
            };
            let cached_source_language = if !needs_rec {
                map.get(line_id)
                    .map(|l| l.source_language.clone())
                    .unwrap_or_default()
            } else {
                String::new()
            };
            out.push(DetectionOutcome {
                line_id,
                kind,
                needs_rec,
                cached_source_text,
                cached_source_language,
            });
        }
        out
    }

    /// Push a single recognized line back into the map: store the
    /// text and language, and snapshot the line's current u-extent
    /// as "rec just saw up to here" so future observations that
    /// extend past it trigger re-recognition.
    pub fn ingest_rec(&self, line_id: SurfaceLineId, source_text: &str, source_language: &str) {
        if let Ok(mut map) = self.surface_map.lock() {
            if let Some(line) = map.get_mut(line_id) {
                line.source_text = source_text.to_string();
                if !source_language.is_empty() {
                    line.source_language = source_language.to_string();
                }
                line.record_rec_extent();
            }
        }
    }

    /// Push translated text back into a set of lines (all
    /// recipients receive the same string — caller has already
    /// performed block-level translation across the joined source
    /// strings).
    pub fn ingest_translation(&self, line_ids: &[SurfaceLineId], translated: &str) {
        if let Ok(mut map) = self.surface_map.lock() {
            for &id in line_ids {
                if let Some(line) = map.get_mut(id) {
                    line.translated_text = translated.to_string();
                }
            }
        }
    }
}

/// Per-detection result from [`LiveSession::observe_detections`].
#[derive(Clone, Debug)]
pub struct DetectionOutcome {
    pub line_id: SurfaceLineId,
    pub kind: AddResultKind,
    /// True when caller should run recognition for this line. False
    /// for cache hits (`MergedUnchanged` on a line with text).
    pub needs_rec: bool,
    /// Cached source text from a prior rec, when `!needs_rec`.
    /// Empty otherwise.
    pub cached_source_text: String,
    /// Cached source language from a prior rec, when `!needs_rec`.
    /// Empty otherwise.
    pub cached_source_language: String,
}

impl DetectionOutcome {
    fn poisoned() -> Self {
        Self {
            line_id: 0,
            kind: AddResultKind::Created,
            needs_rec: true,
            cached_source_text: String::new(),
            cached_source_language: String::new(),
        }
    }
}

/// Human-readable variant of `AddResult` for diagnostics + visual
/// color coding in the simulator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AddResultKind {
    Created,
    MergedAndExtended,
    MergedUnchanged,
}

impl From<&AddResult> for AddResultKind {
    fn from(r: &AddResult) -> Self {
        match r {
            AddResult::Created(_) => AddResultKind::Created,
            AddResult::MergedAndExtended(_) => AddResultKind::MergedAndExtended,
            AddResult::MergedUnchanged(_) => AddResultKind::MergedUnchanged,
        }
    }
}

impl Default for LiveSession {
    fn default() -> Self {
        Self::new()
    }
}

// =====================================================================
// Block rendering and grouping. Pure functions used by both Android's
// run_acquire_pipeline and the desktop surface_sim binary to convert
// (strips, display_text, language) into a single rasterized block
// overlay bitmap.
// =====================================================================

/// Inflation of the detector's "tight" rect into the visible pill's
/// vertical extent. Tight is glyph-only; we leave headroom for
/// ascenders/descenders so the pill looks like it covers the line.
pub const TIGHT_VERTICAL_INFLATE: f32 = 2.4;

/// Horizontal padding (per side) on the visible pill vs the
/// detector's tight rect. Keeps glyph edges off the rounded corner.
pub const HORIZONTAL_PAD_PX: f32 = 8.0;

/// Extra padding on the block's bitmap AABB to give rounded-corner
/// antialiasing room.
pub const ITEM_BITMAP_PAD_PX: f32 = 4.0;

/// Per-item raster result: an RGBA bitmap with bounded dimensions
/// plus the surface-coord position of its top-left pixel.
#[derive(Clone, Debug)]
pub struct ItemRaster {
    pub bitmap: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub surface_origin_x: f32,
    pub surface_origin_y: f32,
}

/// Unpack a `0xAARRGGBB` value into the `[r, g, b, a]` byte tuple the
/// rasterizer's per-pixel blender expects.
pub fn argb_to_rgba_bytes(argb: u32) -> [u8; 4] {
    let a = ((argb >> 24) & 0xff) as u8;
    let r = ((argb >> 16) & 0xff) as u8;
    let g = ((argb >> 8) & 0xff) as u8;
    let b = (argb & 0xff) as u8;
    [r, g, b, a]
}

/// Snap sibling line strips within one paragraph block to a shared
/// column in the block's rotated basis. Detector noise gives each
/// strip its own `cx`/`width`/`angle_radians`; without this step
/// the per-line pills form a left-edge "staircase" rather than
/// following the paragraph's column on a tilted page. See
/// FUTURE_SURFACE_MAP.md → "Per-block column alignment".
///
/// Pure in-plane rotation handling: out-of-plane perspective is
/// out of scope (see FUTURE_ANCHOR_RECTIFICATION.md).
pub fn normalize_block_visuals_rotated_basis(visuals: &mut [OrientedRect]) {
    if visuals.len() < 2 {
        return;
    }
    let mut sum_cos = 0.0_f32;
    let mut sum_sin = 0.0_f32;
    let mut total_w = 0.0_f32;
    for v in visuals.iter() {
        let w = v.width.max(0.0);
        sum_cos += v.angle_radians.cos() * w;
        sum_sin += v.angle_radians.sin() * w;
        total_w += w;
    }
    if total_w <= 0.0 {
        return;
    }
    let theta = sum_sin.atan2(sum_cos);
    let max_dev = 10.0_f32.to_radians();
    for v in visuals.iter() {
        let mut d = v.angle_radians - theta;
        while d > std::f32::consts::PI {
            d -= 2.0 * std::f32::consts::PI;
        }
        while d < -std::f32::consts::PI {
            d += 2.0 * std::f32::consts::PI;
        }
        if d.abs() > max_dev {
            return;
        }
    }
    let c = theta.cos();
    let s = theta.sin();
    let mut u_left = f32::INFINITY;
    let mut u_right = f32::NEG_INFINITY;
    for v in visuals.iter() {
        for (x, y) in v.corners() {
            let u = x * c + y * s;
            if u < u_left {
                u_left = u;
            }
            if u > u_right {
                u_right = u;
            }
        }
    }
    if !(u_right > u_left) {
        return;
    }
    let u_centre = 0.5 * (u_left + u_right);
    let block_width = u_right - u_left;
    for v in visuals.iter_mut() {
        let v_axis = -v.cx * s + v.cy * c;
        v.cx = u_centre * c - v_axis * s;
        v.cy = u_centre * s + v_axis * c;
        v.width = block_width;
        v.angle_radians = theta;
    }
}

/// Rasterize a *block*: N per-line strips share one bitmap, one
/// `translated_text`, and one set of background fills (one per strip).
/// The text gets reflowed across the strips by `image_render` using
/// the strips' widths as target line widths.
///
/// `strips` must be ordered top-to-bottom. `display_text` is the
/// translation; when empty, the block renders as a "pending"
/// placeholder (per-strip bg fills, no glyphs). `font_provider`
/// supplies typefaces — Android passes `AndroidFontProvider`, the
/// simulator can pass any `FontProvider` impl (or a stub if it
/// doesn't need text rendering yet).
pub fn render_block_bitmap(
    strips: &[OrientedRect],
    matted_strips: &[Option<crate::color_matting::MattedStrip>],
    display_text: &str,
    language: &str,
    font_provider: &dyn crate::font_provider::FontProvider,
) -> Option<ItemRaster> {
    use crate::ocr::{
        OverlayLayoutHints, OverlayLayoutMode, PreparedImageOverlay, PreparedTextBlock,
        PreparedTextLine, Rect,
    };
    if strips.is_empty() {
        return None;
    }

    let mut visuals: Vec<OrientedRect> = strips
        .iter()
        .filter_map(|s| {
            let v = OrientedRect {
                cx: s.cx,
                cy: s.cy,
                width: s.width + 2.0 * HORIZONTAL_PAD_PX,
                height: s.height * TIGHT_VERTICAL_INFLATE,
                angle_radians: s.angle_radians,
            };
            if v.width <= 0.0 || v.height <= 0.0 {
                None
            } else {
                Some(v)
            }
        })
        .collect();
    if visuals.is_empty() {
        return None;
    }
    normalize_block_visuals_rotated_basis(&mut visuals);

    let (mut min_x, mut min_y) = (f32::INFINITY, f32::INFINITY);
    let (mut max_x, mut max_y) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for v in &visuals {
        for (x, y) in v.corners() {
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
    }
    let pad = ITEM_BITMAP_PAD_PX;
    let origin_x = (min_x - pad).max(0.0);
    let origin_y = (min_y - pad).max(0.0);
    let bitmap_w = ((max_x + pad - origin_x).ceil() as i32).max(1) as u32;
    let bitmap_h = ((max_y + pad - origin_y).ceil() as i32).max(1) as u32;

    let pixels = (bitmap_w as usize) * (bitmap_h as usize);
    let mut rgba = vec![0u8; pixels * 4];
    let default_bg = [0x10, 0x10, 0x10, 0xC8];
    let visuals_local: Vec<OrientedRect> = visuals
        .iter()
        .map(|v| OrientedRect {
            cx: v.cx - origin_x,
            cy: v.cy - origin_y,
            width: v.width,
            height: v.height,
            angle_radians: v.angle_radians,
        })
        .collect();
    for (i, v) in visuals_local.iter().enumerate() {
        let strip_color = matted_strips
            .get(i)
            .and_then(|m| m.as_ref())
            .and_then(|m| m.bg_uniform_argb)
            .map(argb_to_rgba_bytes)
            .unwrap_or(default_bg);
        crate::planar_engine::fill_oriented_rect_blended(
            &mut rgba, bitmap_w, bitmap_h, v, strip_color,
        );
    }
    let foreground_argb: u32 = matted_strips
        .iter()
        .find_map(|m| m.as_ref().map(|s| s.ink_is_dark))
        .map(|dark| if dark { 0xFF10_1010 } else { 0xFFFF_FFFF })
        .unwrap_or(0xFFFF_FFFF);

    if display_text.trim().is_empty() {
        return Some(ItemRaster {
            bitmap: rgba,
            width: bitmap_w,
            height: bitmap_h,
            surface_origin_x: origin_x,
            surface_origin_y: origin_y,
        });
    }

    let lines: Vec<PreparedTextLine> = visuals_local
        .iter()
        .map(|v| {
            let text_box = OrientedRect {
                cx: v.cx,
                cy: v.cy,
                width: (v.width - 2.0 * crate::planar_engine::OVERLAY_TEXT_HORIZONTAL_INSET_PX)
                    .max(1.0),
                height: v.height,
                angle_radians: v.angle_radians,
            };
            let aabb = text_box.to_aabb();
            let bbox = Rect {
                left: aabb.left.min(bitmap_w.saturating_sub(1)),
                top: aabb.top.min(bitmap_h.saturating_sub(1)),
                right: aabb.right.min(bitmap_w),
                bottom: aabb.bottom.min(bitmap_h),
            };
            PreparedTextLine {
                text: String::new(),
                bounding_box: bbox.clone(),
                oriented_box: text_box,
                word_rects: vec![bbox],
                background_argb: 0,
                foreground_argb,
            }
        })
        .collect();
    let suggested_font_px = visuals
        .iter()
        .map(|v| v.height)
        .fold(0.0_f32, f32::max)
        .clamp(10.0, 120.0);
    let block_bbox = Rect {
        left: 0,
        top: 0,
        right: bitmap_w,
        bottom: bitmap_h,
    };
    let block = PreparedTextBlock {
        source_text: String::new(),
        translated_text: display_text.to_string(),
        bounding_box: block_bbox,
        lines,
        layout_hints: OverlayLayoutHints {
            layout_mode: OverlayLayoutMode::PerLine,
            suggested_font_size_px: suggested_font_px,
        },
        background_argb: 0,
        foreground_argb,
    };

    let prepared = PreparedImageOverlay {
        rgba_bytes: rgba,
        width: bitmap_w,
        height: bitmap_h,
        extracted_text: String::new(),
        translated_text: String::new(),
        blocks: vec![block],
    };
    let opts = crate::image_render::RenderOptions {
        language: language.to_string(),
        min_font_size_px: 6.0,
    };
    let final_bytes = crate::image_render::render_overlay(&prepared, font_provider, &opts).ok()?;
    Some(ItemRaster {
        bitmap: final_bytes,
        width: bitmap_w,
        height: bitmap_h,
        surface_origin_x: origin_x,
        surface_origin_y: origin_y,
    })
}

/// Group `SurfaceLine`s into translation blocks (paragraphs) via the
/// shared OCR grouping. Returns indices into the input slice.
pub fn group_surface_lines_into_blocks(lines: &[crate::surface_map::SurfaceLine]) -> Vec<Vec<usize>> {
    use crate::ocr::TextLine;
    if lines.is_empty() {
        return Vec::new();
    }
    let text_lines: Vec<TextLine> = lines
        .iter()
        .map(|l| TextLine {
            text: String::new(),
            bounding_box: l.bbox.to_aabb(),
            oriented_box: l.bbox.clone(),
            tight_box: l.bbox.clone(),
            word_rects: Vec::new(),
        })
        .collect();
    let blocks = crate::ocr::group_live_lines_into_blocks(text_lines);
    blocks
        .into_iter()
        .map(|b| {
            b.lines
                .iter()
                .filter_map(|tl| lines.iter().position(|sl| sl.bbox == tl.tight_box))
                .collect::<Vec<usize>>()
        })
        .filter(|v: &Vec<usize>| !v.is_empty())
        .collect()
}

/// FNV-1a 64-bit hash of the sorted line ids. Identical line-sets
/// across acquires hash to the same id, so `upsert_overlay_block`'s
/// content-hash cache skips re-raster for unchanged blocks. The
/// high bit is set so block ids generated this way are distinct
/// from any legacy `next_entry_id`-derived ids.
pub fn stable_block_id(sorted_line_ids: &[crate::surface_map::SurfaceLineId]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &id in sorted_line_ids {
        for byte in id.to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    hash | (1u64 << 63)
}
