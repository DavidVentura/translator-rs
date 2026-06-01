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

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use crate::api::LanguageCode;
use crate::color_matting::MattedStrip;
use crate::homography::{invert, project};
use crate::live_frame::OrientedImage;
use crate::ocr::{DetectedTextBox, OcrSourceSelection, OrientedRect, RecognizedTextLine};
use crate::routing::MixedTextTranslationResult;
use crate::surface_map::{AddResult, SurfaceLineId, SurfaceLineObservation, SurfaceMap};

/// Anchor identifier as the engine emits it. Mirrors
/// `planar_engine::AnchorId` (u64) but kept here as a plain alias to
/// avoid pulling the engine module into `live_session`'s public
/// surface — the session doesn't care how anchors are produced.
pub type AnchorId = u64;

/// Axis-aligned bounding box in surface coords. Used by the refresh
/// trigger to track which surface region has already been run
/// through the detector and skip refreshes whose viewport is
/// entirely contained in that region (no new info to gain).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    pub min_x: f32,
    pub min_y: f32,
    pub max_x: f32,
    pub max_y: f32,
}

impl Aabb {
    /// Smallest AABB enclosing the given points. Returns `None` on an
    /// empty iterator or a non-finite coordinate.
    pub fn from_points(points: impl IntoIterator<Item = (f32, f32)>) -> Option<Self> {
        let mut iter = points.into_iter();
        let (x0, y0) = iter.next()?;
        if !x0.is_finite() || !y0.is_finite() {
            return None;
        }
        let mut aabb = Self {
            min_x: x0,
            min_y: y0,
            max_x: x0,
            max_y: y0,
        };
        for (x, y) in iter {
            if !x.is_finite() || !y.is_finite() {
                return None;
            }
            if x < aabb.min_x {
                aabb.min_x = x;
            }
            if y < aabb.min_y {
                aabb.min_y = y;
            }
            if x > aabb.max_x {
                aabb.max_x = x;
            }
            if y > aabb.max_y {
                aabb.max_y = y;
            }
        }
        Some(aabb)
    }

    pub fn union_inplace(&mut self, other: &Aabb) {
        self.min_x = self.min_x.min(other.min_x);
        self.min_y = self.min_y.min(other.min_y);
        self.max_x = self.max_x.max(other.max_x);
        self.max_y = self.max_y.max(other.max_y);
    }

    /// True when `inner` fits entirely inside `self` after inflating
    /// `self` by `pad` on each side. The padding absorbs RANSAC
    /// residual / detector noise so a viewport that's *practically*
    /// covered doesn't trip the predicate via sub-pixel jitter.
    pub fn contains_inflated(&self, inner: &Aabb, pad: f32) -> bool {
        inner.min_x >= self.min_x - pad
            && inner.min_y >= self.min_y - pad
            && inner.max_x <= self.max_x + pad
            && inner.max_y <= self.max_y + pad
    }

    pub fn area(&self) -> f32 {
        (self.max_x - self.min_x).max(0.0) * (self.max_y - self.min_y).max(0.0)
    }

    /// AABB of the intersection. Returns a zero-area `Aabb` (with
    /// degenerate min ≥ max) when the boxes don't overlap; callers
    /// should use [`Self::area`] to distinguish "no intersection"
    /// from a real overlap.
    pub fn intersect(&self, other: &Aabb) -> Aabb {
        Aabb {
            min_x: self.min_x.max(other.min_x),
            min_y: self.min_y.max(other.min_y),
            max_x: self.max_x.min(other.max_x),
            max_y: self.max_y.min(other.max_y),
        }
    }
}

/// Project the viewport's four corners through `H_view→surface` and
/// return the surface-coord AABB they enclose. `None` when any
/// corner failed to project (degenerate homography).
pub fn viewport_surface_aabb(
    h_view_to_surface: &[f32; 9],
    frame_w: f32,
    frame_h: f32,
) -> Option<Aabb> {
    let corners = [
        (0.0, 0.0),
        (frame_w, 0.0),
        (frame_w, frame_h),
        (0.0, frame_h),
    ];
    let projected: Vec<(f32, f32)> = corners
        .into_iter()
        .filter_map(|(x, y)| project(h_view_to_surface, x, y))
        .collect();
    if projected.len() != 4 {
        return None;
    }
    Aabb::from_points(projected)
}

/// Area of the viewport quadrilateral after projection through
/// `H_view->surface`. Unlike the AABB, the quadrilateral area is
/// invariant under pure translation and in-plane rotation; it changes
/// when the camera zoom/scale relative to the locked surface changes.
fn viewport_surface_quad_area(
    h_view_to_surface: &[f32; 9],
    frame_w: f32,
    frame_h: f32,
) -> Option<f32> {
    let corners = [
        (0.0, 0.0),
        (frame_w, 0.0),
        (frame_w, frame_h),
        (0.0, frame_h),
    ];
    let mut projected = [(0.0_f32, 0.0_f32); 4];
    for (i, (x, y)) in corners.into_iter().enumerate() {
        projected[i] = project(h_view_to_surface, x, y)?;
    }
    let mut twice_area = 0.0_f32;
    for i in 0..4 {
        let (x0, y0) = projected[i];
        let (x1, y1) = projected[(i + 1) % 4];
        twice_area += x0 * y1 - x1 * y0;
    }
    let area = 0.5 * twice_area.abs();
    area.is_finite().then_some(area)
}

/// Per-anchor overlay state. Holds the source spec for each OCR
/// block on that anchor (paragraph strips + translated text + matting
/// hints) and the cached anchor-wide RGBA canvas built from them.
///
/// The canvas merges *all* of an anchor's blocks — bg fills and
/// translated glyphs — into a single surface-space bitmap, so the
/// per-frame compositor needs only one warp per anchor and bg /
/// cross-block overlap is resolved once at build time. The canvas is
/// rebuilt eagerly when blocks change; between rebuilds, the
/// compositor reads it verbatim with no per-block work.
pub struct AnchorOverlay {
    pub anchor_id: AnchorId,
    /// Per-block source specs keyed by stable_block_id. `BTreeMap`
    /// gives a deterministic iteration order so canvas content is
    /// reproducible across upserts.
    pub blocks: std::collections::BTreeMap<u64, BlockSpec>,
    /// Pre-rec bbox-only strips painted as standalone bg pills.
    /// Populated by `upsert_provisional_overlay` the moment detect
    /// finishes (engine has flipped Locked but orient-rec hasn't
    /// resolved yet), wholesale-cleared by `drop_provisional_overlay`
    /// before `run_post_detect` upserts the real grouped blocks.
    /// Separate from `blocks` because they have no identity, no
    /// grouping, and a different lifecycle (clear-all on transition,
    /// not retain-by-id).
    pub provisional_strips: Vec<OrientedRect>,
    /// Pill footprints (surface coords) from the most recent GPU draw-list build,
    /// for the movement monitor's mask. Set by [`LiveSession::overlay_draw_list`]
    /// each time the overlay is (re)baked; read by [`Self::overlay_pill_rects`].
    pub gpu_painted_pills: Vec<OrientedRect>,
}

impl AnchorOverlay {
    pub fn new(anchor_id: AnchorId) -> Self {
        Self {
            anchor_id,
            blocks: std::collections::BTreeMap::new(),
            provisional_strips: Vec::new(),
            gpu_painted_pills: Vec::new(),
        }
    }
}

/// Source spec for a single OCR block, owned by its `AnchorOverlay`.
/// Carries everything the anchor canvas builder needs to render this
/// block's bg fills + translated glyphs into the shared bitmap.
#[derive(Clone)]
pub struct BlockSpec {
    pub strips: Vec<OrientedRect>,
    pub matted_strips: Vec<Option<MattedStrip>>,
    pub display_text: String,
    pub language: String,
    /// Hash of (strips + display_text + language). The upsert path
    /// compares against the previous value to skip canvas rebuilds
    /// when the same content arrives twice.
    pub content_hash: u64,
    /// Shaped glyph instances for the GPU compositor (screen overlay
    /// only), with pen positions in this block's own canvas-texel frame
    /// — relative to the block's AABB origin (`canvas_geometry` over the
    /// block's visuals), scaled by oversample. Shaped once on upsert so
    /// the per-present draw-list build only offsets them by the
    /// block-origin → canvas-origin delta instead of re-shaping every
    /// frame. Empty for the camera path and for empty-text placeholders.
    pub glyph_instances: Vec<crate::image_render::GlyphInstanceData>,
    /// Upright coverage masks for the unique glyphs referenced by
    /// `glyph_instances`. Block-local because they're shaped at upsert
    /// time; merged across blocks (deduped by key) at draw-list build.
    pub glyph_masks: HashMap<crate::image_render::GlyphKey, crate::image_render::GlyphMaskData>,
}

/// Per-anchor live state. Each acquired anchor (engine
/// `AnchorId`) gets its own slot so two physical surfaces — sign A
/// and sign B — never share coord frames, line ids, or "what's been
/// detected" state. The session holds a `HashMap<AnchorId,
/// AnchorState>`; anchor-bound methods (`observe_detections`,
/// `ingest_rec`, ...) look up the right state by id.
pub struct AnchorState {
    /// Lines on this anchor's canonical frame.
    pub map: SurfaceMap,
    /// Surface region we've already run detection over. The refresh
    /// trigger compares the current viewport's surface AABB against
    /// this: viewport ⊆ covered → no new pixels → skip detection.
    /// Grows monotonically as detection runs over new viewport
    /// areas; reset only when the anchor is evicted (LRU) or the
    /// session is cleared.
    pub covered_region: Option<Aabb>,
    /// Engine's `H_anchor→view` at the time of the most recent
    /// successful detect+OCR+translate pass. The re-lock trigger
    /// uses this as the reference pose: it projects the current
    /// view corners through `inv(last_lock_h)` and through
    /// `inv(current_h)`, then compares the resulting AABBs in
    /// anchor coords. When the camera hasn't moved, both AABBs
    /// coincide and overlap = 1.0. Replaced (not unioned) on
    /// every successful pass.
    pub last_lock_h: Option<[f32; 9]>,
}

impl AnchorState {
    fn new() -> Self {
        Self {
            map: SurfaceMap::new(),
            covered_region: None,
            last_lock_h: None,
        }
    }
}

/// Project an `OrientedRect` through a homography, returning the
/// oriented rect that approximately bounds the warped quad. Used at
/// acquire time to map detection bboxes from raw camera coords into
/// rectified canonical coords (or any other change-of-basis).
///
/// The fit uses the 4 corners' projected positions: the center is
/// their centroid, the angle is the direction from new-TL to new-TR
/// (reading direction), and width/height are the average lengths of
/// the parallel edges. Under perfect homography of a rigid rectangle
/// this is exact; under perspective foreshortening the source
/// rectangle warps to a non-rectangular quad and this is a
/// least-error rectangular fit. For the OCR pipeline's purposes —
/// "where does the text region land after rectification" — that's
/// good enough; PP-OCR rec then operates on the cropped strip.
pub fn warp_oriented_box(b: &OrientedRect, h: &[f32; 9]) -> Option<OrientedRect> {
    let mut projected = [(0.0_f32, 0.0_f32); 4];
    for (i, (x, y)) in b.corners().iter().enumerate() {
        match project(h, *x, *y) {
            Some(p) => projected[i] = p,
            None => return None,
        }
    }
    let cx = 0.25 * (projected[0].0 + projected[1].0 + projected[2].0 + projected[3].0);
    let cy = 0.25 * (projected[0].1 + projected[1].1 + projected[2].1 + projected[3].1);
    let top_dx = projected[1].0 - projected[0].0;
    let top_dy = projected[1].1 - projected[0].1;
    let bot_dx = projected[2].0 - projected[3].0;
    let bot_dy = projected[2].1 - projected[3].1;
    let left_dx = projected[3].0 - projected[0].0;
    let left_dy = projected[3].1 - projected[0].1;
    let right_dx = projected[2].0 - projected[1].0;
    let right_dy = projected[2].1 - projected[1].1;
    let top_len = (top_dx * top_dx + top_dy * top_dy).sqrt();
    let bot_len = (bot_dx * bot_dx + bot_dy * bot_dy).sqrt();
    let left_len = (left_dx * left_dx + left_dy * left_dy).sqrt();
    let right_len = (right_dx * right_dx + right_dy * right_dy).sqrt();
    let width = 0.5 * (top_len + bot_len);
    let height = 0.5 * (left_len + right_len);
    if width <= 0.0 || height <= 0.0 {
        return None;
    }
    let angle_radians = top_dy.atan2(top_dx);
    Some(OrientedRect {
        cx,
        cy,
        width,
        height,
        angle_radians,
    })
}

/// Surface-coord padding when testing viewport ⊆ covered_region. A
/// few px of slack absorbs RANSAC residual + projection noise so a
/// "practically covered" viewport doesn't fail containment via
/// sub-pixel jitter.
pub const COVERAGE_PADDING_SURFACE_PX: f32 = 8.0;

/// Lifetime-bound state shared between platform wrappers (Android
/// `LivePlanarTracker`, desktop `surface_sim`). One instance per
/// active session; cleared on reset (tap-to-focus, language change).
pub struct LiveSession {
    /// Per-root-anchor state. Each entry is independent — different
    /// roots (= different physical surfaces) have different coord
    /// frames, line ids, and covered regions. No session-side LRU:
    /// the engine's `AnchorCache` is the source of truth for which
    /// roots exist, and bindings call
    /// [`Self::retain_anchors`] after each pipeline run with
    /// `engine.cached_root_ids()` to drop state for roots the
    /// engine has evicted.
    pub anchor_states: Mutex<HashMap<AnchorId, AnchorState>>,
    /// Resident rasterized overlays, keyed by anchor id. Each anchor
    /// owns one merged bitmap covering all of its blocks; the
    /// compositor selects the active anchor's canvas for the per-
    /// frame warp. Cached anchors' canvases sit untouched until
    /// either evicted or re-activated.
    pub overlay_anchors: Mutex<HashMap<AnchorId, AnchorOverlay>>,
    /// Frame counter that ticks on every Locked frame the caller
    /// observes after the most recent acquire. Drives the
    /// detect-on-tracking-frame trigger ([`Self::should_refresh_now`]).
    locked_frames_since_acquire: AtomicU64,
    /// Tick value at which the last refresh fired. The refresh
    /// predicate compares the current tick against this + the
    /// configured interval.
    last_refresh_locked_frame: AtomicU64,
    /// How many Locked frames must elapse between detect-on-track
    /// refresh fires. ~15 frames is ~0.5s at 30fps. Configurable so
    /// the sim can run tighter cadence than production.
    refresh_every_n_locked_frames: AtomicU32,
    /// Overlay-canvas oversample factor (f32 bits), set by the caller
    /// from the display/canonical resolution ratio. `1.0` keeps the
    /// legacy 1:1 surface rasterization; higher renders glyphs denser
    /// so the per-frame warp upscales them less.
    overlay_oversample: AtomicU32,
    /// Default pill background, RGBA packed as `(r<<24)|(g<<16)|(b<<8)|a`. The
    /// camera path keeps the translucent `0x101010C8`; the screen path sets an
    /// opaque pill so it isn't double-dimmed by the touch-capped window alpha.
    overlay_bg: AtomicU32,
    /// Bumped on every block/provisional *data* change. The GPU present polls this
    /// to know when to rebake the overlay.
    content_version: AtomicU64,
    /// Persistent glyph cache (parsed faces + glyph alpha-mask atlas) reused across
    /// renders, so repeat acquires and streaming updates find glyphs already
    /// rasterized instead of rebuilding a per-call cache. Font-file-keyed, so it
    /// never goes stale (chain lookups are cleared per render). The screen renders
    /// under `screen_render_lock`, so this Mutex is effectively uncontended.
    glyph_cache: Mutex<crate::image_render::FontCache>,
}

/// Default refresh cadence: fire `run_post_detect` every N tracked
/// frames while Locked. ~333 ms at 30 fps. The covered-region gate
/// makes the per-check cost essentially free (an AABB containment
/// test) on a held camera, so cadence here is the cadence we want
/// while actively panning — not the wall-clock interval between
/// detector runs. While panning, the gate flips to "not contained"
/// the moment the viewport edges past the covered region, and
/// detection fires; while still, almost every check is a cheap skip.
/// Going much below ~10 frames adds nothing because detection
/// itself is ~80–120 ms; we'd just be sampling the predicate more
/// often than detection can keep up with.
const DEFAULT_REFRESH_EVERY_N_LOCKED_FRAMES: u32 = 10;

impl LiveSession {
    pub fn new() -> Self {
        Self {
            anchor_states: Mutex::new(HashMap::new()),
            overlay_anchors: Mutex::new(HashMap::new()),
            locked_frames_since_acquire: AtomicU64::new(0),
            last_refresh_locked_frame: AtomicU64::new(0),
            refresh_every_n_locked_frames: AtomicU32::new(DEFAULT_REFRESH_EVERY_N_LOCKED_FRAMES),
            overlay_oversample: AtomicU32::new(1.0_f32.to_bits()),
            overlay_bg: AtomicU32::new(0x1010_10C8),
            content_version: AtomicU64::new(0),
            glyph_cache: Mutex::new(crate::image_render::FontCache::default()),
        }
    }

    /// Monotonic counter of block/provisional data changes (see
    /// [`Self::content_version`]).
    pub fn content_version(&self) -> u64 {
        self.content_version.load(Ordering::SeqCst)
    }

    /// Build the GPU overlay draw list for `anchor_id` from its current block +
    /// provisional content.
    /// Snapshots under the lock, builds outside it (glyphs are already shaped on
    /// upsert, so this is just pill geometry + pen-offset arithmetic), then stashes
    /// the pill footprints back on the anchor for the movement monitor's mask.
    /// `None` when there's nothing to show. Runs on the GL thread.
    pub fn overlay_draw_list(&self, anchor_id: AnchorId) -> Option<OverlayDrawList> {
        let (blocks, provisional) = {
            let anchors = self.overlay_anchors.lock().ok()?;
            let anchor = anchors.get(&anchor_id)?;
            (anchor.blocks.clone(), anchor.provisional_strips.clone())
        };
        let dl = build_overlay_draw_list(
            &blocks,
            &provisional,
            self.overlay_oversample(),
            self.overlay_bg(),
        )?;
        if let Ok(mut anchors) = self.overlay_anchors.lock() {
            if let Some(anchor) = anchors.get_mut(&anchor_id) {
                anchor.gpu_painted_pills = dl.painted_pills.clone();
            }
        }
        Some(dl)
    }

    /// Set the overlay-canvas oversample factor (texels per surface
    /// unit). Clamped to `[1.0, 4.0]`: below 1 would downsample the
    /// overlay below the surface; above 4 buys nothing on real displays
    /// and balloons the one-time canvas allocation.
    pub fn set_overlay_oversample(&self, factor: f32) {
        let clamped = factor.clamp(1.0, 4.0);
        self.overlay_oversample
            .store(clamped.to_bits(), Ordering::Relaxed);
    }

    fn overlay_oversample(&self) -> f32 {
        f32::from_bits(self.overlay_oversample.load(Ordering::Relaxed))
    }

    /// Set the default pill background (RGBA). Opaque values avoid double-dimming
    /// when the host window is already alpha-clamped (screen-translate path).
    pub fn set_overlay_bg(&self, rgba: [u8; 4]) {
        let packed = ((rgba[0] as u32) << 24)
            | ((rgba[1] as u32) << 16)
            | ((rgba[2] as u32) << 8)
            | (rgba[3] as u32);
        self.overlay_bg.store(packed, Ordering::Relaxed);
    }

    pub fn overlay_bg(&self) -> [u8; 4] {
        let p = self.overlay_bg.load(Ordering::Relaxed);
        [(p >> 24) as u8, (p >> 16) as u8, (p >> 8) as u8, p as u8]
    }

    /// Drop all session state. Caller invokes on tap-to-focus,
    /// language change, or any other coarse-grained reset signal.
    pub fn clear(&self) {
        if let Ok(mut states) = self.anchor_states.lock() {
            states.clear();
        }
        if let Ok(mut anchors) = self.overlay_anchors.lock() {
            anchors.clear();
        }
        self.locked_frames_since_acquire.store(0, Ordering::SeqCst);
        self.last_refresh_locked_frame.store(0, Ordering::SeqCst);
        // Invalidate any baked overlay keyed off the version — without this the GL
        // layer keeps presenting the last baked frame after the state is gone.
        self.content_version.fetch_add(1, Ordering::SeqCst);
    }

    /// Wipe per-anchor session state for `anchor_id` — drops the
    /// surface map, covered region, and any resident overlay items
    /// belonging to it. Use this when the
    /// engine has *re-created* an anchor with the same id (which
    /// can happen when `engine.clear()` resets the id counter and
    /// the next acquire claims id=1 again): without this wipe the
    /// stale surface map's overlays render through the NEW anchor's
    /// per-frame H at *old* surface coords, producing the "overlay
    /// stuck to an arbitrary offset" symptom right after a fast pan
    /// → loss → re-lock cycle.
    pub fn reset_anchor_state(&self, anchor_id: AnchorId) {
        if let Ok(mut states) = self.anchor_states.lock() {
            states.remove(&anchor_id);
        }
        if let Ok(mut anchors) = self.overlay_anchors.lock() {
            anchors.remove(&anchor_id);
        }
        // Invalidate any baked overlay keyed off the version — without this the GL
        // layer keeps presenting the dropped anchor's last baked frame (the "old
        // label sticks in its original position after a scroll" symptom).
        self.content_version.fetch_add(1, Ordering::SeqCst);
    }

    /// Drop per-anchor state and overlays whose `anchor_id` isn't in
    /// `keep`. Bindings call this after each acquire/refresh with the
    /// engine's currently-cached anchor set so our state stays
    /// aligned with what the engine can still track. Anchors evicted
    /// from the engine's LRU lose their session state on the next
    /// call.
    pub fn retain_anchors(&self, keep: &[AnchorId]) {
        let keep_set: std::collections::HashSet<AnchorId> = keep.iter().copied().collect();
        if let Ok(mut states) = self.anchor_states.lock() {
            states.retain(|id, _| keep_set.contains(id));
        }
        if let Ok(mut anchors) = self.overlay_anchors.lock() {
            anchors.retain(|id, _| keep_set.contains(id));
        }
    }

    /// Per-anchor coverage query. True when the viewport AABB is
    /// already inside the anchor's `covered_region` (padded by `pad`
    /// surface-coord units for noise). Refresh trigger uses this as
    /// its motion gate: contained → nothing new visible → skip.
    /// Returns false when no state exists for the anchor (first
    /// refresh after acquire) so the caller fires detection at least
    /// once.
    pub fn viewport_contained_in_coverage(
        &self,
        anchor_id: AnchorId,
        viewport: &Aabb,
        pad: f32,
    ) -> bool {
        let states = match self.anchor_states.lock() {
            Ok(s) => s,
            Err(_) => return false,
        };
        match states.get(&anchor_id).and_then(|s| s.covered_region) {
            Some(covered) => covered.contains_inflated(viewport, pad),
            None => false,
        }
    }

    /// Union the viewport AABB into this anchor's `covered_region`.
    /// Called after `run_post_detect` completes so subsequent
    /// refreshes can gate themselves out when they'd cover the same
    /// surface area.
    pub fn note_coverage(&self, anchor_id: AnchorId, viewport: Aabb) {
        let mut states = match self.anchor_states.lock() {
            Ok(s) => s,
            Err(_) => return,
        };
        let state = states.entry(anchor_id).or_insert_with(AnchorState::new);
        match &mut state.covered_region {
            Some(c) => c.union_inplace(&viewport),
            None => state.covered_region = Some(viewport),
        }
    }

    /// Replace (not union) the anchor's `last_lock_h` with the
    /// engine's `H_anchor→view` at the time of a successful pass.
    /// The re-lock trigger uses this as the reference pose.
    pub fn set_last_lock_h(&self, anchor_id: AnchorId, h: [f32; 9]) {
        let mut states = match self.anchor_states.lock() {
            Ok(s) => s,
            Err(_) => return,
        };
        let state = states.entry(anchor_id).or_insert_with(AnchorState::new);
        state.last_lock_h = Some(h);
    }

    /// Drop the anchor's `last_lock_h`. Called by the trigger when
    /// it fires so the next Locked frame re-initialises the
    /// reference pose from the engine's *current* H. Without this,
    /// the ~1-2 s pipeline window between trigger-fire and
    /// refresh-completion leaves `last_lock_h` lagged behind the
    /// real camera pose; subsequent comparisons then think the
    /// camera has moved when it hasn't, and the trigger refires.
    pub fn clear_last_lock_h(&self, anchor_id: AnchorId) {
        let mut states = match self.anchor_states.lock() {
            Ok(s) => s,
            Err(_) => return,
        };
        if let Some(state) = states.get_mut(&anchor_id) {
            state.last_lock_h = None;
        }
    }

    /// True when the anchor has a stored `last_lock_h`. The trigger
    /// uses this to skip the overlap check on a freshly-initialised
    /// (or freshly-invalidated) anchor and instead lazily seed
    /// `last_lock_h` from the current engine H.
    pub fn has_last_lock_h(&self, anchor_id: AnchorId) -> bool {
        match self.anchor_states.lock() {
            Ok(states) => states
                .get(&anchor_id)
                .and_then(|s| s.last_lock_h.as_ref())
                .is_some(),
            Err(_) => false,
        }
    }

    /// Re-lock trigger for zoom/scale divergence. Project the view
    /// corners `(0,0)..(view_w, view_h)` through both
    /// `inv(last_lock_h)` and `inv(current_h)` into anchor coords,
    /// compare the quadrilateral areas, and trigger when:
    ///   `min(area(curr), area(lock)) / max(area(curr), area(lock)) < threshold`
    ///
    /// This is symmetric in zoom direction: zoom-in shrinks the
    /// projected viewport area, zoom-out grows it. Pure pan/translation
    /// leaves the area unchanged, so panning across the same locked
    /// surface does not clear and re-seed the overlay map.
    ///
    /// Returns false when the anchor has no stored `last_lock_h`
    /// yet — that means we haven't completed a successful pass for
    /// this anchor, so there's nothing to compare against.
    pub fn should_relock_by_view(
        &self,
        anchor_id: AnchorId,
        current_h: &[f32; 9],
        view_w: f32,
        view_h: f32,
        threshold: f32,
    ) -> bool {
        let states = match self.anchor_states.lock() {
            Ok(s) => s,
            Err(_) => return false,
        };
        let lock_h = match states.get(&anchor_id).and_then(|s| s.last_lock_h.as_ref()) {
            Some(h) => *h,
            None => return false,
        };
        drop(states);
        let curr_inv = match crate::homography::invert(current_h) {
            Some(h) => h,
            None => return false,
        };
        let lock_inv = match crate::homography::invert(&lock_h) {
            Some(h) => h,
            None => return false,
        };
        let curr_area = match viewport_surface_quad_area(&curr_inv, view_w, view_h) {
            Some(a) => a,
            None => return false,
        };
        let lock_area = match viewport_surface_quad_area(&lock_inv, view_w, view_h) {
            Some(a) => a,
            None => return false,
        };
        let max_area = curr_area.max(lock_area);
        if max_area <= 0.0 {
            return true;
        }
        (curr_area.min(lock_area) / max_area) < threshold
    }

    /// Pre-refresh hook for the re-lock pipeline. Wipes the surface
    /// map AND overlay items for `anchor_id` so the next
    /// `run_post_detect` starts from empty state.
    ///
    /// This is the **clear + flash** approach. On production where
    /// OCR runs on a worker thread, the compositor renders 4–5
    /// frames between this wipe and the new overlays arriving, so
    /// the user sees a brief "bubbles disappear then snap back"
    /// flash on every refresh trigger. Trade-off: this is the only
    /// implementation we've found that *reliably* avoids stale
    /// overlays staying on screen indefinitely after the scene
    /// changes.
    ///
    /// **Heuristics that were tried and rejected** (see
    /// `analysis.md` § "Overlay swap on refresh" for full history):
    /// - Skip the wipe + rely on `SurfaceMap::add_or_merge` to
    ///   reuse existing block ids: tracker drift between refreshes
    ///   exceeded the 0.3·line_height baseline tolerance, producing
    ///   `Created` block ids at offsets → stacked duplicates that
    ///   `retain_blocks` doesn't drop (because production preserves
    ///   non-observed-this-run items for pan-away UX).
    /// - Displace by IoU > 0.5 at upsert: misses containment (a
    ///   narrow new bbox inside a wide old bbox has low IoU).
    /// - Displace by IoU > 0.5 OR containment > 0.7: catches the
    ///   narrow-inside-wide case but mishandles legitimate
    ///   re-segmentation, e.g. a 9-line paragraph being re-detected
    ///   as 1 line — containment of 1/9 ≈ 0.11 doesn't trigger
    ///   displace, so the 9-line overlay stays stacked under the 1
    ///   new overlay.
    ///
    /// **The proper fix is a larger refactor**: keep the *old*
    /// overlay items rendering throughout `detect → OCR → translate`
    /// (no flash); collect the *new* overlay items in a staging
    /// buffer; when the full new set is ready, atomic-swap old → new
    /// under the overlay-items mutex in a single transaction. That
    /// requires `run_post_detect` to defer all `upsert_block_overlay_bitmap`
    /// calls until completion and stage them via a "pending overlays"
    /// list, which crosses several module boundaries. Until that's
    /// done, this clear+flash version is correct (no stacking)
    /// even if it's visually unpolished.
    ///
    /// `covered_region` / `lock_viewport` are preserved here; the
    /// caller updates `lock_viewport` after the fresh pass succeeds
    /// (via `set_lock_viewport`).
    pub fn clear_anchor_state_for_relock(&self, anchor_id: AnchorId) {
        if let Ok(mut states) = self.anchor_states.lock() {
            if let Some(state) = states.get_mut(&anchor_id) {
                state.map = SurfaceMap::new();
            }
        }
        if let Ok(mut anchors) = self.overlay_anchors.lock() {
            anchors.remove(&anchor_id);
        }
    }

    pub fn clear_overlays(&self) {
        if let Ok(mut anchors) = self.overlay_anchors.lock() {
            anchors.clear();
        }
        self.content_version.fetch_add(1, Ordering::SeqCst);
    }

    /// The exact oriented pill footprints currently painted for `anchor_id`, in
    /// canonical coords — the `gpu_painted_pills` the draw-list build
    /// emitted. The screen monitor masks these out of its movement diff; later
    /// passes focus on them to watch for text changing under a pill. Reads the
    /// emitted geometry rather than re-deriving it from the raw strips, so the
    /// mask can't drift from what's actually opaque.
    pub fn overlay_pill_rects(&self, anchor_id: AnchorId) -> Vec<OrientedRect> {
        let Ok(anchors) = self.overlay_anchors.lock() else {
            return Vec::new();
        };
        let Some(anchor) = anchors.get(&anchor_id) else {
            return Vec::new();
        };
        anchor.gpu_painted_pills.clone()
    }

    /// Reset the refresh counter. Call after each fresh acquire so
    /// `should_refresh_now` doesn't immediately fire again on the
    /// next Locked frame.
    pub fn on_acquire(&self) {
        self.locked_frames_since_acquire.store(0, Ordering::SeqCst);
        self.last_refresh_locked_frame.store(0, Ordering::SeqCst);
    }

    /// Bump the Locked-frame counter. Call once per per-frame tracker
    /// tick that reports `Locked`. Returns the new tick value so the
    /// caller can log it if desired.
    pub fn on_locked_frame(&self) -> u64 {
        self.locked_frames_since_acquire
            .fetch_add(1, Ordering::SeqCst)
            + 1
    }

    /// Configure the refresh interval. Callers tweak this for tests /
    /// sim cadence; production uses the [default][DEFAULT_REFRESH_EVERY_N_LOCKED_FRAMES].
    pub fn set_refresh_every_n_locked_frames(&self, n: u32) {
        self.refresh_every_n_locked_frames
            .store(n.max(1), Ordering::SeqCst);
    }

    /// True when enough Locked frames have elapsed since the last
    /// refresh that the caller is *eligible* to fire one. Does **not**
    /// advance any internal state — pair with [`Self::mark_refresh_fired`]
    /// when the caller actually decides to fire. The split lets the
    /// caller add additional gates (e.g. a motion gate on H_root→view)
    /// without advancing the cadence on every frame.
    pub fn refresh_cadence_elapsed(&self) -> bool {
        let n = self.refresh_every_n_locked_frames.load(Ordering::SeqCst) as u64;
        let tick = self.locked_frames_since_acquire.load(Ordering::SeqCst);
        let last = self.last_refresh_locked_frame.load(Ordering::SeqCst);
        tick > 0 && tick >= last + n
    }

    /// Snapshot the current Locked-frame tick as the new "last
    /// refresh fired" baseline. Call when actually firing a refresh
    /// after [`Self::refresh_cadence_elapsed`] + any caller-side
    /// gates pass.
    pub fn mark_refresh_fired(&self) {
        let tick = self.locked_frames_since_acquire.load(Ordering::SeqCst);
        self.last_refresh_locked_frame.store(tick, Ordering::SeqCst);
    }

    /// Combined check+mark: true when the cadence has elapsed (in
    /// which case the tick is advanced). Convenience for callers that
    /// don't apply additional gates — production uses the split
    /// `refresh_cadence_elapsed` / `mark_refresh_fired` pair to layer
    /// the motion gate in between.
    pub fn should_refresh_now(&self) -> bool {
        if !self.refresh_cadence_elapsed() {
            return false;
        }
        self.mark_refresh_fired();
        true
    }

    /// Feed a batch of detections (in surface coords) into the
    /// **active anchor's** surface map and return per-detection
    /// outcomes the caller uses to (a) decide which detections need
    /// recognition, and (b) push rec results back via
    /// [`Self::ingest_rec`].
    ///
    /// Creates the `AnchorState` on first call for a new
    /// `anchor_id`. `source_language` is used as the default for
    /// newly-created lines; existing lines keep their
    /// previously-recorded language unless the observation carries a
    /// non-empty value.
    pub fn observe_detections(
        &self,
        anchor_id: AnchorId,
        detections: &[OrientedRect],
        source_language: &str,
    ) -> Vec<DetectionOutcome> {
        let mut out = Vec::with_capacity(detections.len());
        let mut states = match self.anchor_states.lock() {
            Ok(s) => s,
            Err(_) => {
                return detections
                    .iter()
                    .map(|_| DetectionOutcome::poisoned())
                    .collect();
            }
        };
        let state = states.entry(anchor_id).or_insert_with(AnchorState::new);
        for d in detections {
            let obs = SurfaceLineObservation {
                bbox: d.clone(),
                source_text: String::new(),
                translated_text: String::new(),
                source_language: source_language.to_string(),
            };
            let res = state.map.add_or_merge(obs);
            let needs_rec = res.needs_rec();
            let line_id = res.id();
            let kind = AddResultKind::from(&res);
            let cached_source_text = if !needs_rec {
                state
                    .map
                    .get(line_id)
                    .map(|l| l.source_text.clone())
                    .unwrap_or_default()
            } else {
                String::new()
            };
            let cached_source_language = if !needs_rec {
                state
                    .map
                    .get(line_id)
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

    /// Push a single recognized line back into the active anchor's
    /// map: store the text and language, and snapshot the line's
    /// current u-extent as "rec just saw up to here" so future
    /// observations that extend past it trigger re-recognition.
    pub fn ingest_rec(
        &self,
        anchor_id: AnchorId,
        line_id: SurfaceLineId,
        source_text: &str,
        source_language: &str,
    ) {
        if let Ok(mut states) = self.anchor_states.lock() {
            if let Some(state) = states.get_mut(&anchor_id) {
                if let Some(line) = state.map.get_mut(line_id) {
                    line.source_text = source_text.to_string();
                    if !source_language.is_empty() {
                        line.source_language = source_language.to_string();
                    }
                    line.record_rec_extent();
                }
            }
        }
    }

    /// Push translated text back into a set of lines on the active
    /// anchor (all recipients receive the same string — caller has
    /// already performed block-level translation across the joined
    /// source strings).
    pub fn ingest_translation(
        &self,
        anchor_id: AnchorId,
        line_ids: &[SurfaceLineId],
        translated: &str,
    ) {
        if let Ok(mut states) = self.anchor_states.lock() {
            if let Some(state) = states.get_mut(&anchor_id) {
                for &id in line_ids {
                    if let Some(line) = state.map.get_mut(id) {
                        line.translated_text = translated.to_string();
                    }
                }
            }
        }
    }

    /// Upsert one resident overlay block on an anchor. Stores the
    /// block's source spec under the anchor's `AnchorOverlay` and
    /// rebuilds the anchor canvas if this block's content hash
    /// changed. Otherwise it's a no-op — the cached canvas survives.
    /// `matted_strips` is indexed parallel to `strips` and may be
    /// empty to fall back to the default-bg pill rendering.
    pub fn upsert_block(
        &self,
        anchor_id: AnchorId,
        id: u64,
        strips: Vec<OrientedRect>,
        matted_strips: Vec<Option<MattedStrip>>,
        source_text: String,
        translated_text: String,
        language: String,
        font_provider: &dyn crate::font_provider::FontProvider,
    ) {
        if strips.is_empty() {
            return;
        }
        let display_text = pick_display_text(&source_text, &translated_text);
        let hash = block_content_hash(&strips, &display_text, &language);
        {
            let anchors = match self.overlay_anchors.lock() {
                Ok(a) => a,
                Err(_) => return,
            };
            if let Some(existing) = anchors.get(&anchor_id).and_then(|a| a.blocks.get(&id)) {
                if existing.content_hash == hash {
                    return;
                }
            }
        }
        let mut spec = BlockSpec {
            strips,
            matted_strips,
            display_text,
            language,
            content_hash: hash,
            glyph_instances: Vec::new(),
            glyph_masks: HashMap::new(),
        };
        // Both paths composite glyphs on the GPU from these pre-shaped instances;
        // shape them here (off the present thread, once per content change) so
        // `build_overlay_draw_list` only offsets pen positions at present time.
        let (instances, masks) = self.shape_block_glyphs(&spec, font_provider);
        spec.glyph_instances = instances;
        spec.glyph_masks = masks;
        {
            let mut anchors = match self.overlay_anchors.lock() {
                Ok(a) => a,
                Err(_) => return,
            };
            let anchor = anchors
                .entry(anchor_id)
                .or_insert_with(|| AnchorOverlay::new(anchor_id));
            anchor.blocks.insert(id, spec);
            // `after_content_change` bumps the version; the GL present rebakes the
            // overlay on its next frame from the updated blocks.
        }
        // NB: we used to keep a per-block "displace stale duplicates"
        // pass here (IoU + containment in surface coords) so the
        // clear-before-refresh wipe could be dropped (which would
        // eliminate the flash). It worked for in-place duplicates
        // but broke on re-segmentation: when the detector merges a
        // 9-line paragraph into one box at frame A and then sees it
        // as 1 line at frame B, the new 1-line bbox is 1/9 of the
        // old paragraph's area — containment = 1/9 ≈ 0.11, well
        // below any usable threshold — and the old 9-line overlay
        // stayed stacked. No purely-geometric heuristic over
        // (old_bbox, new_bbox) distinguishes "stale duplicate of the
        // same physical text" from "legitimate sub-region of a
        // larger block", so we now rely on
        // `clear_anchor_state_for_relock` to wipe the slate on each
        // refresh. The flash that creates is the price of correctness
        // until the staged-atomic-swap refactor lands. See
        // `analysis.md` § "Overlay swap on refresh" for context.
        self.after_content_change();
    }

    /// Shape one block's glyphs into its own canvas-texel frame for the GPU
    /// compositor: pen positions are relative to the block's AABB origin
    /// (`canvas_geometry` over the block's own visuals) scaled by oversample, so
    /// `build_overlay_draw_list` only has to offset them by the block-origin →
    /// canvas-origin delta. Returns empty for placeholder (empty-text) blocks.
    /// Mirrors the geometry chain of [`build_overlay_draw_list`]'s per-block setup,
    /// localized to the block instead of the shared canvas.
    fn shape_block_glyphs(
        &self,
        spec: &BlockSpec,
        font_provider: &dyn crate::font_provider::FontProvider,
    ) -> (
        Vec<crate::image_render::GlyphInstanceData>,
        HashMap<crate::image_render::GlyphKey, crate::image_render::GlyphMaskData>,
    ) {
        let os = self.overlay_oversample().max(1.0);
        let visuals = inflate_block_visuals(spec);
        let Some((origin_x, origin_y, bitmap_w, bitmap_h)) = canvas_geometry(&visuals, os) else {
            return (Vec::new(), HashMap::new());
        };
        let local = localize_visuals(&visuals, origin_x, origin_y, os);
        let Some(tb) = build_block_text_block(spec, &visuals, &local, os, bitmap_w, bitmap_h)
        else {
            return (Vec::new(), HashMap::new());
        };
        let opts = crate::image_render::RenderOptions {
            language: spec.language.clone(),
            min_font_size_px: 6.0 * os,
        };
        let collector = with_glyph_cache(Some(&self.glyph_cache), |gc| {
            crate::image_render::collect_overlay_glyphs(
                std::slice::from_ref(&tb),
                gc,
                font_provider,
                &opts,
            )
        });
        (collector.instances, collector.masks)
    }

    /// Record a block/provisional data change so the GPU present rebakes the
    /// overlay on its next frame.
    fn after_content_change(&self) {
        // Both paths composite on the GPU: just bump the version. The present
        // thread re-bakes the overlay the next frame it sees the version move.
        self.content_version.fetch_add(1, Ordering::SeqCst);
    }

    /// Publish a provisional bbox-only overlay covering all
    /// `surface_strips`. Called immediately after acquire (before
    /// orient-rec/post-detect) so the user sees translucent pills the
    /// instant the engine flips Locked. Strips come in *surface
    /// coordinates* (== view coords at acquire time, modulo
    /// `sensor_crop`). Stored as a dedicated field on the anchor
    /// overlay so they share the canvas with real blocks but follow
    /// their own clear-all lifecycle.
    pub fn upsert_provisional_overlay(
        &self,
        anchor_id: AnchorId,
        surface_strips: Vec<OrientedRect>,
    ) {
        {
            let mut anchors = match self.overlay_anchors.lock() {
                Ok(a) => a,
                Err(_) => return,
            };
            let anchor = anchors
                .entry(anchor_id)
                .or_insert_with(|| AnchorOverlay::new(anchor_id));
            anchor.provisional_strips = surface_strips;
        }
        self.after_content_change();
    }

    /// Drop the provisional bbox-only strips published by
    /// [`Self::upsert_provisional_overlay`]. Does not eagerly rebuild
    /// the canvas — the next real-block upsert inside
    /// `run_post_detect` repaints without them via the existing
    /// fingerprint-mismatch path.
    pub fn drop_provisional_overlay(&self, anchor_id: AnchorId) {
        if let Ok(mut anchors) = self.overlay_anchors.lock() {
            if let Some(anchor) = anchors.get_mut(&anchor_id) {
                anchor.provisional_strips.clear();
            }
        }
        self.content_version.fetch_add(1, Ordering::SeqCst);
    }

    /// Drop blocks from `anchor_id` whose id isn't in `ids`. Used
    /// when an acquire / refresh finishes (final block id set known)
    /// so stale overlays from a prior pipeline run on the same anchor
    /// don't linger. Blocks on *other* anchors are untouched.
    pub fn retain_blocks(&self, anchor_id: AnchorId, ids: &[u64]) {
        let keep: std::collections::HashSet<u64> = ids.iter().copied().collect();
        if let Ok(mut anchors) = self.overlay_anchors.lock() {
            if let Some(anchor) = anchors.get_mut(&anchor_id) {
                anchor.blocks.retain(|id, _| keep.contains(id));
            }
        }
        // Bump the version so the GL present rebakes without the dropped blocks.
        self.content_version.fetch_add(1, Ordering::SeqCst);
    }

    /// Invalidate the cached SurfaceMap lines overlapping `rects` for an anchor,
    /// so the next detection there forces recognition instead of reusing stale
    /// OCR text. Paired with [`Self::remove_blocks`] on an under-pill change.
    pub fn invalidate_surface_region(&self, anchor_id: AnchorId, rects: &[OrientedRect]) {
        if rects.is_empty() {
            return;
        }
        if let Ok(mut states) = self.anchor_states.lock() {
            if let Some(state) = states.get_mut(&anchor_id) {
                state.map.remove_overlapping(rects);
            }
        }
    }

    /// Drop specific resident blocks (the screen monitor blinks a pill off on an
    /// under-pill change so the next masked acquire re-grabs it). Bumps the
    /// version so the present rebakes without them.
    pub fn remove_blocks(&self, anchor_id: AnchorId, ids: &[u64]) {
        let drop: std::collections::HashSet<u64> = ids.iter().copied().collect();
        if drop.is_empty() {
            return;
        }
        if let Ok(mut anchors) = self.overlay_anchors.lock() {
            if let Some(anchor) = anchors.get_mut(&anchor_id) {
                anchor.blocks.retain(|id, _| !drop.contains(id));
            }
        }
        self.content_version.fetch_add(1, Ordering::SeqCst);
    }

    /// Resident blocks as `(block_id, content_hash, strips)` — the screen monitor
    /// keys per-box pinhole state by `block_id` and rebaselines only when a block
    /// is new or its `content_hash` changed (re-OCR'd), so an unrelated acquire
    /// can't reset a stale block's baseline.
    pub fn overlay_blocks(&self, anchor_id: AnchorId) -> Vec<(u64, u64, Vec<OrientedRect>)> {
        let Ok(anchors) = self.overlay_anchors.lock() else {
            return Vec::new();
        };
        let Some(anchor) = anchors.get(&anchor_id) else {
            return Vec::new();
        };
        anchor
            .blocks
            .iter()
            .map(|(id, spec)| (*id, spec.content_hash, spec.strips.clone()))
            .collect()
    }

    /// Resident blocks as `(block_id, strip_rect)` pairs — one per strip. The
    /// screen monitor keys its per-box pinhole state by `block_id` (so a trip maps
    /// straight to [`Self::remove_blocks`]) and masks the recovery to these rects.
    pub fn overlay_block_pills(&self, anchor_id: AnchorId) -> Vec<(u64, OrientedRect)> {
        let Ok(anchors) = self.overlay_anchors.lock() else {
            return Vec::new();
        };
        let Some(anchor) = anchors.get(&anchor_id) else {
            return Vec::new();
        };
        anchor
            .blocks
            .iter()
            .flat_map(|(id, spec)| spec.strips.iter().map(move |s| (*id, s.clone())))
            .collect()
    }

    /// `(block_id, display_text)` for each resident block — for debug logging the
    /// per-box monitor state joined with the label's text.
    pub fn block_display_texts(&self, anchor_id: AnchorId) -> Vec<(u64, String)> {
        let Ok(anchors) = self.overlay_anchors.lock() else {
            return Vec::new();
        };
        let Some(anchor) = anchors.get(&anchor_id) else {
            return Vec::new();
        };
        anchor
            .blocks
            .iter()
            .map(|(id, spec)| (*id, spec.display_text.clone()))
            .collect()
    }

    /// Per-block axis-aligned footprints of our own translated glyphs in canonical
    /// (surface) coords: `(block_id, [[min_x, min_y, max_x, max_y]; n])`. The screen
    /// monitor drops lattice points covered by these — a hole on our opaque text
    /// reads the text, not the screen behind it, so it never registers the
    /// underlying content changing and would pin the box (an immortal label). Only
    /// the pill-background holes between/around the glyphs are monitored.
    pub fn overlay_block_glyph_aabbs(&self, anchor_id: AnchorId) -> Vec<(u64, Vec<[f32; 4]>)> {
        let os = self.overlay_oversample().max(1.0);
        let Ok(anchors) = self.overlay_anchors.lock() else {
            return Vec::new();
        };
        let Some(anchor) = anchors.get(&anchor_id) else {
            return Vec::new();
        };
        anchor
            .blocks
            .iter()
            .map(|(id, spec)| {
                let visuals = inflate_block_visuals(spec);
                let Some((ox, oy, _, _)) = canvas_geometry(&visuals, os) else {
                    return (*id, Vec::new());
                };
                let mut rects = Vec::with_capacity(spec.glyph_instances.len());
                for inst in &spec.glyph_instances {
                    let Some(mask) = spec.glyph_masks.get(&inst.key) else {
                        continue;
                    };
                    let (w, h) = (mask.w as f32, mask.h as f32);
                    let (l, t) = (mask.left as f32, mask.top as f32);
                    // Same unit-quad → canvas-texel map the glyph bake uses, over the
                    // four corners → texel AABB (covers any line rotation).
                    let mut min_x = f32::MAX;
                    let mut min_y = f32::MAX;
                    let mut max_x = f32::MIN;
                    let mut max_y = f32::MIN;
                    for (u, v) in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)] {
                        let tx = inst.pen_x + (l + u * w) * inst.cos - (t + v * h) * inst.sin;
                        let ty = inst.pen_y + (l + u * w) * inst.sin + (t + v * h) * inst.cos;
                        min_x = min_x.min(tx);
                        min_y = min_y.min(ty);
                        max_x = max_x.max(tx);
                        max_y = max_y.max(ty);
                    }
                    // Canvas-texel → canonical: canon = origin + texel / oversample.
                    rects.push([
                        ox + min_x / os,
                        oy + min_y / os,
                        ox + max_x / os,
                        oy + max_y / os,
                    ]);
                }
                (*id, rects)
            })
            .collect()
    }
}

/// Adapter implementing the live recognition interface. The bindings
/// implements this on `&TranslatorSession`; the desktop sim wraps a
/// `&PpocrEngine`. Errors are stringified — the orchestrator only
/// logs them and continues, so a typed error tree pulls its weight.
pub trait LiveRecognizer {
    fn recognize(
        &self,
        oriented: &OrientedImage,
        boxes: &[DetectedTextBox],
        source_selection: &OcrSourceSelection,
        canonical_quadrant: Option<crate::coords::Quadrant>,
    ) -> Result<Vec<RecognizedTextLine>, String>;
}

/// Adapter implementing the live translation interface. Mirrors
/// `TranslatorSession::translate_mixed_texts`'s signature. The sim
/// passes [`NoopTranslator`] (no translation models loaded).
pub trait LiveTranslator {
    fn translate_mixed_texts(
        &self,
        inputs: &[String],
        forced_source_code: Option<&str>,
        target_code: &str,
        available_language_codes: &[LanguageCode],
    ) -> Result<MixedTextTranslationResult, String>;
}

#[cfg(feature = "ppocr")]
impl LiveRecognizer for &crate::session::TranslatorSession {
    fn recognize(
        &self,
        oriented: &OrientedImage,
        boxes: &[DetectedTextBox],
        source_selection: &OcrSourceSelection,
        canonical_quadrant: Option<crate::coords::Quadrant>,
    ) -> Result<Vec<RecognizedTextLine>, String> {
        (*self)
            .recognize_in_oriented_image(
                oriented,
                boxes,
                source_selection.clone(),
                canonical_quadrant,
            )
            .map_err(|e| format!("{e:?}"))
    }
}

impl LiveTranslator for &crate::session::TranslatorSession {
    fn translate_mixed_texts(
        &self,
        inputs: &[String],
        forced_source_code: Option<&str>,
        target_code: &str,
        available_language_codes: &[LanguageCode],
    ) -> Result<MixedTextTranslationResult, String> {
        (*self)
            .translate_mixed_texts(
                inputs,
                forced_source_code,
                target_code,
                available_language_codes,
            )
            .map_err(|e| format!("{e:?}"))
    }
}

/// Stub translator: returns the source as the "translation". Suitable
/// for the desktop simulator (no translation models loaded) where the
/// pipeline still wants to exercise the translate → upsert path.
pub struct NoopTranslator;

impl LiveTranslator for NoopTranslator {
    fn translate_mixed_texts(
        &self,
        inputs: &[String],
        _forced: Option<&str>,
        _target: &str,
        _available: &[LanguageCode],
    ) -> Result<MixedTextTranslationResult, String> {
        let translations = inputs
            .iter()
            .map(|s| crate::routing::TextTranslation {
                source_text: s.clone(),
                translated_text: s.clone(),
            })
            .collect();
        Ok(MixedTextTranslationResult {
            translations,
            nothing_reason: None,
        })
    }
}

/// Estimate the scene's reading-direction quadrant from a freshly-detected
/// set of text boxes. Runs the textline orientation classifier twice per
/// candidate strip (raw + CW90) so the binary 0°/180° model resolves the
/// full 90° quadrant; per-strip votes are aggregated with confidence,
/// quorum, and majority gates. Returns `None` if the classifier model
/// isn't installed, no wide strips qualified, or consensus didn't reach
/// the gate.
///
/// Pure relative to its arguments: no engine state is mutated. The caller
/// (the platform acquire path) feeds the result into
/// `LivePlanarEngine::acquire_now_with_orientation`.
/// Rec-based orientation estimator. Picks the quadrant that produces the
/// highest average recognition confidence across K sample detections, by
/// running the script-specific recognizer at each of the four candidate
/// canonicals and keeping the winner.
///
/// Why this beats the textline-ori classifier on hard / OOD content:
/// rec is a CTC decoder over a real character vocabulary, so it lights
/// up when the orientation is right and emits low-conf character salad
/// when it isn't — the conf gap is a strong, well-calibrated signal.
/// The textline-ori model is binary with a class prior; on out-of-
/// distribution inputs it falls back to the prior regardless of the
/// pixels, giving us the "Up vs Up @0.999" pathology.
///
/// Cost: 2 × K rec invocations (≈ K boxes × 2 sides of the scene's
/// dominant axis) — typically O(150-250 ms) at K=3, comparable to the
/// windowed textline-ori path while being far more accurate.
///
/// Returns `None` when:
/// - `script` is unknown at the call site (auto-source mode — caller
///   should fall back to the textline-ori path).
/// - no detections qualify as samples (too few, low conf, no contour).
/// - the winning canonical doesn't beat the loser by `MIN_REC_CONF_MARGIN`.
#[cfg(feature = "ppocr")]
pub fn estimate_canonical_via_rec(
    engine: &crate::ppocr::PpocrEngine,
    oriented: &crate::live_frame::OrientedImage,
    boxes: &[DetectedTextBox],
    script: crate::PpocrScript,
) -> Option<crate::coords::Quadrant> {
    use crate::coords::Quadrant;
    use crate::ppocr::PpocrProfile;

    const MIN_DET_SCORE: f32 = 0.5;
    const MIN_LONG_PX: f32 = 40.0;
    const K_SAMPLES: usize = 3;
    /// Winner's avg rec confidence must beat the runner-up's by this
    /// margin to be trusted. Small but real — if all 4 canonicals are
    /// close, content is genuinely orientation-ambiguous (numerals,
    /// symmetric glyphs, single-character boxes).
    const MIN_REC_CONF_MARGIN: f32 = 0.05;
    /// Absolute floor on the winner. Without this, low-quality content
    /// where rec is bad in every direction could still elect a
    /// canonical purely on noise.
    const MIN_WINNER_CONF: f32 = 0.30;

    if boxes.is_empty() {
        return None;
    }
    let rgb = oriented.rgb.as_ref()?;
    let gray = &oriented.gray;

    // Pick top-K by det score, filtered for size + presence of contour.
    let mut sample_indices: Vec<(usize, f32)> = boxes
        .iter()
        .enumerate()
        .filter_map(|(i, b)| {
            if b.score < MIN_DET_SCORE {
                return None;
            }
            if b.contour.is_empty() || b.contour.len() % 2 != 0 {
                return None;
            }
            let w = b.rect.right.saturating_sub(b.rect.left) as f32;
            let h = b.rect.bottom.saturating_sub(b.rect.top) as f32;
            let long = w.max(h);
            if long < MIN_LONG_PX {
                return None;
            }
            Some((i, b.score * long))
        })
        .collect();
    sample_indices
        .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    sample_indices.truncate(K_SAMPLES);
    if sample_indices.is_empty() {
        log::info!(
            "[orient-rec] no qualifying samples among {} boxes",
            boxes.len()
        );
        return None;
    }
    // Scale to the rec-source resolution: these samples are both PCA-analyzed
    // (angle is scale-invariant) and cropped from `oriented.rgb` (rec-res).
    let samples: Vec<DetectedTextBox> = oriented.rec_scaled_boxes(
        &sample_indices
            .iter()
            .map(|(i, _)| boxes[*i].clone())
            .collect::<Vec<_>>(),
    );
    let scripts = vec![script; samples.len()];

    // The dewarp's canonical parameter is a sign-flip on the PCA axis,
    // not an absolute-direction rotation. With (ux, uy) ≈ (cos θ_pca,
    // sin θ_pca), the four cardinal canonicals collapse to two distinct
    // outputs depending on the sign of ux*cos(Q) + uy*sin(Q). So we only
    // need ONE pair of opposite canonicals on the axis the scene's PCA
    // picks. Two rec calls per acquire, not four.
    //
    // Axis selection: average the sample boxes' PCA angles via the
    // doubled-angle trick (Mardia's circular mean for axial data),
    // giving a stable scene axis under per-box tilt. If the resulting
    // axis is closer to horizontal, test R0 vs R180; if vertical, test
    // R90 vs R270.
    let mut axis_cos2 = 0.0f32;
    let mut axis_sin2 = 0.0f32;
    let mut axis_count = 0usize;
    for b in &samples {
        let contour: Vec<(f32, f32)> = b.contour.chunks_exact(2).map(|c| (c[0], c[1])).collect();
        if let Some(theta) = crate::ppocr::contour_principal_axis_angle(&contour) {
            axis_cos2 += (2.0 * theta).cos();
            axis_sin2 += (2.0 * theta).sin();
            axis_count += 1;
        }
    }
    if axis_count == 0 {
        return None;
    }
    let axis_theta = 0.5 * axis_sin2.atan2(axis_cos2); // ∈ [-π/2, π/2]
    let (canon_a, canon_b) = if axis_theta.cos().abs() >= axis_theta.sin().abs() {
        (Quadrant::R0, Quadrant::R180)
    } else {
        (Quadrant::R90, Quadrant::R270)
    };
    let axis_label = if matches!(canon_a, Quadrant::R0) {
        "horizontal"
    } else {
        "vertical"
    };

    let t_start = std::time::Instant::now();
    let mut sides: [(Quadrant, f32, Vec<crate::ocr::RecognizedTextLine>); 2] =
        [(canon_a, 0.0, Vec::new()), (canon_b, 0.0, Vec::new())];
    for slot in sides.iter_mut() {
        let canon = slot.0;
        match engine.recognize_text_in_boxes_image(
            rgb,
            gray,
            &samples,
            &scripts,
            PpocrProfile::Live,
            Some(canon),
        ) {
            Ok(lines) => {
                let conf_sum: f32 = lines.iter().map(|l| l.confidence).sum();
                let conf_avg = if lines.is_empty() {
                    0.0
                } else {
                    conf_sum / lines.len() as f32
                };
                // Truncate each rec text to keep logcat readable on
                // long sentences. Enough characters to tell "real text"
                // from "empty / gibberish" at a glance, no more.
                const LOG_TEXT_MAX_CHARS: usize = 30;
                let texts: Vec<String> = lines
                    .iter()
                    .map(|l| {
                        let t = l.text.trim();
                        if t.chars().count() <= LOG_TEXT_MAX_CHARS {
                            t.to_string()
                        } else {
                            let truncated: String = t.chars().take(LOG_TEXT_MAX_CHARS).collect();
                            format!("{truncated}…")
                        }
                    })
                    .collect();
                log::info!(
                    "[orient-rec canonical={:?}] avg_conf={:.3} n={} texts={:?}",
                    canon,
                    conf_avg,
                    lines.len(),
                    texts,
                );
                slot.1 = conf_avg;
                slot.2 = lines;
            }
            Err(e) => {
                log::debug!("[orient-rec canonical={:?}] recognize failed: {e:?}", canon);
            }
        }
    }
    let elapsed_ms = t_start.elapsed().as_secs_f64() * 1000.0;

    let (win_idx, loser_idx) = if sides[0].1 >= sides[1].1 {
        (0, 1)
    } else {
        (1, 0)
    };
    let win_conf = sides[win_idx].1;
    let loser_conf = sides[loser_idx].1;
    let margin = win_conf - loser_conf;
    let winner = sides[win_idx].0;

    log::info!(
        "[orient-rec] {:.1}ms K={} axis={} ({:?}@{:.3} vs {:?}@{:.3}) → winner={:?} margin={:.3}",
        elapsed_ms,
        samples.len(),
        axis_label,
        sides[0].0,
        sides[0].1,
        sides[1].0,
        sides[1].1,
        winner,
        margin,
    );

    if win_conf < MIN_WINNER_CONF {
        log::info!(
            "[orient-rec] winner conf {:.3} below floor {:.3} — falling back to default",
            win_conf,
            MIN_WINNER_CONF,
        );
        return None;
    }
    if margin < MIN_REC_CONF_MARGIN {
        log::info!(
            "[orient-rec] margin {:.3} below floor {:.3} — content orientation-symmetric, falling back",
            margin,
            MIN_REC_CONF_MARGIN,
        );
        return None;
    }

    Some(winner)
}

/// Original textline-ori-based estimator, kept for the auto-source path
/// (where we don't know the script up-front and can't run rec).
#[cfg(feature = "ppocr")]
/// Geometric reading-direction quadrant from detected-box contour angles —
/// no recognizer pass. Picks the dominant principal axis (horizontal vs
/// vertical, length-weighted) and assumes world-up: horizontal → R0, vertical
/// → R90. Used by the screen pipeline, which is always world-up but may be
/// captured in landscape; the rec-based 180° disambiguation that
/// [`estimate_canonical_quadrant`] adds is skipped for speed.
pub fn dominant_axis_quadrant(boxes: &[DetectedTextBox]) -> crate::coords::Quadrant {
    use crate::ppocr::contour_principal_axis_angle;
    let mut horizontal = 0.0f32;
    let mut vertical = 0.0f32;
    for b in boxes {
        if b.contour.is_empty() || b.contour.len() % 2 != 0 {
            continue;
        }
        let contour: Vec<(f32, f32)> = b.contour.chunks_exact(2).map(|c| (c[0], c[1])).collect();
        let Some(theta) = contour_principal_axis_angle(&contour) else {
            continue;
        };
        let w = b.rect.right.saturating_sub(b.rect.left) as f32;
        let h = b.rect.bottom.saturating_sub(b.rect.top) as f32;
        let weight = w.max(h);
        // θ ∈ [0, π): |sin θ| is 0 for a horizontal axis, 1 for vertical; split
        // at 45°.
        if theta.sin().abs() < std::f32::consts::FRAC_1_SQRT_2 {
            horizontal += weight;
        } else {
            vertical += weight;
        }
    }
    if vertical > horizontal {
        crate::coords::Quadrant::R90
    } else {
        crate::coords::Quadrant::R0
    }
}

pub fn estimate_canonical_quadrant(
    engine: &crate::ppocr::PpocrEngine,
    image: &image::DynamicImage,
    gray: &image::GrayImage,
    boxes: &[DetectedTextBox],
) -> Option<crate::coords::Quadrant> {
    use crate::coords::Quadrant;
    use crate::ppocr::{TextlineOriLabel, contour_principal_axis_angle, dewarp_contour_to_strip};

    // Deskew-and-classify pipeline. For each detected box we PCA the
    // contour to get its principal-axis angle θ ∈ [0, π), dewarp the
    // strip so the text becomes axis-aligned in a tight rectangle,
    // and only then feed it to the (binary, horizontal-only) textline-
    // ori model. The classifier then resolves the ±x ambiguity. World
    // reading direction = θ (Up) or θ + π (Flipped180).
    //
    // This replaces the older Wide/Tall aspect-gate hack which dropped
    // any non-axis-aligned tilt on the floor and crushed the classifier
    // input. The deskew gives the model the kind of input it was
    // trained on; the validation-pair gate below filters out the
    // model's class-prior bias on hard content.
    const MIN_LONG_PX: f32 = 40.0;
    const MIN_DET_SCORE: f32 = 0.5;
    const TOP_N: usize = 10;
    // Must match the textline-ori model's native input size
    // (TEXTLINE_ORI_WIDTH/HEIGHT in ppocr.rs).
    const WINDOW_W: u32 = 160;
    const WINDOW_H: u32 = 80;
    const MAX_WINDOWS_PER_STRIP: u32 = 3;
    // The asymmetry gate: Δ = P(Up|raw) − P(Up|raw180). An unbiased
    // classifier on a 180°-rotated copy of the same input MUST give
    // the opposite probability, so |Δ| ≈ 1 is the ideal signal. A
    // biased classifier hallucinates the same label on both pair
    // members → Δ ≈ 0. Anything below this threshold is treated as
    // "model can't discriminate" and dropped. Raw180's absolute
    // confidence is intentionally ignored — what matters is that the
    // model's answer *changes* under a 180° rotation, not that the
    // rotated input also crosses some absolute confidence bar.
    const MIN_ASYMMETRY: f32 = 0.3;
    const MIN_VOTES: usize = 2;
    const MAJORITY_RATIO: f32 = 0.7;

    if !engine.has_textline_orientation() || boxes.is_empty() {
        return None;
    }
    let _ = image; // dewarp samples `gray`; the RGB copy isn't used here.

    let mut selected: Vec<(usize, f32)> = boxes
        .iter()
        .enumerate()
        .filter_map(|(i, b)| {
            if b.score < MIN_DET_SCORE {
                return None;
            }
            if b.contour.is_empty() || b.contour.len() % 2 != 0 {
                return None;
            }
            let w = b.rect.right.saturating_sub(b.rect.left) as f32;
            let h = b.rect.bottom.saturating_sub(b.rect.top) as f32;
            let long = w.max(h);
            if long < MIN_LONG_PX {
                return None;
            }
            Some((i, long))
        })
        .collect();
    selected.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    selected.truncate(TOP_N);
    if selected.is_empty() {
        log::info!(
            "[estimate_canonical_quadrant] no contour-qualified detections among {} boxes",
            boxes.len()
        );
        return None;
    }

    // Per box: PCA → θ, dewarp to axis-aligned RGB strip, H-normalize
    // to WINDOW_H tall, slice up to MAX_WINDOWS_PER_STRIP windows.
    let mut windows: Vec<image::DynamicImage> = Vec::new();
    // (box_idx, θ, window_index_within_strip)
    let mut window_meta: Vec<(usize, f32, u32)> = Vec::new();
    for &(i, _) in &selected {
        let b = &boxes[i];
        let contour: Vec<(f32, f32)> = b.contour.chunks_exact(2).map(|c| (c[0], c[1])).collect();
        let theta = match contour_principal_axis_angle(&contour) {
            Some(t) => t,
            None => continue,
        };
        let deskewed_luma = match dewarp_contour_to_strip(gray, &contour, None) {
            Some(s) => s,
            None => continue,
        };
        let strip = image::DynamicImage::ImageLuma8(deskewed_luma);
        let ow = strip.width();
        let oh = strip.height();
        if oh == 0 || ow == 0 {
            continue;
        }
        let scale = WINDOW_H as f32 / oh as f32;
        let new_w = ((ow as f32) * scale).round().max(1.0) as u32;
        let normalized = strip.resize_exact(new_w, WINDOW_H, image::imageops::FilterType::Triangle);
        if new_w <= WINDOW_W {
            // Strip too short for a native-scale window. Fall back to
            // a single padded sample — preprocess_for_textline_ori
            // will resize_exact to WINDOW_W (only stretching the
            // narrow direction, no aspect crush of glyphs).
            windows.push(normalized);
            window_meta.push((i, theta, 0));
            continue;
        }
        let span = new_w - WINDOW_W;
        let n = (new_w / WINDOW_W).clamp(1, MAX_WINDOWS_PER_STRIP);
        for k in 0..n {
            let x_start = if n == 1 {
                span / 2
            } else {
                (span * k) / (n - 1)
            };
            let window = normalized.crop_imm(x_start, 0, WINDOW_W, WINDOW_H);
            windows.push(window);
            window_meta.push((i, theta, k));
        }
    }

    if windows.is_empty() {
        return None;
    }
    log::info!(
        "[estimate_canonical_quadrant] {} boxes → {} windows",
        selected.len(),
        windows.len(),
    );

    // Validation-pair classification: classify each window in its
    // original orientation AND rotated 180°. An unbiased classifier
    // MUST give opposite labels on the pair (if raw is Up, raw180 is
    // Flipped180, and vice versa). On hard content (sparse text,
    // big-glyph book covers, etc.) the binary textline-ori model has a
    // strong "Up" bias and will call both pair members Up — those
    // votes are unreliable and get discarded. Doubles the eval count
    // but the model is tiny (160×80) and this is the only place we
    // get pixel-orientation evidence in the live pipeline; cheaper
    // than tightening thresholds and rejecting everything.
    let windows_180: Vec<image::DynamicImage> = windows.iter().map(|w| w.rotate180()).collect();
    let labels = match engine.textline_orientation_classify(&windows) {
        Ok(v) => v,
        Err(e) => {
            log::debug!("[estimate_canonical_quadrant] classify(raw) failed: {e:?}");
            return None;
        }
    };
    let labels_180 = match engine.textline_orientation_classify(&windows_180) {
        Ok(v) => v,
        Err(e) => {
            log::debug!("[estimate_canonical_quadrant] classify(180) failed: {e:?}");
            return None;
        }
    };

    // Per-window vote via asymmetry. The strip was deskewed by -θ so
    // the classifier sees axis-aligned text; the only remaining
    // ambiguity is ±x reading direction, which the validation pair
    // resolves. Decision rule:
    //   Δ = P(Up | raw) − P(Up | raw180)
    //   Δ ≥ +MIN_ASYMMETRY  → vote "Up"-side  (world angle = θ)
    //   Δ ≤ -MIN_ASYMMETRY  → vote "Flipped"-side (world angle = θ + π)
    //   |Δ| < MIN_ASYMMETRY → bias-reject (model says same thing on
    //                         both pair members → no discrimination)
    //
    // |Δ| is also the vote weight — strong asymmetry = trustworthy
    // vote, weak asymmetry just barely clears the gate.
    fn p_up(c: &crate::ppocr::TextlineOriCandidate) -> f32 {
        match c.label {
            TextlineOriLabel::Up => c.score,
            TextlineOriLabel::Flipped180 => 1.0 - c.score,
        }
    }
    let mut votes: Vec<(Quadrant, f32)> = Vec::with_capacity(windows.len());
    for (idx, ((cand, cand180), &(box_idx, theta, win_k))) in labels
        .iter()
        .zip(labels_180.iter())
        .zip(window_meta.iter())
        .enumerate()
    {
        let fmt = |c: &Option<crate::ppocr::TextlineOriCandidate>| match c {
            Some(c) => format!("{:?}@{:.3}", c.label, c.score),
            None => "—".to_string(),
        };
        let theta_deg = theta.to_degrees();
        let (raw, raw180) = match (cand, cand180) {
            (Some(r), Some(r180)) => (r, r180),
            _ => {
                log::info!(
                    "[orient-window {}] box={} θ={:.1}° w{} raw={} raw180={} → rejected (missing)",
                    idx,
                    box_idx,
                    theta_deg,
                    win_k,
                    fmt(cand),
                    fmt(cand180)
                );
                continue;
            }
        };
        let delta = p_up(raw) - p_up(raw180);
        if delta.abs() < MIN_ASYMMETRY {
            log::info!(
                "[orient-window {}] box={} θ={:.1}° w{} raw={} raw180={} → rejected (Δ={:+.3}, model bias)",
                idx,
                box_idx,
                theta_deg,
                win_k,
                fmt(cand),
                fmt(cand180),
                delta
            );
            continue;
        }
        let world_angle = if delta > 0.0 {
            theta
        } else {
            theta + std::f32::consts::PI
        };
        let q = Quadrant::from_radians(world_angle);
        log::info!(
            "[orient-window {}] box={} θ={:.1}° w{} raw={} raw180={} Δ={:+.3} → vote {:?}",
            idx,
            box_idx,
            theta_deg,
            win_k,
            fmt(cand),
            fmt(cand180),
            delta,
            q
        );
        votes.push((q, delta.abs()));
    }

    if votes.len() < MIN_VOTES {
        log::info!(
            "[estimate_canonical_quadrant] only {} confident votes (need {}) — falling back to default",
            votes.len(),
            MIN_VOTES
        );
        return None;
    }

    let mut counts: [(Quadrant, f32, usize); 4] = [
        (Quadrant::R0, 0.0, 0),
        (Quadrant::R90, 0.0, 0),
        (Quadrant::R180, 0.0, 0),
        (Quadrant::R270, 0.0, 0),
    ];
    for (q, conf) in &votes {
        let idx = match q {
            Quadrant::R0 => 0,
            Quadrant::R90 => 1,
            Quadrant::R180 => 2,
            Quadrant::R270 => 3,
        };
        counts[idx].1 += conf;
        counts[idx].2 += 1;
    }
    // Tally diagnostic: every quadrant's count + avg conf, in a stable order.
    let tally_fmt = counts
        .iter()
        .map(|(q, sum, n)| {
            let avg = if *n > 0 { sum / *n as f32 } else { 0.0 };
            format!("{:?}={}@{:.3}", q, n, avg)
        })
        .collect::<Vec<_>>()
        .join(" ");
    log::info!("[orient-tally] votes={} {}", votes.len(), tally_fmt);

    let (winner, sum_conf, winner_count) = counts.iter().copied().max_by_key(|c| c.2)?;
    let majority = (winner_count as f32) / (votes.len() as f32);
    if majority < MAJORITY_RATIO {
        log::info!(
            "[estimate_canonical_quadrant] no majority: winner {:?} {}/{} (need ≥{:.0}%) — falling back to default",
            winner,
            winner_count,
            votes.len(),
            MAJORITY_RATIO * 100.0
        );
        return None;
    }
    // Each surviving vote already cleared the per-window asymmetry
    // gate (|Δ| ≥ MIN_ASYMMETRY); the winner's average |Δ| is logged
    // for diagnostics but is not used as a second gate.
    let avg_delta = sum_conf / winner_count.max(1) as f32;
    log::info!(
        "[estimate_canonical_quadrant] winner={:?} votes={}/{} avg_Δ={:.3}",
        winner,
        winner_count,
        votes.len(),
        avg_delta
    );
    Some(winner)
}

/// Inputs to [`LiveSession::run_post_detect`]. The orchestrator owns
/// per-line state internally; callers only thread inputs and pull a
/// summary out.
pub struct PostDetectInput<'a> {
    /// Detections in *camera/full-crop* coords. `tight_box` is what
    /// the surface map gets after projecting through `h_view_to_surface`;
    /// the whole struct is what the recognizer crops from.
    pub detections: &'a [DetectedTextBox],
    /// The same `OrientedImage` detection was just run on. Used to
    /// crop strips for recognition.
    pub oriented: &'a OrientedImage,
    /// View → surface homography. `None` means identity (initial
    /// acquire: canonical == camera). For mid-tracking refreshes
    /// caller passes `invert(H_root_to_view)`.
    pub h_view_to_surface: Option<[f32; 9]>,
    /// Anchor id that "owns" these detections. Threaded through to
    /// logging only; the overlay store is keyed by block id, not
    /// anchor.
    pub anchor_id: u64,
    pub from_lang: &'a str,
    pub to_lang: &'a str,
    pub is_auto_source: bool,
    pub available_codes: &'a [LanguageCode],
    pub font_provider: &'a dyn crate::font_provider::FontProvider,
    /// Per-detection matted strip (indexed parallel to `detections`).
    /// Empty falls back to the legacy pill for every strip.
    pub matted_strips: &'a [Option<MattedStrip>],
    /// Translate-block batch size. Production uses 4; sim may pick a
    /// smaller value to keep per-frame work bounded.
    pub rec_batch_size: usize,
    /// The active anchor's canonical reading-direction quadrant. `None`
    /// when not running against a tracked anchor (still-image flow, or
    /// the orientation estimator has never produced consensus and there
    /// is no fallback yet).
    pub canonical_quadrant: Option<crate::coords::Quadrant>,
}

/// Result of [`LiveSession::run_post_detect`].
#[derive(Clone, Debug, Default)]
pub struct PostDetectOutcome {
    pub anchor_id: u64,
    pub detected_count: u32,
    pub rec_ok_count: u32,
    pub rec_empty_count: u32,
    /// Number of detections that hit the surface-map cache (text was
    /// already known; no ppocr rec call ran for them). Combined with
    /// `rec_ok_count` to tell whether a refresh did real work or
    /// just confirmed existing state.
    pub cache_hits: u32,
    /// Number of detections that actually went through the
    /// recognizer this run (i.e. `detected_count - cache_hits` minus
    /// any cancelled batches).
    pub rec_called_count: u32,
    /// Stable block ids that survived this run (got a non-empty rec
    /// result and were upserted with their final translation). Caller
    /// uses this for the post-pipeline `retain_blocks` so pending
    /// placeholders for rec-failed blocks get dropped.
    pub surviving_block_ids: Vec<u64>,
    pub canceled: bool,
}

impl LiveSession {
    /// One-shot post-detect orchestration: project bboxes into
    /// surface coords, fold into the surface map, run rec on the
    /// `needs_rec` boxes, translate per block, upsert the resident
    /// overlay items, drop the placeholders for rec-failed blocks.
    /// Returns a summary the caller uses for its outcome reporting.
    ///
    /// Cancellation: `cancel` is checked before each potentially-slow
    /// stage (rec batch, translate batch). On `true`, the function
    /// returns early with `canceled = true`. Outputs already pushed
    /// into the surface map / overlay store are kept (so a cancelled
    /// run doesn't undo any partial progress).
    pub fn run_post_detect(
        &self,
        input: PostDetectInput<'_>,
        recognizer: &dyn LiveRecognizer,
        translator: &dyn LiveTranslator,
        cancel: &dyn Fn() -> bool,
    ) -> PostDetectOutcome {
        let total = input.detections.len();
        if total == 0 {
            return PostDetectOutcome {
                anchor_id: input.anchor_id,
                ..Default::default()
            };
        }

        // Project tight_boxes into surface coords (identity when
        // h_view_to_surface is None).
        let mut surface_boxes: Vec<OrientedRect> = match input.h_view_to_surface {
            None => input
                .detections
                .iter()
                .map(|d| d.tight_box.clone())
                .collect(),
            Some(h) => input
                .detections
                .iter()
                .map(|d| {
                    project_oriented_rect(&d.tight_box, &h).unwrap_or_else(|| d.tight_box.clone())
                })
                .collect(),
        };
        // Snap each box's principal-axis angle to the full reading
        // direction. `oriented_boxes_from_contour` only resolves the
        // angle modulo π (it can't tell `+x` from `-x` reading), so
        // overlay rendering would draw glyphs along `+x` even for
        // a 180°-rotated page. With the scene-level
        // `canonical_quadrant` we know which side of the principal
        // axis is reading-up, and we can flip the angle by π when it
        // disagrees. Perpendicular boxes (sideways callouts) keep
        // their own angle.
        if let Some(canon) = input.canonical_quadrant {
            let canonical_radians = canon.radians();
            let mut snapped_to_canonical = 0usize;
            let mut snapped_by_pi = 0usize;
            let mut left_perp = 0usize;
            for sb in surface_boxes.iter_mut() {
                let before = sb.angle_radians;
                let after = align_angle_to_canonical(before, canonical_radians);
                if (after - before).abs() < 1e-3 {
                    let two_pi = 2.0 * std::f32::consts::PI;
                    let diff = (canonical_radians - before).rem_euclid(two_pi);
                    if diff < std::f32::consts::FRAC_PI_4
                        || diff > 7.0 * std::f32::consts::FRAC_PI_4
                    {
                        snapped_to_canonical += 1;
                    } else {
                        left_perp += 1;
                    }
                } else {
                    snapped_by_pi += 1;
                }
                sb.angle_radians = after;
            }
            log::info!(
                "[run_post_detect] anchor={} canonical={:?} ({:.3} rad) | snap: kept_aligned={} flipped_pi={} left_perp={} of {}",
                input.anchor_id,
                canon,
                canonical_radians,
                snapped_to_canonical,
                snapped_by_pi,
                left_perp,
                surface_boxes.len(),
            );
        } else {
            log::info!(
                "[run_post_detect] anchor={} canonical=None — no angle snap (will use principal-axis modulo π for overlay rendering)",
                input.anchor_id,
            );
        }

        let outcomes = self.observe_detections(input.anchor_id, &surface_boxes, input.from_lang);

        // Per-entry rec state: text + whether rec already filled it
        // from cache. `rec_box` keeps the *camera-coord* DetectedTextBox
        // so the recognizer can crop the strip; `line_id` ties back to
        // the surface map for ingest.
        struct Entry {
            tight_surface: OrientedRect,
            rec_box: DetectedTextBox,
            line_id: SurfaceLineId,
            source_text: String,
            source_code: String,
            rec_attempted: bool,
        }
        let mut entries: Vec<Entry> = input
            .detections
            .iter()
            .zip(surface_boxes.iter())
            .zip(outcomes.iter())
            .map(|((d, surf), outcome)| {
                let mut e = Entry {
                    tight_surface: surf.clone(),
                    rec_box: d.clone(),
                    line_id: outcome.line_id,
                    source_text: String::new(),
                    source_code: input.from_lang.to_string(),
                    rec_attempted: false,
                };
                if !outcome.needs_rec && !outcome.cached_source_text.is_empty() {
                    e.source_text = outcome.cached_source_text.clone();
                    e.source_code = if outcome.cached_source_language.is_empty() {
                        input.from_lang.to_string()
                    } else {
                        outcome.cached_source_language.clone()
                    };
                    e.rec_attempted = true;
                }
                e
            })
            .collect();

        let new_lines = outcomes
            .iter()
            .filter(|o| matches!(o.kind, AddResultKind::Created))
            .count();
        let extended_lines = outcomes
            .iter()
            .filter(|o| matches!(o.kind, AddResultKind::MergedAndExtended))
            .count();
        let unchanged_lines = outcomes
            .iter()
            .filter(|o| matches!(o.kind, AddResultKind::MergedUnchanged))
            .count();
        let cache_hits = entries.iter().filter(|e| e.rec_attempted).count();
        log::debug!(
            "[post_detect] anchor={} +new={} extended={} unchanged={} cache_hits={} total={}",
            input.anchor_id,
            new_lines,
            extended_lines,
            unchanged_lines,
            cache_hits,
            total,
        );

        // Group lines into blocks using the *anchor's* surface map
        // bbox per line (may be wider than this run's projected tight
        // rect when prior observations extended it).
        let block_strip_indices: Vec<Vec<usize>>;
        let block_strips: Vec<Vec<OrientedRect>>;
        let block_ids: Vec<u64>;
        {
            let states_guard = self.anchor_states.lock();
            let snapshot_lines: Vec<crate::surface_map::SurfaceLine> = match states_guard {
                Ok(ref s) => match s.get(&input.anchor_id) {
                    Some(state) => entries
                        .iter()
                        .filter_map(|e| state.map.get(e.line_id).cloned())
                        .collect(),
                    None => Vec::new(),
                },
                Err(_) => Vec::new(),
            };
            let groups = group_surface_lines_into_blocks_in_quadrant(
                &snapshot_lines,
                input
                    .canonical_quadrant
                    .unwrap_or(crate::coords::Quadrant::R0),
            );
            block_strip_indices = groups
                .iter()
                .map(|g| {
                    g.iter()
                        .filter_map(|&snap_idx| {
                            let line_id = snapshot_lines[snap_idx].id;
                            entries.iter().position(|e| e.line_id == line_id)
                        })
                        .collect::<Vec<usize>>()
                })
                .filter(|v| !v.is_empty())
                .collect();
            block_strips = block_strip_indices
                .iter()
                .map(|idxs| {
                    idxs.iter()
                        .map(|&i| {
                            states_guard
                                .as_ref()
                                .ok()
                                .and_then(|s| s.get(&input.anchor_id))
                                .and_then(|state| state.map.get(entries[i].line_id))
                                .map(|line| line.bbox.clone())
                                .unwrap_or_else(|| entries[i].tight_surface.clone())
                        })
                        .collect()
                })
                .collect();
            block_ids = block_strip_indices
                .iter()
                .map(|idxs| {
                    let mut ids: Vec<SurfaceLineId> =
                        idxs.iter().map(|&i| entries[i].line_id).collect();
                    ids.sort_unstable();
                    stable_block_id(input.anchor_id, &ids)
                })
                .collect();
        }

        // Pending placeholders: per-strip bg rects, no text. Only
        // upsert for blocks that don't already have a resident
        // overlay (i.e. first time we see this set of lines). For a
        // refresh on a known block, blanking the overlay back to an
        // empty pill and then re-rendering the translation ~300 ms
        // later is a visible flash; the translated overlay from a
        // prior acquire is the right thing to keep on screen until
        // the new translation arrives.
        let existing_ids: std::collections::HashSet<u64> = match self.overlay_anchors.lock() {
            Ok(anchors) => anchors
                .values()
                .flat_map(|a| a.blocks.keys().copied())
                .collect(),
            Err(_) => std::collections::HashSet::new(),
        };
        for (i, &id) in block_ids.iter().enumerate() {
            if existing_ids.contains(&id) {
                continue;
            }
            let block_mats = pick_matted_for_block(input.matted_strips, &block_strip_indices[i]);
            self.upsert_block(
                input.anchor_id,
                id,
                block_strips[i].clone(),
                block_mats,
                String::new(),
                String::new(),
                input.to_lang.to_string(),
                input.font_provider,
            );
        }
        // NB: don't `retain_blocks(&block_ids)` here. That would
        // drop overlay items whose stable_block_id isn't in *this
        // run's* set — i.e. lines the detector happened to miss in
        // this single frame. PaddleOCR is non-deterministic on
        // borderline glyphs; on a held camera, missing 19 of 25
        // lines for one frame is normal and the lines still exist
        // in the surface map. Dropping their overlays makes pills
        // visibly evaporate. The only blocks we should drop are
        // those that were *re-observed and rec-failed* this run —
        // see the failed-block cleanup below the rec/translate
        // loop.

        if cancel() {
            return PostDetectOutcome {
                anchor_id: input.anchor_id,
                detected_count: total as u32,
                canceled: true,
                ..Default::default()
            };
        }

        let source_selection = if input.is_auto_source {
            OcrSourceSelection::Auto
        } else {
            OcrSourceSelection::Specific {
                language_code: LanguageCode::from(input.from_lang),
            }
        };

        let rec_batch_size = input.rec_batch_size.max(1);
        let mut block_of_entry = vec![0usize; total];
        for (bi, idxs) in block_strip_indices.iter().enumerate() {
            for &ei in idxs {
                block_of_entry[ei] = bi;
            }
        }
        let mut block_rec_remaining: Vec<usize> = block_strip_indices
            .iter()
            .map(|idxs| idxs.iter().filter(|&&i| !entries[i].rec_attempted).count())
            .collect();
        let mut block_translated = vec![false; block_ids.len()];
        // Blocks that rec'd fine but whose translation came back empty.
        // Marked translated (so we don't retry them every batch) yet
        // excluded from the survivors, so their placeholder bg is dropped
        // rather than left as an opaque, text-less box over the original.
        let mut block_dropped = vec![false; block_ids.len()];

        let mut start = 0;
        while start < total {
            if cancel() {
                return PostDetectOutcome {
                    anchor_id: input.anchor_id,
                    detected_count: total as u32,
                    canceled: true,
                    ..Default::default()
                };
            }
            let end = (start + rec_batch_size).min(total);

            let original_indices: Vec<usize> = (start..end)
                .filter(|&i| !entries[i].rec_attempted)
                .collect();
            // Boxes are canonical; the recognizer crops from `oriented.rgb`,
            // which may be rec-resolution (half) — scale to that space.
            let batch_boxes: Vec<DetectedTextBox> = input.oriented.rec_scaled_boxes(
                &original_indices
                    .iter()
                    .map(|&i| entries[i].rec_box.clone())
                    .collect::<Vec<_>>(),
            );
            let lines = if batch_boxes.is_empty() {
                Vec::new()
            } else {
                match recognizer.recognize(
                    input.oriented,
                    &batch_boxes,
                    &source_selection,
                    input.canonical_quadrant,
                ) {
                    Ok(l) => l,
                    Err(e) => {
                        log::warn!("[post_detect] recognize failed: {e}");
                        break;
                    }
                }
            };

            for (i, line) in lines.iter().enumerate() {
                let idx = match original_indices.get(i) {
                    Some(&v) => v,
                    None => break,
                };
                entries[idx].source_text = line.text.trim().to_string();
                entries[idx].rec_attempted = true;
                if input.is_auto_source {
                    if let Some(code) = &line.source_code {
                        entries[idx].source_code = code.clone();
                    }
                }
                let bi = block_of_entry[idx];
                if block_rec_remaining[bi] > 0 {
                    block_rec_remaining[bi] -= 1;
                }
                self.ingest_rec(
                    input.anchor_id,
                    entries[idx].line_id,
                    &entries[idx].source_text,
                    &entries[idx].source_code,
                );
            }

            if cancel() {
                return PostDetectOutcome {
                    anchor_id: input.anchor_id,
                    detected_count: total as u32,
                    canceled: true,
                    ..Default::default()
                };
            }

            // Which blocks just finished rec'ing all their strips?
            let mut ready_blocks: Vec<usize> = (0..block_ids.len())
                .filter(|&bi| block_rec_remaining[bi] == 0 && !block_translated[bi])
                .collect();

            if !ready_blocks.is_empty() {
                let block_sources: Vec<String> = ready_blocks
                    .iter()
                    .map(|&bi| {
                        block_strip_indices[bi]
                            .iter()
                            .map(|&i| entries[i].source_text.as_str())
                            .filter(|t| !t.is_empty())
                            .collect::<Vec<_>>()
                            .join("\n")
                    })
                    .collect();
                let kept: Vec<(usize, String)> = ready_blocks
                    .drain(..)
                    .zip(block_sources)
                    .filter(|(_, s)| !s.trim().is_empty())
                    .collect();
                if !kept.is_empty() {
                    let inputs: Vec<String> = kept.iter().map(|(_, s)| s.clone()).collect();
                    let forced = if input.is_auto_source {
                        None
                    } else {
                        Some(input.from_lang)
                    };
                    let result = translator.translate_mixed_texts(
                        &inputs,
                        forced,
                        input.to_lang,
                        input.available_codes,
                    );
                    let by_src: std::collections::HashMap<String, String> = match result {
                        Ok(res) => res
                            .translations
                            .into_iter()
                            .map(|t| (t.source_text, t.translated_text))
                            .collect(),
                        Err(e) => {
                            log::warn!("[post_detect] translate batch failed: {e}");
                            std::collections::HashMap::new()
                        }
                    };
                    for (bi, src) in kept {
                        if cancel() {
                            return PostDetectOutcome {
                                anchor_id: input.anchor_id,
                                detected_count: total as u32,
                                canceled: true,
                                ..Default::default()
                            };
                        }
                        let translated = by_src.get(&src).cloned().unwrap_or_default();
                        if translated.trim().is_empty() {
                            // Recognized fine, but translation came back
                            // empty (untranslatable input, or missing from
                            // the batch result). Mark handled so it isn't
                            // retried each batch, and drop it so its
                            // placeholder bg is removed below — the sharp
                            // original beneath beats an opaque, text-less
                            // box pinned over it.
                            block_translated[bi] = true;
                            block_dropped[bi] = true;
                            continue;
                        }
                        let kept_indices: Vec<usize> = block_strip_indices[bi]
                            .iter()
                            .copied()
                            .filter(|&i| !entries[i].source_text.is_empty())
                            .collect();
                        if kept_indices.is_empty() {
                            continue;
                        }
                        // Pull each strip's geometry from the **surface
                        // map**, not from the per-detection
                        // `entries[i].tight_surface`. The per-detection
                        // projection carries detector + RANSAC noise
                        // (1–3 px) every refresh; the map's stored
                        // `line.bbox` only mutates on
                        // `MergedAndExtended`, so for a held camera
                        // every kept strip's geometry stays bit-for-bit
                        // identical to the previous upsert → content
                        // hash matches → no re-raster, no overlay shift.
                        let line_ids: Vec<SurfaceLineId> =
                            kept_indices.iter().map(|&i| entries[i].line_id).collect();
                        let kept_strips: Vec<OrientedRect> = {
                            let states = match self.anchor_states.lock() {
                                Ok(s) => s,
                                Err(_) => continue,
                            };
                            let map = states.get(&input.anchor_id).map(|s| &s.map);
                            line_ids
                                .iter()
                                .zip(kept_indices.iter())
                                .map(|(&id, &i)| match map.and_then(|m| m.get(id)) {
                                    Some(line) => line.bbox.clone(),
                                    None => entries[i].tight_surface.clone(),
                                })
                                .collect()
                        };
                        let kept_mats = pick_matted_for_block(input.matted_strips, &kept_indices);
                        self.ingest_translation(input.anchor_id, &line_ids, &translated);
                        self.upsert_block(
                            input.anchor_id,
                            block_ids[bi],
                            kept_strips,
                            kept_mats,
                            src,
                            translated,
                            input.to_lang.to_string(),
                            input.font_provider,
                        );
                        block_translated[bi] = true;
                    }
                }
            }

            start = end;
        }

        let surviving_block_ids: Vec<u64> = block_ids
            .iter()
            .enumerate()
            .filter_map(|(bi, &id)| {
                if block_translated[bi] && !block_dropped[bi] {
                    Some(id)
                } else {
                    None
                }
            })
            .collect();
        // Drop *only* the blocks that were observed AND failed —
        // either rec returned empty, or rec succeeded but translation
        // came back empty (`block_dropped`). Either way the placeholder
        // bg is removed so the original shows through. Blocks not in
        // this run's set are untouched — see the comment up top
        // explaining why "blocks the detector missed this frame
        // shouldn't get evicted." `failed_block_ids` is the
        // complement of `surviving_block_ids` restricted to
        // `block_ids` from this run.
        let failed_block_ids: Vec<u64> = block_ids
            .iter()
            .filter(|id| !surviving_block_ids.contains(id))
            .copied()
            .collect();
        if !failed_block_ids.is_empty() {
            let failed_set: std::collections::HashSet<u64> =
                failed_block_ids.iter().copied().collect();
            let mut needs_rebuild = false;
            if let Ok(mut anchors) = self.overlay_anchors.lock() {
                if let Some(anchor) = anchors.get_mut(&input.anchor_id) {
                    let before = anchor.blocks.len();
                    anchor.blocks.retain(|id, _| !failed_set.contains(id));
                    needs_rebuild = anchor.blocks.len() != before && !anchor.blocks.is_empty();
                }
            }
            if needs_rebuild {
                // Dropping failed blocks changed the content; bump the version so the
                // GPU present rebakes the overlay without them on its next frame.
                self.content_version.fetch_add(1, Ordering::SeqCst);
            }
        }

        // Mark the viewport's surface AABB as covered for this
        // anchor. Subsequent refresh triggers compare their viewport
        // AABB against this region and gate themselves out when
        // there's nothing new visible. For the initial-acquire case
        // (h_view_to_surface = None / identity), the viewport in
        // surface coords is `(0, 0)..(rgb.W, rgb.H)`.
        let frame_w = input.oriented.gray.width() as f32;
        let frame_h = input.oriented.gray.height() as f32;
        let viewport_aabb = match input.h_view_to_surface {
            None => Aabb::from_points([
                (0.0, 0.0),
                (frame_w, 0.0),
                (frame_w, frame_h),
                (0.0, frame_h),
            ]),
            Some(h) => viewport_surface_aabb(&h, frame_w, frame_h),
        };
        if let Some(aabb) = viewport_aabb {
            self.note_coverage(input.anchor_id, aabb);
        }
        // `last_lock_h` is set by the *caller* (uniffi_catalog) once
        // `run_post_detect` returns, because the engine's H is the
        // authoritative reference and only the bindings layer knows
        // which H to pin (identity for acquire, the pending-target H
        // for refresh).

        let rec_ok = entries
            .iter()
            .filter(|e| e.rec_attempted && !e.source_text.is_empty())
            .count();
        let rec_empty = entries
            .iter()
            .filter(|e| e.rec_attempted && e.source_text.is_empty())
            .count();
        let rec_called_count = total.saturating_sub(cache_hits) as u32;

        PostDetectOutcome {
            anchor_id: input.anchor_id,
            detected_count: total as u32,
            rec_ok_count: rec_ok as u32,
            rec_empty_count: rec_empty as u32,
            cache_hits: cache_hits as u32,
            rec_called_count,
            surviving_block_ids,
            canceled: false,
        }
    }
}

fn pick_matted_for_block(
    mats: &[Option<MattedStrip>],
    entry_indices: &[usize],
) -> Vec<Option<MattedStrip>> {
    if mats.is_empty() {
        return entry_indices.iter().map(|_| None).collect();
    }
    entry_indices
        .iter()
        .map(|&i| mats.get(i).and_then(|m| m.clone()))
        .collect()
}

/// Project an `OrientedRect` through the homography `h` by projecting
/// each corner and re-fitting an `OrientedRect` from the resulting
/// Pull a box's principal-axis angle to the right side of the scene's
/// canonical reading direction.
///
/// `oriented_boxes_from_contour` only resolves the angle modulo π — it
/// can't tell apart "text reading +x" from "text reading -x" since the
/// contour shape is identical. For a 180°-rotated or 270°-rotated
/// scene, every box's angle ends up on the wrong side of the principal
/// axis and the overlay renderer draws glyphs in the original-reading
/// direction (so they appear upside down).
///
/// With the scene-level `canonical_radians` (from
/// `estimate_canonical_quadrant`) we know which side of the principal
/// axis is "reading-up", and can flip the angle by π when the input
/// disagrees. Perpendicular boxes (sideways callouts in an otherwise
/// horizontally-oriented page) are left alone — they're already on a
/// different axis and overriding them would break their layout.
pub fn align_angle_to_canonical(angle: f32, canonical_radians: f32) -> f32 {
    use std::f32::consts::{FRAC_PI_4, PI};
    let two_pi = 2.0 * PI;
    let diff = (canonical_radians - angle).rem_euclid(two_pi);
    // diff ∈ [0, 2π).
    if diff < FRAC_PI_4 || diff > 7.0 * FRAC_PI_4 {
        // Within ±π/4 of canonical → already aligned.
        angle
    } else if diff > 3.0 * FRAC_PI_4 && diff < 5.0 * FRAC_PI_4 {
        // Within ±π/4 of canonical+π → 180° flipped, snap by +π.
        angle + PI
    } else {
        // Perpendicular axis (~ canonical ±π/2). Likely a sideways
        // element in an otherwise canonically-oriented scene; keep
        // its own angle so its overlay still lines up with it.
        angle
    }
}

/// quad. For mild homographies (pan/zoom/in-plane rotation) the
/// projected quad is near-rectangular; we approximate with the
/// centroid + averaged edge direction + averaged side lengths. Returns
/// `None` if any corner failed to project (e.g. h is non-invertible
/// or quad collapses past the projective horizon).
pub fn project_oriented_rect(rect: &OrientedRect, h: &[f32; 9]) -> Option<OrientedRect> {
    let mut projected: [(f32, f32); 4] = [(0.0, 0.0); 4];
    for (i, (x, y)) in rect.corners().into_iter().enumerate() {
        let p = project(h, x, y)?;
        projected[i] = p;
    }
    let cx = 0.25 * (projected[0].0 + projected[1].0 + projected[2].0 + projected[3].0);
    let cy = 0.25 * (projected[0].1 + projected[1].1 + projected[2].1 + projected[3].1);
    let top_dx = projected[1].0 - projected[0].0;
    let top_dy = projected[1].1 - projected[0].1;
    let bot_dx = projected[2].0 - projected[3].0;
    let bot_dy = projected[2].1 - projected[3].1;
    let angle = (top_dy + bot_dy).atan2(top_dx + bot_dx);
    let width =
        0.5 * ((top_dx.powi(2) + top_dy.powi(2)).sqrt() + (bot_dx.powi(2) + bot_dy.powi(2)).sqrt());
    let left_dx = projected[3].0 - projected[0].0;
    let left_dy = projected[3].1 - projected[0].1;
    let right_dx = projected[2].0 - projected[1].0;
    let right_dy = projected[2].1 - projected[1].1;
    let height = 0.5
        * ((left_dx.powi(2) + left_dy.powi(2)).sqrt()
            + (right_dx.powi(2) + right_dy.powi(2)).sqrt());
    if !(width.is_finite() && height.is_finite() && width > 0.0 && height > 0.0) {
        return None;
    }
    Some(OrientedRect {
        cx,
        cy,
        width,
        height,
        angle_radians: angle,
    })
}

/// `invert(h)` convenience that maps `H_root→view` into `H_view→surface`
/// for the detect-on-tracking-frame trigger. Caller passes the
/// homography it just got from the planar engine.
pub fn h_view_to_surface_from(h_root_to_view: &[f32; 9]) -> Option<[f32; 9]> {
    invert(h_root_to_view)
}

/// Decide which string to render for an item. Translation wins when
/// non-empty; otherwise empty (pending placeholder). Mirrors the
/// bindings-side `pick_display_text` for the run_acquire_pipeline-era
/// pill renderer.
fn pick_display_text(_source_text: &str, translated_text: &str) -> String {
    if !translated_text.trim().is_empty() {
        translated_text.to_string()
    } else {
        String::new()
    }
}

/// Content hash for `upsert_block` change detection. Same hash → same
/// rasterized bitmap, no need to re-render.
fn block_content_hash(strips: &[OrientedRect], display_text: &str, language: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    (strips.len() as u64).hash(&mut h);
    for s in strips {
        s.cx.to_bits().hash(&mut h);
        s.cy.to_bits().hash(&mut h);
        s.width.to_bits().hash(&mut h);
        s.height.to_bits().hash(&mut h);
        s.angle_radians.to_bits().hash(&mut h);
    }
    display_text.hash(&mut h);
    language.hash(&mut h);
    h.finish()
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

/// Diagnostic: tint each block's bg with a deterministic palette
/// colour (selected by `block_id % 8`) so we can see at a glance
/// which pixels belong to which block. Useful for spotting
/// inter-block overlap (different colours intermixing) vs
/// intra-block double-fill artefacts (same colour appearing
/// darker — should never happen due to cap-by-max in
/// `fill_oriented_rect_blended`, but worth eyeballing). Flip off
/// for production.
pub const DEBUG_PER_BLOCK_BG_COLOR: bool = false;

/// 8-color palette for [`DEBUG_PER_BLOCK_BG_COLOR`]. All entries
/// share the same alpha as the default bg (0xC8 = 200/255) and
/// similar luma so text legibility doesn't change wildly between
/// blocks — only the hue.
pub const DEBUG_BG_PALETTE: [[u8; 4]; 8] = [
    [0x50, 0x10, 0x10, 0xC8], // crimson
    [0x10, 0x40, 0x10, 0xC8], // forest
    [0x10, 0x20, 0x50, 0xC8], // navy
    [0x50, 0x40, 0x10, 0xC8], // olive
    [0x50, 0x10, 0x50, 0xC8], // magenta
    [0x10, 0x40, 0x40, 0xC8], // teal
    [0x60, 0x30, 0x10, 0xC8], // rust
    [0x30, 0x10, 0x50, 0xC8], // indigo
];

/// Horizontal padding (per side) on the visible pill vs the
/// detector's tight rect. Keeps glyph edges off the rounded corner.
pub const HORIZONTAL_PAD_PX: f32 = 8.0;

/// Extra padding on the block's bitmap AABB to give rounded-corner
/// antialiasing room.
pub const ITEM_BITMAP_PAD_PX: f32 = 4.0;

/// Angles within this of axis-aligned are snapped to exactly 0 — detection noise on
/// world-up text isn't worth the slow rotated pill fill + atlas bypass. 1°.
const AXIS_SNAP_RAD: f32 = 0.017_453_293;

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

/// Inflate one block's detector strips into the visible pill rects (surface
/// coords): pad width, inflate height, snap the shared basis via
/// [`normalize_block_visuals_rotated_basis`], then snap near-axis angles to 0.
/// Empty when no strip survives the width/height guard. Used by
/// [`build_overlay_draw_list`] and the per-block glyph shaping so both produce
/// identical pill geometry.
fn inflate_block_visuals(spec: &BlockSpec) -> Vec<OrientedRect> {
    let mut visuals: Vec<OrientedRect> = spec
        .strips
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
        return visuals;
    }
    normalize_block_visuals_rotated_basis(&mut visuals);
    for v in &mut visuals {
        if v.angle_radians.abs() < AXIS_SNAP_RAD {
            v.angle_radians = 0.0;
        }
    }
    visuals
}

/// Inflate provisional strips into pill rects (surface coords): same width/height
/// inflation as [`inflate_block_visuals`] but each strip stands alone (no
/// grouping/normalize), with the axis snap applied to the raw angle first.
fn inflate_provisional_visuals(strips: &[OrientedRect]) -> Vec<OrientedRect> {
    strips
        .iter()
        .filter_map(|s| {
            let angle = if s.angle_radians.abs() < AXIS_SNAP_RAD {
                0.0
            } else {
                s.angle_radians
            };
            let v = OrientedRect {
                cx: s.cx,
                cy: s.cy,
                width: s.width + 2.0 * HORIZONTAL_PAD_PX,
                height: s.height * TIGHT_VERTICAL_INFLATE,
                angle_radians: angle,
            };
            if v.width <= 0.0 || v.height <= 0.0 {
                None
            } else {
                Some(v)
            }
        })
        .collect()
}

/// Union AABB (surface coords) over a set of oriented pills' corners. `None` when
/// the set is empty (no finite bound).
fn pills_union_aabb(pills: &[OrientedRect]) -> Option<(f32, f32, f32, f32)> {
    let (mut min_x, mut min_y) = (f32::INFINITY, f32::INFINITY);
    let (mut max_x, mut max_y) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for v in pills {
        for (x, y) in v.corners() {
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
    }
    if min_x.is_finite() && min_y.is_finite() && max_x.is_finite() && max_y.is_finite() {
        Some((min_x, min_y, max_x, max_y))
    } else {
        None
    }
}

/// Canvas geometry (origin in surface px, bitmap dims in texels) for a pill set:
/// AABB → padded origin → floor-sized dims. The block-local glyph shaping and the
/// shared draw-list build use the same chain so pen positions line up after the
/// block-origin → canvas-origin offset.
fn canvas_geometry(pills: &[OrientedRect], os: f32) -> Option<(f32, f32, u32, u32)> {
    let (min_x, min_y, max_x, max_y) = pills_union_aabb(pills)?;
    let pad = ITEM_BITMAP_PAD_PX;
    let origin_x = (min_x - pad).max(0.0);
    let origin_y = (min_y - pad).max(0.0);
    let bitmap_w = (((max_x + pad - origin_x) * os).floor() as i32).max(1) as u32;
    let bitmap_h = (((max_y + pad - origin_y) * os).floor() as i32).max(1) as u32;
    Some((origin_x, origin_y, bitmap_w, bitmap_h))
}

/// One opaque pill the GPU draws as an oriented rounded-rect quad. `rect` is in
/// **surface (canonical)** coords; the compositor localizes it into the overlay
/// texture (`(p − origin) * oversample`). `color` is straight RGBA.
#[derive(Clone, Debug)]
pub struct OverlayPill {
    pub rect: OrientedRect,
    pub color: [u8; 4],
}

/// Backend-agnostic GPU draw list for one anchor's overlay: oriented pill quads
/// (surface coords) + per-glyph instance data for the GPU atlas compositor, plus
/// the canvas geometry the compositor renders into and the `painted_pills` the
/// movement monitor masks. The GPU compositor renders this into an overlay texture
/// (rounded pills + glyph quads), which the present then warps by the homography
/// (camera) or blits at identity (screen). Shared by both paths; the camera differs
/// only in the present transform.
#[derive(Default)]
pub struct OverlayDrawList {
    /// Canvas (overlay-texture) origin in surface coords.
    pub origin_x: f32,
    pub origin_y: f32,
    /// Texels per surface unit baked into the texture.
    pub oversample: f32,
    /// Overlay texture dims (texels).
    pub bitmap_w: u32,
    pub bitmap_h: u32,
    pub pills: Vec<OverlayPill>,
    pub glyphs: crate::image_render::GlyphCollector,
    /// Pill footprints (surface coords) for the movement monitor's mask.
    pub painted_pills: Vec<OrientedRect>,
}

/// Build the GPU draw list for one anchor's overlay: oriented pill quads + per-glyph
/// instance data for the GPU atlas compositor. Glyphs are shaped once per content
/// change in [`LiveSession::shape_block_glyphs`] (block-local pen positions); this
/// only offsets each block's instances by its block-origin → canvas-origin delta and
/// merges the per-block masks — O(N_glyphs) arithmetic, no font access. The GPU
/// places and rotates the upright masks, so a tilted line reuses the same atlas
/// entry as axis-aligned text. `None` when there is nothing to show.
pub(crate) fn build_overlay_draw_list(
    blocks: &std::collections::BTreeMap<u64, BlockSpec>,
    provisional_strips: &[OrientedRect],
    oversample: f32,
    bg_rgba: [u8; 4],
) -> Option<OverlayDrawList> {
    if blocks.is_empty() && provisional_strips.is_empty() {
        return None;
    }
    let os = oversample.max(1.0);

    struct PreparedBlock<'a> {
        block_id: u64,
        visuals: Vec<OrientedRect>,
        spec: &'a BlockSpec,
    }
    let mut prepared: Vec<PreparedBlock<'_>> = Vec::with_capacity(blocks.len());
    for (block_id, spec) in blocks {
        let visuals = inflate_block_visuals(spec);
        if visuals.is_empty() {
            continue;
        }
        prepared.push(PreparedBlock {
            block_id: *block_id,
            visuals,
            spec,
        });
    }
    let prepared_provisional = inflate_provisional_visuals(provisional_strips);
    if prepared.is_empty() && prepared_provisional.is_empty() {
        return None;
    }

    let painted_pills: Vec<OrientedRect> = prepared
        .iter()
        .flat_map(|pb| pb.visuals.iter().copied())
        .chain(prepared_provisional.iter().copied())
        .collect();

    let (origin_x, origin_y, bitmap_w, bitmap_h) = canvas_geometry(&painted_pills, os)?;

    // Pills: block strips (matted/default color per strip index) then provisional
    // strips (default bg), in surface coords — drawn as oriented rounded quads.
    let mut pills: Vec<OverlayPill> = Vec::with_capacity(painted_pills.len());
    for pb in &prepared {
        for (i, v) in pb.visuals.iter().enumerate() {
            let color = block_pill_color(pb.spec, i, pb.block_id, bg_rgba);
            pills.push(OverlayPill { rect: *v, color });
        }
    }
    for v in &prepared_provisional {
        pills.push(OverlayPill {
            rect: *v,
            color: bg_rgba,
        });
    }

    // Glyphs were shaped block-local at upsert time (pen positions relative to each
    // block's own AABB origin × oversample). Offset each block's instances by the
    // delta from its block origin to the shared canvas origin — `shape_block_glyphs`
    // uses the same `canvas_geometry(block visuals)` origin, so adding the delta
    // reconstructs canvas-texel pen positions exactly — and merge the masks (deduped
    // by key) for the GPU atlas upload.
    let mut glyphs = crate::image_render::GlyphCollector::default();
    for pb in &prepared {
        let Some((block_origin_x, block_origin_y, _, _)) = canvas_geometry(&pb.visuals, os) else {
            continue;
        };
        let dx = (block_origin_x - origin_x) * os;
        let dy = (block_origin_y - origin_y) * os;
        for inst in &pb.spec.glyph_instances {
            glyphs
                .instances
                .push(crate::image_render::GlyphInstanceData {
                    key: inst.key,
                    pen_x: inst.pen_x + dx,
                    pen_y: inst.pen_y + dy,
                    cos: inst.cos,
                    sin: inst.sin,
                    color: inst.color,
                });
        }
        for (key, mask) in &pb.spec.glyph_masks {
            glyphs.masks.entry(*key).or_insert_with(|| mask.clone());
        }
    }

    Some(OverlayDrawList {
        origin_x,
        origin_y,
        oversample: os,
        bitmap_w,
        bitmap_h,
        pills,
        glyphs,
        painted_pills,
    })
}

/// Run `f` with a `FontCache`: the persistent one (locked) when provided, else a
/// fresh throwaway. Lets the screen reuse its glyph atlas across renders while the
/// camera/sim stay on per-call caches.
fn with_glyph_cache<R>(
    glyph_cache: Option<&Mutex<crate::image_render::FontCache>>,
    f: impl FnOnce(&mut crate::image_render::FontCache) -> R,
) -> R {
    match glyph_cache {
        Some(m) => f(&mut m.lock().expect("glyph cache poisoned")),
        None => f(&mut crate::image_render::FontCache::default()),
    }
}

/// Translate a block's surface-coord visuals into oversampled canvas-local coords.
/// Shared by the full build (step 3) and the screen incremental update.
fn localize_visuals(
    visuals: &[OrientedRect],
    origin_x: f32,
    origin_y: f32,
    os: f32,
) -> Vec<OrientedRect> {
    visuals
        .iter()
        .map(|v| OrientedRect {
            cx: (v.cx - origin_x) * os,
            cy: (v.cy - origin_y) * os,
            width: v.width * os,
            height: v.height * os,
            angle_radians: v.angle_radians,
        })
        .collect()
}

/// Background color for a block's i-th pill: the matted per-strip color when the
/// matting resolved a uniform bg, else the anchor default (or a debug palette
/// entry keyed by block id when `DEBUG_PER_BLOCK_BG_COLOR`).
fn block_pill_color(spec: &BlockSpec, i: usize, block_id: u64, bg_rgba: [u8; 4]) -> [u8; 4] {
    let default_bg = if DEBUG_PER_BLOCK_BG_COLOR {
        DEBUG_BG_PALETTE[(block_id as usize) % DEBUG_BG_PALETTE.len()]
    } else {
        bg_rgba
    };
    spec.matted_strips
        .get(i)
        .and_then(|m| m.as_ref())
        .and_then(|m| m.bg_uniform_argb)
        .map(argb_to_rgba_bytes)
        .unwrap_or(default_bg)
}

/// Build one block's `PreparedTextBlock` (per-line oriented text boxes + layout
/// hints) from its spec + localized visuals. Returns `None` for empty-text
/// placeholder blocks (they keep their bg-only pill, no glyph pass). Shared by the
/// full build (step 4) and the screen incremental update.
fn build_block_text_block(
    spec: &BlockSpec,
    visuals: &[OrientedRect],
    local: &[OrientedRect],
    os: f32,
    bitmap_w: u32,
    bitmap_h: u32,
) -> Option<crate::ocr::PreparedTextBlock> {
    use crate::ocr::{
        OverlayLayoutHints, OverlayLayoutMode, PreparedTextBlock, PreparedTextLine, Rect,
    };
    if spec.display_text.trim().is_empty() {
        return None;
    }
    let foreground_argb: u32 = spec
        .matted_strips
        .iter()
        .find_map(|m| m.as_ref().map(|s| s.ink_is_dark))
        .map(|dark| if dark { 0xFF10_1010 } else { 0xFFFF_FFFF })
        .unwrap_or(0xFFFF_FFFF);
    let lines: Vec<PreparedTextLine> = local
        .iter()
        .map(|v| {
            let text_box = OrientedRect {
                cx: v.cx,
                cy: v.cy,
                width: (v.width
                    - 2.0 * crate::planar_engine::OVERLAY_TEXT_HORIZONTAL_INSET_PX * os)
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
        .clamp(10.0, 120.0)
        * os;
    let block_bbox = block_aabb_within_canvas(local, bitmap_w, bitmap_h);
    Some(PreparedTextBlock {
        source_text: String::new(),
        translated_text: spec.display_text.clone(),
        bounding_box: block_bbox,
        lines,
        layout_hints: OverlayLayoutHints {
            layout_mode: OverlayLayoutMode::PerLine,
            suggested_font_size_px: suggested_font_px,
        },
        background_argb: 0,
        foreground_argb,
    })
}

fn block_aabb_within_canvas(
    local: &[OrientedRect],
    canvas_w: u32,
    canvas_h: u32,
) -> crate::ocr::Rect {
    let mut min_l = u32::MAX;
    let mut min_t = u32::MAX;
    let mut max_r: u32 = 0;
    let mut max_b: u32 = 0;
    for v in local {
        let aabb = v.to_aabb();
        if aabb.left < min_l {
            min_l = aabb.left;
        }
        if aabb.top < min_t {
            min_t = aabb.top;
        }
        if aabb.right > max_r {
            max_r = aabb.right;
        }
        if aabb.bottom > max_b {
            max_b = aabb.bottom;
        }
    }
    crate::ocr::Rect {
        left: min_l.min(canvas_w.saturating_sub(1)),
        top: min_t.min(canvas_h.saturating_sub(1)),
        right: max_r.min(canvas_w),
        bottom: max_b.min(canvas_h),
    }
}

/// Group `SurfaceLine`s into translation blocks (paragraphs) via the
/// shared OCR grouping. Returns indices into the input slice. R0
/// variant kept for callers that don't track a scene canonical
/// quadrant.
pub fn group_surface_lines_into_blocks(
    lines: &[crate::surface_map::SurfaceLine],
) -> Vec<Vec<usize>> {
    group_surface_lines_into_blocks_in_quadrant(lines, crate::coords::Quadrant::R0)
}

/// Canonical-quadrant-aware variant: routes through
/// `crate::ocr::group_live_lines_into_blocks_in_quadrant` so that scenes
/// captured with the camera rotated produce blocks in the page's actual
/// reading order rather than image-y order.
pub fn group_surface_lines_into_blocks_in_quadrant(
    lines: &[crate::surface_map::SurfaceLine],
    canonical_quadrant: crate::coords::Quadrant,
) -> Vec<Vec<usize>> {
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
    let blocks =
        crate::ocr::group_live_lines_into_blocks_in_quadrant(text_lines, canonical_quadrant);
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
pub fn stable_block_id(
    anchor_id: AnchorId,
    sorted_line_ids: &[crate::surface_map::SurfaceLineId],
) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in anchor_id.to_le_bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    for &id in sorted_line_ids {
        for byte in id.to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    hash | (1u64 << 63)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warp_oriented_box_identity_round_trip() {
        let b = OrientedRect {
            cx: 100.0,
            cy: 200.0,
            width: 50.0,
            height: 20.0,
            angle_radians: 0.1,
        };
        let identity = [1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let w = warp_oriented_box(&b, &identity).expect("identity warp");
        assert!((w.cx - b.cx).abs() < 1e-3);
        assert!((w.cy - b.cy).abs() < 1e-3);
        assert!((w.width - b.width).abs() < 1e-3);
        assert!((w.height - b.height).abs() < 1e-3);
        assert!((w.angle_radians - b.angle_radians).abs() < 1e-3);
    }

    #[test]
    fn warp_oriented_box_translation_shifts_center() {
        let b = OrientedRect {
            cx: 50.0,
            cy: 50.0,
            width: 30.0,
            height: 10.0,
            angle_radians: 0.0,
        };
        let translate = [1.0_f32, 0.0, 17.0, 0.0, 1.0, -23.0, 0.0, 0.0, 1.0];
        let w = warp_oriented_box(&b, &translate).expect("translate warp");
        assert!((w.cx - (b.cx + 17.0)).abs() < 1e-3);
        assert!((w.cy - (b.cy - 23.0)).abs() < 1e-3);
        assert!((w.width - b.width).abs() < 1e-3);
        assert!((w.height - b.height).abs() < 1e-3);
        assert!((w.angle_radians - b.angle_radians).abs() < 1e-3);
    }

    #[test]
    fn relock_by_view_ignores_pure_translation_pan() {
        let session = LiveSession::new();
        let anchor_id = 7;
        let identity = [1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        session.set_last_lock_h(anchor_id, identity);

        let pan = [1.0_f32, 0.0, 420.0, 0.0, 1.0, -130.0, 0.0, 0.0, 1.0];
        assert!(
            !session.should_relock_by_view(anchor_id, &pan, 1000.0, 800.0, 0.75),
            "pure pan should keep using the existing surface map; coverage refresh handles newly visible areas"
        );
    }

    #[test]
    fn relock_by_view_triggers_on_zoom_change() {
        let session = LiveSession::new();
        let anchor_id = 7;
        let identity = [1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        session.set_last_lock_h(anchor_id, identity);

        let zoom_in = [2.0_f32, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0];
        assert!(
            session.should_relock_by_view(anchor_id, &zoom_in, 1000.0, 800.0, 0.75),
            "2x zoom shrinks the projected surface viewport area to 25%"
        );

        let zoom_out = [0.5_f32, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0];
        assert!(
            session.should_relock_by_view(anchor_id, &zoom_out, 1000.0, 800.0, 0.75),
            "0.5x zoom grows the projected surface viewport area to 400%"
        );
    }

    #[test]
    fn relock_by_view_ignores_in_plane_rotation() {
        let session = LiveSession::new();
        let anchor_id = 7;
        let identity = [1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        session.set_last_lock_h(anchor_id, identity);

        let (cx, cy) = (500.0_f32, 400.0_f32);
        let angle = 25.0_f32.to_radians();
        let c = angle.cos();
        let s = angle.sin();
        let rotate_about_center = [
            c,
            -s,
            cx - c * cx + s * cy,
            s,
            c,
            cy - s * cx - c * cy,
            0.0,
            0.0,
            1.0,
        ];
        assert!(
            !session.should_relock_by_view(anchor_id, &rotate_about_center, 1000.0, 800.0, 0.75),
            "in-plane rotation preserves viewport area in surface coordinates"
        );
    }
}
