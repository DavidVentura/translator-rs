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

/// One rasterized overlay item resident across composite calls. The
/// caller hashes the source content (strips + texts + language) and
/// only re-rasterizes items whose hash changed, so dense pages with
/// stable content stay cheap to render.
#[derive(Clone)]
pub struct OverlayItem {
    pub id: u64,
    /// Anchor whose canonical frame this overlay lives in. The
    /// compositor warps the bitmap by *that anchor's*
    /// `H_root→view`; items for inactive anchors are skipped at
    /// compose time so cached state for a cached anchor doesn't
    /// render at the wrong place under the current anchor's H.
    pub anchor_id: AnchorId,
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
    /// Resident rasterized overlays. Each item carries the
    /// `anchor_id` it belongs to; the compositor filters to only the
    /// currently-active anchor so cached anchors' overlays don't
    /// render at wrong positions under another anchor's H.
    pub overlay_items: Mutex<Vec<OverlayItem>>,
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
            overlay_items: Mutex::new(Vec::new()),
            locked_frames_since_acquire: AtomicU64::new(0),
            last_refresh_locked_frame: AtomicU64::new(0),
            refresh_every_n_locked_frames: AtomicU32::new(DEFAULT_REFRESH_EVERY_N_LOCKED_FRAMES),
        }
    }

    /// Drop all session state. Caller invokes on tap-to-focus,
    /// language change, or any other coarse-grained reset signal.
    pub fn clear(&self) {
        if let Ok(mut states) = self.anchor_states.lock() {
            states.clear();
        }
        if let Ok(mut items) = self.overlay_items.lock() {
            items.clear();
        }
        self.locked_frames_since_acquire.store(0, Ordering::SeqCst);
        self.last_refresh_locked_frame.store(0, Ordering::SeqCst);
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
        if let Ok(mut items) = self.overlay_items.lock() {
            items.retain(|it| it.anchor_id != anchor_id);
        }
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
        if let Ok(mut items) = self.overlay_items.lock() {
            items.retain(|it| keep_set.contains(&it.anchor_id));
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

    /// Wipe the surface map + overlay items for `anchor_id` so the
    /// next `run_post_detect` starts from empty state. Used by the
    /// re-lock pipeline to convert the existing merge semantics into
    /// an atomic replace: clear → detect → OCR → translate → insert
    /// into the empty map. `covered_region` and `lock_viewport` are
    /// preserved here; the caller updates `lock_viewport` after the
    /// fresh pass succeeds (via `set_lock_viewport`).
    pub fn clear_anchor_state_for_relock(&self, anchor_id: AnchorId) {
        if let Ok(mut states) = self.anchor_states.lock() {
            if let Some(state) = states.get_mut(&anchor_id) {
                state.map = SurfaceMap::new();
            }
        }
        if let Ok(mut items) = self.overlay_items.lock() {
            items.retain(|it| it.anchor_id != anchor_id);
        }
    }

    pub fn clear_overlays(&self) {
        if let Ok(mut items) = self.overlay_items.lock() {
            items.clear();
        }
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

    /// Upsert one resident overlay item. Re-rasters only when the
    /// content hash (strips + display text + language) changed since
    /// the previous upsert for `id`; otherwise this is a no-op and
    /// the cached bitmap survives. `matted_strips` is indexed parallel
    /// to `strips` and may be empty to fall back to the legacy pill
    /// rendering for every strip.
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
        // Fast path: same content → keep the cached bitmap.
        if let Ok(items) = self.overlay_items.lock() {
            if let Some(existing) = items.iter().find(|it| it.id == id) {
                if existing.content_hash == hash {
                    return;
                }
            }
        }
        let raster = match render_block_bitmap(
            id,
            &strips,
            &matted_strips,
            &display_text,
            &language,
            font_provider,
        ) {
            Some(r) => r,
            None => return,
        };
        if let Ok(mut items) = self.overlay_items.lock() {
            let new_item = OverlayItem {
                id,
                anchor_id,
                bitmap: raster.bitmap,
                width: raster.width,
                height: raster.height,
                surface_origin_x: raster.surface_origin_x,
                surface_origin_y: raster.surface_origin_y,
                content_hash: hash,
            };
            if let Some(slot) = items.iter_mut().find(|it| it.id == id) {
                *slot = new_item;
            } else {
                items.push(new_item);
            }
        }
    }

    /// Drop overlay items for `anchor_id` whose id isn't in `ids`.
    /// Used when an acquire / refresh finishes (final block id set
    /// known) so stale overlays from a prior pipeline run on the
    /// same anchor don't linger. Items for *other* anchors are
    /// untouched.
    pub fn retain_blocks(&self, anchor_id: AnchorId, ids: &[u64]) {
        let keep: std::collections::HashSet<u64> = ids.iter().copied().collect();
        if let Ok(mut items) = self.overlay_items.lock() {
            items.retain(|it| it.anchor_id != anchor_id || keep.contains(&it.id));
        }
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
    ) -> Result<Vec<RecognizedTextLine>, String> {
        (*self)
            .recognize_in_oriented_image(oriented, boxes, source_selection.clone())
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
        let surface_boxes: Vec<OrientedRect> = match input.h_view_to_surface {
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
            let groups = group_surface_lines_into_blocks(&snapshot_lines);
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
        let existing_ids: std::collections::HashSet<u64> = match self.overlay_items.lock() {
            Ok(items) => items.iter().map(|it| it.id).collect(),
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
            let batch_boxes: Vec<DetectedTextBox> = original_indices
                .iter()
                .map(|&i| entries[i].rec_box.clone())
                .collect();
            let lines = if batch_boxes.is_empty() {
                Vec::new()
            } else {
                match recognizer.recognize(input.oriented, &batch_boxes, &source_selection) {
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
            .filter_map(|(bi, &id)| if block_translated[bi] { Some(id) } else { None })
            .collect();
        // Drop *only* the blocks that were observed AND failed
        // (placeholder upserted, rec returned empty). Blocks not in
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
            if let Ok(mut items) = self.overlay_items.lock() {
                items.retain(|it| it.anchor_id != input.anchor_id || !failed_set.contains(&it.id));
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
pub const DEBUG_PER_BLOCK_BG_COLOR: bool = true;

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
    block_id: u64,
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
    let default_bg = if DEBUG_PER_BLOCK_BG_COLOR {
        DEBUG_BG_PALETTE[(block_id as usize) % DEBUG_BG_PALETTE.len()]
    } else {
        [0x10, 0x10, 0x10, 0xC8]
    };
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
            &mut rgba,
            bitmap_w,
            bitmap_h,
            v,
            strip_color,
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
pub fn group_surface_lines_into_blocks(
    lines: &[crate::surface_map::SurfaceLine],
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
