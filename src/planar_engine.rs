//! Lifecycle state machine + anchor LRU cache for the planar tracker.
//!
//! `planar_tracker` does pure per-frame detection / matching / fitting.
//! This module owns:
//!   - the Acquiring → Locked → Lost state machine (Phase E)
//!   - what counts as "scene change" / "refresh needed" (E + F)
//!   - the LRU cache of recently-seen anchors with their translated
//!     overlay sets, so flipping back to a known scene is instant (G)
//!
//! See `FUTURE_PLANAR_TRACKER.md` for the design rationale.

use image::GrayImage;

use crate::homography::mat3_mul;
use crate::planar_tracker::{
    SceneAnchor, TrackResult, TrackerConfig, build_anchor, build_anchor_in_regions,
    track_against_anchor, track_against_anchor_with_prior,
};

#[cfg(feature = "image-render")]
use crate::font_provider::FontProvider;
#[cfg(feature = "image-render")]
use crate::image_render::{RenderOptions, render_overlay};
#[cfg(feature = "image-render")]
use crate::ocr::{
    OrientedRect, OverlayLayoutHints, OverlayLayoutMode, PreparedImageOverlay, PreparedTextBlock,
    PreparedTextLine, Rect,
};

/// Stable identifier for a captured scene. Increases monotonically; we
/// never reuse an id even after an anchor is evicted from the LRU
/// cache, so downstream caches can safely key on it.
pub type AnchorId = u64;

/// One overlay's geometry in *canonical-frame coordinates* (the frame
/// at which the anchor was acquired). The engine doesn't interpret
/// `payload` — it's opaque to Rust. Kotlin uses it to carry whatever
/// it needs (translated text, font hints, source bbox id).
#[derive(Clone, Debug)]
pub struct CanonicalOverlay {
    pub id: u64,
    /// Four corners (top-left, top-right, bottom-right, bottom-left)
    /// in canonical-frame pixel coordinates.
    pub quad: [(f32, f32); 4],
    pub payload: String,
}

/// One overlay's geometry in *current-frame coordinates* — what the
/// renderer actually draws. Output of `project_overlays`.
#[derive(Clone, Debug)]
pub struct OverlayProjection {
    pub id: u64,
    pub quad: [(f32, f32); 4],
    pub payload: String,
}

/// What the engine wants Kotlin to do this frame.
#[derive(Clone, Debug)]
pub enum TrackerCommand {
    /// No anchor; nothing to draw. Wait for `imu_stable && stable_required_ns`
    /// to elapse, then call `acquire_now`.
    Idle,
    /// We're inside the stable-frame quiet window. Same render hint as
    /// `Idle` (no overlays to project yet); useful to display a
    /// "Looking…" indicator.
    Acquiring,
    /// Locked on `anchor_id`. Project the anchor's canonical overlays
    /// through `homography` and render them. `is_new` is true the
    /// first frame after a fresh acquisition (Kotlin should run OCR
    /// then); false on subsequent frames or when we've snapped back to
    /// a cached anchor (skip OCR — Phase F caching benefit).
    Locked {
        anchor_id: AnchorId,
        homography: [f32; 9],
        is_new: bool,
        inliers: usize,
    },
    /// We had `last_anchor_id` locked recently but lost the track.
    /// Kotlin can briefly hide overlays or extrapolate via IMU until a
    /// later frame re-locks or we time out and go back to `Idle`.
    Lost { last_anchor_id: AnchorId },
}

/// Engine tuning knobs not covered by `TrackerConfig`.
#[derive(Clone, Copy, Debug)]
pub struct EngineConfig {
    pub tracker: TrackerConfig,
    /// LRU capacity for cached anchors.
    pub anchor_cache_size: usize,
    /// Min time between successive acquires (nanoseconds). Prevents
    /// thrashing when the user is still settling on a scene.
    pub acquire_cooldown_ns: u64,
    /// After this many consecutive `track == None` frames, transition
    /// Locked → Lost.
    pub lost_after_frames: u32,
    /// After this many additional frames in Lost without a re-lock,
    /// transition Lost → Idle and clear active anchor.
    pub give_up_after_frames: u32,
    /// Quiet-IMU period required before an Idle auto-acquire fires
    /// (nanoseconds).
    pub stable_required_ns: u64,
    /// Refresh trigger: if a Locked anchor is older than this and the
    /// scene is still locked, `should_refresh` will return true so
    /// Kotlin can re-run OCR to absorb new text.
    pub anchor_refresh_age_ns: u64,
    /// Spawn a new (handoff) anchor when the active anchor's per-frame
    /// RANSAC inlier count falls below this. Picked well above the
    /// wobble floor so the handoff homography fit is also stable.
    pub handoff_min_inliers: usize,
    /// Spawn a new (handoff) anchor when this fraction of the active
    /// anchor's keypoints fail to project inside the viewport — i.e.
    /// most features are about to leave the frame.
    pub handoff_min_visible_ratio: f32,
    /// Spawn a new (handoff) anchor when `|ln(approx_scale(H_anchor→view))|`
    /// exceeds this threshold. BRIEF descriptors are scale-variant; once
    /// the view's effective scale diverges past ~1.35× from acquire
    /// (`ln(1.35) ≈ 0.30`) per-keypoint Hamming matching starts losing
    /// inliers from descriptor drift before the inlier-count or
    /// visible-ratio gates trip. Catching the scale change earlier lets
    /// the handoff fit on still-co-visible features rather than at the
    /// bottom of an inlier cliff.
    pub handoff_scale_log_threshold: f32,
    /// Cooldown between successive handoffs. Prevents churn when
    /// inliers oscillate near the trigger threshold.
    pub handoff_cooldown_ns: u64,
    /// Inlier-discontinuity sanity gate: reject a new RANSAC fit when
    /// its inlier count drops below `sanity_gate_drop_ratio * EMA`. A
    /// drop this severe in one frame is almost always a wrong-basin
    /// fit or a descriptor-collapse event, not a real world change.
    pub sanity_gate_drop_ratio: f32,
    /// Only apply the sanity gate when the running EMA is at least
    /// this high. Below this, the lock is already marginal and a
    /// "discontinuity" carries less signal than noise.
    pub sanity_gate_min_ema: f32,
    /// Maximum number of consecutive frames for which the sanity gate
    /// will substitute a predicted H. After this, fall through and
    /// let the bad fit advance `frames_lost` so we can transition to
    /// Lost on sustained failure.
    pub sanity_gate_max_consecutive: u32,
    /// EMA smoothing factor for accepted inlier counts. Higher = more
    /// responsive to recent inlier counts; lower = more stable.
    pub inlier_ema_alpha: f32,
    /// Spatial-spread sanity check: minimum fraction of the anchor's
    /// keypoint bbox area that the inlier set must cover for the fit
    /// to be trusted. Catches the "lots of inliers, all clustered in
    /// one region" failure mode that produces overlay wobble even at
    /// inlier counts the count-based gate would accept.
    pub sanity_gate_min_bbox_coverage: f32,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            tracker: TrackerConfig::default(),
            // Capacity 1: drop the "page-flip" matching that would
            // re-lock onto an old anchor whenever its subset of
            // features still showed up in the new scene (e.g. half-page
            // anchor matching a full-page view). The brief-loss
            // recovery within a single anchor still works because the
            // current anchor lives in the cache.
            // v0 surface-map persistence-on-pan: keep up to 5 anchors
            // (1 root + up to 4 handoff anchors, or multiple distinct
            // acquired roots in the page-flip case). Per-anchor cost is
            // ~20 KB; the per-frame fallback (try other anchors when the
            // active one fails) is bounded by cache size × ~5 ms/anchor
            // = ~20 ms worst case at 5.
            anchor_cache_size: 5,
            acquire_cooldown_ns: 250_000_000, // 250 ms — fast re-acquire after loss
            // Drop to Lost quickly under bad fits so the overlay
            // hides as soon as the matcher loses confidence —
            // matches Google Translate's "shake → overlay
            // disappears → re-acquire when still" UX. Paired with
            // the H-delta cap that converts wrong-basin fits into
            // None returns, so violent shake reliably triggers a
            // Lost transition in ~150 ms instead of riding a
            // degenerate fit for half a second.
            lost_after_frames: 5, // ~150 ms @ 30 fps
            // Recovery budget: 30 frames (~1 s) before we go Idle so a
            // fresh acquire can fire. Bigger values make stuck-Lost
            // feel glacial when the user has clearly aimed at something
            // new; smaller values waste re-OCR on momentary blur.
            give_up_after_frames: 30,
            stable_required_ns: 200_000_000,       // 200 ms
            anchor_refresh_age_ns: 30_000_000_000, // 30 s
            handoff_min_inliers: 50,
            handoff_min_visible_ratio: 0.6,
            // Disabled (A/B): set absurdly high so `scale_changed`
            // never fires. Original 0.30 threshold (≈ 1.35× scale
            // change) was theoretically perspective-invariant via
            // `sqrt(|det|)`, but on real-world tilts where the
            // camera isn't perfectly centred on the surface the
            // affine det can still move 10-20% and spuriously fire
            // a handoff that creates a new full-frame anchor with
            // mismatched feature density. Bring back once we have a
            // perspective-stable scale measure (e.g. min singular
            // value of the upper-left 2×2 + bounds on max).
            handoff_scale_log_threshold: f32::INFINITY,
            handoff_cooldown_ns: 500_000_000, // 500 ms
            sanity_gate_drop_ratio: 0.3,
            sanity_gate_min_ema: 60.0,
            sanity_gate_max_consecutive: 3,
            inlier_ema_alpha: 0.2,
            // Minimum fraction of the anchor's keypoint bbox area
            // that the inlier set's bbox must cover for the fit to
            // be considered spatially well-constrained. Below this
            // the inliers are clustered (typically on the
            // feature-dense top half of the subject), the H is
            // over-fit to that region, and the overlay projection
            // outside the cluster wobbles toward the cluster —
            // visible as the "snap up" you observe at sub-30 inlier
            // counts. Defaults to 25% (i.e. 50% × 50% of anchor
            // bbox dimensions).
            sanity_gate_min_bbox_coverage: 0.25,
        }
    }
}

/// The full live-OCR tracker engine.
pub struct LivePlanarEngine {
    config: EngineConfig,
    cache: AnchorCache,
    state: EngineState,
    next_anchor_id: AnchorId,
    last_acquire_ns: u64,
    /// First timestamp at which IMU went quiet (None when moving).
    stable_since_ns: Option<u64>,
    /// Timestamp of the last handoff (anchor spawn from an existing
    /// chain). Used to throttle handoffs via `handoff_cooldown_ns`.
    last_spawn_ns: u64,
    /// History of accepted H + inlier EMA for the active anchor.
    /// Drives the inlier-discontinuity sanity gate.
    track_quality: TrackQualityState,
}

/// Single-step history of accepted `H_anchor→view` for the active anchor,
/// plus a running EMA of accepted inlier counts. Drives the inlier-
/// discontinuity sanity gate (rejects catastrophic single-frame drops
/// that are usually wrong-basin fits or descriptor collapse) and
/// supplies the "freeze" baseline (`h_prev`) the gate substitutes when
/// it suppresses a suspicious fit.
#[derive(Clone, Debug)]
struct TrackQualityState {
    h_prev: Option<[f32; 9]>,
    anchor_id: Option<AnchorId>,
    inlier_ema: Option<f32>,
    suspicious_frames: u32,
}

impl TrackQualityState {
    fn new() -> Self {
        Self {
            h_prev: None,
            anchor_id: None,
            inlier_ema: None,
            suspicious_frames: 0,
        }
    }
    fn reset(&mut self) {
        *self = Self::new();
    }
    /// Drops history when the anchor changes — the H coordinate system
    /// is anchor-relative.
    fn ensure_anchor(&mut self, anchor_id: AnchorId) {
        if self.anchor_id != Some(anchor_id) {
            *self = Self {
                anchor_id: Some(anchor_id),
                ..Self::new()
            };
        }
    }
    fn push_h(&mut self, h: [f32; 9]) {
        self.h_prev = Some(h);
    }
    fn update_ema(&mut self, inliers: f32, alpha: f32) {
        self.inlier_ema = Some(match self.inlier_ema {
            Some(prev) => prev * (1.0 - alpha) + inliers * alpha,
            None => inliers,
        });
    }
    /// Reset the EMA and substitution budget after the sanity gate
    /// rejects a suspicious fit, but **keep `h_prev`** so the next
    /// suspicious frame can still freeze to the last accepted H.
    /// Clearing EMA prevents a stale high baseline from making every
    /// subsequent fit look suspicious (which produced multi-frame
    /// reject cascades).
    fn reset_ema_and_budget(&mut self) {
        self.inlier_ema = None;
        self.suspicious_frames = 0;
    }
}

#[derive(Clone, Debug)]
enum EngineState {
    Idle,
    Locked {
        anchor_id: AnchorId,
        frames_lost: u32,
        last_homography: [f32; 9],
    },
    Lost {
        last_anchor_id: AnchorId,
        frames_lost: u32,
    },
}

struct CachedAnchor {
    anchor: SceneAnchor,
    /// Populated only on root anchors (those created by `acquire_now`).
    /// Handoff anchors carry empty overlays — their root holds them.
    overlays: Vec<CanonicalOverlay>,
    /// The root this anchor projects through. `root_id == self_id` for
    /// roots; for handoff descendants, points at the root in whose
    /// canonical frame overlays live.
    root_id: AnchorId,
    /// Homography mapping root canonical → this anchor's canonical
    /// frame. Identity for roots. `H_root_to_canonical · root_point`
    /// gives the equivalent point in this anchor's canonical coords.
    h_root_to_canonical: [f32; 9],
    created_at_ns: u64,
    last_locked_ns: u64,
}

/// Hand-rolled LRU. Capacity is small (≤5 in production) so a Vec keyed
/// in MRU-first order is fine; not worth pulling in a crate. Insertions
/// at the front, evictions from the back.
struct AnchorCache {
    capacity: usize,
    entries: Vec<(AnchorId, CachedAnchor)>,
}

impl AnchorCache {
    fn new(capacity: usize) -> Self {
        let cap = capacity.max(1);
        Self {
            capacity: cap,
            entries: Vec::with_capacity(cap),
        }
    }

    fn insert(&mut self, id: AnchorId, entry: CachedAnchor) {
        self.entries.insert(0, (id, entry));
        while self.entries.len() > self.capacity {
            self.entries.pop();
        }
    }

    fn touch(&mut self, id: AnchorId) {
        let idx = match self.entries.iter().position(|(k, _)| *k == id) {
            Some(i) => i,
            None => return,
        };
        if idx == 0 {
            return;
        }
        let entry = self.entries.remove(idx);
        self.entries.insert(0, entry);
    }

    fn get(&self, id: AnchorId) -> Option<&CachedAnchor> {
        self.entries
            .iter()
            .find_map(|(k, v)| if *k == id { Some(v) } else { None })
    }

    fn get_mut(&mut self, id: AnchorId) -> Option<&mut CachedAnchor> {
        self.entries
            .iter_mut()
            .find_map(|(k, v)| if *k == id { Some(v) } else { None })
    }

    fn ids_mru(&self) -> Vec<AnchorId> {
        self.entries.iter().map(|(k, _)| *k).collect()
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl LivePlanarEngine {
    pub fn new(config: EngineConfig) -> Self {
        let cache_size = config.anchor_cache_size;
        Self {
            config,
            cache: AnchorCache::new(cache_size),
            state: EngineState::Idle,
            next_anchor_id: 1,
            last_acquire_ns: 0,
            stable_since_ns: None,
            last_spawn_ns: 0,
            track_quality: TrackQualityState::new(),
        }
    }

    /// Process one camera frame. Decides whether to track against the
    /// current anchor, fall back to a cached anchor, declare loss, or
    /// stay idle waiting for an acquire.
    pub fn process_frame(
        &mut self,
        gray: &GrayImage,
        imu_stable: bool,
        timestamp_ns: u64,
    ) -> TrackerCommand {
        let cmd = self.process_frame_inner(gray, imu_stable, timestamp_ns, None);
        if matches!(cmd, TrackerCommand::Idle) {
            self.track_quality.reset();
        }
        cmd
    }

    fn process_frame_inner(
        &mut self,
        gray: &GrayImage,
        imu_stable: bool,
        timestamp_ns: u64,
        prior: Option<[f32; 9]>,
    ) -> TrackerCommand {
        self.tick_stable(imu_stable, timestamp_ns);
        match self.state.clone() {
            EngineState::Idle => {
                // Even when "Idle", we still try matching against cached
                // anchors. If the user picked up a previously-seen scene,
                // we want to snap back to it without forcing a new acquire.
                if !self.cache.is_empty() {
                    if let Some((id, result)) = self.try_cached_anchors(gray, None) {
                        self.transition_to_locked(id, &result, timestamp_ns, false);
                        let (root_id, h_root_to_view) =
                            self.chain_homography(id, &result.homography);
                        return TrackerCommand::Locked {
                            anchor_id: root_id,
                            homography: h_root_to_view,
                            is_new: false,
                            inliers: result.inliers,
                        };
                    }
                }
                if self.is_stable_enough(timestamp_ns) {
                    TrackerCommand::Acquiring
                } else {
                    TrackerCommand::Idle
                }
            }
            EngineState::Locked {
                anchor_id,
                frames_lost,
                last_homography,
            } => {
                self.track_quality.ensure_anchor(anchor_id);
                // In Locked state we apply hysteresis: a lower inlier
                // bar to *keep* the lock than to acquire it. Avoids
                // per-frame Locked↔Lost flicker when inliers wander
                // around the acquire threshold. Pair with the
                // IMU-derived prior (if any) to short-circuit RANSAC.
                //
                // Fallback prior: when no external `prior` is provided
                // (e.g. IMU bypassed), seed RANSAC with the previous
                // frame's `H_anchor→view`. Without a seed, RANSAC
                // starts from random 4-point samples; on scenes with
                // repetitive features (e.g. pages of text), ~20% of
                // those random seeds suggest a *flipped* homography
                // and RANSAC can then confirm that flip from other
                // wrong correspondences in the same basin — visible
                // as "UI floating inverted in the middle of the air"
                // with healthy inlier counts (107+) even on a static
                // scene. The previous-frame H is the most informed
                // seed we can give RANSAC at zero extra cost; it's
                // essentially what the IMU prior reduces to when
                // rotation is small.
                let keep_min = self.config.tracker.min_inliers_keep_locked;
                let seed_prior = prior.or(Some(last_homography));
                let result = self.cache.get(anchor_id).and_then(|a| {
                    let dims = a.anchor.image_dims;
                    let brute = track_against_anchor_with_prior(
                        &a.anchor,
                        gray,
                        &self.config.tracker,
                        keep_min,
                        seed_prior,
                    );
                    match brute {
                        None => {
                            log::debug!("[engine] brute force returned None (matcher failed)");
                            None
                        }
                        Some(t) => {
                            let raw_inliers = t.inliers;
                            if !homography_is_sane(&t.homography, dims.0, dims.1)
                                || !homography_delta_is_sane(
                                    &t.homography,
                                    &last_homography,
                                    dims.0,
                                    dims.1,
                                )
                            {
                                log::debug!(
                                    "[engine] brute force fit rejected by validate() (insane H or large delta vs last_homography); raw inliers={}",
                                    raw_inliers
                                );
                                None
                            } else {
                                Some(t)
                            }
                        }
                    }
                });
                let anchor_bbox = self
                    .cache
                    .get(anchor_id)
                    .map(|a| Self::keypoint_bbox(&a.anchor.positions))
                    .unwrap_or((0.0, 0.0, 1.0, 1.0));
                let result = result.and_then(|r| self.apply_sanity_gate(r, anchor_bbox));
                if let Some(r) = result {
                    self.cache.touch(anchor_id);
                    if let Some(entry) = self.cache.get_mut(anchor_id) {
                        entry.last_locked_ns = timestamp_ns;
                    }
                    // External: emit the root id and the chain-composed
                    // H_root→view. The handoff machinery is invisible
                    // to Kotlin.
                    let (root_id, h_root_to_view) = self.chain_homography(anchor_id, &r.homography);
                    // Handoff trigger: spawn a new anchor while the
                    // current one still has enough overlap to fit a
                    // stable H_active→new. Inlier and visibility checks
                    // both matter — see EngineConfig docs.
                    let cooldown_elapsed = timestamp_ns.saturating_sub(self.last_spawn_ns)
                        >= self.config.handoff_cooldown_ns;
                    let visible_ratio = self
                        .cache
                        .get(anchor_id)
                        .map(|a| {
                            visible_keypoint_ratio(
                                &a.anchor.positions,
                                &r.homography,
                                gray.width(),
                                gray.height(),
                            )
                        })
                        .unwrap_or(1.0);
                    // Scale-change trigger: H_anchor→view's local linear
                    // scale captures cumulative zoom since acquire. The
                    // anchor's canonical frame is by construction
                    // identity-mapped to itself, so the current H *is*
                    // the change. BRIEF is not scale-invariant; fire a
                    // handoff once we're past ~1.35× to rebuild
                    // descriptors at the current scale before matching
                    // collapses.
                    let scale = crate::coords::Homography::<
                        crate::coords::AnchorSpace,
                        crate::coords::TrackerSpace,
                    >::from_raw(r.homography)
                    .approx_scale();
                    let scale_log = if scale > 0.0 { scale.ln().abs() } else { 0.0 };
                    let scale_changed = scale_log > self.config.handoff_scale_log_threshold;
                    let needs_handoff = r.inliers < self.config.handoff_min_inliers
                        || visible_ratio < self.config.handoff_min_visible_ratio
                        || scale_changed;
                    let new_active = if cooldown_elapsed && needs_handoff {
                        self.spawn_handoff(gray, anchor_id, &r.homography, timestamp_ns)
                            .unwrap_or(anchor_id)
                    } else {
                        anchor_id
                    };
                    // If we handed off, the new anchor's canonical frame
                    // IS this view, so its `last_homography` is identity.
                    // Otherwise we stay on the old anchor with its
                    // tracked H.
                    let (new_state_h, new_state_id) = if new_active == anchor_id {
                        (r.homography, anchor_id)
                    } else {
                        (IDENTITY, new_active)
                    };
                    self.state = EngineState::Locked {
                        anchor_id: new_state_id,
                        frames_lost: 0,
                        last_homography: new_state_h,
                    };
                    return TrackerCommand::Locked {
                        anchor_id: root_id,
                        homography: h_root_to_view,
                        is_new: false,
                        inliers: r.inliers,
                    };
                }
                // Current anchor lost the frame — try cached siblings.
                if let Some((id, alt)) = self.try_cached_anchors(gray, Some(anchor_id)) {
                    self.transition_to_locked(id, &alt, timestamp_ns, false);
                    let (root_id, h_root_to_view) = self.chain_homography(id, &alt.homography);
                    return TrackerCommand::Locked {
                        anchor_id: root_id,
                        homography: h_root_to_view,
                        is_new: false,
                        inliers: alt.inliers,
                    };
                }
                let new_frames_lost = frames_lost + 1;
                let root = self.root_of(anchor_id);
                log::info!(
                    "[engine] Locked frame produced no usable fit; frames_lost {} -> {} (gate ema={:?} susp={})",
                    frames_lost,
                    new_frames_lost,
                    self.track_quality.inlier_ema,
                    self.track_quality.suspicious_frames,
                );
                if new_frames_lost >= self.config.lost_after_frames {
                    self.state = EngineState::Lost {
                        last_anchor_id: anchor_id,
                        frames_lost: 0,
                    };
                    TrackerCommand::Lost {
                        last_anchor_id: root,
                    }
                } else {
                    self.state = EngineState::Locked {
                        anchor_id,
                        frames_lost: new_frames_lost,
                        last_homography: match self.state {
                            EngineState::Locked {
                                last_homography, ..
                            } => last_homography,
                            _ => unreachable!(),
                        },
                    };
                    TrackerCommand::Lost {
                        last_anchor_id: root,
                    }
                }
            }
            EngineState::Lost {
                last_anchor_id,
                frames_lost,
            } => {
                if let Some((id, result)) = self.try_cached_anchors(gray, None) {
                    self.transition_to_locked(id, &result, timestamp_ns, false);
                    let (root_id, h_root_to_view) = self.chain_homography(id, &result.homography);
                    return TrackerCommand::Locked {
                        anchor_id: root_id,
                        homography: h_root_to_view,
                        is_new: false,
                        inliers: result.inliers,
                    };
                }
                let new_frames_lost = frames_lost + 1;
                let root = self.root_of(last_anchor_id);
                if new_frames_lost >= self.config.give_up_after_frames {
                    self.state = EngineState::Idle;
                    TrackerCommand::Idle
                } else {
                    self.state = EngineState::Lost {
                        last_anchor_id,
                        frames_lost: new_frames_lost,
                    };
                    TrackerCommand::Lost {
                        last_anchor_id: root,
                    }
                }
            }
        }
    }

    /// Force-acquire a new scene anchor from `gray`. Use this when
    /// Kotlin has already run OCR on the frame and wants to lock the
    /// surface in place. Returns the new anchor id, or `None` if the
    /// frame had insufficient features.
    pub fn acquire_now(&mut self, gray: &GrayImage, timestamp_ns: u64) -> Option<AnchorId> {
        self.acquire_inner(gray, &[], 0, timestamp_ns)
    }

    /// Like [`acquire_now`] but restricts anchor features to those
    /// inside any of the given axis-aligned regions (padded by
    /// `pad_px`). Use this when you know which surface in the frame
    /// you care about (e.g. the union of OCR-detected text bboxes) so
    /// the tracker doesn't lock onto background clutter.
    pub fn acquire_now_in_regions(
        &mut self,
        gray: &GrayImage,
        regions: &[(u32, u32, u32, u32)],
        pad_px: u32,
        timestamp_ns: u64,
    ) -> Option<AnchorId> {
        self.acquire_inner(gray, regions, pad_px, timestamp_ns)
    }

    fn acquire_inner(
        &mut self,
        gray: &GrayImage,
        regions: &[(u32, u32, u32, u32)],
        pad_px: u32,
        timestamp_ns: u64,
    ) -> Option<AnchorId> {
        if timestamp_ns < self.last_acquire_ns
            || timestamp_ns - self.last_acquire_ns < self.config.acquire_cooldown_ns
        {
            if self.last_acquire_ns != 0 {
                return None;
            }
        }
        let anchor = if regions.is_empty() {
            build_anchor(gray, &self.config.tracker, timestamp_ns)?
        } else {
            build_anchor_in_regions(gray, &self.config.tracker, regions, pad_px, timestamp_ns)?
        };
        if anchor.len() < self.config.tracker.min_inliers {
            return None;
        }
        let id = self.next_anchor_id;
        self.next_anchor_id += 1;
        self.cache.insert(
            id,
            CachedAnchor {
                anchor,
                overlays: Vec::new(),
                // `acquire_now` always creates a fresh root: this anchor
                // owns its own overlays and starts a new chain.
                root_id: id,
                h_root_to_canonical: IDENTITY,
                created_at_ns: timestamp_ns,
                last_locked_ns: timestamp_ns,
            },
        );
        self.state = EngineState::Locked {
            anchor_id: id,
            frames_lost: 0,
            last_homography: IDENTITY,
        };
        self.last_acquire_ns = timestamp_ns;
        self.track_quality.reset();
        // Reset spawn cooldown — a fresh root means there's no chain
        // yet to throttle.
        self.last_spawn_ns = timestamp_ns;
        Some(id)
    }

    /// Attach (or replace) the canonical overlay set for an anchor.
    /// Returns false if the anchor isn't cached anymore.
    pub fn set_overlays(&mut self, anchor_id: AnchorId, overlays: Vec<CanonicalOverlay>) -> bool {
        match self.cache.get_mut(anchor_id) {
            Some(entry) => {
                entry.overlays = overlays;
                true
            }
            None => false,
        }
    }

    /// Read the overlays for an anchor.
    pub fn overlays(&self, anchor_id: AnchorId) -> &[CanonicalOverlay] {
        match self.cache.get(anchor_id) {
            Some(entry) => &entry.overlays,
            None => &[],
        }
    }

    /// Backwards-compatible shim — delegates to the free
    /// [`render_text_overlay_bitmap`] function so existing callers that
    /// already hold an `&LivePlanarEngine` keep working. New callers
    /// (uniffi `prepare_overlay_for_composite`) should call the free
    /// function directly so they don't need to lock the engine mutex
    /// for the duration of the raster — which is what was blocking the
    /// detector thread (and therefore the display pipeline) during the
    /// 50–100 ms it takes to rasterize a dense page.
    #[cfg(feature = "image-render")]
    pub fn render_text_overlay_bitmap(
        &self,
        frame_width: u32,
        frame_height: u32,
        items: &[TextRenderItem],
        fonts: &dyn FontProvider,
    ) -> Option<Vec<u8>> {
        render_text_overlay_bitmap(frame_width, frame_height, items, fonts)
    }

    /// Project all overlays of `anchor_id` through `homography` into
    /// current-frame coordinates. Degenerate projections are skipped.
    pub fn project_overlays(
        &self,
        anchor_id: AnchorId,
        homography: &[f32; 9],
    ) -> Vec<OverlayProjection> {
        let entry = match self.cache.get(anchor_id) {
            Some(e) => e,
            None => return Vec::new(),
        };
        entry
            .overlays
            .iter()
            .filter_map(|ov| {
                let mut quad = [(0.0, 0.0); 4];
                for (i, &(x, y)) in ov.quad.iter().enumerate() {
                    let (px, py) = crate::homography::project(homography, x, y)?;
                    quad[i] = (px, py);
                }
                Some(OverlayProjection {
                    id: ov.id,
                    quad,
                    payload: ov.payload.clone(),
                })
            })
            .collect()
    }

    /// True if the anchor's overlays should be re-derived from a new
    /// OCR pass: anchor older than `anchor_refresh_age_ns` and still
    /// being actively locked recently.
    pub fn should_refresh(&self, anchor_id: AnchorId, now_ns: u64) -> bool {
        match self.cache.get(anchor_id) {
            Some(entry) => {
                let age = now_ns.saturating_sub(entry.created_at_ns);
                age >= self.config.anchor_refresh_age_ns
            }
            None => false,
        }
    }

    pub fn current_anchor(&self) -> Option<AnchorId> {
        match self.state {
            EngineState::Locked { anchor_id, .. } => Some(self.root_of(anchor_id)),
            EngineState::Lost { last_anchor_id, .. } => Some(self.root_of(last_anchor_id)),
            EngineState::Idle => None,
        }
    }

    /// Internal cache handles, MRU-first. **These name specific cache
    /// entries, not root coordinate frames.** A single root (one
    /// physical surface) may have several handles after handoffs.
    /// `pub(crate)` so the handle id space never escapes
    /// `planar_engine` — confusing it with the root id space (the
    /// id externally-emitted by `TrackerCommand::Locked`,
    /// `acquire_now`, etc.) was the source of "surface-map state
    /// evaporates after panning" bugs. External callers want
    /// [`Self::cached_root_ids`].
    #[allow(dead_code)]
    pub(crate) fn cached_handle_ids(&self) -> Vec<AnchorId> {
        self.cache.ids_mru()
    }

    /// Unique root anchor ids currently represented in the cache, in
    /// MRU-of-any-descendant order. One root per physical surface;
    /// the right id space for session-state retention because the
    /// emitted `TrackerCommand::Locked.anchor_id` is always a root.
    ///
    /// A root survives in this list as long as **any** of its
    /// descendants is still cached — handoff chains preserve the
    /// root's coord frame even when the original root anchor itself
    /// gets LRU-evicted from `AnchorCache`.
    pub fn cached_root_ids(&self) -> Vec<AnchorId> {
        let mut seen = std::collections::HashSet::new();
        let mut roots = Vec::new();
        for id in self.cache.ids_mru() {
            if let Some(entry) = self.cache.get(id) {
                if seen.insert(entry.root_id) {
                    roots.push(entry.root_id);
                }
            }
        }
        roots
    }

    pub fn cache_len(&self) -> usize {
        self.cache.len()
    }

    pub fn clear(&mut self) {
        self.cache = AnchorCache::new(self.config.anchor_cache_size);
        self.state = EngineState::Idle;
        self.next_anchor_id = 1;
        self.last_acquire_ns = 0;
        self.stable_since_ns = None;
        self.last_spawn_ns = 0;
        self.track_quality.reset();
    }

    // -- internal helpers --------------------------------------------------

    /// Map an active-anchor id to its chain root. External APIs (the
    /// `TrackerCommand` enum, `current_anchor`, etc.) speak in root
    /// ids so Kotlin doesn't see the handoff machinery.
    fn root_of(&self, active_id: AnchorId) -> AnchorId {
        self.cache
            .get(active_id)
            .map(|a| a.root_id)
            .unwrap_or(active_id)
    }

    /// Compute the bbox of a set of keypoint positions in canonical
    /// (anchor) coords. Returns (min_x, min_y, max_x, max_y).
    fn keypoint_bbox(positions: &[(f32, f32)]) -> (f32, f32, f32, f32) {
        let mut min_x = f32::INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        for &(x, y) in positions {
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
        (min_x, min_y, max_x, max_y)
    }

    /// Inlier-discontinuity sanity gate (fix A) + spatial-spread check.
    ///
    /// Three outcomes per call:
    ///   - *Accept*: fit looks fine (not a catastrophic drop, inliers
    ///     well-spread over the anchor bbox). Push to history, update
    ///     EMA, return the fit.
    ///   - *Freeze*: fit is suspicious (count drop OR spatial cluster)
    ///     and we have a last-accepted H within budget. Substitute the
    ///     *frozen* last-accepted H instead of the suspicious fit. We
    ///     don't extrapolate forward (that was fix #1; in practice the
    ///     extrapolation overshoots on non-linear motion and drifts the
    ///     overlay off-target). Don't advance history or EMA — the
    ///     freeze baseline stays anchored to last-known-good
    ///     observations.
    ///   - *Reject*: fit is suspicious and we either have no history to
    ///     freeze on, or we've already burned the substitution budget.
    ///     Return None so the caller treats this like a matcher miss;
    ///     `frames_lost` advances toward Lost via the normal path. We
    ///     don't push the bad H to history (avoids "split" artefacts on
    ///     the next frame) and don't update EMA (a wrong fit shouldn't
    ///     drag the running average down and weaken the gate).
    fn apply_sanity_gate(
        &mut self,
        r: TrackResult,
        anchor_bbox: (f32, f32, f32, f32),
    ) -> Option<TrackResult> {
        let cfg = &self.config;
        let new_inliers_f = r.inliers as f32;
        let ema = self.track_quality.inlier_ema.unwrap_or(new_inliers_f);
        let freeze_h = self.track_quality.h_prev;
        let count_suspicious =
            ema >= cfg.sanity_gate_min_ema && new_inliers_f < cfg.sanity_gate_drop_ratio * ema;
        let coverage = bbox_coverage(r.inlier_anchor_bbox, anchor_bbox);
        let spread_suspicious = coverage < cfg.sanity_gate_min_bbox_coverage;
        let suspicious = count_suspicious || spread_suspicious;
        let can_freeze = self.track_quality.suspicious_frames < cfg.sanity_gate_max_consecutive
            && freeze_h.is_some();
        let trigger = if count_suspicious && spread_suspicious {
            "count+spread"
        } else if count_suspicious {
            "count"
        } else {
            "spread"
        };
        if suspicious && can_freeze {
            self.track_quality.suspicious_frames += 1;
            let hf = freeze_h.expect("freeze_h is Some when can_freeze");
            log::info!(
                "[sanity_gate] freeze (substitute with last accepted H): inliers={} ema={:.1} coverage={:.2} trigger={} run={}",
                r.inliers,
                ema,
                coverage,
                trigger,
                self.track_quality.suspicious_frames,
            );
            Some(TrackResult {
                homography: hf,
                ..r
            })
        } else if suspicious {
            let reason = if freeze_h.is_none() {
                "no_history(h_prev=false)".to_string()
            } else {
                format!(
                    "budget_exhausted(run={})",
                    self.track_quality.suspicious_frames
                )
            };
            log::info!(
                "[sanity_gate] reject suspicious fit: inliers={} ema={:.1} coverage={:.2} trigger={} reason={}",
                r.inliers,
                ema,
                coverage,
                trigger,
                reason,
            );
            self.track_quality.reset_ema_and_budget();
            None
        } else {
            self.track_quality.suspicious_frames = 0;
            self.track_quality
                .update_ema(new_inliers_f, cfg.inlier_ema_alpha);
            self.track_quality.push_h(r.homography);
            Some(r)
        }
    }

    /// return `(root_id, H_root→view)` — the pair the engine should
    /// emit externally. The chain compose lets Kotlin keep treating
    /// the engine as single-anchor: it sees the root id and a single
    /// homography per frame, never the handoff anchors in between.
    /// Falls back to `(active_id, h_active_to_view)` if the active
    /// anchor isn't in the cache (shouldn't happen during normal flow).
    fn chain_homography(
        &self,
        active_id: AnchorId,
        h_active_to_view: &[f32; 9],
    ) -> (AnchorId, [f32; 9]) {
        match self.cache.get(active_id) {
            Some(a) => {
                let h_root_to_view = mat3_mul(h_active_to_view, &a.h_root_to_canonical);
                (a.root_id, h_root_to_view)
            }
            None => (active_id, *h_active_to_view),
        }
    }

    /// Spawn a handoff anchor from the current frame as a descendant
    /// of the currently-active chain. Returns the new anchor id on
    /// success, `None` if the new anchor doesn't carry enough features
    /// to be useful. The new anchor's canonical frame *is* this view,
    /// so `H_active→newCanonical = h_active_to_view`, and
    /// `H_root→newCanonical = h_active_to_view · h_root_to_active`.
    fn spawn_handoff(
        &mut self,
        gray: &GrayImage,
        active_id: AnchorId,
        h_active_to_view: &[f32; 9],
        timestamp_ns: u64,
    ) -> Option<AnchorId> {
        let (root_id, h_root_to_active) = {
            let active = self.cache.get(active_id)?;
            (active.root_id, active.h_root_to_canonical)
        };
        let new_anchor = build_anchor(gray, &self.config.tracker, timestamp_ns)?;
        if new_anchor.len() < self.config.tracker.min_inliers {
            return None;
        }
        let h_root_to_new = mat3_mul(h_active_to_view, &h_root_to_active);
        let new_id = self.next_anchor_id;
        self.next_anchor_id += 1;
        self.cache.insert(
            new_id,
            CachedAnchor {
                anchor: new_anchor,
                overlays: Vec::new(),
                root_id,
                h_root_to_canonical: h_root_to_new,
                created_at_ns: timestamp_ns,
                last_locked_ns: timestamp_ns,
            },
        );
        self.last_spawn_ns = timestamp_ns;
        Some(new_id)
    }

    fn tick_stable(&mut self, imu_stable: bool, timestamp_ns: u64) {
        if imu_stable {
            if self.stable_since_ns.is_none() {
                self.stable_since_ns = Some(timestamp_ns);
            }
        } else {
            self.stable_since_ns = None;
        }
    }

    fn is_stable_enough(&self, now_ns: u64) -> bool {
        match self.stable_since_ns {
            Some(t) => now_ns.saturating_sub(t) >= self.config.stable_required_ns,
            None => false,
        }
    }

    fn try_cached_anchors(
        &self,
        gray: &GrayImage,
        skip_id: Option<AnchorId>,
    ) -> Option<(AnchorId, TrackResult)> {
        let mut best: Option<(AnchorId, TrackResult)> = None;
        for &id in &self.cache.ids_mru() {
            if Some(id) == skip_id {
                continue;
            }
            let entry = match self.cache.get(id) {
                Some(e) => e,
                None => continue,
            };
            if let Some(r) = track_against_anchor(&entry.anchor, gray, &self.config.tracker) {
                let beats_current = match &best {
                    Some((_, b)) => r.inliers > b.inliers,
                    None => true,
                };
                if beats_current {
                    best = Some((id, r));
                }
            }
        }
        best
    }

    fn transition_to_locked(
        &mut self,
        anchor_id: AnchorId,
        result: &TrackResult,
        timestamp_ns: u64,
        is_new: bool,
    ) {
        let _ = is_new;
        self.cache.touch(anchor_id);
        if let Some(entry) = self.cache.get_mut(anchor_id) {
            entry.last_locked_ns = timestamp_ns;
        }
        // Stale velocity + EMA from before the Lost period would
        // mislead the sanity gate.
        self.track_quality.reset();
        self.state = EngineState::Locked {
            anchor_id,
            frames_lost: 0,
            last_homography: result.homography,
        };
    }
}

const IDENTITY: [f32; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

/// Free-function form of the overlay rasterizer (extracted from
/// `LivePlanarEngine::render_text_overlay_bitmap`). Doesn't touch any
/// engine state — pure on `items` + `fonts`. Pulled out so the uniffi
/// `prepare_overlay_for_composite` call can rasterize without locking
/// the engine mutex; otherwise the rec-batch worker's raster (50–100
/// ms on a dense page) starves the detector thread that's trying to
/// run `process_frame_with_imu`, which freezes the SurfaceView at the
/// rec-batch rate instead of the camera rate.
#[cfg(feature = "image-render")]
pub fn render_text_overlay_bitmap(
    frame_width: u32,
    frame_height: u32,
    items: &[TextRenderItem],
    fonts: &dyn FontProvider,
) -> Option<Vec<u8>> {
    if frame_width == 0 || frame_height == 0 {
        return None;
    }
    let pixels = (frame_width as usize) * (frame_height as usize);
    let mut rgba = vec![0u8; pixels * 4];
    for it in items {
        let oriented = oriented_rect_from_corners(&it.quad);
        fill_oriented_rect_blended(
            &mut rgba,
            frame_width,
            frame_height,
            &oriented,
            argb_u32_to_rgba8(it.bg_argb),
        );
    }
    let blocks: Vec<PreparedTextBlock> = items
        .iter()
        .filter(|it| !it.translated_text.trim().is_empty())
        .map(|it| {
            let bg_box = oriented_rect_from_corners(&it.quad);
            // Inset the text box horizontally so the left-aligned text
            // starts inside the bg's rounded edge instead of flush
            // against it. Matches the Kotlin `HORIZONTAL_PAD_PX`.
            let text_box = OrientedRect {
                cx: bg_box.cx,
                cy: bg_box.cy,
                width: (bg_box.width - 2.0 * OVERLAY_TEXT_HORIZONTAL_INSET_PX).max(1.0),
                height: bg_box.height,
                angle_radians: bg_box.angle_radians,
            };
            let aabb = text_box.to_aabb();
            let bbox = Rect {
                left: aabb.left.min(frame_width.saturating_sub(1)),
                top: aabb.top.min(frame_height.saturating_sub(1)),
                right: aabb.right.min(frame_width),
                bottom: aabb.bottom.min(frame_height),
            };
            let line = PreparedTextLine {
                text: it.translated_text.clone(),
                bounding_box: bbox.clone(),
                oriented_box: text_box,
                word_rects: vec![bbox.clone()],
                background_argb: it.bg_argb,
                foreground_argb: it.fg_argb,
            };
            PreparedTextBlock {
                source_text: it.source_text.clone(),
                translated_text: it.translated_text.clone(),
                bounding_box: bbox,
                lines: vec![line],
                layout_hints: OverlayLayoutHints {
                    layout_mode: OverlayLayoutMode::PerLine,
                    suggested_font_size_px: it.suggested_font_px.max(6.0),
                },
                background_argb: it.bg_argb,
                foreground_argb: it.fg_argb,
            }
        })
        .collect();
    if blocks.is_empty() {
        return Some(rgba);
    }
    let prepared = PreparedImageOverlay {
        rgba_bytes: rgba,
        width: frame_width,
        height: frame_height,
        extracted_text: String::new(),
        translated_text: String::new(),
        blocks,
    };
    let opts = RenderOptions {
        language: items
            .first()
            .map(|i| i.language.clone())
            .unwrap_or_default(),
        min_font_size_px: 6.0,
    };
    render_overlay(&prepared, fonts, &opts).ok()
}

/// Horizontal text inset (canonical pixels) within the live-overlay
/// rounded-rect background. Must equal the Kotlin-side
/// `HORIZONTAL_PAD_PX` used to inflate `visualBox`; together they yield
/// left-aligned text starting `pad` px inside the bg's rounded edge,
/// with `pad` px of bg breathing room on the right of short lines.
#[cfg(feature = "image-render")]
pub const OVERLAY_TEXT_HORIZONTAL_INSET_PX: f32 = 8.0;

/// Fraction of an anchor's canonical-frame keypoints that, projected
/// through `h` (canonical → view), land inside `(0..view_w, 0..view_h)`.
/// Returns 0.0 for empty anchors and saturates at 1.0. Used by the
/// handoff trigger: when too few features will be visible next frame,
/// we want a new anchor while the current one still has enough
/// overlap to fit `H_active→new` cleanly.
fn visible_keypoint_ratio(positions: &[(f32, f32)], h: &[f32; 9], view_w: u32, view_h: u32) -> f32 {
    if positions.is_empty() {
        return 0.0;
    }
    let w = view_w as f32;
    let h_ = view_h as f32;
    let mut visible = 0usize;
    for &(x, y) in positions {
        let Some((vx, vy)) = crate::homography::project(h, x, y) else {
            continue;
        };
        if vx >= 0.0 && vx < w && vy >= 0.0 && vy < h_ {
            visible += 1;
        }
    }
    visible as f32 / positions.len() as f32
}

/// Reject a homography if it would produce a visually-degenerate
/// projection of the canonical frame's four corners. A "valid" matrix
/// from RANSAC can still send some bitmap pixels to infinity (when the
/// homogeneous `w` of a corner is near zero) or produce a wildly
/// skewed trapezoid; either case manifests as the "huge diagonal
/// streaks" rendering glitch. Three checks:
///   1. all 4 corner projections must succeed (finite, non-degenerate `w`)
///   2. no projected edge longer than 4× the canonical diagonal
///   3. opposite edges within a 6× length ratio of each other
/// Maximum corner displacement (in canonical pixels) we'll accept
/// between consecutive Locked frames' homographies. RANSAC on
/// repetitive content occasionally settles into a wrong basin that
/// is locally self-consistent (high inlier count, passes
/// `homography_is_sane`) but is geometrically nonsense — projecting
/// the canonical corners hundreds of pixels away from where the
/// previous frame's H put them. Real hand motion at 30 fps doesn't
/// move a corner ~150 px in one frame for typical zoom levels, so
/// anything larger is rejected as a wrong-basin fit. The reject
/// path returns None → `frames_lost++` → after
/// `lost_after_frames` consecutive rejects, transition to Lost
/// (overlay hides, scene re-acquires when matcher recovers).
const MAX_CORNER_JUMP_PX: f32 = 300.0;

/// Fraction of `anchor_bbox`'s area covered by `inlier_bbox`. Returns
/// 0.0 if the anchor bbox is degenerate. Clamped to [0, 1].
fn bbox_coverage(inlier_bbox: (f32, f32, f32, f32), anchor_bbox: (f32, f32, f32, f32)) -> f32 {
    let (imin_x, imin_y, imax_x, imax_y) = inlier_bbox;
    let (amin_x, amin_y, amax_x, amax_y) = anchor_bbox;
    let aw = (amax_x - amin_x).max(0.0);
    let ah = (amax_y - amin_y).max(0.0);
    if aw <= 0.0 || ah <= 0.0 {
        return 0.0;
    }
    let iw = (imax_x - imin_x).max(0.0);
    let ih = (imax_y - imin_y).max(0.0);
    let coverage = (iw * ih) / (aw * ah);
    coverage.clamp(0.0, 1.0)
}

/// True when the corner displacement between `h_new` and `h_prev`
/// projected over the canonical rect stays below
/// [`MAX_CORNER_JUMP_PX`]. Treats degenerate projections as a
/// failure (= reject the fit).
fn homography_delta_is_sane(
    h_new: &[f32; 9],
    h_prev: &[f32; 9],
    canonical_w: u32,
    canonical_h: u32,
) -> bool {
    let cw = canonical_w as f32;
    let ch = canonical_h as f32;
    let corners = [(0.0_f32, 0.0_f32), (cw, 0.0), (cw, ch), (0.0, ch)];
    for &(x, y) in &corners {
        let pn = crate::homography::project(h_new, x, y);
        let pp = crate::homography::project(h_prev, x, y);
        match (pn, pp) {
            (Some(pn), Some(pp)) => {
                let dx = pn.0 - pp.0;
                let dy = pn.1 - pp.1;
                let d = (dx * dx + dy * dy).sqrt();
                if !d.is_finite() || d > MAX_CORNER_JUMP_PX {
                    return false;
                }
            }
            _ => return false,
        }
    }
    true
}

fn homography_is_sane(h: &[f32; 9], canonical_w: u32, canonical_h: u32) -> bool {
    let cw = canonical_w as f32;
    let ch = canonical_h as f32;
    let corners = [(0.0_f32, 0.0_f32), (cw, 0.0), (cw, ch), (0.0, ch)];
    let mut p = [(0.0f32, 0.0f32); 4];
    for (i, &(x, y)) in corners.iter().enumerate() {
        match crate::homography::project(h, x, y) {
            Some(q) if q.0.is_finite() && q.1.is_finite() => p[i] = q,
            _ => return false,
        }
    }
    let edge = |a: (f32, f32), b: (f32, f32)| -> f32 {
        ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
    };
    let w_top = edge(p[0], p[1]);
    let w_bot = edge(p[3], p[2]);
    let h_left = edge(p[0], p[3]);
    let h_right = edge(p[1], p[2]);
    let max_edge = w_top.max(w_bot).max(h_left).max(h_right);
    let orig_diag = (cw * cw + ch * ch).sqrt();
    // Tightened from 4.0× to 2.0× canonical diagonal: catches more of
    // the moderately-bad fits that pass the looser bound but still
    // produce visible warp. A genuinely-tilted surface at 45° has
    // edges ~√2 × canonical, so 2.0 leaves headroom; anything beyond
    // that is the perspective-DoF-noise regime, not a real view.
    if !max_edge.is_finite() || max_edge > orig_diag * 2.0 {
        return false;
    }
    let safe_min = |a: f32, b: f32| (a.min(b)).max(1.0);
    let w_ratio = w_top.max(w_bot) / safe_min(w_top, w_bot);
    let h_ratio = h_left.max(h_right) / safe_min(h_left, h_right);
    // Tightened from 6.0× to 3.0×: catches lopsided trapezoids that
    // are mathematically valid homographies but indicate a fit
    // driven by an under-spread inlier cluster.
    if w_ratio > 3.0 || h_ratio > 3.0 {
        return false;
    }
    true
}

/// One translated text region to render into the Phase-2 overlay
/// bitmap. The bindings layer fills this in from the OCR + translation
/// pipeline; the engine just stitches them into a `PreparedImageOverlay`
/// and calls the shared rasterizer.
#[cfg(feature = "image-render")]
pub struct TextRenderItem {
    pub id: u64,
    /// Canonical-frame corners (TL, TR, BR, BL) in pixels.
    pub quad: [(f32, f32); 4],
    pub translated_text: String,
    pub source_text: String,
    /// BCP-47 of the target language, used as a font-fallback hint.
    pub language: String,
    pub bg_argb: u32,
    pub fg_argb: u32,
    pub suggested_font_px: f32,
}

/// Invert `OrientedRect::corners()`. Width = distance TL→TR, height =
/// distance TL→BL, angle = TL→TR direction. Robust to slight quad
/// non-rectangularity (post-perspective warps); we use the centroid.
#[cfg(feature = "image-render")]
fn oriented_rect_from_corners(quad: &[(f32, f32); 4]) -> OrientedRect {
    let (tlx, tly) = quad[0];
    let (trx, try_) = quad[1];
    let (brx, bry) = quad[2];
    let (blx, bly) = quad[3];
    let cx = (tlx + trx + brx + blx) * 0.25;
    let cy = (tly + try_ + bry + bly) * 0.25;
    let wdx = trx - tlx;
    let wdy = try_ - tly;
    let hdx = blx - tlx;
    let hdy = bly - tly;
    OrientedRect {
        cx,
        cy,
        width: (wdx * wdx + wdy * wdy).sqrt(),
        height: (hdx * hdx + hdy * hdy).sqrt(),
        angle_radians: wdy.atan2(wdx),
    }
}

// -- Phase 1 debug bitmap helpers ----------------------------------------
//
// Tiny rasterizer for Phase 1 verification. Once Phase 2 lands we route
// through `image_render::render_overlay` instead and these go away.

fn argb_u32_to_rgba8(argb: u32) -> [u8; 4] {
    let a = ((argb >> 24) & 0xff) as u8;
    let r = ((argb >> 16) & 0xff) as u8;
    let g = ((argb >> 8) & 0xff) as u8;
    let b = (argb & 0xff) as u8;
    [r, g, b, a]
}

/// Stamp a flat colour into the canvas across the oriented rect, with
/// rounded corners and a 1-pixel anti-aliased edge. Uses *max alpha*
/// rather than alpha-blending so overlapping boxes (adjacent lines that
/// share an edge) take the louder coverage instead of summing — without
/// this, every overlap would visibly darken.
///
/// All bg fills are expected to use the same RGB; callers paint into a
/// freshly-zeroed canvas before the text rasterizer runs.
pub fn fill_oriented_rect_blended(
    rgba: &mut [u8],
    w: u32,
    h: u32,
    rect: &crate::ocr::OrientedRect,
    color: [u8; 4],
) {
    if color[3] == 0 {
        return;
    }
    let cos = rect.angle_radians.cos();
    let sin = rect.angle_radians.sin();
    let hw = rect.width * 0.5;
    let hh = rect.height * 0.5;
    // Corner radius: 50 % of the short half-extent, capped at 12 px.
    // Visible softness without eating into glyph space on short boxes.
    let radius = (hw.min(hh) * 0.5).min(12.0).max(0.0);
    // AABB of the rect, clipped to the canvas. Rotation can make the
    // AABB bigger than width × height, so use the half-diagonal as the
    // scan window radius.
    let half_diag = (hw * hw + hh * hh).sqrt();
    let min_x = ((rect.cx - half_diag).floor() as i32).max(0) as u32;
    let min_y = ((rect.cy - half_diag).floor() as i32).max(0) as u32;
    let max_x = ((rect.cx + half_diag).ceil() as i32).max(0).min(w as i32) as u32;
    let max_y = ((rect.cy + half_diag).ceil() as i32).max(0).min(h as i32) as u32;
    for y in min_y..max_y {
        for x in min_x..max_x {
            let dx = x as f32 + 0.5 - rect.cx;
            let dy = y as f32 + 0.5 - rect.cy;
            let lx = dx * cos + dy * sin;
            let ly = -dx * sin + dy * cos;
            // Signed distance to a rounded rect centred at the origin
            // with half-extents (hw, hh) and corner radius `radius`.
            // < 0 inside, > 0 outside.
            let qx = lx.abs() - (hw - radius);
            let qy = ly.abs() - (hh - radius);
            let outside = (qx.max(0.0) * qx.max(0.0) + qy.max(0.0) * qy.max(0.0)).sqrt();
            let inside = qx.max(qy).min(0.0);
            let sdf = outside + inside - radius;
            // 1-px feathered edge: full coverage at sdf ≤ -0.5, zero at
            // sdf ≥ 0.5, linear in between.
            let coverage = (0.5 - sdf).clamp(0.0, 1.0);
            if coverage <= 0.0 {
                continue;
            }
            let new_alpha = (color[3] as f32 * coverage).round() as u8;
            if new_alpha == 0 {
                continue;
            }
            let idx = ((y * w + x) * 4) as usize;
            if new_alpha > rgba[idx + 3] {
                rgba[idx] = color[0];
                rgba[idx + 1] = color[1];
                rgba[idx + 2] = color[2];
                rgba[idx + 3] = new_alpha;
            }
        }
    }
}

#[allow(dead_code)]
fn blend_pixel(dst: &mut [u8], src: [u8; 4]) {
    let sa = src[3] as u32;
    if sa == 0 {
        return;
    }
    if sa == 255 {
        dst[0] = src[0];
        dst[1] = src[1];
        dst[2] = src[2];
        dst[3] = 255;
        return;
    }
    let inv = 255 - sa;
    dst[0] = ((src[0] as u32 * sa + dst[0] as u32 * inv) / 255) as u8;
    dst[1] = ((src[1] as u32 * sa + dst[1] as u32 * inv) / 255) as u8;
    dst[2] = ((src[2] as u32 * sa + dst[2] as u32 * inv) / 255) as u8;
    dst[3] = (sa + dst[3] as u32 * inv / 255) as u8;
}
