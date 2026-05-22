//! Planar surface tracker built on FAST-9 corners + BRIEF-256 descriptors.
//!
//! Replaces the per-region SAD tracker in `live_tracking.rs`. Instead of
//! tracking each text bbox independently, we fit a single homography
//! per frame mapping the original "canonical" frame (the moment we
//! acquired the surface) to the current frame. All overlays are then
//! projected through that one homography so they move rigidly together.
//!
//! See `FUTURE_PLANAR_TRACKER.md` at the repo root for the design and
//! Phase A benchmark results (which is why we use FAST+BRIEF rather
//! than AKAZE).

use image::GrayImage;

use crate::homography::{fit_affine, fit_homography, fit_similarity, invert, mat3_mul, project};

/// Master toggle for per-frame tracker timing logs (target
/// `planar_timing`). Flip to `false` to silence the per-frame
/// detect/describe/match/ransac line — useful when something other
/// than the planar pipeline is being investigated and the timing
/// chatter is in the way. The bindings crate has its own equivalent
/// gate on the `process_and_composite` outer line.
pub const PER_FRAME_TIMING_LOG: bool = false;

/// BRIEF descriptor length in bits. 256 is the original BRIEF default;
/// 32 bytes per keypoint, Hamming distance fits in a u32.
pub const BRIEF_BITS: usize = 256;
/// Byte width of one descriptor.
pub const DESCRIPTOR_BYTES: usize = BRIEF_BITS / 8;
/// Patch around each keypoint sampled by BRIEF: a `(2 * R + 1)`-square
/// area in pixels. 15 → 31x31 patch.
pub const BRIEF_PATCH_RADIUS: i32 = 15;
/// Distance from image edge required for a keypoint to be describable.
/// Sample offsets are within `BRIEF_PATCH_RADIUS` of the centre, but
/// after rotating the BRIEF pattern by an arbitrary angle, the worst
/// case is a corner sample at `(R, R)` rotating to `(0, R√2)` — so we
/// need at least `ceil(R·√2)` plus the 3x3 box-blur neighbourhood.
/// R=15 → ceil(15·√2)=22 → +1 box blur = 23. Round up to 24 for slack.
pub const KEYPOINT_BORDER: i32 = 24;

/// FAST-9 corner. Score is the sum-of-absolute-differences between the
/// 16 circle pixels and the center; used only for non-max suppression.
#[derive(Clone, Copy, Debug)]
pub struct KeyPoint {
    pub x: f32,
    pub y: f32,
    pub score: i32,
}

/// 256-bit BRIEF descriptor. Hamming distance fits in a u32.
#[derive(Clone, Copy, Debug)]
pub struct Descriptor(pub [u8; DESCRIPTOR_BYTES]);

impl Descriptor {
    pub fn hamming(&self, other: &Self) -> u32 {
        let mut d = 0u32;
        for i in 0..DESCRIPTOR_BYTES {
            d += (self.0[i] ^ other.0[i]).count_ones();
        }
        d
    }
}

/// Reference scene captured at acquire time. Each keypoint slot has a
/// position (in this frame's pixel coords — the "canonical" frame for
/// the current acquisition) and a descriptor.
#[derive(Clone, Debug)]
pub struct SceneAnchor {
    pub descriptors: Vec<Descriptor>,
    pub positions: Vec<(f32, f32)>,
    pub image_dims: (u32, u32),
    pub created_at_ns: u64,
}

impl SceneAnchor {
    pub fn len(&self) -> usize {
        self.descriptors.len()
    }
    pub fn is_empty(&self) -> bool {
        self.descriptors.is_empty()
    }
}

/// Result of a successful per-frame fit.
#[derive(Clone, Debug)]
pub struct TrackResult {
    /// Row-major 3x3 mapping canonical-frame pixel coords → current-frame
    /// pixel coords. See `crate::homography::project`.
    pub homography: [f32; 9],
    /// Number of correspondences that survived RANSAC + final LS fit.
    pub inliers: usize,
    /// Total matched correspondences (before RANSAC).
    pub matches: usize,
    /// Median re-projection residual in pixels (over inliers).
    pub median_residual_px: f32,
    /// The actual inlier correspondences `(anchor_x, anchor_y, view_x,
    /// view_y)` — populated by RANSAC for callers that need to feed
    /// them back next frame (e.g. the KLT propagator).
    pub inlier_pairs: Vec<(f32, f32, f32, f32)>,
    /// Subset of `inliers` that came from the descriptor matcher
    /// (i.e. not from externally-injected KLT correspondences). The
    /// engine uses this to drive handoff decisions on descriptor-
    /// matcher health rather than the KLT-padded total, which would
    /// otherwise mask sustained descriptor degradation (e.g. scale
    /// drift past BRIEF's invariance) and starve handoffs.
    pub descriptor_inliers: usize,
}

/// Public-facing tracker state. Wraps an optional anchor and the
/// detection / matching tuning knobs.
pub struct LivePlanarTracker {
    pub anchor: Option<SceneAnchor>,
    pub config: TrackerConfig,
}

#[derive(Clone, Copy, Debug)]
pub struct TrackerConfig {
    /// FAST corner threshold (0..255). Higher → fewer, stronger corners.
    pub fast_threshold: u8,
    /// Fallback FAST threshold for the Locked-state matcher: if a
    /// first-pass detect at `fast_threshold` returns fewer than
    /// `fast_min_keypoints`, we redetect at this looser threshold to
    /// catch corners that motion blur has softened below the strict
    /// score cutoff. Used only on the per-frame match path; acquire and
    /// re-lock keep `fast_threshold` so anchor quality stays high.
    pub fast_threshold_fallback: u8,
    /// Minimum first-pass keypoint count below which the Locked matcher
    /// triggers the `fast_threshold_fallback` redetect.
    pub fast_min_keypoints: usize,
    /// Cap on keypoints returned by detection — strongest survive.
    pub max_features: usize,
    /// Lowe ratio test cutoff (best/second-best Hamming) used by
    /// fresh-acquire and re-lock paths. Lower = stricter, fewer false
    /// matches, fewer total matches; safe default for cold starts where
    /// we have no geometric prior to filter the surviving pool.
    pub lowe_ratio: f32,
    /// Lowe ratio used when the tracker is *already* Locked. Relaxed
    /// (default 0.9) so that descriptors degraded by motion blur — best
    /// vs second-best ratio creeps from 0.7 toward 1.0 as the patch
    /// smears — still get through to RANSAC. The previous-frame H seed
    /// and the engine's sanity gate (corner-jump cap, inlier-bbox
    /// coverage) filter the resulting false matches; the safety net
    /// only triggers when guess-quality has actually collapsed.
    pub lowe_ratio_locked: f32,
    /// RANSAC inlier residual threshold in pixels.
    pub ransac_residual_px: f32,
    /// RANSAC iterations.
    pub ransac_iters: usize,
    /// Minimum inliers to *acquire* a successful track (called from a
    /// cold start or from Lost-recovery). Higher = stricter, but
    /// avoids locking onto a noisy match.
    pub min_inliers: usize,
    /// Minimum inliers to *keep* a currently-Locked track. Lower than
    /// `min_inliers` to give hysteresis — a Locked track that drops a
    /// frame or two near the threshold shouldn't flicker between
    /// Locked and Lost on every camera frame.
    pub min_inliers_keep_locked: usize,
    /// Non-max suppression radius for FAST corners (in pixels).
    pub nms_radius: i32,
    /// Search-window radius for guided descriptor matching, in
    /// pixel-coords of the gray the matcher operates on. The
    /// per-frame Locked matcher uses this when an `H_anchor→view`
    /// prior is provided (= we're tracking, not re-acquiring). Set to
    /// roughly cover the worst-case feature displacement between two
    /// adjacent frames: at 30 fps with handheld slow pan that's
    /// ~10-20 px; we leave headroom for moderate motion.
    pub guided_search_radius_px: f32,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            fast_threshold: 15,
            fast_threshold_fallback: 7,
            fast_min_keypoints: 200,
            max_features: 500,
            lowe_ratio: 0.8,
            lowe_ratio_locked: 0.85,
            ransac_residual_px: 4.0,
            ransac_iters: 400,
            min_inliers: 25,
            // Hysteresis floor for keeping a Locked track. Set to 10
            // so the cascading-model refinement in
            // `ransac_homography_with_prior` can fall through to the
            // 4-DoF similarity fit at low inlier counts — earlier we
            // pinned this at 18 (mid-affine band), which cut RANSAC
            // off before the similarity path ever engaged. Similarity
            // is over-flat on tilted planes (no perspective), but at
            // sub-15-inlier counts the 8-DoF homography is so under-
            // constrained that it produces worse drift than the flat
            // similarity does; better a stable 4-DoF lock through a
            // brief blur than a wild 8-DoF wrong-basin fit. The
            // per-frame H-delta cap + sanity gate spread-coverage
            // check still catch egregiously bad similarity fits.
            min_inliers_keep_locked: 25,
            nms_radius: 3,
            guided_search_radius_px: 30.0,
        }
    }
}

impl LivePlanarTracker {
    pub fn new() -> Self {
        Self::with_config(TrackerConfig::default())
    }

    pub fn with_config(config: TrackerConfig) -> Self {
        Self {
            anchor: None,
            config,
        }
    }

    /// Capture `gray` as the new reference scene. Replaces any existing
    /// anchor. Returns `true` if enough features were found to be useful.
    pub fn acquire(&mut self, gray: &GrayImage, created_at_ns: u64) -> bool {
        let anchor = build_anchor(gray, &self.config, created_at_ns);
        let ok = anchor
            .as_ref()
            .is_some_and(|a| a.len() >= self.config.min_inliers);
        self.anchor = anchor;
        ok
    }

    /// Attempt to fit a homography from the current anchor's canonical
    /// frame to `gray`. Returns `None` if there is no anchor or if the
    /// inlier count falls below `config.min_inliers`.
    pub fn track(&self, gray: &GrayImage) -> Option<TrackResult> {
        let anchor = self.anchor.as_ref()?;
        track_against_anchor(anchor, gray, &self.config)
    }

    pub fn clear(&mut self) {
        self.anchor = None;
    }
}

impl Default for LivePlanarTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Top-level scene-build helper exposed for tests and benches.
pub fn build_anchor(
    gray: &GrayImage,
    cfg: &TrackerConfig,
    created_at_ns: u64,
) -> Option<SceneAnchor> {
    build_anchor_filtered(gray, cfg, created_at_ns, |_x, _y| true)
}

/// Like [`build_anchor`] but only keeps keypoints inside any of the
/// given axis-aligned regions (padded by `pad_px` on each side).
///
/// Why: when the camera is looking at a moving object against a richly
/// textured background (e.g. a book on a duvet), naive whole-frame
/// feature extraction picks up many more background corners than object
/// corners. RANSAC then locks onto the background, and the overlays
/// stop following the object when it moves but the phone doesn't.
///
/// Restricting anchor features to the detected text regions keeps the
/// tracker focused on the surface that actually carries the OCR
/// content.
pub fn build_anchor_in_regions(
    gray: &GrayImage,
    cfg: &TrackerConfig,
    regions: &[(u32, u32, u32, u32)],
    pad_px: u32,
    created_at_ns: u64,
) -> Option<SceneAnchor> {
    if regions.is_empty() {
        return build_anchor(gray, cfg, created_at_ns);
    }
    let (img_w, img_h) = gray.dimensions();
    let padded: Vec<(f32, f32, f32, f32)> = regions
        .iter()
        .map(|&(l, t, r, b)| {
            let lf = (l.saturating_sub(pad_px)) as f32;
            let tf = (t.saturating_sub(pad_px)) as f32;
            let rf = (r.saturating_add(pad_px).min(img_w)) as f32;
            let bf = (b.saturating_add(pad_px).min(img_h)) as f32;
            (lf, tf, rf, bf)
        })
        .collect();
    build_anchor_filtered(gray, cfg, created_at_ns, move |x, y| {
        padded
            .iter()
            .any(|&(l, t, r, b)| x >= l && x < r && y >= t && y < b)
    })
}

fn build_anchor_filtered<F: Fn(f32, f32) -> bool>(
    gray: &GrayImage,
    cfg: &TrackerConfig,
    created_at_ns: u64,
    accept: F,
) -> Option<SceneAnchor> {
    let (w, h) = gray.dimensions();
    let mut kps = detect_fast(gray, cfg.fast_threshold, cfg.max_features, cfg.nms_radius);
    kps.retain(|k| accept(k.x, k.y));
    if kps.is_empty() {
        return None;
    }
    let (kept_kps, descs) = describe_brief(gray, &kps);
    if kept_kps.is_empty() {
        return None;
    }
    let positions: Vec<(f32, f32)> = kept_kps.iter().map(|k| (k.x, k.y)).collect();
    Some(SceneAnchor {
        descriptors: descs,
        positions,
        image_dims: (w, h),
        created_at_ns,
    })
}

/// Single-frame fit. Used internally by `track` and exposed for tests.
pub fn track_against_anchor(
    anchor: &SceneAnchor,
    gray: &GrayImage,
    cfg: &TrackerConfig,
) -> Option<TrackResult> {
    track_against_anchor_with_min(anchor, gray, cfg, cfg.min_inliers)
}

/// Like [`track_against_anchor`] but lets the caller override the
/// inlier threshold. Used by the engine to apply hysteresis: a higher
/// bar for *acquiring* a Locked state vs. a lower bar for *keeping*
/// one we already have. Without this split the per-frame inlier
/// count flickers across `min_inliers` and the state machine
/// thrashes Locked ↔ Lost every frame.
pub fn track_against_anchor_with_min(
    anchor: &SceneAnchor,
    gray: &GrayImage,
    cfg: &TrackerConfig,
    min_inliers: usize,
) -> Option<TrackResult> {
    track_against_anchor_with_prior(anchor, gray, cfg, min_inliers, None)
}

/// Like [`track_against_anchor_with_min`] but seeded with a `prior`
/// homography. The prior is evaluated as the first hypothesis in
/// RANSAC; if it already has many inliers, RANSAC's random iterations
/// rarely beat it, so we converge to a clean fit immediately. Use this
/// with an IMU-derived prediction when fast pan / rotation would
/// otherwise make random sampling miss the answer.
pub fn track_against_anchor_with_prior(
    anchor: &SceneAnchor,
    gray: &GrayImage,
    cfg: &TrackerConfig,
    min_inliers: usize,
    prior: Option<[f32; 9]>,
) -> Option<TrackResult> {
    track_against_anchor_with_prior_and_extra(anchor, gray, cfg, min_inliers, prior, &[])
}

/// Per-frame keypoint + descriptor cache. Reuse one set of features
/// across multiple anchor-match attempts (Lost-state cached-anchor
/// retry loop) instead of redetecting/redescribing per anchor.
#[derive(Clone, Debug)]
pub struct FrameFeatures {
    pub frame_kps: Vec<KeyPoint>,
    pub frame_descs: Vec<Descriptor>,
    /// Raw FAST keypoint count before describe_brief filtered the
    /// near-edge ones. Carried for diagnostics / timing logs only.
    pub raw_kp_count: usize,
}

/// FAST + BRIEF on a single frame, with the same blur-fallback
/// behaviour used inside `track_against_anchor_with_prior_and_extra`.
/// Returns `None` if no usable keypoints survive.
pub fn compute_frame_features(gray: &GrayImage, cfg: &TrackerConfig) -> Option<FrameFeatures> {
    let mut kps = detect_fast(gray, cfg.fast_threshold, cfg.max_features, cfg.nms_radius);
    if kps.len() < cfg.fast_min_keypoints && cfg.fast_threshold_fallback < cfg.fast_threshold {
        let retry = detect_fast(
            gray,
            cfg.fast_threshold_fallback,
            cfg.max_features,
            cfg.nms_radius,
        );
        if retry.len() > kps.len() {
            kps = retry;
        }
    }
    let raw_kp_count = kps.len();
    if kps.is_empty() {
        return None;
    }
    let (frame_kps, frame_descs) = describe_brief(gray, &kps);
    if frame_kps.is_empty() {
        return None;
    }
    Some(FrameFeatures {
        frame_kps,
        frame_descs,
        raw_kp_count,
    })
}

/// Like [`track_against_anchor_with_prior_and_extra`] but with the
/// per-frame FAST + BRIEF already computed by
/// [`compute_frame_features`]. Use when matching the same frame
/// against multiple candidate anchors (e.g. the Lost-state cached-
/// anchor retry loop).
pub fn track_against_anchor_with_features(
    anchor: &SceneAnchor,
    features: &FrameFeatures,
    cfg: &TrackerConfig,
    min_inliers: usize,
    prior: Option<[f32; 9]>,
    extra_pairs: &[(f32, f32, f32, f32)],
) -> Option<TrackResult> {
    let frame_kps = &features.frame_kps;
    let frame_descs = &features.frame_descs;
    let t2 = std::time::Instant::now();
    // Guided when we have a prior (= Locked, tracking through the
    // scene): restrict each anchor's match candidates to a small
    // window around its predicted view position. The narrow window
    // lets blur-degraded descriptors pass Lowe ratio because they
    // compete only against the ~handful of in-window candidates
    // instead of the global pool, and the resulting inlier set is
    // spread across the anchor (every anchor keypoint with a frame
    // feature near its prediction can match), eliminating the
    // clustered-inlier drift seen with brute matching. Brute path
    // remains for re-acquire (no prior) where the previous-frame
    // location is meaningless.
    let mut matches = match prior {
        Some(h_prior) => match_descriptors_guided(
            &anchor.descriptors,
            &anchor.positions,
            frame_kps,
            frame_descs,
            cfg.lowe_ratio_locked,
            &h_prior,
            cfg.guided_search_radius_px,
        ),
        None => match_descriptors(&anchor.descriptors, frame_descs, cfg.lowe_ratio_locked),
    };
    matches.sort_unstable_by_key(|m| m.distance);
    let t_match = t2.elapsed();
    let n_matches = matches.len();
    if matches.len() < 4 {
        if PER_FRAME_TIMING_LOG {
            log::info!(
                target: "planar_timing",
                "match-only: match={:.1}ms (matches={}) → bail",
                t_match.as_secs_f64() * 1000.0,
                n_matches,
            );
        }
        return None;
    }
    let mut pairs: Vec<(f32, f32, f32, f32)> =
        Vec::with_capacity(extra_pairs.len() + matches.len());
    pairs.extend_from_slice(extra_pairs);
    pairs.extend(matches.iter().map(|m| {
        let (ax, ay) = anchor.positions[m.anchor_idx];
        let fk = &frame_kps[m.frame_idx];
        (ax, ay, fk.x, fk.y)
    }));
    let t3 = std::time::Instant::now();
    let mut out = ransac_homography_with_prior(&pairs, cfg, min_inliers, prior);
    let t_ransac = t3.elapsed();
    if let Some(r) = out.as_mut() {
        let r_thresh_sq = cfg.ransac_residual_px * cfg.ransac_residual_px;
        let n_klt = extra_pairs.len();
        let descriptor_pairs = &pairs[n_klt..];
        r.descriptor_inliers = descriptor_pairs
            .iter()
            .filter(
                |&&(px, py, qx, qy)| match crate::homography::project(&r.homography, px, py) {
                    Some((px2, py2)) => {
                        let dx = px2 - qx;
                        let dy = py2 - qy;
                        dx * dx + dy * dy <= r_thresh_sq
                    }
                    None => false,
                },
            )
            .count();
        // Temporal-prior post-refit was tried here and reverted:
        // `homography::fit_homography_with_temporal_prior` with
        // `lambda = max(0, 50 - desc_inliers)` (sparse-only) helped
        // book (M2 19.3 → 13.9) but regressed magn (M2 6.4 → 19.4)
        // and increased Lost frames on park/gintonic. The
        // fundamental tension: λ strong enough to break wrong-basin
        // ties is strong enough to lag during real motion, and
        // descriptor inlier count alone doesn't distinguish
        // "sparse because under-determined" from "sparse because
        // fast motion / transitional frame". A proper Kalman-style
        // per-DoF covariance tracking would solve this; the simple
        // damped DLT cannot. The implementation
        // `fit_homography_with_temporal_prior` is kept in
        // `homography.rs` for future use. See `analysis.md` § "EKF
        // path" for the principled-fit roadmap.
    }
    if PER_FRAME_TIMING_LOG {
        log::info!(
            target: "planar_timing",
            "match-only: match={:.1}ms ransac={:.1}ms (matches={} inliers={:?} desc_inliers={:?})",
            t_match.as_secs_f64() * 1000.0,
            t_ransac.as_secs_f64() * 1000.0,
            n_matches,
            out.as_ref().map(|r| r.inliers),
            out.as_ref().map(|r| r.descriptor_inliers),
        );
    }
    out
}

/// Like [`track_against_anchor_with_prior`] but accepts pre-computed
/// `(anchor_x, anchor_y, view_x, view_y)` correspondences from outside
/// the descriptor matcher (e.g. KLT-tracked inliers from the previous
/// frame). Extras are prepended to the descriptor-matched pairs so
/// PROSAC's high-quality early phases sample them first.
pub fn track_against_anchor_with_prior_and_extra(
    anchor: &SceneAnchor,
    gray: &GrayImage,
    cfg: &TrackerConfig,
    min_inliers: usize,
    prior: Option<[f32; 9]>,
    extra_pairs: &[(f32, f32, f32, f32)],
) -> Option<TrackResult> {
    let features = compute_frame_features(gray, cfg)?;
    track_against_anchor_with_features(anchor, &features, cfg, min_inliers, prior, extra_pairs)
}

/// One match between an anchor descriptor and a current-frame descriptor.
#[derive(Clone, Copy, Debug)]
pub struct Match {
    pub anchor_idx: usize,
    pub frame_idx: usize,
    pub distance: u32,
}

/// Brute-force Hamming matcher with Lowe ratio test. For every anchor
/// descriptor, find the two closest frame descriptors and accept iff
/// `best.distance < lowe_ratio * second.distance`.
pub fn match_descriptors(
    anchor: &[Descriptor],
    frame: &[Descriptor],
    lowe_ratio: f32,
) -> Vec<Match> {
    let mut out = Vec::new();
    if frame.len() < 2 {
        return out;
    }
    for (a_idx, a_desc) in anchor.iter().enumerate() {
        let mut best = u32::MAX;
        let mut best_idx = 0usize;
        let mut second = u32::MAX;
        for (f_idx, f_desc) in frame.iter().enumerate() {
            let d = a_desc.hamming(f_desc);
            if d < best {
                second = best;
                best = d;
                best_idx = f_idx;
            } else if d < second {
                second = d;
            }
        }
        if (best as f32) < lowe_ratio * (second as f32) {
            out.push(Match {
                anchor_idx: a_idx,
                frame_idx: best_idx,
                distance: best,
            });
        }
    }
    out
}

/// Guided Hamming matcher. Uses `h_prior` (typically the last-frame
/// `H_anchor→view`) to project each anchor keypoint into expected view
/// coords; only frame keypoints inside a `search_radius_px`-pixel disk
/// around that prediction are candidates. The Lowe ratio test is
/// applied within that restricted set, so a blur-degraded descriptor —
/// whose "best" candidate in brute matching is barely better than the
/// "second best" amongst the global ~500 frame keypoints — competes
/// only against the handful of candidates in its predicted neighbourhood
/// and usually wins decisively. This is what lets the matcher *survive*
/// motion blur and produce spread inliers when brute matching would
/// collapse to clustered noise.
///
/// Returns brute-style matches; downstream RANSAC + sanity gate are
/// unchanged.
pub fn match_descriptors_guided(
    anchor_descs: &[Descriptor],
    anchor_positions: &[(f32, f32)],
    frame_kps: &[KeyPoint],
    frame_descs: &[Descriptor],
    lowe_ratio: f32,
    h_prior: &[f32; 9],
    search_radius_px: f32,
) -> Vec<Match> {
    let mut out = Vec::new();
    if frame_kps.len() < 2 {
        return out;
    }
    let r_sq = search_radius_px * search_radius_px;
    for (a_idx, (a_desc, a_pos)) in anchor_descs.iter().zip(anchor_positions.iter()).enumerate() {
        let Some(predicted) = project(h_prior, a_pos.0, a_pos.1) else {
            continue;
        };
        let mut best = u32::MAX;
        let mut best_idx = 0usize;
        let mut second = u32::MAX;
        let mut window_count = 0usize;
        for (f_idx, fkp) in frame_kps.iter().enumerate() {
            let dx = fkp.x - predicted.0;
            let dy = fkp.y - predicted.1;
            if dx * dx + dy * dy > r_sq {
                continue;
            }
            window_count += 1;
            let d = a_desc.hamming(&frame_descs[f_idx]);
            if d < best {
                second = best;
                best = d;
                best_idx = f_idx;
            } else if d < second {
                second = d;
            }
        }
        // Need at least 2 candidates for the Lowe ratio test to be
        // meaningful; with only 1 in-window candidate we can't measure
        // ambiguity vs the second-best.
        if window_count < 2 {
            continue;
        }
        if (best as f32) < lowe_ratio * (second as f32) {
            out.push(Match {
                anchor_idx: a_idx,
                frame_idx: best_idx,
                distance: best,
            });
        }
    }
    out
}

/// RANSAC over `(canonical_x, canonical_y, frame_x, frame_y)` pairs.
/// Samples 4 random correspondences per iteration, fits a homography,
/// counts inliers (residual under `cfg.ransac_residual_px`), keeps the
/// best, then re-fits on all inliers.
pub fn ransac_homography(
    pairs: &[(f32, f32, f32, f32)],
    cfg: &TrackerConfig,
) -> Option<TrackResult> {
    ransac_homography_with_min(pairs, cfg, cfg.min_inliers)
}

pub fn ransac_homography_with_min(
    pairs: &[(f32, f32, f32, f32)],
    cfg: &TrackerConfig,
    min_inliers: usize,
) -> Option<TrackResult> {
    ransac_homography_with_prior(pairs, cfg, min_inliers, None)
}

pub fn ransac_homography_with_prior(
    pairs: &[(f32, f32, f32, f32)],
    cfg: &TrackerConfig,
    min_inliers: usize,
    prior: Option<[f32; 9]>,
) -> Option<TrackResult> {
    if pairs.len() < 4 {
        return None;
    }
    let mut rng = SmallRng::from_seed(0xA5A5_5A5A_3C3C_C3C3);
    let mut best_h: Option<[f32; 9]> = None;
    let mut best_inliers_idx: Vec<usize> = Vec::new();
    let r_thresh_sq = cfg.ransac_residual_px * cfg.ransac_residual_px;
    let n = pairs.len();
    // Only trust the prior if it's *still* well-supported by the
    // current frame's correspondences. Threshold = half of min_inliers
    // — if fewer than that many current matches fit the prior, the
    // prior is stale (the scene has moved enough that it no longer
    // explains the data) and biasing RANSAC toward it would be
    // counter-productive. This prevents the "stuck prior" failure
    // where a fast camera move makes the previous H useless but the
    // prior's barely-passing inlier count blocks random samples from
    // displacing it.
    let prior_min_inliers = (min_inliers / 2).max(1);
    if let Some(h) = prior {
        let mut inliers_idx = Vec::with_capacity(n);
        for (i, &(px, py, qx, qy)) in pairs.iter().enumerate() {
            if let Some((px2, py2)) = project(&h, px, py) {
                let dx = px2 - qx;
                let dy = py2 - qy;
                if dx * dx + dy * dy <= r_thresh_sq {
                    inliers_idx.push(i);
                }
            }
        }
        if inliers_idx.len() >= prior_min_inliers {
            best_inliers_idx = inliers_idx;
            best_h = Some(h);
        }
    }
    // PROSAC schedule: assumes `pairs` is pre-sorted by match quality
    // (best first). Early iters sample from a small top-quality
    // prefix; the prefix grows in four phases over the iteration
    // budget so that bad-prior tops get a chance to lose to broader
    // samples later. Quarters were picked over a smooth curve for
    // implementation simplicity — the gain over uniform random is
    // primarily from phase 0, not the exact growth law.
    let quarter = (cfg.ransac_iters / 4).max(1);
    for t in 0..cfg.ransac_iters {
        let phase = (t / quarter).min(3);
        let growth_cap = match phase {
            0 => (n / 4).max(8).min(n),
            1 => (n / 2).max(8).min(n),
            2 => (n * 3 / 4).max(8).min(n),
            _ => n,
        };
        let mut sample = [(0.0f32, 0.0f32, 0.0f32, 0.0f32); 4];
        let mut idxs = [0usize; 4];
        for k in 0..4 {
            loop {
                let candidate = (rng.next_u32() as usize) % growth_cap;
                if idxs[..k].iter().all(|&p| p != candidate) {
                    idxs[k] = candidate;
                    sample[k] = pairs[candidate];
                    break;
                }
            }
        }
        let h = match fit_homography(&sample) {
            Some(h) => h,
            None => continue,
        };
        let mut inliers_idx = Vec::with_capacity(n);
        for (i, &(px, py, qx, qy)) in pairs.iter().enumerate() {
            let Some((px2, py2)) = project(&h, px, py) else {
                continue;
            };
            let dx = px2 - qx;
            let dy = py2 - qy;
            if dx * dx + dy * dy <= r_thresh_sq {
                inliers_idx.push(i);
            }
        }
        // `>=` rather than `>`: a random sample that *ties* the
        // current best (often the prior) gets to replace it. Strict
        // `>` made the prior sticky — once seeded, a fast camera
        // move that produced equally-noisy correspondences for
        // *every* random sample couldn't displace it, so the engine
        // returned the same H frame after frame while the scene
        // visibly moved underneath. Tie-replacement is slightly
        // non-deterministic (last winning sample wins) but allows
        // escape from a stale seed when random sampling finds an
        // equally-good fresh fit.
        if inliers_idx.len() >= best_inliers_idx.len() && !inliers_idx.is_empty() {
            best_inliers_idx = inliers_idx;
            best_h = Some(h);
        }
    }
    if best_inliers_idx.len() < min_inliers {
        return None;
    }
    let _ = best_h?;
    let inlier_pairs: Vec<(f32, f32, f32, f32)> =
        best_inliers_idx.iter().map(|&i| pairs[i]).collect();
    // Adaptive model: pick fitter complexity by inlier count. At sparse
    // inlier counts the 8-DoF homography fit is under-constrained — the
    // perspective entries (h6/h7) and the implicit rotation get driven
    // by noise, producing frame-to-frame jitter (sub-30 inliers) or
    // sudden 30°+ rotation jumps (<20 inliers). Affine drops to 6 DoF
    // (no perspective); similarity drops to 4 DoF (uniform scale +
    // explicit rotation parameter), both stable at the sparse end.
    // Above ~30 inliers the full homography wins back its real
    // perspective expressiveness on actually-tilted surfaces.
    let refined = if inlier_pairs.len() >= 30 {
        fit_homography(&inlier_pairs)?
    } else if inlier_pairs.len() >= 15 {
        fit_affine(&inlier_pairs)?
    } else {
        fit_similarity(&inlier_pairs)?
    };
    let mut residuals: Vec<f32> = Vec::with_capacity(inlier_pairs.len());
    for &(px, py, qx, qy) in &inlier_pairs {
        if let Some((px2, py2)) = project(&refined, px, py) {
            let dx = px2 - qx;
            let dy = py2 - qy;
            residuals.push((dx * dx + dy * dy).sqrt());
        }
    }
    residuals.sort_by(|a, b| a.total_cmp(b));
    let median = if residuals.is_empty() {
        f32::INFINITY
    } else {
        residuals[residuals.len() / 2]
    };
    Some(TrackResult {
        homography: refined,
        inliers: inlier_pairs.len(),
        matches: pairs.len(),
        median_residual_px: median,
        inlier_pairs,
        descriptor_inliers: 0,
    })
}

/// FAST-9 corner detector with non-maximum suppression. Scans every
/// interior pixel; tests if 9 contiguous pixels of the Bresenham-3
/// circle are all brighter than `c + t` or all darker than `c - t`.
/// Returns the top `max_features` corners by score, after suppressing
/// any corner that has a higher-scoring neighbour within `nms_radius`.
pub fn detect_fast(
    gray: &GrayImage,
    threshold: u8,
    max_features: usize,
    nms_radius: i32,
) -> Vec<KeyPoint> {
    let (w, h) = gray.dimensions();
    let w_i = w as i32;
    let h_i = h as i32;
    if w_i < 2 * KEYPOINT_BORDER + 1 || h_i < 2 * KEYPOINT_BORDER + 1 {
        return Vec::new();
    }
    let buf = gray.as_raw();
    let mut raw = Vec::with_capacity(4096);
    let t = threshold as i16;
    for y in KEYPOINT_BORDER..(h_i - KEYPOINT_BORDER) {
        let row = (y as u32 * w) as usize;
        for x in KEYPOINT_BORDER..(w_i - KEYPOINT_BORDER) {
            let c = buf[row + x as usize] as i16;
            let hi = c + t;
            let lo = c - t;
            let p =
                |dx: i32, dy: i32| buf[((y + dy) as u32 * w) as usize + (x + dx) as usize] as i16;
            // Bresenham circle of radius 3 (clockwise from top).
            let p1 = p(0, -3);
            let p5 = p(3, 0);
            let p9 = p(0, 3);
            let p13 = p(-3, 0);
            let n_hi = (p1 > hi) as u8 + (p5 > hi) as u8 + (p9 > hi) as u8 + (p13 > hi) as u8;
            let n_lo = (p1 < lo) as u8 + (p5 < lo) as u8 + (p9 < lo) as u8 + (p13 < lo) as u8;
            if n_hi < 3 && n_lo < 3 {
                continue;
            }
            let pix = [
                p1,
                p(1, -3),
                p(2, -2),
                p(3, -1),
                p5,
                p(3, 1),
                p(2, 2),
                p(1, 3),
                p9,
                p(-1, 3),
                p(-2, 2),
                p(-3, 1),
                p13,
                p(-3, -1),
                p(-2, -2),
                p(-1, -3),
            ];
            let (is_corner, score) = fast9_test(&pix, c, t);
            if is_corner {
                raw.push(KeyPoint {
                    x: x as f32,
                    y: y as f32,
                    score,
                });
            }
        }
    }
    let mut kps = nms_filter(raw, nms_radius, max_features);
    // Sub-pixel refinement: each FAST keypoint comes out at an
    // integer pixel because the corner test runs at integer pixel
    // positions. The actual visual corner lies somewhere within
    // that pixel's footprint — typically ±0.5 px. Fitting a parabola
    // to the SAD score response across the 4 cardinal neighbours
    // and interpolating to its peak recovers ~0.1-0.2 px localisation
    // accuracy, which propagates directly to a tighter RANSAC fit
    // (per-keypoint position error is the source noise the H fit
    // distributes across its DoFs). See `analysis.md` § "Source-
    // side fixes". Cost: ~32 k extra ops per frame on 500 kps,
    // negligible vs detect itself.
    for kp in kps.iter_mut() {
        refine_keypoint_subpixel(buf, w as usize, w_i, h_i, kp);
    }
    kps
}

/// Compute the FAST SAD score at an arbitrary pixel — sum of
/// |circle_pixel − centre| over the 16 Bresenham-radius-3 circle
/// pixels. Unlike `fast9_test`, this returns a score regardless of
/// whether the pixel actually passes the FAST-9 corner test;
/// callers use it to evaluate the score *surface* around a confirmed
/// corner for parabolic peak fitting.
#[inline]
fn fast_sad_at(buf: &[u8], w: usize, x: i32, y: i32) -> i32 {
    let c = buf[(y as usize) * w + (x as usize)] as i32;
    let p = |dx: i32, dy: i32| -> i32 { buf[((y + dy) as usize) * w + ((x + dx) as usize)] as i32 };
    let offsets: [(i32, i32); 16] = [
        (0, -3),
        (1, -3),
        (2, -2),
        (3, -1),
        (3, 0),
        (3, 1),
        (2, 2),
        (1, 3),
        (0, 3),
        (-1, 3),
        (-2, 2),
        (-3, 1),
        (-3, 0),
        (-3, -1),
        (-2, -2),
        (-1, -3),
    ];
    let mut s = 0i32;
    for &(dx, dy) in &offsets {
        s += (p(dx, dy) - c).abs();
    }
    s
}

/// Refine a FAST keypoint's (x, y) to sub-pixel position by fitting
/// a 1D parabola in x and y separately to the SAD response at the
/// integer-pixel centre and its 4 cardinal neighbours. The keypoint's
/// `score` field is unchanged (still the integer-pixel SAD); only
/// `x` and `y` are refined.
///
/// Parabola through `(-1, S_l), (0, S_c), (1, S_r)` peaks at
/// `x_peak = (S_l − S_r) / (2 · (S_l + S_r − 2·S_c))`. Only valid
/// when `S_c` is a local max in that axis (denom < 0). If the
/// integer pixel isn't a local max along an axis (because the SAD
/// surface around this corner is degenerate or saddle-like), no
/// refinement is applied on that axis. The result is clamped to
/// `[-0.5, +0.5]` — parabolic fit is only trustworthy for offsets
/// smaller than the sample spacing.
fn refine_keypoint_subpixel(buf: &[u8], w: usize, w_i: i32, h_i: i32, kp: &mut KeyPoint) {
    let kx = kp.x as i32;
    let ky = kp.y as i32;
    // SAD evaluation at (kx±1, ky) and (kx, ky±1) needs the radius-3
    // circle around those neighbours to be in-frame: i.e. positions
    // (kx±4, ky) and (kx, ky±4) must be valid pixels. The detector's
    // KEYPOINT_BORDER (24) already guarantees a much wider margin
    // for all kept keypoints, so this is just a defensive check.
    if kx < 4 || kx >= w_i - 4 || ky < 4 || ky >= h_i - 4 {
        return;
    }
    let s_c = kp.score;
    let s_l = fast_sad_at(buf, w, kx - 1, ky);
    let s_r = fast_sad_at(buf, w, kx + 1, ky);
    let s_u = fast_sad_at(buf, w, kx, ky - 1);
    let s_d = fast_sad_at(buf, w, kx, ky + 1);
    let denom_x = (s_l + s_r - 2 * s_c) as f32;
    let denom_y = (s_u + s_d - 2 * s_c) as f32;
    let dx = if denom_x < -1e-3 {
        (0.5 * (s_l - s_r) as f32 / denom_x).clamp(-0.5, 0.5)
    } else {
        0.0
    };
    let dy = if denom_y < -1e-3 {
        (0.5 * (s_u - s_d) as f32 / denom_y).clamp(-0.5, 0.5)
    } else {
        0.0
    };
    kp.x += dx;
    kp.y += dy;
}

/// Check FAST-9 condition on 16 circle pixels: 9 contiguous brighter
/// than `c + t` OR 9 contiguous darker than `c - t`. The score is the
/// sum-of-absolute-differences of all circle pixels vs the center —
/// fine for non-max suppression even though it's not the "standard"
/// FAST score function.
fn fast9_test(pix: &[i16; 16], c: i16, t: i16) -> (bool, i32) {
    let hi = c + t;
    let lo = c - t;
    let bright = [
        pix[0] > hi,
        pix[1] > hi,
        pix[2] > hi,
        pix[3] > hi,
        pix[4] > hi,
        pix[5] > hi,
        pix[6] > hi,
        pix[7] > hi,
        pix[8] > hi,
        pix[9] > hi,
        pix[10] > hi,
        pix[11] > hi,
        pix[12] > hi,
        pix[13] > hi,
        pix[14] > hi,
        pix[15] > hi,
    ];
    let dark = [
        pix[0] < lo,
        pix[1] < lo,
        pix[2] < lo,
        pix[3] < lo,
        pix[4] < lo,
        pix[5] < lo,
        pix[6] < lo,
        pix[7] < lo,
        pix[8] < lo,
        pix[9] < lo,
        pix[10] < lo,
        pix[11] < lo,
        pix[12] < lo,
        pix[13] < lo,
        pix[14] < lo,
        pix[15] < lo,
    ];
    let mut score = 0i32;
    for v in pix {
        score += (*v as i32 - c as i32).abs();
    }
    (has_run(&bright, 9) || has_run(&dark, 9), score)
}

/// Does the cyclic boolean array contain a run of at least `n` trues?
fn has_run(arr: &[bool; 16], n: usize) -> bool {
    let mut run = 0;
    let mut total = 0;
    for &b in arr.iter().chain(arr.iter().take(n - 1)) {
        if b {
            run += 1;
            total = total.max(run);
            if total >= n {
                return true;
            }
        } else {
            run = 0;
        }
    }
    false
}

/// Greedy non-maximum suppression: sort by score descending, then keep
/// only points whose `(nms_radius)` neighbourhood contains no
/// already-kept higher-scoring point. Returns at most `max_features`.
fn nms_filter(mut kps: Vec<KeyPoint>, nms_radius: i32, max_features: usize) -> Vec<KeyPoint> {
    kps.sort_unstable_by(|a, b| b.score.cmp(&a.score));
    let r2 = (nms_radius * nms_radius) as f32;
    let mut kept: Vec<KeyPoint> = Vec::with_capacity(max_features.min(kps.len()));
    for kp in kps {
        let too_close = kept.iter().any(|other| {
            let dx = other.x - kp.x;
            let dy = other.y - kp.y;
            dx * dx + dy * dy <= r2
        });
        if too_close {
            continue;
        }
        kept.push(kp);
        if kept.len() >= max_features {
            break;
        }
    }
    kept
}

/// Compute oriented BRIEF-256 descriptors for each keypoint. The ORB-
/// style intensity centroid gives a dominant orientation per keypoint;
/// the BRIEF sample pattern is rotated by that angle before sampling.
/// This makes descriptors rotation-invariant — without it, rotating the
/// camera (or the scene) by more than ~30° drops the match rate to
/// near-zero.
///
/// Drops keypoints too close to the edge; the returned `keypoints`
/// vector is filtered in lock-step with `descriptors`.
pub fn describe_brief(gray: &GrayImage, kps: &[KeyPoint]) -> (Vec<KeyPoint>, Vec<Descriptor>) {
    let (w, h) = gray.dimensions();
    let w_i = w as i32;
    let h_i = h as i32;
    let buf = gray.as_raw();
    let pattern = &BRIEF_PATTERN;
    let mut kept_kps = Vec::with_capacity(kps.len());
    let mut descs = Vec::with_capacity(kps.len());
    // Sample pairs at distance ≤ R from the keypoint, plus the 3x3 box-blur
    // neighbourhood; KEYPOINT_BORDER already accounts for both.
    for kp in kps {
        let cx = kp.x.round() as i32;
        let cy = kp.y.round() as i32;
        if cx < KEYPOINT_BORDER
            || cy < KEYPOINT_BORDER
            || cx >= w_i - KEYPOINT_BORDER
            || cy >= h_i - KEYPOINT_BORDER
        {
            continue;
        }
        let angle = patch_orientation(buf, w, cx, cy);
        let (sin_a, cos_a) = angle.sin_cos();
        // 3x3 box-blur of the gray patch at integer offsets — matches the
        // smoothing the original BRIEF paper prescribes. Bounds are
        // guaranteed by KEYPOINT_BORDER + cap on rotated radius.
        let sample = |dx: i32, dy: i32| -> u8 {
            let mut acc = 0u32;
            for oy in -1..=1 {
                let row = ((cy + dy + oy) as u32 * w) as usize;
                for ox in -1..=1 {
                    acc += buf[row + (cx + dx + ox) as usize] as u32;
                }
            }
            (acc / 9) as u8
        };
        let mut bytes = [0u8; DESCRIPTOR_BYTES];
        for (i, &(ax, ay, bx, by)) in pattern.iter().enumerate() {
            // Rotate the sample offsets by the keypoint's dominant angle.
            // Rotation magnitudes are bounded by BRIEF_PATCH_RADIUS, well
            // inside KEYPOINT_BORDER.
            let rax = (cos_a * ax as f32 - sin_a * ay as f32).round() as i32;
            let ray = (sin_a * ax as f32 + cos_a * ay as f32).round() as i32;
            let rbx = (cos_a * bx as f32 - sin_a * by as f32).round() as i32;
            let rby = (sin_a * bx as f32 + cos_a * by as f32).round() as i32;
            let a_val = sample(rax, ray);
            let b_val = sample(rbx, rby);
            if a_val < b_val {
                bytes[i / 8] |= 1 << (i % 8);
            }
        }
        kept_kps.push(*kp);
        descs.push(Descriptor(bytes));
    }
    (kept_kps, descs)
}

/// ORB-style intensity centroid. Compute first-order moments of a
/// circular patch of radius `BRIEF_PATCH_RADIUS` around (cx, cy); the
/// angle from the patch centre to the intensity-weighted centroid is
/// the dominant orientation. Bounds-safe: the caller has already
/// checked `KEYPOINT_BORDER`.
fn patch_orientation(buf: &[u8], w: u32, cx: i32, cy: i32) -> f32 {
    let r = BRIEF_PATCH_RADIUS;
    let r_sq = r * r;
    let mut m10: i32 = 0;
    let mut m01: i32 = 0;
    for dy in -r..=r {
        let row = ((cy + dy) as u32 * w) as usize;
        for dx in -r..=r {
            if dx * dx + dy * dy > r_sq {
                continue;
            }
            let v = buf[row + (cx + dx) as usize] as i32;
            m10 += dx * v;
            m01 += dy * v;
        }
    }
    if m10 == 0 && m01 == 0 {
        return 0.0;
    }
    (m01 as f32).atan2(m10 as f32)
}

/// The BRIEF sampling pattern. 256 (ax, ay, bx, by) tuples uniformly
/// drawn within the patch via a deterministic xorshift PRNG, computed at
/// compile time. No runtime initialization, no rand-crate dep, and the
/// pattern is identical on every platform.
const BRIEF_PATTERN: [(i8, i8, i8, i8); BRIEF_BITS] = generate_brief_pattern();

const fn generate_brief_pattern() -> [(i8, i8, i8, i8); BRIEF_BITS] {
    let mut out = [(0i8, 0i8, 0i8, 0i8); BRIEF_BITS];
    let mut state: u64 = 0x1234_5678_9ABC_DEF0;
    let r = BRIEF_PATCH_RADIUS as i32;
    let span = (2 * r + 1) as u32;
    let mut i = 0;
    while i < BRIEF_BITS {
        let (s1, ax) = const_uniform(state, span, r);
        let (s2, ay) = const_uniform(s1, span, r);
        let (s3, bx) = const_uniform(s2, span, r);
        let (s4, by) = const_uniform(s3, span, r);
        state = s4;
        out[i] = (ax, ay, bx, by);
        i += 1;
    }
    out
}

const fn const_uniform(state: u64, span: u32, radius: i32) -> (u64, i8) {
    let next = xorshift64(state);
    let v = (next as u32) % span;
    (next, (v as i32 - radius) as i8)
}

const fn xorshift64(mut x: u64) -> u64 {
    if x == 0 {
        x = 0xDEAD_BEEF_CAFE_BABE;
    }
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

/// Tiny xorshift64 PRNG. We deliberately don't depend on `rand` — keeps
/// the dep tree small and the determinism guarantees explicit.
pub struct SmallRng(u64);

impl SmallRng {
    pub fn from_seed(seed: u64) -> Self {
        Self(if seed == 0 {
            0xDEAD_BEEF_CAFE_BABE
        } else {
            seed
        })
    }

    pub fn next_u32(&mut self) -> u32 {
        self.0 = xorshift64(self.0);
        self.0 as u32
    }
}

/// Compose two homographies: `out = a * b` (matrix multiplication on
/// row-major 3x3). Convenience wrapper over `crate::homography::mat3_mul`.
pub fn compose(a: &[f32; 9], b: &[f32; 9]) -> [f32; 9] {
    mat3_mul(a, b)
}

/// `crate::homography::invert` re-exported for tests.
pub fn invert_h(h: &[f32; 9]) -> Option<[f32; 9]> {
    invert(h)
}
