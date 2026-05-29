//! Static screen-translate pipeline: the no-tracker counterpart to
//! [`crate::live_tracker_pipeline::LiveTrackerPipeline`].
//!
//! A MediaProjection-style screen capture is a flat, fronto-parallel surface
//! fixed in the capture frame, so there is no homography to track — the
//! transform from detected text to overlay is identity. This pipeline drops
//! the engine / coarse-tracker / async-weave machinery entirely and just runs
//! the shared acquire core (detect → orient → rec/translate) on a timestamp
//! cadence, then composites the resident overlays at identity every frame.
//!
//! The detect→rec→translate→overlay-build and the composite are the exact same
//! functions the tracked camera path uses
//! ([`crate::live_tracker_pipeline::acquire_detect`] etc.), so there is no
//! duplicated OCR/overlay logic — only the orchestration differs.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::font_provider::FontProvider;
use crate::live_compositor::ComposeTarget;
use crate::live_frame::LiveFrame;
use crate::live_session::{LiveSession, dominant_axis_quadrant};
use crate::live_tracker_pipeline::{acquire_detect, acquire_rec_translate, composite_overlays};
use crate::ocr::{OrientedRect, Rect};
use crate::session::TranslatorSession;

/// The one anchor the screen pipeline owns; everything composites against it.
const SCREEN_ANCHOR_ID: u64 = 1;

const IDENTITY: [f32; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

/// Coarse-diff downscale: the longest side of the gray we diff per frame. Heavy
/// on purpose — small FX / noise wash out, so only structural change (a real
/// scroll) crosses [`MOVE_THRESHOLD`] — but fine enough that the pill mask hugs
/// the strips (a 1-cell-tall pill at 64 leaked past the mask).
const COARSE_LONG_SIDE: u32 = 128;

/// Mean-abs-diff (0..255) over the coarse gray, vs the window's base frame, that
/// counts as "the screen moved": resets the base + pushes the settle deadline.
const MOVE_THRESHOLD: f32 = 8.0;

/// Quiet window: once no frame exceeds [`MOVE_THRESHOLD`] for this long, the
/// screen is settled → acquire. Purely time-based, never a frame count.
const SETTLE_QUIET_NS: i64 = 120_000_000;

#[derive(Clone)]
struct ScreenConfig {
    from_lang: String,
    to_lang: String,
    is_auto_source: bool,
    det_max_pixels: u32,
    rec_batch_size: usize,
}

impl Default for ScreenConfig {
    fn default() -> Self {
        Self {
            from_lang: String::new(),
            to_lang: String::new(),
            is_auto_source: true,
            det_max_pixels: 650_000,
            rec_batch_size: 4,
        }
    }
}

/// Per-frame result, packed by the JNI layer for the debug pill.
#[derive(Debug, Clone, Default)]
pub struct ScreenFrameResult {
    pub overlay_count: u32,
    pub did_detect: bool,
    pub detected_count: u32,
    pub rec_ok_count: u32,
}

/// What the monitor wants the GL worker to do after a captured frame / tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MonitorAction {
    /// Screen unchanged enough — leave the overlay as-is.
    None,
    /// Movement started — hide the overlay so the user sees real content.
    Hide,
    /// Settled — run a clear → clean-capture → detect/rec/translate → show cycle.
    Acquire,
}

/// The change-detection state. The base frame is the coarse-gray reference the
/// current settle window opened with; deltas are measured against it (cumulative
/// change since the window opened), not against the previous frame.
enum MonitorState {
    /// Debouncing toward an acquire (overlay hidden). Also the bootstrap state
    /// (`base`/`deadline` start `None`), so the first frame opens the window.
    Settling {
        base: Option<Vec<u8>>,
        deadline_ns: Option<i64>,
    },
    /// We just presented a freshly-acquired overlay; the next frame verifies the
    /// screen didn't change *during* the (synchronous, multi-second) OCR by
    /// diffing against `clean_base` — the coarse gray the OCR actually ran on. If
    /// it moved (e.g. the user tabbed to another app mid-acquire), the overlay is
    /// stale → hide + re-acquire; otherwise → Idle.
    Validating { clean_base: Vec<u8> },
    /// Overlay reflects the current static screen; wait for movement. The diff
    /// here masks out the pill cells (our own opaque overlay), so our redraw
    /// never reads as movement and only real content motion in the gaps counts.
    Idle { base: Option<Vec<u8>> },
}

fn mean_abs_diff(a: &[u8], b: &[u8]) -> f32 {
    if a.is_empty() || a.len() != b.len() {
        return 255.0;
    }
    let sum: u64 = a.iter().zip(b).map(|(x, y)| x.abs_diff(*y) as u64).sum();
    sum as f32 / a.len() as f32
}

/// Mean-abs-diff over only the cells not flagged in `excluded` (the pill mask).
/// A fully-masked frame (pills cover everything) yields 0 — no movement.
fn mean_abs_diff_masked(a: &[u8], b: &[u8], excluded: &[bool]) -> f32 {
    if a.is_empty() || a.len() != b.len() || excluded.len() != a.len() {
        return 255.0;
    }
    let mut sum = 0u64;
    let mut n = 0u64;
    for i in 0..a.len() {
        if excluded[i] {
            continue;
        }
        sum += a[i].abs_diff(b[i]) as u64;
        n += 1;
    }
    if n == 0 {
        return 0.0;
    }
    sum as f32 / n as f32
}

/// Whether coarse cell `(px,py)` (centre, with half-extents `half_cx`/`half_cy`)
/// overlaps the oriented pill `r`. Conservative: tests the cell centre against
/// `r` inflated by the cell's half-extent projected onto the rect's local axes,
/// so any cell the pill *touches* is masked (not just centre-inside ones) — the
/// coarse grid is too sparse for centre-only masking to fully cover a 1-cell-tall
/// pill, and a leaked pill edge reads as movement.
fn cell_overlaps_pill(px: f32, py: f32, half_cx: f32, half_cy: f32, r: &OrientedRect) -> bool {
    let dx = px - r.cx;
    let dy = py - r.cy;
    let cos = r.angle_radians.cos();
    let sin = r.angle_radians.sin();
    let lx = (dx * cos + dy * sin).abs();
    let ly = (-dx * sin + dy * cos).abs();
    let infl_x = half_cx * cos.abs() + half_cy * sin.abs();
    let infl_y = half_cx * sin.abs() + half_cy * cos.abs();
    lx <= r.width * 0.5 + infl_x && ly <= r.height * 0.5 + infl_y
}

/// Coarse `gw×gh` mask (true = excluded) of every cell any pill overlaps.
/// `cw/ch` are the canonical dims the `rects` are expressed in.
fn build_pill_mask(rects: &[OrientedRect], gw: u32, gh: u32, cw: u32, ch: u32) -> Vec<bool> {
    let mut mask = vec![false; (gw as usize) * (gh as usize)];
    if rects.is_empty() {
        return mask;
    }
    let sx = cw as f32 / gw as f32;
    let sy = ch as f32 / gh as f32;
    let half_cx = sx * 0.5;
    let half_cy = sy * 0.5;
    for cy in 0..gh {
        let py = (cy as f32 + 0.5) * sy;
        for cx in 0..gw {
            let px = (cx as f32 + 0.5) * sx;
            if rects
                .iter()
                .any(|r| cell_overlaps_pill(px, py, half_cx, half_cy, r))
            {
                mask[(cy * gw + cx) as usize] = true;
            }
        }
    }
    mask
}

fn settling(base: Vec<u8>, now_ns: i64) -> MonitorState {
    MonitorState::Settling {
        base: Some(base),
        deadline_ns: Some(now_ns + SETTLE_QUIET_NS),
    }
}

/// Pure transition for a captured frame; returns the next state + action.
/// `excluded` is the pill mask, applied only in `Idle` (overlay up) — `None`
/// while `Settling` (overlay hidden, diff the whole frame).
fn step_frame(
    state: MonitorState,
    gray: &[u8],
    excluded: Option<&[bool]>,
    now_ns: i64,
) -> (MonitorState, MonitorAction) {
    match state {
        // Resolved by `monitor_validate_clean` (a pre-present, pill-free check),
        // not by captured frames — the GL thread is single-threaded and runs that
        // check before any monitor_frame call. Defensive no-op if one slips in.
        MonitorState::Validating { clean_base } => {
            (MonitorState::Validating { clean_base }, MonitorAction::None)
        }
        MonitorState::Idle { base: None } => (
            MonitorState::Idle {
                base: Some(gray.to_vec()),
            },
            MonitorAction::None,
        ),
        MonitorState::Idle { base: Some(b) } => {
            let g = match excluded {
                Some(ex) => mean_abs_diff_masked(&b, gray, ex),
                None => mean_abs_diff(&b, gray),
            };
            log::debug!("[screen-monitor] idle g={g:.1}");
            if g > MOVE_THRESHOLD {
                log::info!("[screen-monitor] movement g={g:.1} → hide");
                (settling(gray.to_vec(), now_ns), MonitorAction::Hide)
            } else {
                (MonitorState::Idle { base: Some(b) }, MonitorAction::None)
            }
        }
        MonitorState::Settling { base: None, .. } => {
            (settling(gray.to_vec(), now_ns), MonitorAction::None)
        }
        MonitorState::Settling {
            base: Some(b),
            deadline_ns,
        } => {
            let g = mean_abs_diff(&b, gray);
            log::debug!("[screen-monitor] settling g={g:.1}");
            if g > MOVE_THRESHOLD {
                (settling(gray.to_vec(), now_ns), MonitorAction::None)
            } else if deadline_ns.is_some_and(|d| now_ns >= d) {
                log::info!("[screen-monitor] settled (frame) g={g:.1} → acquire");
                (
                    MonitorState::Validating { clean_base: b },
                    MonitorAction::Acquire,
                )
            } else {
                (
                    MonitorState::Settling {
                        base: Some(b),
                        deadline_ns,
                    },
                    MonitorAction::None,
                )
            }
        }
    }
}

/// Pure transition for the pre-present staleness check: `Validating` resolves to
/// `Idle` (screen unchanged → present) or `Settling` + `Hide` (screen moved
/// during OCR → drop stale overlay, re-acquire). `gray` is a pill-free frame.
fn step_validate(state: MonitorState, gray: &[u8], now_ns: i64) -> (MonitorState, MonitorAction) {
    let MonitorState::Validating { clean_base } = state else {
        return (state, MonitorAction::None);
    };
    let g = mean_abs_diff(&clean_base, gray);
    if g > MOVE_THRESHOLD {
        log::info!(
            "[screen-monitor] stale overlay (screen moved during OCR) g={g:.1} → re-acquire"
        );
        (settling(gray.to_vec(), now_ns), MonitorAction::Hide)
    } else {
        log::debug!("[screen-monitor] overlay validated g={g:.1}");
        (MonitorState::Idle { base: None }, MonitorAction::None)
    }
}

/// Pure transition for a timed tick (no new frame): only a pending settle fires.
fn step_tick(state: MonitorState, now_ns: i64) -> (MonitorState, MonitorAction) {
    match state {
        MonitorState::Settling {
            base: Some(clean_base),
            deadline_ns: Some(d),
        } if now_ns >= d => {
            log::info!("[screen-monitor] settled (tick) → acquire");
            (
                MonitorState::Validating { clean_base },
                MonitorAction::Acquire,
            )
        }
        other => (other, MonitorAction::None),
    }
}

pub struct LiveScreenPipeline {
    catalog: Arc<TranslatorSession>,
    session: Arc<LiveSession>,
    font_provider: Arc<dyn FontProvider + Send + Sync>,
    config: Mutex<ScreenConfig>,
    /// Bumped on reset / language change so an in-flight rec/translate bails.
    generation: AtomicU64,
    /// Movement / settle change-detection state (the v1 "Monitoring" logic).
    monitor: Mutex<MonitorState>,
}

impl LiveScreenPipeline {
    pub fn new(
        catalog: Arc<TranslatorSession>,
        font_provider: Arc<dyn FontProvider + Send + Sync>,
    ) -> Arc<Self> {
        let session = Arc::new(LiveSession::new());
        // Opaque pill: the screen overlay window is already alpha-clamped (~0.79)
        // for touch passthrough, so the default translucent pill (0xC8) would
        // double-dim into unreadable mush. Camera keeps the translucent default.
        session.set_overlay_bg([0x00, 0x00, 0x00, 0xFF]);
        Arc::new(Self {
            catalog,
            session,
            font_provider,
            config: Mutex::new(ScreenConfig::default()),
            generation: AtomicU64::new(0),
            monitor: Mutex::new(MonitorState::Settling {
                base: None,
                deadline_ns: None,
            }),
        })
    }

    fn reset_monitor(&self) {
        if let Ok(mut m) = self.monitor.lock() {
            *m = MonitorState::Settling {
                base: None,
                deadline_ns: None,
            };
        }
    }

    /// Coarse-diff dims for a `cw×ch` capture: longest side [`COARSE_LONG_SIDE`],
    /// aspect preserved.
    pub fn coarse_dims(&self, cw: u32, ch: u32) -> (u32, u32) {
        let cw = cw.max(1);
        let ch = ch.max(1);
        if cw >= ch {
            let gw = COARSE_LONG_SIDE.min(cw);
            let gh = ((gw as u64 * ch as u64) / cw as u64).max(1) as u32;
            (gw, gh)
        } else {
            let gh = COARSE_LONG_SIDE.min(ch);
            let gw = ((gh as u64 * cw as u64) / ch as u64).max(1) as u32;
            (gw, gh)
        }
    }

    /// Feed one captured frame's coarse gray (`gw×gh`, oriented like the present;
    /// `cw/ch` are the full canonical dims it samples). Returns what the GL
    /// worker should do. `now_ns` is a monotonic clock (`System.nanoTime()`).
    pub fn monitor_frame(
        &self,
        gray: &[u8],
        gw: u32,
        gh: u32,
        cw: u32,
        ch: u32,
        now_ns: i64,
    ) -> MonitorAction {
        let mut guard = self.monitor.lock().expect("monitor lock");
        // Mask pill cells only while the overlay is up (Idle). In Settling the
        // overlay is hidden, and the stale rects would wrongly exclude the very
        // content whose motion we await.
        let mask = if matches!(&*guard, MonitorState::Idle { base: Some(_) }) {
            let rects = self.session.overlay_pill_rects(SCREEN_ANCHOR_ID);
            Some(build_pill_mask(&rects, gw, gh, cw, ch))
        } else {
            None
        };
        let state = std::mem::replace(&mut *guard, MonitorState::Idle { base: None });
        let (next, action) = step_frame(state, gray, mask.as_deref(), now_ns);
        *guard = next;
        action
    }

    /// Timed tick with no new frame; fires a pending settle so the screen settles
    /// even when the mirror stops emitting frames.
    pub fn monitor_tick(&self, now_ns: i64) -> MonitorAction {
        let mut guard = self.monitor.lock().expect("monitor lock");
        let state = std::mem::replace(&mut *guard, MonitorState::Idle { base: None });
        let (next, action) = step_tick(state, now_ns);
        *guard = next;
        action
    }

    /// Pre-present staleness check: after the (synchronous, multi-second) OCR,
    /// the GL worker drains the latest captured frame — still pill-free, since
    /// the new overlay hasn't been shown — and calls this with its coarse gray.
    /// If it differs from `clean_base` (the frame the OCR ran on), the screen
    /// changed underneath us (e.g. the user tabbed away): `Hide` → drop the stale
    /// overlay and re-acquire. Otherwise `None` → safe to present. No mask: both
    /// frames are pill-free, so we compare the whole frame.
    pub fn monitor_validate_clean(&self, gray: &[u8], now_ns: i64) -> MonitorAction {
        let mut guard = self.monitor.lock().expect("monitor lock");
        let state = std::mem::replace(&mut *guard, MonitorState::Idle { base: None });
        let (next, action) = step_validate(state, gray, now_ns);
        *guard = next;
        action
    }

    /// Whether the GL worker should poll on a timer (a settle deadline is armed).
    pub fn wants_tick(&self) -> bool {
        matches!(
            &*self.monitor.lock().expect("monitor lock"),
            MonitorState::Settling {
                deadline_ns: Some(_),
                ..
            }
        )
    }

    pub fn set_languages(&self, from: &str, to: &str, is_auto_source: bool) {
        if let Ok(mut cfg) = self.config.lock() {
            cfg.from_lang = from.to_string();
            cfg.to_lang = to.to_string();
            cfg.is_auto_source = is_auto_source;
        }
        // Drop stale overlays; the next detect (gated by the GL worker)
        // repopulates them.
        self.generation.fetch_add(1, Ordering::SeqCst);
        self.session.clear_overlays();
        self.reset_monitor();
    }

    pub fn set_overlay_oversample(&self, factor: f32) {
        self.session.set_overlay_oversample(factor);
    }

    /// Detector pixel budget — the JNI layer uses it to size the GPU det-gray
    /// readback ([`crate::live_frame::aligned_det_dims`]).
    pub fn det_max_pixels(&self) -> u32 {
        self.config
            .lock()
            .map(|c| c.det_max_pixels)
            .unwrap_or(650_000)
    }

    pub fn clear_overlay(&self) {
        self.session.clear_overlays();
    }

    pub fn reset(&self) {
        self.generation.fetch_add(1, Ordering::SeqCst);
        self.session.clear();
        self.reset_monitor();
    }

    /// Drive one frame: on the detect cadence run detect → orient →
    /// rec/translate (into [`SCREEN_ANCHOR_ID`]); every frame composite the
    /// resident overlays into `target` at identity. `frame` carries the
    /// captured canonical RGBA; `canonical_w/h` are its dims.
    pub fn process_frame_overlay(
        &self,
        frame: &Arc<LiveFrame>,
        target: &mut dyn ComposeTarget,
        canonical_w: u32,
        canonical_h: u32,
    ) -> ScreenFrameResult {
        let cfg = self.config.lock().map(|c| c.clone()).unwrap_or_default();
        let mut result = ScreenFrameResult::default();

        // The GL worker gates the detect cadence and only calls this on a
        // detect-due frame, so we always detect + composite here.
        result.did_detect = true;
        self.run_detect_cycle(frame, canonical_w, canonical_h, &cfg, &mut result);

        result.overlay_count = composite_overlays(
            &self.session,
            frame,
            target,
            canonical_w,
            canonical_h,
            Some(IDENTITY),
            SCREEN_ANCHOR_ID,
        )
        .unwrap_or(0);
        result
    }

    fn run_detect_cycle(
        &self,
        frame: &Arc<LiveFrame>,
        canonical_w: u32,
        canonical_h: u32,
        cfg: &ScreenConfig,
        result: &mut ScreenFrameResult,
    ) {
        // Each acquire re-detects the whole screen fresh; drop the previous
        // pass's surface map + blocks + canvas so boxes don't accumulate.
        self.session.reset_anchor_state(SCREEN_ANCHOR_ID);
        let crop = Rect {
            left: 0,
            top: 0,
            right: canonical_w,
            bottom: canonical_h,
        };
        let t_det = std::time::Instant::now();
        let detected = match acquire_detect(&self.catalog, frame, crop, cfg.det_max_pixels) {
            Ok(d) => d,
            Err(e) => {
                log::warn!("[screen] detect failed: {e}");
                return;
            }
        };
        let det_ms = t_det.elapsed().as_secs_f64() * 1000.0;
        if detected.is_empty() {
            self.session.clear_overlays();
            log::info!("[screen] detect={det_ms:.0}ms boxes=0");
            return;
        }
        result.detected_count = detected.len() as u32;
        // Geometric 90° quadrant (supports landscape); skip the rec-based 180°
        // disambiguation since a captured screen is world-up.
        let quadrant = Some(dominant_axis_quadrant(&detected));
        let gen_id = self.generation.load(Ordering::SeqCst);
        let cancel = || self.generation.load(Ordering::SeqCst) != gen_id;
        let t_rec = std::time::Instant::now();
        let rec_result = acquire_rec_translate(
            &self.catalog,
            &self.session,
            &*self.font_provider,
            frame,
            crop,
            &cfg.from_lang,
            &cfg.to_lang,
            cfg.is_auto_source,
            cfg.rec_batch_size,
            &detected,
            SCREEN_ANCHOR_ID,
            quadrant,
            &[],
            &cancel,
        );
        let rec_ms = t_rec.elapsed().as_secs_f64() * 1000.0;
        match rec_result {
            Ok(outcome) => {
                result.rec_ok_count = outcome.rec_ok_count;
                // Drop placeholders for rec-failed blocks so only translated
                // text stays resident.
                self.session
                    .retain_blocks(SCREEN_ANCHOR_ID, &outcome.surviving_block_ids);
            }
            Err(e) => log::warn!("[screen] rec/translate failed: {e}"),
        }
        log::info!(
            "[screen] detect={:.0}ms rec+translate={:.0}ms boxes={} rec_ok={}",
            det_ms,
            rec_ms,
            detected.len(),
            result.rec_ok_count,
        );
    }
}

#[cfg(test)]
mod monitor_tests {
    use super::*;

    const MS: i64 = 1_000_000;
    const QUIET_MS: i64 = SETTLE_QUIET_NS / MS;

    fn flat(v: u8) -> Vec<u8> {
        vec![v; 64 * 36]
    }

    #[test]
    fn mean_abs_diff_basics() {
        assert_eq!(mean_abs_diff(&[10, 10, 10], &[10, 10, 10]), 0.0);
        assert_eq!(mean_abs_diff(&[0], &[255]), 255.0);
        // length mismatch / empty → treated as max motion
        assert_eq!(mean_abs_diff(&[1, 2], &[1]), 255.0);
        assert_eq!(mean_abs_diff(&[], &[]), 255.0);
    }

    #[test]
    fn bootstrap_settles_into_first_acquire() {
        // Start = Settling{None,None}; first frame opens the window.
        let s = MonitorState::Settling {
            base: None,
            deadline_ns: None,
        };
        let (s, a) = step_frame(s, &flat(100), None, 0);
        assert_eq!(a, MonitorAction::None);
        // Before the deadline: nothing.
        let (s, a) = step_tick(s, (QUIET_MS - 1) * MS);
        assert_eq!(a, MonitorAction::None);
        // At the deadline: acquire, and we land in Validating awaiting the
        // post-present frame to confirm the screen didn't move during OCR.
        let (s, a) = step_tick(s, QUIET_MS * MS);
        assert_eq!(a, MonitorAction::Acquire);
        assert!(matches!(s, MonitorState::Validating { .. }));
    }

    #[test]
    fn idle_rebaselines_then_hides_on_movement() {
        // Post-present: Idle{None} takes the next frame as the new base.
        let s = MonitorState::Idle { base: None };
        let (s, a) = step_frame(s, &flat(100), None, 0);
        assert_eq!(a, MonitorAction::None);
        assert!(matches!(s, MonitorState::Idle { base: Some(_) }));
        // A sub-threshold frame keeps us Idle (no hide).
        let (s, a) = step_frame(s, &flat(102), None, 10 * MS);
        assert_eq!(a, MonitorAction::None);
        // A big change → hide + start debouncing.
        let (s, a) = step_frame(s, &flat(200), None, 20 * MS);
        assert_eq!(a, MonitorAction::Hide);
        assert!(matches!(
            s,
            MonitorState::Settling {
                deadline_ns: Some(_),
                ..
            }
        ));
        // Quiet again → acquire.
        let (_s, a) = step_tick(s, (20 + QUIET_MS) * MS);
        assert_eq!(a, MonitorAction::Acquire);
    }

    #[test]
    fn pill_mask_excludes_overlay_cells() {
        // A 64×36 grid; one pill covering the left half. Cells under it are masked.
        let (gw, gh, cw, ch) = (64u32, 36u32, 1280u32, 720u32);
        let pill = OrientedRect {
            cx: cw as f32 * 0.25,
            cy: ch as f32 * 0.5,
            width: cw as f32 * 0.5,
            height: ch as f32,
            angle_radians: 0.0,
        };
        let mask = build_pill_mask(&[pill], gw, gh, cw, ch);
        assert!(mask[0], "top-left (under pill) masked");
        assert!(!mask[(gw - 1) as usize], "top-right (no pill) not masked");
        let masked_cells = mask.iter().filter(|m| **m).count();
        assert!(masked_cells > 0 && masked_cells < mask.len());
    }

    #[test]
    fn masked_diff_ignores_change_under_pill() {
        // The overlay redraw changes only pill cells; with those masked the diff
        // is ~0, so our own redraw never reads as movement (no self-change guard).
        let n = 64 * 36;
        let mut base = vec![100u8; n];
        let mut cur = vec![100u8; n];
        let mut excluded = vec![false; n];
        for i in 0..n / 2 {
            excluded[i] = true; // pill covers the first half
            cur[i] = 255; // pill flips those pixels wildly
        }
        assert_eq!(mean_abs_diff_masked(&base, &cur, &excluded), 0.0);
        // A real change in an *unmasked* cell is still seen.
        cur[n - 1] = 200;
        base[n - 1] = 100;
        assert!(mean_abs_diff_masked(&base, &cur, &excluded) > 0.0);
    }

    #[test]
    fn idle_masks_pill_self_change() {
        // Idle with a base; the pill mask excludes the changed region → no hide.
        let mut excluded = vec![false; 64 * 36];
        for e in excluded.iter_mut().take(64 * 36 / 2) {
            *e = true;
        }
        let base = flat(100);
        let mut cur = flat(100);
        for c in cur.iter_mut().take(64 * 36 / 2) {
            *c = 255; // huge change, but all under the pill
        }
        let s = MonitorState::Idle { base: Some(base) };
        let (s, a) = step_frame(s, &cur, Some(&excluded), 0);
        assert_eq!(a, MonitorAction::None);
        assert!(matches!(s, MonitorState::Idle { base: Some(_) }));
    }

    #[test]
    fn constant_subthreshold_frames_still_settle() {
        // A small animation keeps feeding frames, but each is sub-threshold vs
        // the window base, so the deadline is never pushed → it still settles.
        let mut s = settling(flat(100), 0);
        for t in [20, 40, 60, 80, 100] {
            let (ns, a) = step_frame(s, &flat(101), None, t * MS);
            assert_eq!(a, MonitorAction::None, "frame at {t}ms should not fire");
            s = ns;
        }
        // A frame past the deadline (still sub-threshold) fires the acquire.
        let (s, a) = step_frame(s, &flat(101), None, (QUIET_MS + 1) * MS);
        assert_eq!(a, MonitorAction::Acquire);
        assert!(matches!(s, MonitorState::Validating { .. }));
    }

    #[test]
    fn validate_clean_reacquires_when_screen_moved_during_ocr() {
        // Screen unchanged across OCR → overlay valid → Idle (rebaseline next).
        let s = MonitorState::Validating {
            clean_base: flat(100),
        };
        let (s, a) = step_validate(s, &flat(101), 0);
        assert_eq!(a, MonitorAction::None);
        assert!(matches!(s, MonitorState::Idle { base: None }));
        // Screen changed a lot during OCR (tabbed to another app) → stale → hide
        // + re-acquire. Compared pill-free, no mask needed.
        let s = MonitorState::Validating {
            clean_base: flat(0),
        };
        let (s, a) = step_validate(s, &flat(200), 5 * MS);
        assert_eq!(a, MonitorAction::Hide);
        assert!(matches!(
            s,
            MonitorState::Settling {
                deadline_ns: Some(_),
                ..
            }
        ));
    }

    #[test]
    fn movement_pushes_the_deadline() {
        // Each big-change frame resets base + deadline, so settle waits for quiet.
        let mut s = settling(flat(0), 0);
        for (t, v) in [(50, 80u8), (100, 160), (150, 240)] {
            let (ns, a) = step_frame(s, &flat(v), None, t * MS);
            assert_eq!(a, MonitorAction::None);
            s = ns;
        }
        // 150ms + quiet still hasn't elapsed relative to the last reset (150ms).
        let (s, a) = step_tick(s, (150 + QUIET_MS - 1) * MS);
        assert_eq!(a, MonitorAction::None);
        let (_s, a) = step_tick(s, (150 + QUIET_MS) * MS);
        assert_eq!(a, MonitorAction::Acquire);
    }
}
