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
use crate::live_frame::LiveFrame;
use crate::live_session::{LiveSession, dominant_axis_quadrant};
use crate::live_tracker_pipeline::{acquire_detect, acquire_rec_translate};
use crate::live_worker::SlotWorker;
use crate::ocr::{OrientedRect, Rect};
use crate::session::TranslatorSession;

/// The one anchor the screen pipeline owns; everything composites against it.
const SCREEN_ANCHOR_ID: u64 = 1;

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
    /// An async acquire (detect/rec/translate) is running on the worker thread.
    /// `clean_base` is the coarse gray the OCR ran on; each captured frame is
    /// masked-diffed against it (same pill mask as `Idle`), and real content
    /// motion → hide + abort + re-acquire. The worker flips this to `Idle` on a
    /// clean finish; an abort moves it to `Settling` first (guarded so the worker
    /// won't clobber that).
    Acquiring { clean_base: Vec<u8> },
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
        // Worker running OCR. Masked-diff vs the frame it ran on; real content
        // motion → hide + re-acquire (the GL thread also aborts the worker). Our
        // own pills appearing are masked out (same as Idle). The worker flips
        // this to Idle on clean finish.
        MonitorState::Acquiring { clean_base } => {
            let g = match excluded {
                Some(ex) => mean_abs_diff_masked(&clean_base, gray, ex),
                None => mean_abs_diff(&clean_base, gray),
            };
            if g > MOVE_THRESHOLD {
                log::info!(
                    "[screen-monitor] movement g={g:.1} during acquire → abort + re-acquire"
                );
                (settling(gray.to_vec(), now_ns), MonitorAction::Hide)
            } else {
                (MonitorState::Acquiring { clean_base }, MonitorAction::None)
            }
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
                // Stay Settling until the dispatch actually lands (the worker may
                // still be draining a just-aborted job); `monitor_confirm_acquire`
                // flips us to Acquiring on success, else we re-fire next tick.
                (
                    MonitorState::Settling {
                        base: Some(b),
                        deadline_ns,
                    },
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

/// Pure transition for a timed tick (no new frame): only a pending settle fires.
fn step_tick(state: MonitorState, now_ns: i64) -> (MonitorState, MonitorAction) {
    match state {
        MonitorState::Settling {
            base: Some(clean_base),
            deadline_ns: Some(d),
        } if now_ns >= d => {
            log::info!("[screen-monitor] settled (tick) → acquire");
            // Stay Settling until dispatch lands (see step_frame).
            (
                MonitorState::Settling {
                    base: Some(clean_base),
                    deadline_ns: Some(d),
                },
                MonitorAction::Acquire,
            )
        }
        other => (other, MonitorAction::None),
    }
}

/// One acquire job: the GPU-rendered OCR frame + the generation it was dispatched
/// at (so a later `abort`/reset can cancel it mid-flight).
struct ScreenJob {
    frame: Arc<LiveFrame>,
    generation: u64,
}

pub struct LiveScreenPipeline {
    catalog: Arc<TranslatorSession>,
    session: Arc<LiveSession>,
    font_provider: Arc<dyn FontProvider + Send + Sync>,
    config: Mutex<ScreenConfig>,
    /// Bumped on reset / language change / abort so an in-flight rec/translate bails.
    generation: AtomicU64,
    /// Movement / settle change-detection state (the v1 "Monitoring" logic).
    monitor: Mutex<MonitorState>,
    /// Background OCR worker; `None` only transiently during `new`.
    worker: Mutex<Option<SlotWorker<ScreenJob>>>,
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
        // Mark the screen-overlay render path: solid square pills (opaque,
        // touch-cap-dimmed → the camera's SDF feather/rounding is invisible), skip
        // the row-extent scan (only the camera's GPU warp needs it), and defer the
        // canvas raster to the GL thread at present time rather than running it
        // inline on the OCR worker per block (it was O(N²) and stalled rec/translate).
        session.set_screen_overlay(true);
        let pipeline = Arc::new(Self {
            catalog,
            session,
            font_provider,
            config: Mutex::new(ScreenConfig::default()),
            generation: AtomicU64::new(0),
            monitor: Mutex::new(MonitorState::Settling {
                base: None,
                deadline_ns: None,
            }),
            worker: Mutex::new(None),
        });
        let worker = {
            let pipeline_weak = Arc::downgrade(&pipeline);
            SlotWorker::spawn("LiveScreenWorker", move |job: ScreenJob| {
                let Some(pipeline) = pipeline_weak.upgrade() else {
                    return;
                };
                pipeline.run_screen_acquire(job.frame, job.generation);
            })
        };
        *pipeline.worker.lock().expect("worker slot poisoned") = Some(worker);
        pipeline
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
        // Mask pill cells whenever the overlay is (or is becoming) visible — Idle,
        // and Acquiring (our provisional/full pills appear mid-acquire). In
        // Settling the overlay is hidden and stale rects would wrongly exclude the
        // very content whose motion we await.
        let overlay_up = matches!(
            &*guard,
            MonitorState::Idle { base: Some(_) } | MonitorState::Acquiring { .. }
        );
        let mask = if overlay_up {
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

    /// Flip `Settling → Acquiring` once a dispatch actually landed (so a dropped
    /// dispatch — worker still draining an aborted job — leaves us Settling to
    /// retry rather than stuck in Acquiring with no worker).
    fn monitor_confirm_acquire(&self) {
        let mut guard = self.monitor.lock().expect("monitor lock");
        let state = std::mem::replace(&mut *guard, MonitorState::Idle { base: None });
        *guard = match state {
            MonitorState::Settling { base: Some(b), .. } => {
                MonitorState::Acquiring { clean_base: b }
            }
            other => other,
        };
    }

    /// Called by the worker on a clean acquire finish: flip `Acquiring → Idle`
    /// (the next captured frame rebaselines). Guarded so an abort that already
    /// moved us to `Settling` isn't clobbered by a worker that hadn't yet seen
    /// the cancel.
    fn monitor_set_idle_if_acquiring(&self) {
        let mut guard = self.monitor.lock().expect("monitor lock");
        if matches!(&*guard, MonitorState::Acquiring { .. }) {
            *guard = MonitorState::Idle { base: None };
        }
    }

    /// Whether the GL worker should poll on a timer: a settle deadline is armed,
    /// or an acquire is in flight (so it picks up the worker's provisional/full
    /// overlays even though the static screen emits no frames).
    pub fn wants_tick(&self) -> bool {
        matches!(
            &*self.monitor.lock().expect("monitor lock"),
            MonitorState::Settling {
                deadline_ns: Some(_),
                ..
            } | MonitorState::Acquiring { .. }
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

    /// Dispatch an acquire (detect/rec/translate) to the background worker for
    /// the GPU-rendered `frame`. Non-blocking; returns whether the job was
    /// queued (the monitor serializes acquires, so this should always succeed).
    pub fn dispatch_acquire(&self, frame: Arc<LiveFrame>) -> bool {
        let generation = self.generation.load(Ordering::SeqCst);
        let job = ScreenJob { frame, generation };
        let dispatched = self
            .worker
            .lock()
            .ok()
            .and_then(|g| g.as_ref().map(|w| w.try_dispatch(job)))
            .unwrap_or(false);
        if dispatched {
            self.monitor_confirm_acquire();
        }
        dispatched
    }

    /// Abort an in-flight acquire (the screen moved). Bumps the generation so the
    /// worker's `cancel` closure trips at the next rec batch.
    pub fn abort_acquire(&self) {
        self.generation.fetch_add(1, Ordering::SeqCst);
    }

    /// Whether the worker is busy (an acquire is in flight or queued).
    pub fn acquire_busy(&self) -> bool {
        self.worker
            .lock()
            .ok()
            .and_then(|g| g.as_ref().map(|w| w.busy()))
            .unwrap_or(false)
    }

    /// Monotonically increasing each time overlay *content* changes (the
    /// provisional pills, then each rec batch's translated block, then the
    /// post-retain cleanup). The GL thread re-presents whenever this changes —
    /// and the actual canvas raster happens in [`Self::composite_current`] on
    /// that thread, so blocks appear incrementally without stalling the worker.
    pub fn overlay_version(&self) -> u64 {
        self.session.content_version()
    }

    /// Build the GPU overlay draw list (pills + per-block text tiles) from the
    /// resident screen overlay content. The GL thread bakes this into a texture
    /// and presents it — no CPU canvas raster. `None` when there's nothing to
    /// show. Runs on the GL thread.
    pub fn overlay_draw_list(&self) -> Option<crate::live_session::OverlayDrawList> {
        self.session.overlay_draw_list(SCREEN_ANCHOR_ID)
    }

    /// Worker-thread body: detect → provisional overlay → rec/translate → full
    /// overlay, bumping [`overlay_version`](Self::overlay_version) after each
    /// upsert so the GL thread presents provisional pills immediately and the
    /// translated text when ready. Bails at any stage if `generation` moved
    /// (language change / reset / abort-on-movement).
    fn run_screen_acquire(&self, frame: Arc<LiveFrame>, generation: u64) {
        let cfg = self.config.lock().map(|c| c.clone()).unwrap_or_default();
        let cancel = || self.generation.load(Ordering::SeqCst) != generation;
        let (cw, ch) = {
            let s = frame.state().lock().expect("frame state poisoned");
            (s.width, s.height)
        };
        let crop = Rect {
            left: 0,
            top: 0,
            right: cw,
            bottom: ch,
        };
        // Drop the previous pass's surface map + blocks + canvas so boxes don't
        // accumulate.
        self.session.reset_anchor_state(SCREEN_ANCHOR_ID);
        let t_det = std::time::Instant::now();
        let detected = match acquire_detect(&self.catalog, &frame, crop, cfg.det_max_pixels) {
            Ok(d) => d,
            Err(e) => {
                log::warn!("[screen] detect failed: {e}");
                return;
            }
        };
        let det_ms = t_det.elapsed().as_secs_f64() * 1000.0;
        if cancel() {
            return;
        }
        if detected.is_empty() {
            self.session.clear_overlays();
            self.monitor_set_idle_if_acquiring();
            log::info!("[screen] detect={det_ms:.0}ms boxes=0");
            return;
        }
        // Provisional bbox-only pills the instant detection lands (identity
        // transform → the detected tight boxes are already in canonical/surface
        // coords). The canvas rebuild bumps the session version, so the GL thread
        // presents these before rec/translate finishes.
        let strips: Vec<OrientedRect> = detected.iter().map(|d| d.tight_box.clone()).collect();
        self.session
            .upsert_provisional_overlay(SCREEN_ANCHOR_ID, strips, &*self.font_provider);
        // The upsert bumped the content version; the GL present thread builds the
        // provisional draw list + bakes it on its next poll. No CPU canvas raster
        // on the worker (the GPU compositor replaced it).
        if cancel() {
            return;
        }
        // Geometric 90° quadrant (supports landscape); skip the rec-based 180°
        // disambiguation since a captured screen is world-up.
        let quadrant = Some(dominant_axis_quadrant(&detected));
        let t_rec = std::time::Instant::now();
        let rec_result = acquire_rec_translate(
            &self.catalog,
            &self.session,
            &*self.font_provider,
            &frame,
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
        let mut rec_ok = 0;
        match rec_result {
            Ok(outcome) => {
                rec_ok = outcome.rec_ok_count;
                // Drop placeholders for rec-failed blocks so only translated
                // text stays resident, and clear the provisional bbox pills —
                // boxes that never became a surviving block (low detection
                // confidence, rec/translate failure) were still showing their
                // provisional pill (the camera drops these in
                // `acquire_stage_apply_orientation`; the screen path kept them
                // through streaming for feedback). Then rebuild the canvas
                // (retain_blocks/drop only mark it stale) so the cleaned-up
                // overlay — surviving translated blocks only — is presented.
                // (Canvas raster is deferred to the GL thread's present, so these
                // just bump content_version; no inline render on the worker.)
                self.session
                    .retain_blocks(SCREEN_ANCHOR_ID, &outcome.surviving_block_ids);
                self.session.drop_provisional_overlay(SCREEN_ANCHOR_ID);
            }
            Err(e) => log::warn!("[screen] rec/translate failed: {e}"),
        }
        if !cancel() {
            self.monitor_set_idle_if_acquiring();
        }
        log::info!(
            "[screen] detect={det_ms:.0}ms rec+translate={rec_ms:.0}ms boxes={} rec_ok={rec_ok}",
            detected.len(),
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
        // At the deadline: emit Acquire but stay Settling — the GL worker confirms
        // (→ Acquiring) only once the dispatch actually lands.
        let (s, a) = step_tick(s, QUIET_MS * MS);
        assert_eq!(a, MonitorAction::Acquire);
        assert!(matches!(s, MonitorState::Settling { base: Some(_), .. }));
    }

    #[test]
    fn acquiring_aborts_on_movement() {
        // While an acquire runs, a big content change → Hide + back to Settling
        // (the GL thread aborts the worker). A quiet frame keeps Acquiring.
        let s = MonitorState::Acquiring {
            clean_base: flat(100),
        };
        let (s, a) = step_frame(s, &flat(101), None, 0);
        assert_eq!(a, MonitorAction::None);
        assert!(matches!(s, MonitorState::Acquiring { .. }));
        let (s, a) = step_frame(s, &flat(220), None, 5 * MS);
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
        // A frame past the deadline (still sub-threshold) fires the acquire (and
        // stays Settling until the dispatch is confirmed).
        let (s, a) = step_frame(s, &flat(101), None, (QUIET_MS + 1) * MS);
        assert_eq!(a, MonitorAction::Acquire);
        assert!(matches!(s, MonitorState::Settling { base: Some(_), .. }));
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
