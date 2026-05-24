//! End-to-end orchestration for the live-camera planar OCR path.
//!
//! Owns the planar tracker engine, the smoothed homography, the
//! resident overlay store, the matted-strip cache, and an internal
//! worker thread that runs the heavy acquire / refresh pipelines off
//! the per-frame critical path. Public surface is intentionally tiny:
//!
//!   * [`LiveTrackerPipeline::process_frame`] — synchronous per-frame
//!     entry. Runs the tracker step, composites directly into a
//!     caller-supplied RGBA slice, and (if needed) materializes the
//!     frame bytes + dispatches an async acquire/refresh job.
//!   * [`LiveTrackerPipeline::set_languages`] — update the language
//!     config used by future async jobs.
//!   * [`LiveTrackerPipeline::set_target_mode`] — toggle between
//!     `Active` (normal operation) and `Suppressed` (e.g. during AF
//!     scans — every frame resets engine state and bumps generation).
//!   * [`LiveTrackerPipeline::reset`] — bump generation, clear engine
//!     + smoothed H + session state. Any in-flight worker job will
//!     observe the new generation and bail.
//!   * [`LiveTrackerPipeline::last_acquire_telemetry`] — pull the most
//!     recent async-job outcome for debug telemetry.
//!
//! Per-frame fast path returns a small `ProcessFrameResult`; the bulk
//! of acquire telemetry lives in `last_acquire_telemetry` because most
//! frames don't trigger an acquire and the per-frame caller doesn't
//! need to allocate / marshal a big struct.

#![cfg(feature = "planar-tracker")]

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

use crate::LanguageCode;
use crate::api::TranslatorError;
use crate::color_matting::MattedStrip;
use crate::coords::Quadrant;
use crate::font_provider::FontProvider;
use crate::homography;
use crate::live_compositor::{self, CompositeError, OverlayItem, Renderer};
use crate::live_frame::LiveFrame;
use crate::live_session::{
    LiveSession, PostDetectInput, h_view_to_surface_from, viewport_surface_aabb,
};
use crate::ocr::{DetectedTextBox, OrientedRect, Rect};
use crate::planar_engine::{EngineConfig, LivePlanarEngine, TrackerCommand};
use crate::session::TranslatorSession;

/// Tracker state surfaced to the caller. Mirrors the engine command
/// shape but flattened for FFI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanarTrackerState {
    Idle,
    Acquiring,
    Locked,
    Lost,
}

/// Operating mode for the pipeline. `Suppressed` mirrors the
/// pre-refactor behaviour where every frame during a camera AF scan
/// bumped the generation + cleared engine state to defeat any
/// mid-scan acquire while still letting the compositor paint
/// camera-only frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetMode {
    Active,
    Suppressed,
}

#[derive(Clone)]
struct PipelineConfig {
    from_lang: String,
    to_lang: String,
    is_auto_source: bool,
    target_mode: TargetMode,
    /// Detector pixel budget — passed to `OrientedImage::build*` for
    /// downsampling. Both OCR (visible region) and tracker (full
    /// display) caches use the same cap.
    det_max_pixels: u32,
    /// Padding around each detected text box when building an anchor
    /// (see `acquire_now_in_regions`).
    anchor_padding_px: u32,
    /// Per-batch translate size during the rec → translate fan-out.
    rec_batch_size: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            from_lang: String::new(),
            to_lang: String::new(),
            is_auto_source: false,
            target_mode: TargetMode::Active,
            det_max_pixels: 650_000,
            anchor_padding_px: 60,
            rec_batch_size: 4,
        }
    }
}

/// Telemetry returned by an acquire or refresh job. Surfaced to
/// callers via [`LiveTrackerPipeline::last_acquire_telemetry`] for the
/// debug status pill; not on the per-frame fast path.
#[derive(Debug, Clone, Default)]
pub struct AcquireTelemetry {
    pub anchor_id: u64,
    pub detected_count: u32,
    pub rec_ok_count: u32,
    pub rec_empty_count: u32,
    pub cache_hits: u32,
    pub rec_called_count: u32,
    pub total_ms: f64,
    pub canceled: bool,
    pub error: Option<String>,
    /// `true` if this telemetry came from a refresh job, `false` for
    /// an acquire. Debug pills can label accordingly.
    pub is_refresh: bool,
}

/// Result of one synchronous [`LiveTrackerPipeline::process_frame`] call.
#[derive(Debug, Clone)]
pub struct ProcessFrameResult {
    pub state: PlanarTrackerState,
    pub anchor_id: u64,
    pub inliers: u32,
    /// Number of bytes the compositor wrote into the destination
    /// slice. `0` means no composite happened (display dims zero,
    /// frame buffer empty, etc.).
    pub composite_bytes: u32,
    /// `true` when this call spawned an async acquire pipeline job.
    pub started_acquire: bool,
    /// `true` when this call spawned an async refresh pipeline job.
    pub started_refresh: bool,
}

/// Per-anchor record of the last emitted homography. Used so the
/// `LOSS_HIDE_AFTER_FRAMES` grace period during Lost can keep
/// projecting overlays through the last-good H before they hide.
#[derive(Default)]
struct LastEmittedH {
    h: Option<[f32; 9]>,
    anchor_id: u64,
    consecutive_lost: u32,
}

const LOSS_HIDE_AFTER_FRAMES: u32 = 4;
const RELOCK_OVERLAP_THRESHOLD: f32 = 0.65;
/// Inflation around the viewport AABB when asking the session
/// "is this view already covered?". A small pad swallows the
/// per-frame tracker jitter without admitting genuinely-new
/// surface area as already-covered. Surface coords are anchor-
/// resolution, so 24 px = ~2-3% of typical viewport extent.
const COVERAGE_PAD_PX: f32 = 24.0;
const ENABLE_COLOR_MATTING: bool = false;
/// Worker threads the overlay warp fans out over. Matches the tracker
/// pool size; the two run sequentially so total live threads peak at 2.
const COMPOSITE_POOL_THREADS: usize = 2;

/// What kind of async job the per-frame fast path needs to dispatch.
#[derive(Clone)]
enum PendingJob {
    Acquire {
        frame: Arc<LiveFrame>,
        display_crop: Rect,
        config: PipelineConfig,
        timestamp_ns: u64,
        generation: u64,
    },
    Refresh {
        frame: Arc<LiveFrame>,
        display_crop: Rect,
        config: PipelineConfig,
        generation: u64,
    },
}

/// Single-thread async worker. Lives for the pipeline's lifetime;
/// drained via Condvar so we don't pay thread-spawn cost on every
/// acquire (~50 μs is negligible per acquire but the pattern is also
/// noisy in profiles).
struct Worker {
    inner: Arc<WorkerInner>,
}

struct WorkerInner {
    /// Single slot: at most one job is pending or in-flight at a time.
    /// Per-frame backpressure: if a new job arrives while a worker is
    /// busy, the new job is dropped (analogous to today's
    /// `acquireInFlight: AtomicBoolean` gate on the Kotlin side).
    slot: Mutex<WorkerState>,
    cv: Condvar,
}

struct WorkerState {
    pending: Option<PendingJob>,
    busy: bool,
    shutting_down: bool,
}

impl Worker {
    fn spawn(pipeline: Arc<LiveTrackerPipeline>) -> Self {
        let inner = Arc::new(WorkerInner {
            slot: Mutex::new(WorkerState {
                pending: None,
                busy: false,
                shutting_down: false,
            }),
            cv: Condvar::new(),
        });
        let inner_clone = Arc::clone(&inner);
        let pipeline_weak = Arc::downgrade(&pipeline);
        std::thread::Builder::new()
            .name("LiveTrackerPipelineWorker".into())
            .spawn(move || {
                loop {
                    let job = {
                        let mut state = inner_clone.slot.lock().expect("worker slot poisoned");
                        while state.pending.is_none() && !state.shutting_down {
                            state = inner_clone.cv.wait(state).expect("worker cv poisoned");
                        }
                        if state.shutting_down {
                            return;
                        }
                        let job = state.pending.take();
                        if job.is_some() {
                            state.busy = true;
                        }
                        job
                    };
                    let Some(job) = job else { continue };
                    let Some(pipeline) = pipeline_weak.upgrade() else {
                        return;
                    };
                    let telemetry = match job {
                        PendingJob::Acquire {
                            frame,
                            display_crop,
                            config,
                            timestamp_ns,
                            generation,
                        } => pipeline.run_acquire_inner(
                            &frame,
                            display_crop,
                            &config,
                            timestamp_ns,
                            generation,
                        ),
                        PendingJob::Refresh {
                            frame,
                            display_crop,
                            config,
                            generation,
                        } => pipeline.run_refresh_inner(&frame, display_crop, &config, generation),
                    };
                    if let Ok(mut slot) = pipeline.last_telemetry.lock() {
                        *slot = Some(telemetry);
                    }
                    let mut state = inner_clone.slot.lock().expect("worker slot poisoned");
                    state.busy = false;
                    inner_clone.cv.notify_all();
                }
            })
            .expect("failed to spawn LiveTrackerPipelineWorker");
        Worker { inner }
    }

    /// Try to dispatch a new job. Drops the request if a job is
    /// already in flight or already queued (the per-frame caller will
    /// try again on a later frame).
    fn try_dispatch(&self, job: PendingJob) -> bool {
        let mut state = self.inner.slot.lock().expect("worker slot poisoned");
        if state.busy || state.pending.is_some() {
            return false;
        }
        state.pending = Some(job);
        self.inner.cv.notify_one();
        true
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        if let Ok(mut state) = self.inner.slot.lock() {
            state.shutting_down = true;
            self.inner.cv.notify_all();
        }
    }
}

/// Rolling per-frame timing for `process_frame`. Reset every
/// `TIMING_WINDOW_FRAMES` calls; emits one log line per window so the
/// caller can spot regressions or pauses without per-frame chatter.
///
/// `tracker_ms_sum` is the engine's wall time; the `*_sub_sum` fields
/// break it down into the same sub-steps the engine reports via
/// `LivePlanarEngine::last_step_timings`. `composite_overlay_count_sum`
/// lets the consumer see overlay count alongside warp time.
struct TimingStats {
    window_count: u32,
    total_ms_sum: f64,
    tracker_ms_sum: f64,
    tracker_pyramid_ms_sum: f64,
    tracker_features_ms_sum: f64,
    tracker_track_ms_sum: f64,
    tracker_chain_refine_ms_sum: f64,
    composite_ms_sum: f64,
    composite_overlay_count_sum: u64,
    window_start: Instant,
}

impl Default for TimingStats {
    fn default() -> Self {
        Self {
            window_count: 0,
            total_ms_sum: 0.0,
            tracker_ms_sum: 0.0,
            tracker_pyramid_ms_sum: 0.0,
            tracker_features_ms_sum: 0.0,
            tracker_track_ms_sum: 0.0,
            tracker_chain_refine_ms_sum: 0.0,
            composite_ms_sum: 0.0,
            composite_overlay_count_sum: 0,
            window_start: Instant::now(),
        }
    }
}

const TIMING_WINDOW_FRAMES: u32 = 10;

/// The pipeline itself.
pub struct LiveTrackerPipeline {
    engine: Mutex<LivePlanarEngine>,
    session: Arc<LiveSession>,
    last_emitted_h: Mutex<LastEmittedH>,
    last_root_to_view: Mutex<Option<(u64, [f32; 9])>>,
    pending_refresh_target: Mutex<Option<(u64, [f32; 9])>>,
    pending_compose: Mutex<Option<(u64, [f32; 9])>>,
    matted_strips: Mutex<HashMap<u64, Vec<Option<MattedStrip>>>>,
    generation: AtomicU64,
    config: Mutex<PipelineConfig>,
    catalog: Arc<TranslatorSession>,
    font_provider: Arc<dyn FontProvider + Send + Sync>,
    last_telemetry: Mutex<Option<AcquireTelemetry>>,
    worker: Mutex<Option<Worker>>,
    timing: Mutex<TimingStats>,
    /// Fixed-size pool the per-frame overlay warp fans out over. Separate
    /// from the engine's tracker pool because composite runs after the
    /// tracker step (sequentially), so the two never contend; a dedicated
    /// pool keeps the warp off the global rayon pool the OCR worker uses.
    composite_pool: rayon::ThreadPool,
}

impl LiveTrackerPipeline {
    /// Construct a new pipeline. The internal worker thread is
    /// lazily spawned on the first frame so callers that build a
    /// pipeline but never feed frames don't pay for it.
    pub fn new(
        catalog: Arc<TranslatorSession>,
        font_provider: Arc<dyn FontProvider + Send + Sync>,
    ) -> Arc<Self> {
        let composite_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(COMPOSITE_POOL_THREADS)
            .thread_name(|i| format!("planar-comp-{i}"))
            .build()
            .expect("failed to build composite thread pool");
        let pipeline = Arc::new(Self {
            engine: Mutex::new(LivePlanarEngine::new(EngineConfig::default())),
            session: Arc::new(LiveSession::new()),
            last_emitted_h: Mutex::new(LastEmittedH::default()),
            last_root_to_view: Mutex::new(None),
            pending_refresh_target: Mutex::new(None),
            pending_compose: Mutex::new(None),
            matted_strips: Mutex::new(HashMap::new()),
            generation: AtomicU64::new(0),
            config: Mutex::new(PipelineConfig::default()),
            catalog,
            font_provider,
            last_telemetry: Mutex::new(None),
            worker: Mutex::new(None),
            timing: Mutex::new(TimingStats::default()),
            composite_pool,
        });
        let worker = Worker::spawn(Arc::clone(&pipeline));
        *pipeline.worker.lock().expect("worker slot poisoned") = Some(worker);
        pipeline
    }

    pub fn set_languages(&self, from: &str, to: &str, is_auto_source: bool) {
        if let Ok(mut cfg) = self.config.lock() {
            cfg.from_lang = from.to_string();
            cfg.to_lang = to.to_string();
            cfg.is_auto_source = is_auto_source;
        }
    }

    pub fn set_target_mode(&self, mode: TargetMode) {
        if let Ok(mut cfg) = self.config.lock() {
            cfg.target_mode = mode;
        }
    }

    /// Bump generation, clear engine state + smoothed H + session
    /// state. Any in-flight worker job will observe the new generation
    /// and bail at its next gen-check.
    pub fn reset(&self) {
        self.generation.fetch_add(1, Ordering::SeqCst);
        if let Ok(mut engine) = self.engine.lock() {
            engine.clear();
        }
        if let Ok(mut sm) = self.last_emitted_h.lock() {
            *sm = LastEmittedH::default();
        }
        if let Ok(mut slot) = self.last_root_to_view.lock() {
            *slot = None;
        }
        if let Ok(mut slot) = self.pending_refresh_target.lock() {
            *slot = None;
        }
        if let Ok(mut slot) = self.pending_compose.lock() {
            *slot = None;
        }
        self.session.clear();
    }

    /// Drop all resident overlay items. Compositor will draw a
    /// camera-only frame after this.
    pub fn clear_overlay(&self) {
        self.session.clear_overlays();
    }

    /// Pull (and clear) the most recent async-job telemetry. The
    /// per-frame fast path doesn't return this; callers poll it
    /// separately for the debug status pill.
    pub fn last_acquire_telemetry(&self) -> Option<AcquireTelemetry> {
        self.last_telemetry.lock().ok().and_then(|mut g| g.take())
    }

    /// One-shot per-frame entry. Locks the engine + frame state
    /// internally, runs the tracker step, writes the composited
    /// camera+overlay into `dst`, and (when needed) materializes the
    /// frame bytes + dispatches an async acquire/refresh job.
    ///
    /// `dst` is the output RGBA buffer (`dst_w * dst_h * 4` bytes).
    /// `visible_sensor_w/h` are the visible-region dims in sensor
    /// coords (typically equal to `dst_w/h` when the SurfaceView uses
    /// FILL_CENTER on the sensor frame).
    /// `full_view_w/h` are the *full-display* dims in sensor coords,
    /// used by the relock decision (which compares the current
    /// viewport against the anchor's lock viewport in full coords).
    #[allow(clippy::too_many_arguments)]
    pub fn process_frame(
        &self,
        frame: &Arc<LiveFrame>,
        display_crop: Rect,
        dst: &mut [u8],
        dst_w: u32,
        dst_h: u32,
        visible_sensor_w: u32,
        visible_sensor_h: u32,
        full_view_w: u32,
        full_view_h: u32,
        imu_stable: bool,
        timestamp_ns: u64,
    ) -> Result<ProcessFrameResult, TranslatorError> {
        let cfg = self.config.lock().map(|c| c.clone()).unwrap_or_default();
        // AF-scan or other "suppressed" intent: bump generation every
        // frame so any in-flight acquire bails at its next check; clear
        // engine + smoothed state so the stable-window restarts on
        // every frame and the overlay disappears.
        if matches!(cfg.target_mode, TargetMode::Suppressed) {
            self.generation.fetch_add(1, Ordering::SeqCst);
            if let Ok(mut engine) = self.engine.lock() {
                engine.clear();
            }
            if let Ok(mut sm) = self.last_emitted_h.lock() {
                *sm = LastEmittedH::default();
            }
        }

        let t_frame = Instant::now();
        let t_tracker = Instant::now();
        let (tracker_state, tracker_anchor, tracker_inliers, tracker_h, step_timings) =
            self.step_tracker(frame, cfg.det_max_pixels, imu_stable, timestamp_ns)?;
        let tracker_ms = t_tracker.elapsed().as_secs_f64() * 1000.0;
        let frame_state_dims = {
            let state = frame.state().lock().map_err(|_| poisoned())?;
            (state.width, state.height, state.rotation_degrees)
        };

        let h_for_compose = self.select_compose_h(tracker_state, tracker_anchor, tracker_h);
        if let Ok(mut slot) = self.pending_compose.lock() {
            *slot = h_for_compose.map(|h| (tracker_anchor, h));
        }

        let t_composite = Instant::now();
        let (composite_bytes, overlay_count) = match self.composite_into_slice(
            frame,
            dst,
            visible_sensor_w,
            visible_sensor_h,
            h_for_compose,
            tracker_anchor,
        ) {
            Ok(n) => (dst_w.saturating_mul(dst_h).saturating_mul(4), n),
            Err(e) => {
                log::warn!("composite failed: {e:?}");
                (0, 0)
            }
        };
        let composite_ms = t_composite.elapsed().as_secs_f64() * 1000.0;

        // Detect-on-tracking refresh trigger.
        let should_refresh = matches!(tracker_state, PlanarTrackerState::Locked)
            && self.update_refresh_trigger(
                tracker_anchor,
                tracker_h,
                frame_state_dims,
                full_view_w,
                full_view_h,
            );

        // Mode decision: Acquiring → maybe spawn acquire. Locked +
        // should_refresh → maybe spawn refresh. Both gated by worker
        // backpressure (only one in-flight at a time).
        let (started_acquire, started_refresh) =
            if matches!(cfg.target_mode, TargetMode::Suppressed) {
                (false, false)
            } else {
                self.maybe_dispatch_async(
                    frame,
                    display_crop,
                    &cfg,
                    timestamp_ns,
                    tracker_state,
                    should_refresh,
                )
            };

        // Materialize bytes if any async work was dispatched, else
        // drop the borrow. Cheap (~3 ms memcpy at 1.2 MP) and avoided
        // entirely on the common pure-tracking frame.
        {
            let mut state = frame.state().lock().map_err(|_| poisoned())?;
            if started_acquire || started_refresh {
                state.materialize_owned();
            } else {
                state.clear_external();
            }
        }

        let total_ms = t_frame.elapsed().as_secs_f64() * 1000.0;
        if let Ok(mut t) = self.timing.lock() {
            t.window_count += 1;
            t.total_ms_sum += total_ms;
            t.tracker_ms_sum += tracker_ms;
            t.tracker_pyramid_ms_sum += step_timings.pyramid_ms;
            t.tracker_features_ms_sum += step_timings.features_ms;
            t.tracker_track_ms_sum += step_timings.track_ms;
            t.tracker_chain_refine_ms_sum += step_timings.chain_refine_ms;
            t.composite_ms_sum += composite_ms;
            t.composite_overlay_count_sum += overlay_count as u64;
            if t.window_count >= TIMING_WINDOW_FRAMES {
                let n = t.window_count as f64;
                let wall_s = t.window_start.elapsed().as_secs_f64();
                let fps = if wall_s > 1e-6 { n / wall_s } else { 0.0 };
                log::info!(
                    "[lt] {} frames fps={:.1} total={:.1}ms tracker={:.1}ms (pyr={:.1} feat={:.1} match={:.1} chain={:.1}) composite={:.1}ms (overlays={:.1})",
                    t.window_count,
                    fps,
                    t.total_ms_sum / n,
                    t.tracker_ms_sum / n,
                    t.tracker_pyramid_ms_sum / n,
                    t.tracker_features_ms_sum / n,
                    t.tracker_track_ms_sum / n,
                    t.tracker_chain_refine_ms_sum / n,
                    t.composite_ms_sum / n,
                    t.composite_overlay_count_sum as f64 / n,
                );
                *t = TimingStats::default();
            }
        }

        Ok(ProcessFrameResult {
            state: tracker_state,
            anchor_id: tracker_anchor,
            inliers: tracker_inliers,
            composite_bytes,
            started_acquire,
            started_refresh,
        })
    }

    // ---- internal helpers ----

    fn step_tracker(
        &self,
        frame: &Arc<LiveFrame>,
        det_max_pixels: u32,
        imu_stable: bool,
        timestamp_ns: u64,
    ) -> Result<
        (
            PlanarTrackerState,
            u64,
            u32,
            Option<[f32; 9]>,
            crate::planar_engine::StepTimings,
        ),
        TranslatorError,
    > {
        let mut state = frame.state().lock().map_err(|_| poisoned())?;
        state.ensure_tracker_oriented(det_max_pixels)?;
        let oriented = state
            .cached_tracker
            .as_ref()
            .expect("ensure_tracker filled cache");
        let det_to_full = oriented.det_to_full_scale;
        let mut engine = self.engine.lock().map_err(|_| poisoned())?;
        let cmd = engine.process_frame(&oriented.gray, imu_stable, timestamp_ns);
        let step_timings = engine.last_step_timings();
        drop(engine);
        drop(state);
        let (state_kind, anchor_id, h, inliers) = match cmd {
            TrackerCommand::Idle => (PlanarTrackerState::Idle, 0u64, None, 0u32),
            TrackerCommand::Acquiring => (PlanarTrackerState::Acquiring, 0, None, 0),
            TrackerCommand::Locked {
                anchor_id,
                homography,
                inliers,
                ..
            } => {
                let h = if det_to_full != 1.0 {
                    scale_homography(&homography, det_to_full)
                } else {
                    homography
                };
                (
                    PlanarTrackerState::Locked,
                    anchor_id,
                    Some(h),
                    inliers as u32,
                )
            }
            TrackerCommand::Lost { last_anchor_id } => {
                (PlanarTrackerState::Lost, last_anchor_id, None, 0)
            }
        };
        Ok((state_kind, anchor_id, inliers, h, step_timings))
    }

    fn select_compose_h(
        &self,
        state: PlanarTrackerState,
        anchor_id: u64,
        incoming: Option<[f32; 9]>,
    ) -> Option<[f32; 9]> {
        let mut sm = self.last_emitted_h.lock().ok()?;
        match state {
            PlanarTrackerState::Locked => {
                sm.consecutive_lost = 0;
                let incoming = incoming?;
                sm.h = Some(incoming);
                sm.anchor_id = anchor_id;
                Some(incoming)
            }
            PlanarTrackerState::Lost => {
                sm.consecutive_lost = sm.consecutive_lost.saturating_add(1);
                if sm.consecutive_lost < LOSS_HIDE_AFTER_FRAMES {
                    sm.h
                } else {
                    None
                }
            }
            PlanarTrackerState::Idle | PlanarTrackerState::Acquiring => {
                sm.consecutive_lost = 0;
                None
            }
        }
    }

    fn composite_into_slice(
        &self,
        frame: &Arc<LiveFrame>,
        dst: &mut [u8],
        bitmap_w: u32,
        bitmap_h: u32,
        h_surface_to_viewport: Option<[f32; 9]>,
        active_anchor_id: u64,
    ) -> Result<u32, CompositeError> {
        let state = frame.state().lock().expect("frame mutex poisoned");
        let sensor_w = state.width;
        let sensor_h = state.height;
        let src_offset_x = sensor_w.saturating_sub(bitmap_w) / 2;
        let src_offset_y = sensor_h.saturating_sub(bitmap_h) / 2;
        // One bitmap per anchor (bg + glyphs already composed in
        // surface space at canvas-build time). The compositor does a
        // single bilinear warp per item; cross-block bg overlap is
        // resolved in the canvas, not per frame.
        let overlay_guard = self.session.overlay_anchors.lock().ok();
        let items_vec: Vec<OverlayItem<'_>> = match (&overlay_guard, h_surface_to_viewport) {
            (Some(anchors), Some(_)) => anchors
                .get(&active_anchor_id)
                .and_then(|a| a.canvas.as_ref())
                .map(|c| {
                    vec![OverlayItem {
                        bitmap_rgba: &c.bitmap,
                        bitmap_width: c.width,
                        bitmap_height: c.height,
                        bitmap_origin_surface_x: c.surface_origin_x,
                        bitmap_origin_surface_y: c.surface_origin_y,
                        row_extents: &c.row_extents,
                    }]
                })
                .unwrap_or_default(),
            _ => Vec::new(),
        };
        let h_for_call =
            h_surface_to_viewport.unwrap_or([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let translate = [
            1.0,
            0.0,
            -(src_offset_x as f32),
            0.0,
            1.0,
            -(src_offset_y as f32),
            0.0,
            0.0,
            1.0,
        ];
        let h_translated = homography::mat3_mul(&translate, &h_for_call);
        let overlay_count = items_vec.len() as u32;
        let camera = state.rgba_bytes();
        let input = live_compositor::CompositeInput {
            dst_w: bitmap_w,
            dst_h: bitmap_h,
            camera_rgba: camera,
            src_full_w: sensor_w,
            src_full_h: sensor_h,
            src_offset_x,
            src_offset_y,
            h_surface_to_viewport: &h_translated,
            items: &items_vec,
        };
        self.composite_pool
            .install(|| {
                let mut renderer = live_compositor::CpuRenderer;
                renderer.composite(&input, dst)
            })
            .map(|()| overlay_count)
    }

    fn update_refresh_trigger(
        &self,
        anchor_id: u64,
        tracker_h: Option<[f32; 9]>,
        frame_state_dims: (u32, u32, i32),
        _full_view_w_hint: u32,
        _full_view_h_hint: u32,
    ) -> bool {
        let Some(h) = tracker_h else { return false };
        if let Ok(mut slot) = self.last_root_to_view.lock() {
            *slot = Some((anchor_id, h));
        }
        if !self.session.has_last_lock_h(anchor_id) {
            self.session.set_last_lock_h(anchor_id, h);
            return false;
        }
        let r = ((frame_state_dims.2 % 360) + 360) % 360;
        let (full_view_w, full_view_h) = if r == 90 || r == 270 {
            (frame_state_dims.1 as f32, frame_state_dims.0 as f32)
        } else {
            (frame_state_dims.0 as f32, frame_state_dims.1 as f32)
        };
        if self.session.should_relock_by_view(
            anchor_id,
            &h,
            full_view_w,
            full_view_h,
            RELOCK_OVERLAP_THRESHOLD,
        ) {
            // Quadrilateral area changed enough to *consider* a
            // refresh, but a pure tilt / perspective change can flip
            // this trigger without actually revealing new content.
            // Gate on coverage: if the new viewport AABB is still
            // inside the surface area we've already detected text in,
            // every visible line is already in the surface map and a
            // detect+OCR pass would be wasted work — at best a no-op,
            // at worst (if tracker drift exceeds the merge tolerance)
            // it produces stacked duplicate overlays.
            let Some(h_view_to_surface) = h_view_to_surface_from(&h) else {
                return false;
            };
            let viewport = match viewport_surface_aabb(&h_view_to_surface, full_view_w, full_view_h)
            {
                Some(v) => v,
                None => return false,
            };
            if self
                .session
                .viewport_contained_in_coverage(anchor_id, &viewport, COVERAGE_PAD_PX)
            {
                return false;
            }
            log::debug!(
                "[refresh] firing: viewport AABB extends beyond covered surface region (anchor={anchor_id})",
            );
            self.session.clear_last_lock_h(anchor_id);
            if let Ok(mut slot) = self.pending_refresh_target.lock() {
                *slot = Some((anchor_id, h));
            }
            true
        } else {
            false
        }
    }

    fn maybe_dispatch_async(
        &self,
        frame: &Arc<LiveFrame>,
        display_crop: Rect,
        cfg: &PipelineConfig,
        timestamp_ns: u64,
        state: PlanarTrackerState,
        should_refresh: bool,
    ) -> (bool, bool) {
        let worker_guard = self.worker.lock().ok();
        let Some(worker) = worker_guard.as_ref().and_then(|w| w.as_ref()) else {
            return (false, false);
        };
        let current_gen = self.generation.load(Ordering::SeqCst);
        match state {
            PlanarTrackerState::Acquiring => {
                let started = worker.try_dispatch(PendingJob::Acquire {
                    frame: Arc::clone(frame),
                    display_crop,
                    config: cfg.clone(),
                    timestamp_ns,
                    generation: current_gen,
                });
                (started, false)
            }
            PlanarTrackerState::Locked if should_refresh => {
                let started = worker.try_dispatch(PendingJob::Refresh {
                    frame: Arc::clone(frame),
                    display_crop,
                    config: cfg.clone(),
                    generation: current_gen,
                });
                (false, started)
            }
            _ => (false, false),
        }
    }

    // ---- async stages (run on the worker thread) ----

    fn run_acquire_inner(
        &self,
        frame: &Arc<LiveFrame>,
        display_crop: Rect,
        cfg: &PipelineConfig,
        timestamp_ns: u64,
        generation: u64,
    ) -> AcquireTelemetry {
        let gen_check = || self.generation.load(Ordering::SeqCst) == generation;
        if !gen_check() {
            return AcquireTelemetry {
                canceled: true,
                ..Default::default()
            };
        }
        let t_overall = Instant::now();

        // Detect.
        let detected: Vec<DetectedTextBox> = {
            let mut state = match frame.state().lock() {
                Ok(s) => s,
                Err(_) => {
                    return AcquireTelemetry {
                        error: Some("frame.state poisoned".into()),
                        ..Default::default()
                    };
                }
            };
            if state
                .ensure_oriented_with_rgb(display_crop, cfg.det_max_pixels)
                .is_err()
            {
                return AcquireTelemetry {
                    error: Some("ensure_oriented failed".into()),
                    ..Default::default()
                };
            }
            let oriented = state.cached.as_ref().expect("ensure_oriented filled cache");
            let raw = match self.catalog.detect_text_in_oriented_image(oriented) {
                Ok(r) => r,
                Err(e) => {
                    log::warn!("detect failed: {e:?}");
                    return AcquireTelemetry {
                        error: Some("detect failed".into()),
                        ..Default::default()
                    };
                }
            };
            let scale = oriented.det_to_full_scale;
            let rgb = oriented.rgb.as_ref().expect("with_rgb path");
            let max_w = rgb.width();
            let max_h = rgb.height();
            raw.into_iter()
                .map(|b| scale_detected_box(b, scale, max_w, max_h))
                .collect()
        };

        if !gen_check() {
            return AcquireTelemetry {
                canceled: true,
                ..Default::default()
            };
        }
        if detected.is_empty() {
            return AcquireTelemetry {
                total_ms: t_overall.elapsed().as_secs_f64() * 1000.0,
                ..Default::default()
            };
        }

        // Orientation estimate.
        let forced_script = if cfg.is_auto_source {
            None
        } else {
            self.catalog.ppocr_script_for_language_code(&cfg.from_lang)
        };
        let estimated_quadrant: Option<Quadrant> = {
            let state = match frame.state().lock() {
                Ok(s) => s,
                Err(_) => {
                    return AcquireTelemetry {
                        error: Some("frame.state poisoned".into()),
                        ..Default::default()
                    };
                }
            };
            let oriented = match state.cached.as_ref() {
                Some(o) => o,
                None => {
                    return AcquireTelemetry {
                        error: Some("oriented cache miss".into()),
                        ..Default::default()
                    };
                }
            };
            if let Some(script) = forced_script {
                self.catalog
                    .estimate_canonical_via_rec_in_oriented_image(oriented, &detected, script)
                    .unwrap_or(None)
            } else {
                self.catalog
                    .estimate_canonical_quadrant_in_oriented_image(oriented, &detected)
                    .unwrap_or(None)
            }
        };

        // Acquire anchor.
        let anchor_id = {
            let mut state = match frame.state().lock() {
                Ok(s) => s,
                Err(_) => {
                    return AcquireTelemetry {
                        error: Some("frame.state poisoned".into()),
                        ..Default::default()
                    };
                }
            };
            if state.ensure_tracker_oriented(cfg.det_max_pixels).is_err() {
                return AcquireTelemetry {
                    error: Some("ensure_tracker failed".into()),
                    ..Default::default()
                };
            }
            let tracker_oriented = state
                .cached_tracker
                .as_ref()
                .expect("ensure_tracker filled cache");
            let cached_sensor_crop =
                state
                    .cached
                    .as_ref()
                    .map(|oi| oi.sensor_crop)
                    .unwrap_or(Rect {
                        left: 0,
                        top: 0,
                        right: 0,
                        bottom: 0,
                    });
            let scale_down = if tracker_oriented.det_to_full_scale > 0.0 {
                1.0 / tracker_oriented.det_to_full_scale
            } else {
                1.0
            };
            let regions: Vec<(u32, u32, u32, u32)> = detected
                .iter()
                .map(|d| {
                    let scale_u32 = |v: u32| ((v as f32) * scale_down).round() as u32;
                    (
                        scale_u32(d.rect.left + cached_sensor_crop.left),
                        scale_u32(d.rect.top + cached_sensor_crop.top),
                        scale_u32(d.rect.right + cached_sensor_crop.left),
                        scale_u32(d.rect.bottom + cached_sensor_crop.top),
                    )
                })
                .collect();
            let mut engine = match self.engine.lock() {
                Ok(e) => e,
                Err(_) => {
                    return AcquireTelemetry {
                        error: Some("engine poisoned".into()),
                        ..Default::default()
                    };
                }
            };
            engine
                .acquire_now_with_orientation(
                    &tracker_oriented.gray,
                    &regions,
                    cfg.anchor_padding_px,
                    timestamp_ns,
                    estimated_quadrant,
                )
                .unwrap_or(0)
        };
        if anchor_id == 0 {
            return AcquireTelemetry {
                error: Some("acquire_now returned 0".into()),
                ..Default::default()
            };
        }
        self.session.reset_anchor_state(anchor_id);
        if !gen_check() {
            return AcquireTelemetry {
                canceled: true,
                ..Default::default()
            };
        }

        // Color matting (disabled by default).
        if ENABLE_COLOR_MATTING {
            let matted: Vec<Option<MattedStrip>> = {
                let state = match frame.state().lock() {
                    Ok(s) => s,
                    Err(_) => {
                        return AcquireTelemetry {
                            error: Some("frame.state poisoned".into()),
                            ..Default::default()
                        };
                    }
                };
                let oriented = state.cached.as_ref().expect("oriented still cached");
                crate::color_matting::mat_detections(
                    &oriented.rgb.as_ref().expect("with_rgb path").to_rgba8(),
                    &detected,
                )
            };
            if let Ok(mut store) = self.matted_strips.lock() {
                store.insert(anchor_id, matted);
            }
        }

        let total = detected.len();
        let available_codes: Vec<LanguageCode> = self
            .catalog
            .language_rows()
            .into_iter()
            .map(|row| LanguageCode::from(row.language.code.as_str()))
            .collect();
        let matted_strips: Vec<Option<MattedStrip>> = match self.matted_strips.lock() {
            Ok(g) => g.get(&anchor_id).cloned().unwrap_or_default(),
            Err(_) => Vec::new(),
        };

        let cancel = || self.generation.load(Ordering::SeqCst) != generation;
        let session_ref: &TranslatorSession = &self.catalog;
        let outcome = {
            let state = match frame.state().lock() {
                Ok(s) => s,
                Err(_) => {
                    return AcquireTelemetry {
                        error: Some("frame.state poisoned".into()),
                        ..Default::default()
                    };
                }
            };
            let oriented = match state
                .cached
                .as_ref()
                .filter(|oi| oi.display_crop == display_crop)
            {
                Some(o) => o,
                None => {
                    return AcquireTelemetry {
                        error: Some("oriented cache miss".into()),
                        ..Default::default()
                    };
                }
            };
            let h_view_to_sensor = [
                1.0,
                0.0,
                oriented.sensor_crop.left as f32,
                0.0,
                1.0,
                oriented.sensor_crop.top as f32,
                0.0,
                0.0,
                1.0,
            ];
            let canonical_quadrant = self
                .engine
                .lock()
                .ok()
                .and_then(|e| e.canonical_rotation_for(anchor_id));
            let outcome = self.session.run_post_detect(
                PostDetectInput {
                    detections: &detected,
                    oriented,
                    h_view_to_surface: Some(h_view_to_sensor),
                    anchor_id,
                    from_lang: &cfg.from_lang,
                    to_lang: &cfg.to_lang,
                    is_auto_source: cfg.is_auto_source,
                    available_codes: &available_codes,
                    font_provider: &*self.font_provider,
                    matted_strips: &matted_strips,
                    rec_batch_size: cfg.rec_batch_size,
                    canonical_quadrant,
                },
                &session_ref,
                &session_ref,
                &cancel,
            );
            drop(state);
            outcome
        };
        self.session.on_acquire();

        if outcome.canceled {
            return AcquireTelemetry {
                canceled: true,
                anchor_id,
                ..Default::default()
            };
        }
        let rec_ok = outcome.rec_ok_count as usize;
        let rec_empty = outcome.rec_empty_count as usize;
        if rec_ok == 0 && rec_empty + rec_ok == total {
            if let Ok(mut engine) = self.engine.lock() {
                engine.clear();
            }
            self.session.clear_overlays();
        }
        if let Ok(engine) = self.engine.lock() {
            let keep = engine.cached_root_ids();
            drop(engine);
            self.session.retain_anchors(&keep);
        }

        AcquireTelemetry {
            anchor_id,
            detected_count: total as u32,
            rec_ok_count: rec_ok as u32,
            rec_empty_count: rec_empty as u32,
            cache_hits: outcome.cache_hits,
            rec_called_count: outcome.rec_called_count,
            total_ms: t_overall.elapsed().as_secs_f64() * 1000.0,
            canceled: false,
            error: None,
            is_refresh: false,
        }
    }

    fn run_refresh_inner(
        &self,
        frame: &Arc<LiveFrame>,
        display_crop: Rect,
        cfg: &PipelineConfig,
        generation: u64,
    ) -> AcquireTelemetry {
        let gen_check = || self.generation.load(Ordering::SeqCst) == generation;
        if !gen_check() {
            return AcquireTelemetry {
                canceled: true,
                is_refresh: true,
                ..Default::default()
            };
        }
        let t_overall = Instant::now();
        let (anchor_id, h_root_to_view) = match self.pending_refresh_target.lock() {
            Ok(mut g) => match g.take() {
                Some((id, h)) => (id, h),
                None => {
                    return AcquireTelemetry {
                        error: Some("refresh without armed trigger".into()),
                        is_refresh: true,
                        ..Default::default()
                    };
                }
            },
            Err(_) => {
                return AcquireTelemetry {
                    error: Some("pending_refresh_target poisoned".into()),
                    is_refresh: true,
                    ..Default::default()
                };
            }
        };
        let h_sensor_view_to_surface = match homography::invert(&h_root_to_view) {
            Some(h) => h,
            None => {
                return AcquireTelemetry {
                    error: Some("H_root→view not invertible".into()),
                    is_refresh: true,
                    ..Default::default()
                };
            }
        };

        let detected: Vec<DetectedTextBox> = {
            let mut state = match frame.state().lock() {
                Ok(s) => s,
                Err(_) => {
                    return AcquireTelemetry {
                        error: Some("frame.state poisoned".into()),
                        is_refresh: true,
                        ..Default::default()
                    };
                }
            };
            if state
                .ensure_oriented_with_rgb(display_crop, cfg.det_max_pixels)
                .is_err()
            {
                return AcquireTelemetry {
                    error: Some("ensure_oriented failed".into()),
                    is_refresh: true,
                    ..Default::default()
                };
            }
            let oriented = state.cached.as_ref().expect("ensure_oriented filled cache");
            let raw = match self.catalog.detect_text_in_oriented_image(oriented) {
                Ok(r) => r,
                Err(e) => {
                    log::warn!("[refresh] detect failed: {e:?}");
                    return AcquireTelemetry {
                        error: Some("detect failed".into()),
                        is_refresh: true,
                        ..Default::default()
                    };
                }
            };
            let scale = oriented.det_to_full_scale;
            let rgb = oriented.rgb.as_ref().expect("with_rgb path");
            let max_w = rgb.width();
            let max_h = rgb.height();
            raw.into_iter()
                .map(|b| scale_detected_box(b, scale, max_w, max_h))
                .collect()
        };
        if !gen_check() {
            return AcquireTelemetry {
                canceled: true,
                is_refresh: true,
                ..Default::default()
            };
        }
        if detected.is_empty() {
            return AcquireTelemetry {
                anchor_id,
                total_ms: t_overall.elapsed().as_secs_f64() * 1000.0,
                is_refresh: true,
                ..Default::default()
            };
        }

        let available_codes: Vec<LanguageCode> = self
            .catalog
            .language_rows()
            .into_iter()
            .map(|row| LanguageCode::from(row.language.code.as_str()))
            .collect();

        let state = match frame.state().lock() {
            Ok(s) => s,
            Err(_) => {
                return AcquireTelemetry {
                    error: Some("frame.state poisoned".into()),
                    is_refresh: true,
                    ..Default::default()
                };
            }
        };
        let oriented = match state
            .cached
            .as_ref()
            .filter(|oi| oi.display_crop == display_crop)
        {
            Some(o) => o,
            None => {
                return AcquireTelemetry {
                    error: Some("oriented cache miss".into()),
                    is_refresh: true,
                    ..Default::default()
                };
            }
        };
        let cancel = || self.generation.load(Ordering::SeqCst) != generation;
        let h_view_to_sensor = [
            1.0,
            0.0,
            oriented.sensor_crop.left as f32,
            0.0,
            1.0,
            oriented.sensor_crop.top as f32,
            0.0,
            0.0,
            1.0,
        ];
        let h_view_to_surface_composed =
            homography::mat3_mul(&h_sensor_view_to_surface, &h_view_to_sensor);
        let session_ref: &TranslatorSession = &self.catalog;
        self.session.clear_anchor_state_for_relock(anchor_id);
        let canonical_quadrant = self
            .engine
            .lock()
            .ok()
            .and_then(|e| e.canonical_rotation_for(anchor_id));
        let outcome = self.session.run_post_detect(
            PostDetectInput {
                detections: &detected,
                oriented,
                h_view_to_surface: Some(h_view_to_surface_composed),
                anchor_id,
                from_lang: &cfg.from_lang,
                to_lang: &cfg.to_lang,
                is_auto_source: cfg.is_auto_source,
                available_codes: &available_codes,
                font_provider: &*self.font_provider,
                matted_strips: &[],
                rec_batch_size: cfg.rec_batch_size,
                canonical_quadrant,
            },
            &session_ref,
            &session_ref,
            &cancel,
        );
        drop(state);

        if outcome.canceled {
            return AcquireTelemetry {
                canceled: true,
                anchor_id,
                is_refresh: true,
                ..Default::default()
            };
        }
        AcquireTelemetry {
            anchor_id,
            detected_count: outcome.detected_count,
            rec_ok_count: outcome.rec_ok_count,
            rec_empty_count: outcome.rec_empty_count,
            cache_hits: outcome.cache_hits,
            rec_called_count: outcome.rec_called_count,
            total_ms: t_overall.elapsed().as_secs_f64() * 1000.0,
            canceled: false,
            error: None,
            is_refresh: true,
        }
    }
}

fn poisoned() -> TranslatorError {
    TranslatorError::new(
        crate::api::TranslatorErrorKind::Internal,
        "live pipeline mutex poisoned",
    )
}

/// Conjugate a homography `H` by an isotropic scale `s` so an H built
/// in downsampled coords maps the equivalent points in full coords.
fn scale_homography(h: &[f32; 9], s: f32) -> [f32; 9] {
    let inv_s = 1.0 / s;
    [
        h[0],
        h[1],
        h[2] * s,
        h[3],
        h[4],
        h[5] * s,
        h[6] * inv_s,
        h[7] * inv_s,
        h[8],
    ]
}

/// Scale a `DetectedTextBox` from detector-image coords up to
/// full-crop coords, clamping inside the destination dimensions.
fn scale_detected_box(b: DetectedTextBox, scale: f32, max_w: u32, max_h: u32) -> DetectedTextBox {
    let left = ((b.rect.left as f32) * scale).max(0.0) as u32;
    let top = ((b.rect.top as f32) * scale).max(0.0) as u32;
    let right = ((b.rect.right as f32) * scale).min(max_w as f32) as u32;
    let bottom = ((b.rect.bottom as f32) * scale).min(max_h as f32) as u32;
    let rect = Rect {
        left: left.min(right.saturating_sub(1)),
        top: top.min(bottom.saturating_sub(1)),
        right: right.max(left + 1),
        bottom: bottom.max(top + 1),
    };
    let oriented = OrientedRect {
        cx: b.oriented_box.cx * scale,
        cy: b.oriented_box.cy * scale,
        width: b.oriented_box.width * scale,
        height: b.oriented_box.height * scale,
        angle_radians: b.oriented_box.angle_radians,
    };
    let tight = OrientedRect {
        cx: b.tight_box.cx * scale,
        cy: b.tight_box.cy * scale,
        width: b.tight_box.width * scale,
        height: b.tight_box.height * scale,
        angle_radians: b.tight_box.angle_radians,
    };
    let mut contour = Vec::with_capacity(b.contour.len());
    for v in &b.contour {
        contour.push(v * scale);
    }
    DetectedTextBox {
        rect,
        oriented_box: oriented,
        tight_box: tight,
        contour,
        score: b.score,
    }
}
