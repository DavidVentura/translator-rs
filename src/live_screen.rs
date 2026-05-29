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
use crate::ocr::Rect;
use crate::session::TranslatorSession;

/// The one anchor the screen pipeline owns; everything composites against it.
const SCREEN_ANCHOR_ID: u64 = 1;

const IDENTITY: [f32; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

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

pub struct LiveScreenPipeline {
    catalog: Arc<TranslatorSession>,
    session: Arc<LiveSession>,
    font_provider: Arc<dyn FontProvider + Send + Sync>,
    config: Mutex<ScreenConfig>,
    /// Bumped on reset / language change so an in-flight rec/translate bails.
    generation: AtomicU64,
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
        })
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
    }

    pub fn set_overlay_oversample(&self, factor: f32) {
        self.session.set_overlay_oversample(factor);
    }

    pub fn clear_overlay(&self) {
        self.session.clear_overlays();
    }

    pub fn reset(&self) {
        self.generation.fetch_add(1, Ordering::SeqCst);
        self.session.clear();
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
