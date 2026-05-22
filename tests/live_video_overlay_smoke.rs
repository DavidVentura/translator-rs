//! Opt-in integration smoke for the live camera overlay path.
//!
//! Takes an mp4 in and emits an mp4 of the composited overlay. Decoding
//! and encoding are done by shelling out to the `ffmpeg` binary (must be
//! on PATH).
//!
//! Input:
//!   - `LIVE_VIDEO_MP4_FILE=/path/to/clip.mp4`
//!
//! Each decoded frame is fed through:
//! frame build -> planar tracker -> PPOCR detect/recognize -> LiveSession overlay
//! upsert -> live compositor.
//!
//! Outputs:
//!   - `$LIVE_VIDEO_DUMP_DIR/composite.mp4` (camera + overlay, H.264)
//!   - `$LIVE_VIDEO_DUMP_DIR/summary.jsonl`
//!
//! Example:
//!   LIVE_VIDEO_MP4_FILE=/tmp/clip.mp4 \
//!   LIVE_VIDEO_DUMP_DIR=smoke-out/live-video-overlay \
//!   cargo test --release --features ppocr,planar-tracker --test live_video_overlay_smoke -- --nocapture

#![cfg(all(feature = "ppocr", feature = "planar-tracker"))]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};

use ab_glyph::{FontArc, PxScale};
use image::{Rgba, RgbaImage, codecs::jpeg::JpegEncoder};
use imageproc::drawing::{draw_filled_rect_mut, draw_text_mut};
use imageproc::rect::Rect as ImpRect;
use translator::font_provider::{FontHandle, FontProvider, FontRequest};
use translator::live_compositor::{OverlayItem as CompositeOverlayItem, composite_frame_into};
use translator::live_frame::OrientedImage;
use translator::live_session::{
    LiveRecognizer, LiveSession, NoopTranslator, PostDetectInput, h_view_to_surface_from,
    viewport_surface_aabb,
};
use translator::ocr::{
    DetectedTextBox, OcrSourceSelection, OrientedRect, RecognizedTextLine, Rect,
};
use translator::planar_engine::{EngineConfig, LivePlanarEngine, TrackerCommand};
use translator::ppocr::{PpocrEngine, PpocrProfile, PpocrRecognizerSpec};
use translator::{LanguageCode, PpocrScript};

const DET_MAX_PIXELS_DEFAULT: u32 = 650_000;
const MODEL_DIR_DEFAULT: &str = "/home/david/AndroidStudioProjects/bucket/ocr/1/PP-OCRv5";
const IDENTITY_H: [f32; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

#[test]
fn video_frames_drive_live_overlay_pipeline() {
    let _ = env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("translator=info,planar_timing=info"),
    )
    .is_test(false)
    .try_init();

    let Some(input) = input_frames() else {
        eprintln!("[live_video_overlay_smoke] skipping: set LIVE_VIDEO_MP4_FILE");
        return;
    };
    let Some(model) = ppocr_model_paths() else {
        eprintln!("[live_video_overlay_smoke] skipping: PPOCR model files missing");
        return;
    };

    let dump_dir = env_path("LIVE_VIDEO_DUMP_DIR")
        .unwrap_or_else(|| PathBuf::from("smoke-out/live-video-overlay"));
    fs::create_dir_all(&dump_dir).expect("create dump dir");
    let output_fps = env_u32("LIVE_VIDEO_OUTPUT_FPS").unwrap_or(30);
    let jpeg_quality = env_u8("LIVE_VIDEO_MJPEG_QUALITY").unwrap_or(90);
    let skip_mp4 = env_bool("LIVE_VIDEO_SKIP_MP4").unwrap_or(false);
    let mut composite_writer: Option<Mp4Writer> = if skip_mp4 {
        eprintln!("[live_video_overlay_smoke] SKIP_MP4 set: composite.mp4 will not be written");
        None
    } else {
        Some(Mp4Writer::new(
            &dump_dir.join("composite.mp4"),
            output_fps,
            jpeg_quality,
        ))
    };

    let ppocr = PpocrEngine::load(
        &model.det,
        None,
        None,
        vec![PpocrRecognizerSpec {
            script: model.script,
            model_path: model.rec,
            keys_path: model.keys,
        }],
        env_usize("LIVE_VIDEO_DET_THREADS").unwrap_or(2),
    )
    .expect("load ppocr");
    let recognizer = PpocrLiveRecognizer {
        engine: &ppocr,
        script: model.script,
    };
    let translator = NoopTranslator;
    let font_provider = HostFonts::new().unwrap_or(HostFonts::none());

    let from_lang = env_string("LIVE_VIDEO_SOURCE_LANG").unwrap_or_else(|| "en".to_string());
    let to_lang = env_string("LIVE_VIDEO_TARGET_LANG").unwrap_or_else(|| from_lang.clone());
    let available_codes = vec![LanguageCode::from(from_lang.as_str())];
    let refresh_every_n = env_u32("LIVE_VIDEO_REFRESH_EVERY_N")
        .or_else(|| env_u32("LIVE_VIDEO_DETECT_EVERY_N"))
        .unwrap_or(12)
        .max(1);
    let coverage_pad_px = env_f32("LIVE_VIDEO_COVERAGE_PAD_PX").unwrap_or(24.0);
    let relock_area_ratio = env_f32("LIVE_VIDEO_RELOCK_AREA_RATIO").unwrap_or(0.75);
    let det_max_pixels = env_u32("LIVE_VIDEO_DET_MAX_PIXELS").unwrap_or(DET_MAX_PIXELS_DEFAULT);
    let max_frames = env_usize("LIVE_VIDEO_MAX_FRAMES");
    let skip_ocr = env_bool("LIVE_VIDEO_SKIP_OCR").unwrap_or(false);
    let mut ocr_fired = false;
    let overlay_font = load_overlay_font();
    if skip_ocr {
        eprintln!(
            "[live_video_overlay_smoke] SKIP_OCR mode: OCR fires once on first Acquire, then tracker-only"
        );
    }

    let mut cfg = EngineConfig::default();
    cfg.stable_required_ns = 0;
    let mut engine = LivePlanarEngine::new(cfg);
    let session = LiveSession::new();
    session.set_refresh_every_n_locked_frames(refresh_every_n);

    let mut active_anchor: Option<u64> = None;
    let mut h_surface_to_view = IDENTITY_H;
    let mut summaries = Vec::new();
    let mut frames_with_overlay = 0usize;
    let mut total_detected = 0u64;
    let mut total_rec_ok = 0u64;

    let frames = input.frames;
    let frame_count = max_frames
        .map(|n| n.min(frames.len()))
        .unwrap_or(frames.len());
    eprintln!(
        "[live_video_overlay_smoke] processing {} frame(s), dump={}",
        frame_count,
        dump_dir.display()
    );
    let progress_every = env_usize("LIVE_VIDEO_PROGRESS_EVERY").unwrap_or(30);
    let mut prev_state: &str = "init";
    let mut prev_inliers: Option<usize> = None;
    let mut prev_anchor: Option<u64> = None;

    let timing_every = env_usize("LIVE_VIDEO_TIMING_EVERY").unwrap_or(0);
    let mut t_decode_sum = std::time::Duration::ZERO;
    let mut t_oriented_sum = std::time::Duration::ZERO;
    let mut t_track_sum = std::time::Duration::ZERO;
    let mut t_render_sum = std::time::Duration::ZERO;
    let mut t_encode_sum = std::time::Duration::ZERO;
    for (idx, frame) in frames.iter().take(frame_count).enumerate() {
        let t_decode_start = std::time::Instant::now();
        let rgba = frame.decode();
        let t_decode = t_decode_start.elapsed();
        t_decode_sum += t_decode;
        let (w, h) = rgba.dimensions();
        let crop = Rect {
            left: 0,
            top: 0,
            right: w,
            bottom: h,
        };
        let t_oriented_start = std::time::Instant::now();
        let oriented =
            OrientedImage::build_with_rgb(&rgba, w, h, 0, crop, det_max_pixels).expect("frame");
        t_oriented_sum += t_oriented_start.elapsed();
        let timestamp_ns = idx as u64 * 33_333_333;

        let mut detect_reason = DetectReason::None;
        let mut h_view_to_surface = None;
        let mut tracker_inliers: Option<usize> = None;
        let mut lost_anchor: Option<u64> = None;
        let t_track_start = std::time::Instant::now();
        let cmd = engine.process_frame(&oriented.gray, true, timestamp_ns);
        let t_track = t_track_start.elapsed();
        t_track_sum += t_track;
        let last_fit = engine.last_track_result().cloned();
        let tracker_state = match cmd {
            TrackerCommand::Idle | TrackerCommand::Acquiring => {
                if let Some(anchor_id) = engine.acquire_now(&oriented.gray, timestamp_ns) {
                    if active_anchor != Some(anchor_id) {
                        session.reset_anchor_state(anchor_id);
                    }
                    active_anchor = Some(anchor_id);
                    h_surface_to_view = IDENTITY_H;
                    session.on_acquire();
                    detect_reason = DetectReason::Acquire;
                }
                "idle"
            }
            TrackerCommand::Locked {
                anchor_id,
                homography,
                inliers,
                ..
            } => {
                active_anchor = Some(anchor_id);
                h_surface_to_view = homography;
                tracker_inliers = Some(inliers);
                session.on_locked_frame();
                let frame_w = oriented.gray.width() as f32;
                let frame_h = oriented.gray.height() as f32;
                let allow_ocr_trigger = !(skip_ocr && ocr_fired);
                if !session.has_last_lock_h(anchor_id) {
                    session.set_last_lock_h(anchor_id, homography);
                } else if allow_ocr_trigger
                    && session.should_relock_by_view(
                        anchor_id,
                        &homography,
                        frame_w,
                        frame_h,
                        relock_area_ratio,
                    )
                {
                    session.clear_anchor_state_for_relock(anchor_id);
                    session.clear_last_lock_h(anchor_id);
                    session.mark_refresh_fired();
                    h_view_to_surface = h_view_to_surface_from(&homography);
                    detect_reason = if h_view_to_surface.is_some() {
                        DetectReason::RelockByView
                    } else {
                        DetectReason::None
                    };
                } else if allow_ocr_trigger && session.refresh_cadence_elapsed() {
                    h_view_to_surface = h_view_to_surface_from(&homography);
                    if let Some(h) = h_view_to_surface {
                        let viewport = viewport_surface_aabb(&h, frame_w, frame_h);
                        if viewport.as_ref().is_some_and(|aabb| {
                            !session.viewport_contained_in_coverage(
                                anchor_id,
                                aabb,
                                coverage_pad_px,
                            )
                        }) {
                            session.mark_refresh_fired();
                            detect_reason = DetectReason::CoverageExpanded;
                        }
                    }
                }
                "locked"
            }
            TrackerCommand::Lost { last_anchor_id } => {
                lost_anchor = Some(last_anchor_id);
                "lost"
            }
        };

        let state_changed = tracker_state != prev_state;
        if state_changed {
            eprintln!(
                "[smoke] frame {idx}: {prev_state} -> {tracker_state} (prev inliers={}, prev anchor={})",
                prev_inliers
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "-".into()),
                prev_anchor
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "-".into()),
            );
        } else if tracker_state == "locked" {
            if let (Some(prev), Some(now)) = (prev_inliers, tracker_inliers) {
                if prev >= 30 && (now as f32) < (prev as f32) * 0.5 {
                    eprintln!(
                        "[smoke] frame {idx}: inlier drop {prev} -> {now} (still locked, anchor={})",
                        active_anchor
                            .map(|v| v.to_string())
                            .unwrap_or_else(|| "-".into()),
                    );
                }
            }
        }
        let _ = progress_every; // every-frame logging supersedes progress_every
        eprintln!(
            "[smoke] frame {idx}/{frame_count}: state={tracker_state} inliers={} anchor={}",
            tracker_inliers
                .map(|v| v.to_string())
                .unwrap_or_else(|| "-".into()),
            active_anchor
                .map(|v| v.to_string())
                .unwrap_or_else(|| "-".into()),
        );
        prev_state = tracker_state;
        prev_inliers = tracker_inliers;
        prev_anchor = active_anchor;

        let mut detected_count = 0u32;
        let mut rec_ok_count = 0u32;
        let mut cache_hits = 0u32;
        if detect_reason.runs_detection() {
            let anchor_id = active_anchor.expect("active anchor after detect trigger");
            let raw = ppocr
                .detect_only_image(
                    oriented.rgb_det.as_ref().expect("rgb_det"),
                    PpocrProfile::Live,
                )
                .expect("ppocr detect");
            let rgb = oriented.rgb.as_ref().expect("rgb");
            let boxes: Vec<_> = raw
                .into_iter()
                .map(|b| {
                    scale_detected_box(b, oriented.det_to_full_scale, rgb.width(), rgb.height())
                })
                .collect();
            let outcome = session.run_post_detect(
                PostDetectInput {
                    detections: &boxes,
                    oriented: &oriented,
                    h_view_to_surface,
                    anchor_id,
                    from_lang: &from_lang,
                    to_lang: &to_lang,
                    is_auto_source: false,
                    available_codes: &available_codes,
                    font_provider: &font_provider,
                    matted_strips: &[],
                    rec_batch_size: env_usize("LIVE_VIDEO_REC_BATCH_SIZE").unwrap_or(8),
                    canonical_quadrant: None,
                },
                &recognizer,
                &translator,
                &|| false,
            );
            detected_count = outcome.detected_count;
            rec_ok_count = outcome.rec_ok_count;
            cache_hits = outcome.cache_hits;
            total_detected += detected_count as u64;
            total_rec_ok += rec_ok_count as u64;
            if !outcome.surviving_block_ids.is_empty() {
                session.retain_blocks(anchor_id, &outcome.surviving_block_ids);
            }
            session.set_last_lock_h(anchor_id, h_surface_to_view);
            ocr_fired = true;
        }

        let render_anchor = if tracker_state == "lost" {
            None
        } else {
            active_anchor
        };
        let t_render_start = std::time::Instant::now();
        let (mut composite, item_count) =
            render_outputs(&session, render_anchor, h_surface_to_view, &rgba);
        if item_count > 0 {
            frames_with_overlay += 1;
        }
        if let Some(font) = overlay_font.as_ref() {
            composite =
                draw_frame_label(composite, w, h, font, idx, tracker_state, tracker_inliers);
        }
        t_render_sum += t_render_start.elapsed();

        let t_encode_start = std::time::Instant::now();
        if let Some(writer) = composite_writer.as_mut() {
            writer.write_frame(w, h, &composite);
        }
        let t_encode = t_encode_start.elapsed();
        t_encode_sum += t_encode;
        if timing_every > 0 && idx % timing_every == 0 {
            eprintln!(
                "[timing] frame {idx}: decode={:.1}ms track={:.1}ms encode={:.1}ms",
                t_decode.as_secs_f64() * 1000.0,
                t_track.as_secs_f64() * 1000.0,
                t_encode.as_secs_f64() * 1000.0,
            );
        }
        let (matches_str, residual_str, model_str, desc_inliers_str) = match last_fit.as_ref() {
            Some(r) => {
                let model = if r.inliers >= 30 {
                    "homography"
                } else if r.inliers >= 15 {
                    "affine"
                } else {
                    "similarity"
                };
                (
                    r.matches.to_string(),
                    format!("{:.4}", r.median_residual_px),
                    format!("\"{model}\""),
                    r.descriptor_inliers.to_string(),
                )
            }
            None => (
                "null".to_string(),
                "null".to_string(),
                "null".to_string(),
                "null".to_string(),
            ),
        };
        let gc = engine.gate_counters();
        summaries.push(format!(
            "{{\"frame\":{idx},\"input\":\"{}\",\"tracker\":\"{}\",\"tracker_inliers\":{},\"descriptor_inliers\":{},\"matches\":{},\"median_residual_px\":{},\"fit_model\":{},\"lost_anchor\":{},\"detect_reason\":\"{}\",\"anchor\":{},\"render_anchor\":{},\"h_surface_to_view\":[{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.8},{:.8},{:.6}],\"overlay_items\":{},\"detected\":{},\"rec_ok\":{},\"cache_hits\":{},\"gates\":{{\"sanity_freeze\":{},\"sanity_reject\":{},\"delta_cap\":{},\"h_sanity\":{}}}}}",
            json_escape(frame.label()),
            tracker_state,
            tracker_inliers
                .map(|v| v.to_string())
                .unwrap_or_else(|| "null".to_string()),
            desc_inliers_str,
            matches_str,
            residual_str,
            model_str,
            lost_anchor
                .map(|v| v.to_string())
                .unwrap_or_else(|| "null".to_string()),
            detect_reason.as_str(),
            active_anchor.map(|v| v.to_string()).unwrap_or_else(|| "null".to_string()),
            render_anchor.map(|v| v.to_string()).unwrap_or_else(|| "null".to_string()),
            h_surface_to_view[0],
            h_surface_to_view[1],
            h_surface_to_view[2],
            h_surface_to_view[3],
            h_surface_to_view[4],
            h_surface_to_view[5],
            h_surface_to_view[6],
            h_surface_to_view[7],
            h_surface_to_view[8],
            item_count,
            detected_count,
            rec_ok_count,
            cache_hits,
            gc.sanity_gate_freeze,
            gc.sanity_gate_reject,
            gc.delta_cap_reject,
            gc.h_sanity_reject,
        ));
    }

    let n_f = frame_count.max(1) as f64;
    eprintln!(
        "[timing] avg per-frame over {} frames: decode={:.1}ms oriented={:.1}ms track={:.1}ms render={:.1}ms encode={:.1}ms",
        frame_count,
        t_decode_sum.as_secs_f64() * 1000.0 / n_f,
        t_oriented_sum.as_secs_f64() * 1000.0 / n_f,
        t_track_sum.as_secs_f64() * 1000.0 / n_f,
        t_render_sum.as_secs_f64() * 1000.0 / n_f,
        t_encode_sum.as_secs_f64() * 1000.0 / n_f,
    );
    if let Some(writer) = composite_writer {
        writer.finish();
    }
    fs::write(dump_dir.join("summary.jsonl"), summaries.join("\n")).expect("write summary");
    eprintln!(
        "[live_video_overlay_smoke] overlay frames: {}/{}; detected total={}; rec_ok total={}",
        frames_with_overlay, frame_count, total_detected, total_rec_ok
    );

    assert!(frame_count > 0, "no input frames");
    assert!(
        frames_with_overlay > 0,
        "pipeline produced no overlay items; inspect {}",
        dump_dir.display()
    );
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DetectReason {
    None,
    Acquire,
    CoverageExpanded,
    RelockByView,
}

impl DetectReason {
    fn runs_detection(self) -> bool {
        !matches!(self, DetectReason::None)
    }

    fn as_str(self) -> &'static str {
        match self {
            DetectReason::None => "none",
            DetectReason::Acquire => "acquire",
            DetectReason::CoverageExpanded => "coverage_expanded",
            DetectReason::RelockByView => "relock_by_view",
        }
    }
}

struct PpocrPaths {
    det: PathBuf,
    rec: PathBuf,
    keys: PathBuf,
    script: PpocrScript,
}

fn ppocr_model_paths() -> Option<PpocrPaths> {
    let model_dir = env_path("LIVE_VIDEO_PPOCR_MODEL_DIR").unwrap_or_else(|| {
        env_path("LIVE_OVERLAY_FIXTURE_MODEL_DIR")
            .unwrap_or_else(|| PathBuf::from(MODEL_DIR_DEFAULT))
    });
    let script = env_string("LIVE_VIDEO_PPOCR_SCRIPT")
        .as_deref()
        .and_then(PpocrScript::from_slug)
        .unwrap_or(PpocrScript::Latin);
    let slug = script.as_slug();

    let det = env_path("LIVE_VIDEO_PPOCR_DET").or_else(|| {
        first_existing([
            model_dir.join("PP-OCRv5_mobile_det.mnn"),
            model_dir.join("PP-OCRv5_mobile_det_fp16.mnn"),
        ])
    })?;
    let rec = env_path("LIVE_VIDEO_PPOCR_REC").or_else(|| {
        first_existing([model_dir.join(format!("{slug}_PP-OCRv5_mobile_rec_infer.mnn"))])
    })?;
    let keys = env_path("LIVE_VIDEO_PPOCR_KEYS")
        .or_else(|| first_existing([model_dir.join(format!("{slug}_PP-OCRv5_keys.txt"))]))?;
    Some(PpocrPaths {
        det,
        rec,
        keys,
        script,
    })
}

fn first_existing<const N: usize>(paths: [PathBuf; N]) -> Option<PathBuf> {
    paths.into_iter().find(|p| p.exists())
}

struct InputFrames {
    frames: Vec<InputFrame>,
}

fn input_frames() -> Option<InputFrames> {
    let path = env_path("LIVE_VIDEO_MP4_FILE")?;
    let bytes = decode_mp4_to_mjpeg(&path);
    let frames = split_mjpeg_frames(&bytes)
        .into_iter()
        .enumerate()
        .map(|(idx, bytes)| InputFrame {
            label: format!("{}#{idx}", path.display()),
            bytes,
        })
        .collect::<Vec<_>>();
    assert!(
        !frames.is_empty(),
        "ffmpeg produced no frames from {}",
        path.display()
    );
    Some(InputFrames { frames })
}

struct InputFrame {
    label: String,
    bytes: Vec<u8>,
}

impl InputFrame {
    fn decode(&self) -> RgbaImage {
        image::load_from_memory(&self.bytes)
            .unwrap_or_else(|e| panic!("decode MJPEG frame {}: {e}", self.label))
            .to_rgba8()
    }

    fn label(&self) -> &str {
        &self.label
    }
}

fn decode_mp4_to_mjpeg(path: &Path) -> Vec<u8> {
    let out = Command::new("ffmpeg")
        .args(["-loglevel", "error", "-i"])
        .arg(path)
        .args(["-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "3", "-"])
        .stderr(Stdio::inherit())
        .output()
        .unwrap_or_else(|e| panic!("spawn ffmpeg to decode {}: {e}", path.display()));
    if !out.status.success() {
        panic!("ffmpeg decode {} failed: {}", path.display(), out.status);
    }
    out.stdout
}

struct Mp4Writer {
    child: Child,
    stdin: Option<ChildStdin>,
    quality: u8,
    output: PathBuf,
}

impl Mp4Writer {
    fn new(path: &Path, fps: u32, quality: u8) -> Self {
        let mut child = Command::new("ffmpeg")
            .args(["-y", "-loglevel", "error"])
            .args(["-framerate", &fps.to_string()])
            .args(["-f", "image2pipe", "-vcodec", "mjpeg", "-i", "-"])
            .args([
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
            ])
            .arg(path)
            .stdin(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap_or_else(|e| panic!("spawn ffmpeg to encode {}: {e}", path.display()));
        let stdin = child.stdin.take().expect("ffmpeg stdin");
        Self {
            child,
            stdin: Some(stdin),
            quality,
            output: path.to_path_buf(),
        }
    }

    fn write_frame(&mut self, w: u32, h: u32, rgba: &[u8]) {
        let stdin = self.stdin.as_mut().expect("Mp4Writer stdin closed");
        let mut rgb = Vec::with_capacity((w as usize) * (h as usize) * 3);
        for px in rgba.chunks_exact(4) {
            rgb.extend_from_slice(&px[..3]);
        }
        let mut encoder = JpegEncoder::new_with_quality(&mut *stdin, self.quality);
        encoder
            .encode(&rgb, w, h, image::ExtendedColorType::Rgb8)
            .expect("encode mjpeg frame to ffmpeg stdin");
    }

    fn finish(mut self) {
        self.stdin.take();
        let status = self
            .child
            .wait()
            .unwrap_or_else(|e| panic!("wait ffmpeg encoder for {}: {e}", self.output.display()));
        assert!(
            status.success(),
            "ffmpeg encoder for {} failed: {status}",
            self.output.display()
        );
    }
}

fn split_mjpeg_frames(bytes: &[u8]) -> Vec<Vec<u8>> {
    let mut frames = Vec::new();
    let mut i = 0usize;
    while i + 1 < bytes.len() {
        if bytes[i] != 0xff || bytes[i + 1] != 0xd8 {
            i += 1;
            continue;
        }
        let start = i;
        i += 2;
        while i + 1 < bytes.len() {
            if bytes[i] == 0xff && bytes[i + 1] == 0xd9 {
                let end = i + 2;
                frames.push(bytes[start..end].to_vec());
                i = end;
                break;
            }
            i += 1;
        }
    }
    frames
}

struct PpocrLiveRecognizer<'a> {
    engine: &'a PpocrEngine,
    script: PpocrScript,
}

impl LiveRecognizer for PpocrLiveRecognizer<'_> {
    fn recognize(
        &self,
        oriented: &OrientedImage,
        boxes: &[DetectedTextBox],
        _source_selection: &OcrSourceSelection,
        canonical_quadrant: Option<translator::coords::Quadrant>,
    ) -> Result<Vec<RecognizedTextLine>, String> {
        let rgb = oriented.rgb.as_ref().expect("recognize requires rgb");
        let gray = image::imageops::grayscale(&rgb.to_rgb8());
        let scripts = vec![self.script; boxes.len()];
        self.engine
            .recognize_text_in_boxes_image(
                rgb,
                &gray,
                boxes,
                &scripts,
                PpocrProfile::Live,
                canonical_quadrant,
            )
            .map_err(|e| format!("{e:?}"))
    }
}

struct HostFonts {
    path: Option<PathBuf>,
}

impl HostFonts {
    fn new() -> Option<Self> {
        let path = env_path("LIVE_VIDEO_FONT").or_else(|| {
            first_existing([PathBuf::from(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            )])
        })?;
        Some(Self { path: Some(path) })
    }

    fn none() -> Self {
        Self { path: None }
    }
}

impl FontProvider for HostFonts {
    fn locate(&self, _request: &FontRequest) -> Vec<FontHandle> {
        self.path
            .as_ref()
            .map(|p| vec![FontHandle::from(p.clone())])
            .unwrap_or_default()
    }
}

fn render_outputs(
    session: &LiveSession,
    active_anchor: Option<u64>,
    h_surface_to_view: [f32; 9],
    camera: &RgbaImage,
) -> (Vec<u8>, usize) {
    let w = camera.width();
    let h = camera.height();
    let camera_bytes = camera.as_raw();
    let mut composite = vec![0u8; camera_bytes.len()];

    let Ok(items_guard) = session.overlay_items.lock() else {
        return (composite, 0);
    };
    let items: Vec<CompositeOverlayItem<'_>> = {
        let live = items_guard
            .iter()
            .filter(|it| Some(it.anchor_id) == active_anchor)
            .collect::<Vec<_>>();
        let mut out: Vec<CompositeOverlayItem<'_>> = Vec::with_capacity(live.len() * 2);
        for it in &live {
            out.push(CompositeOverlayItem {
                bitmap_rgba: &it.bg_bitmap,
                bitmap_width: it.width,
                bitmap_height: it.height,
                bitmap_origin_surface_x: it.surface_origin_x,
                bitmap_origin_surface_y: it.surface_origin_y,
                row_extents: &it.bg_row_extents,
            });
        }
        for it in &live {
            out.push(CompositeOverlayItem {
                bitmap_rgba: &it.text_bitmap,
                bitmap_width: it.width,
                bitmap_height: it.height,
                bitmap_origin_surface_x: it.surface_origin_x,
                bitmap_origin_surface_y: it.surface_origin_y,
                row_extents: &it.text_row_extents,
            });
        }
        out
    };
    let item_count = items.len();
    composite_frame_into(
        &mut composite,
        w,
        h,
        camera_bytes,
        &h_surface_to_view,
        &items,
    )
    .expect("composite camera");
    (composite, item_count)
}

fn scale_detected_box(
    mut b: DetectedTextBox,
    scale: f32,
    max_w: u32,
    max_h: u32,
) -> DetectedTextBox {
    b.rect = scale_rect(&b.rect, scale, max_w, max_h);
    b.oriented_box = scale_oriented(&b.oriented_box, scale);
    b.tight_box = scale_oriented(&b.tight_box, scale);
    for v in &mut b.contour {
        *v *= scale;
    }
    b
}

fn scale_rect(r: &Rect, scale: f32, max_w: u32, max_h: u32) -> Rect {
    let left = ((r.left as f32) * scale).floor().clamp(0.0, max_w as f32) as u32;
    let top = ((r.top as f32) * scale).floor().clamp(0.0, max_h as f32) as u32;
    let right = ((r.right as f32) * scale).ceil().clamp(0.0, max_w as f32) as u32;
    let bottom = ((r.bottom as f32) * scale).ceil().clamp(0.0, max_h as f32) as u32;
    Rect {
        left: left.min(right.saturating_sub(1)),
        top: top.min(bottom.saturating_sub(1)),
        right: right.max(left.saturating_add(1)),
        bottom: bottom.max(top.saturating_add(1)),
    }
}

fn scale_oriented(r: &OrientedRect, scale: f32) -> OrientedRect {
    OrientedRect {
        cx: r.cx * scale,
        cy: r.cy * scale,
        width: r.width * scale,
        height: r.height * scale,
        angle_radians: r.angle_radians,
    }
}

fn env_path(key: &str) -> Option<PathBuf> {
    let raw = std::env::var(key).ok()?;
    if let Some(rest) = raw.strip_prefix("~/") {
        Some(PathBuf::from(std::env::var("HOME").ok()?).join(rest))
    } else {
        Some(PathBuf::from(raw))
    }
}

fn env_string(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|s| !s.trim().is_empty())
}

fn env_bool(key: &str) -> Option<bool> {
    let v = env_string(key)?.to_ascii_lowercase();
    match v.as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

fn load_overlay_font() -> Option<FontArc> {
    let path = env_path("LIVE_VIDEO_FONT").or_else(|| {
        first_existing([PathBuf::from(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        )])
    })?;
    let bytes = fs::read(&path).ok()?;
    FontArc::try_from_vec(bytes).ok()
}

fn draw_frame_label(
    composite: Vec<u8>,
    w: u32,
    h: u32,
    font: &FontArc,
    frame_idx: usize,
    tracker_state: &str,
    inliers: Option<usize>,
) -> Vec<u8> {
    let Some(mut img) = RgbaImage::from_raw(w, h, composite) else {
        return Vec::new();
    };
    let label = format!(
        "frame {frame_idx} | {tracker_state} | in={}",
        inliers
            .map(|v| v.to_string())
            .unwrap_or_else(|| "-".to_string()),
    );
    let scale = PxScale::from(28.0);
    draw_filled_rect_mut(
        &mut img,
        ImpRect::at(8, 8).of_size(420, 40),
        Rgba([0, 0, 0, 200]),
    );
    draw_text_mut(
        &mut img,
        Rgba([255, 255, 255, 255]),
        14,
        12,
        scale,
        font,
        &label,
    );
    img.into_raw()
}

fn env_u32(key: &str) -> Option<u32> {
    env_string(key)?.parse().ok()
}

fn env_usize(key: &str) -> Option<usize> {
    env_string(key)?.parse().ok()
}

fn env_u8(key: &str) -> Option<u8> {
    env_string(key)?.parse().ok()
}

fn env_f32(key: &str) -> Option<f32> {
    env_string(key)?.parse().ok()
}

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}
