//! Per-frame plumbing shared by the platform GPU bridges (Linux QtQuick,
//! Android EGL). Each bridge owns its own [`GlesRenderer`] and drives a
//! per-frame loop; the work between "external camera texture available" and
//! "composite drawn into the bound framebuffer" is identical.
//!
//! Split into two calls because the caller has to bind its own present
//! framebuffer + viewport in the middle — after the gray readback (which
//! leaves its own R8 FBO bound) and before [`LiveTrackerPipeline::process_frame`]
//! draws the composite.
//!
//! ```ignore
//! let Some(frame) = frame_from_camera_gray(gles, id, fw, fh, uv, dx) else { return };
//! // platform-specific: bind the present target
//! unsafe { gl.bind_framebuffer(FRAMEBUFFER, present_fbo); gl.viewport(0,0,sw,sh); }
//! run_tracker_with_acquire(pipeline, gles, &frame, fw, fh, dx, ts)?;
//! ```

use std::sync::Arc;

use crate::api::{TranslatorError, TranslatorErrorKind};
use crate::gl_renderer::{ExternalPresentTarget, GlesRenderer};
use crate::live_frame::{LiveFrame, OrientedImage, aligned_det_dims};
use crate::live_tracker_pipeline::{LiveTrackerPipeline, ProcessFrameResult};
use crate::ocr::Rect;

/// Row-major 3×3 mapping `w×h` dst-pixel coords → clip `[-1,1]` — the
/// resolution-independent normalize `read_camera_*` wants as `dst_to_clip`
/// (orientation lives in the `uv` transform set by [`frame_from_camera_gray`]).
fn clip_xform(w: u32, h: u32) -> [f32; 9] {
    let w = w.max(1) as f32;
    let h = h.max(1) as f32;
    [2.0 / w, 0.0, -1.0, 0.0, 2.0 / h, -1.0, 0.0, 0.0, 1.0]
}

/// Borrow the external camera texture, GPU-render the canonical luma into a
/// `width*height` byte buffer, and wrap it in a fresh [`LiveFrame`] ready for
/// the tracker. Returns `None` if no external camera source is set or the
/// GLES extension is missing (matches [`GlesRenderer::read_camera_gray`]).
///
/// A fresh `Arc<LiveFrame>` per tick is deliberate: an in-flight acquire holds
/// its own frame's state lock for the whole det+rec (~1s); reusing one frame
/// would make this call's `reset_gray` block on that lock and freeze the
/// render thread for the entire acquire.
pub fn frame_from_camera_gray(
    gles: &mut GlesRenderer,
    camera_tex: u32,
    canonical_w: u32,
    canonical_h: u32,
    uv_xform: [f32; 9],
    display_xform: [f32; 9],
) -> Option<Arc<LiveFrame>> {
    gles.set_camera_external(camera_tex, uv_xform);
    let gray = gles.read_camera_gray(canonical_w, canonical_h, &display_xform)?;
    let frame = Arc::new(LiveFrame::new(0));
    frame.reset_gray(gray, canonical_w, canonical_h);
    Some(frame)
}

/// Run the tracker for one frame and, if the tracker asked for an
/// acquire/refresh on this gray-only frame, satisfy the
/// [`AcquireRequest`](crate::live_tracker_pipeline::AcquireRequest) by reading
/// back the full-res RGBA from the same external camera texture and handing it
/// to [`LiveTrackerPipeline::provide_acquire_rgb`].
///
/// The caller must have bound the present target (framebuffer + viewport)
/// before calling — that's where [`ExternalPresentTarget`] composites into.
pub fn run_tracker_with_acquire(
    pipeline: &LiveTrackerPipeline,
    gles: &mut GlesRenderer,
    frame: &Arc<LiveFrame>,
    canonical_w: u32,
    canonical_h: u32,
    display_xform: [f32; 9],
    timestamp_ns: u64,
) -> Result<ProcessFrameResult, TranslatorError> {
    let crop = Rect {
        left: 0,
        top: 0,
        right: canonical_w,
        bottom: canonical_h,
    };
    let result = {
        let mut target = ExternalPresentTarget {
            renderer: gles,
            display_xform,
        };
        pipeline.process_frame(
            frame,
            crop,
            &mut target,
            canonical_w,
            canonical_h,
            canonical_w,
            canonical_h,
            canonical_w,
            canonical_h,
            timestamp_ns,
        )?
    };
    if let Some(req) = result.rgb_request.clone() {
        // Render the two OCR inputs directly on the GPU instead of reading back
        // full-res RGBA and CPU-resizing: detector gray at the 32-aligned size,
        // recognition RGBA at half canonical. Orientation is already in `uv`
        // (set by `frame_from_camera_gray`); `clip_xform` just sizes each FBO.
        let (det_w, det_h) = aligned_det_dims(canonical_w, canonical_h, req.det_max_pixels());
        // Recognition reads back at full canonical (the camera's canonical is
        // already modest, ≤1000 long edge); half-res lost too much on small
        // glyphs. `rec_scale` then becomes 1.0 → crops straight from full rgb.
        let (rec_w, rec_h) = (canonical_w, canonical_h);
        let det_gray = gles
            .read_camera_gray(det_w, det_h, &clip_xform(det_w, det_h))
            .ok_or_else(|| {
                TranslatorError::new(
                    TranslatorErrorKind::Internal,
                    "read_camera_gray returned None on acquire",
                )
            })?;
        let rec_rgba = gles
            .read_camera_rgba(rec_w, rec_h, &clip_xform(rec_w, rec_h))
            .ok_or_else(|| {
                TranslatorError::new(
                    TranslatorErrorKind::Internal,
                    "read_camera_rgba returned None on acquire",
                )
            })?;
        let oriented = OrientedImage::from_gpu_split(
            det_gray,
            det_w,
            det_h,
            &rec_rgba,
            rec_w,
            rec_h,
            canonical_w,
            canonical_h,
            req.display_crop(),
        )?;
        // The planar tracker's anchor registration needs a canonical-res gray
        // (matching the per-frame tracker gray's size + transform, so anchor
        // features align). Read it with the same `display_xform`.
        let tracker_gray = gles
            .read_camera_gray(canonical_w, canonical_h, &display_xform)
            .ok_or_else(|| {
                TranslatorError::new(
                    TranslatorErrorKind::Internal,
                    "read_camera_gray (tracker) returned None on acquire",
                )
            })?;
        let rgb_frame = Arc::new(LiveFrame::new(0));
        rgb_frame.reset_oriented_split(oriented, Some((tracker_gray, canonical_w, canonical_h)));
        pipeline.provide_acquire_rgb(req, &rgb_frame);
    }
    Ok(result)
}
