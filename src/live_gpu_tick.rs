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
use crate::live_frame::LiveFrame;
use crate::live_tracker_pipeline::{LiveTrackerPipeline, ProcessFrameResult};
use crate::ocr::Rect;

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
        let rgba = gles
            .read_camera_rgba(canonical_w, canonical_h, &display_xform)
            .ok_or_else(|| {
                TranslatorError::new(
                    TranslatorErrorKind::Internal,
                    "read_camera_rgba returned None on acquire",
                )
            })?;
        let rgb_frame = Arc::new(LiveFrame::new(0));
        rgb_frame.reset_owned(rgba, canonical_w, canonical_h, 0);
        pipeline.provide_acquire_rgb(req, &rgb_frame);
    }
    Ok(result)
}
