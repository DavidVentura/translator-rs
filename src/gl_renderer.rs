//! GLES2 composite backend: the GPU counterpart to the CPU bilinear warp
//! in [`crate::live_compositor`]. Same per-frame contract
//! ([`Renderer`]/[`CompositeInput`]), but the camera passthrough and each
//! overlay are drawn as textured quads and the homography is applied in
//! the vertex shader, so the perspective warp + blend run on the GPU.
//!
//! The crate never creates a GL context: the app owns EGL and the render
//! thread, and hands in a loader (`eglGetProcAddress`-style) at
//! construction. The shader is GLSL ES 1.00 / GLES2 — the universal floor
//! across Android 5+ and Ubuntu Touch — and runs unchanged on GLES3
//! contexts. `GlesRenderer` holds a `glow::Context`, which is `!Send`, so
//! it is bound to its owning thread at compile time.
//!
//! Two output paths share one `draw`:
//! - [`GlesRenderer::composite`] (the [`Renderer`] trait) renders into an
//!   owned FBO and reads the result back into a CPU buffer. Free on CPU,
//!   expensive on GPU — it exists for tests and the pre-present migration
//!   step, not the hot path.
//! - [`GlesRenderer::present`] renders into whatever framebuffer the
//!   caller bound (the window surface / scene-graph FBO). This is the
//!   production path; there is no readback.

use std::num::NonZeroU32;

use glow::HasContext;

use crate::homography::mat3_mul;
use crate::live_compositor::{
    CameraFrame, ComposeTarget, CompositeError, CompositeInput, Renderer,
};

#[derive(Debug)]
pub enum GlError {
    ShaderCompile(String),
    ProgramLink(String),
    MissingLocation(&'static str),
}

const VERT_SRC: &str = r#"#version 100
attribute vec2 a_pos;
uniform mat3 u_transform;
uniform mat3 u_uv_xform;
varying vec2 v_uv;
void main() {
    vec3 clip = u_transform * vec3(a_pos, 1.0);
    vec3 uv = u_uv_xform * vec3(a_pos, 1.0);
    v_uv = uv.xy;
    gl_Position = vec4(clip.xy, 0.0, clip.z);
}
"#;

const FRAG_SRC: &str = r#"#version 100
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
uniform sampler2D u_tex;
varying vec2 v_uv;
void main() {
    gl_FragColor = texture2D(u_tex, v_uv);
}
"#;

/// Camera fragment shader for a borrowed `GL_TEXTURE_EXTERNAL_OES` source
/// (e.g. an Android/UT SurfaceTexture). Shares [`VERT_SRC`]; the only
/// difference from [`FRAG_SRC`] is the external sampler. `#extension` must
/// precede the `precision` block.
const EXT_FRAG_SRC: &str = r#"#version 100
#extension GL_OES_EGL_image_external : require
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
uniform samplerExternalOES u_tex;
varying vec2 v_uv;
void main() {
    gl_FragColor = texture2D(u_tex, v_uv);
}
"#;

/// `GL_TEXTURE_EXTERNAL_OES` (not in glow's constant set).
const TEXTURE_EXTERNAL_OES: u32 = 0x8D65;

/// Row-major identity-affine helpers, matching the convention of
/// [`crate::homography`] (`mat3_mul` is row-major, `a * b`).
fn translate(tx: f32, ty: f32) -> [f32; 9] {
    [1.0, 0.0, tx, 0.0, 1.0, ty, 0.0, 0.0, 1.0]
}

fn scale(sx: f32, sy: f32) -> [f32; 9] {
    [sx, 0.0, 0.0, 0.0, sy, 0.0, 0.0, 0.0, 1.0]
}

/// Maps top-left-origin viewport pixels to clip space, flipping y so the
/// image's top row lands at the top of the framebuffer. Affine (`w` row
/// is `[0,0,1]`), so it leaves the homography's perspective divide intact.
fn ndc_from_viewport(w: f32, h: f32) -> [f32; 9] {
    [2.0 / w, 0.0, -1.0, 0.0, -2.0 / h, 1.0, 0.0, 0.0, 1.0]
}

/// GLES2 `glUniformMatrix3fv` forbids the transpose flag, so we transpose
/// our row-major matrices into the column-major order GL expects.
fn to_column_major(m: &[f32; 9]) -> [f32; 9] {
    [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]]
}

pub struct GlesRenderer {
    gl: glow::Context,
    program: glow::Program,
    u_transform: glow::UniformLocation,
    u_uv_xform: glow::UniformLocation,
    u_tex: glow::UniformLocation,
    a_pos: u32,
    quad_vbo: glow::Buffer,
    camera_tex: glow::Texture,
    overlay_tex: glow::Texture,
    /// Allocated size of each texture, so a same-size frame updates in
    /// place with `glTexSubImage2D` instead of reallocating storage.
    camera_size: Option<(u32, u32)>,
    overlay_size: Option<(u32, u32)>,
    /// Identity (data ptr + len) of the last overlay uploaded. The
    /// overlay bitmap only changes when the OCR worker rebuilds the
    /// canvas (new `Vec`, new ptr); unchanged between refreshes, so we
    /// skip the per-frame re-upload while it matches.
    overlay_id: Option<(usize, usize)>,
    fbo: Option<glow::Framebuffer>,
    fbo_tex: Option<glow::Texture>,
    fbo_size: Option<(u32, u32)>,
    /// Lazily-built program for a borrowed external-OES camera source.
    ext: Option<ExtProgram>,
    /// `(external texture id, canonical uv transform row-major)` when the
    /// camera source is a borrowed `GL_TEXTURE_EXTERNAL_OES` (zero-copy) rather
    /// than the uploaded 2D `camera_tex`. The uv transform carries the
    /// upright/crop/flip from sensor space into the canonical frame.
    camera_external: Option<(u32, [f32; 9])>,
}

/// The external-OES camera program (shares [`VERT_SRC`] with the 2D program;
/// only the fragment sampler differs). Built lazily on first external use so
/// contexts that never use it (Android's uploaded path) don't require the
/// `GL_OES_EGL_image_external` extension.
struct ExtProgram {
    program: glow::Program,
    u_transform: glow::UniformLocation,
    u_uv_xform: glow::UniformLocation,
    u_tex: glow::UniformLocation,
    a_pos: u32,
}

impl GlesRenderer {
    /// `loader` resolves GL function pointers (e.g. `eglGetProcAddress`).
    /// The caller must have created and made-current a GLES2 (or newer)
    /// context on the calling thread before this runs.
    pub fn new(loader: impl FnMut(&str) -> *const std::ffi::c_void) -> Result<Self, GlError> {
        let gl = unsafe { glow::Context::from_loader_function(loader) };
        unsafe {
            let max_tex = gl.get_parameter_i32(glow::MAX_TEXTURE_SIZE);
            log::debug!("GlesRenderer: GL_MAX_TEXTURE_SIZE={max_tex}");

            let program = link_program(&gl)?;
            let u_transform = gl
                .get_uniform_location(program, "u_transform")
                .ok_or(GlError::MissingLocation("u_transform"))?;
            let u_uv_xform = gl
                .get_uniform_location(program, "u_uv_xform")
                .ok_or(GlError::MissingLocation("u_uv_xform"))?;
            let u_tex = gl
                .get_uniform_location(program, "u_tex")
                .ok_or(GlError::MissingLocation("u_tex"))?;
            let a_pos = gl
                .get_attrib_location(program, "a_pos")
                .ok_or(GlError::MissingLocation("a_pos"))?;

            let quad_vbo = gl.create_buffer().map_err(GlError::ProgramLink)?;
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(quad_vbo));
            let verts: [f32; 8] = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
            let verts_bytes = std::slice::from_raw_parts(
                verts.as_ptr() as *const u8,
                std::mem::size_of_val(&verts),
            );
            gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, verts_bytes, glow::STATIC_DRAW);

            gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
            let camera_tex = new_texture(&gl)?;
            let overlay_tex = new_texture(&gl)?;

            Ok(Self {
                gl,
                program,
                u_transform,
                u_uv_xform,
                u_tex,
                a_pos,
                quad_vbo,
                camera_tex,
                overlay_tex,
                camera_size: None,
                overlay_size: None,
                overlay_id: None,
                fbo: None,
                fbo_tex: None,
                fbo_size: None,
                ext: None,
                camera_external: None,
            })
        }
    }

    /// Upload the camera frame into the camera texture. Split out of
    /// [`draw`](Self::draw) so the pipeline can run this ~3 ms CPU→GPU copy
    /// concurrently with the (H-independent) tracker; `draw`/`present` then
    /// sample the already-resident texture. Reuses GL storage in place via
    /// `glTexSubImage2D` when the frame size is unchanged.
    pub fn upload_camera_tex(&mut self, camera: &CameraFrame<'_>) {
        upload_tex(
            &self.gl,
            self.camera_tex,
            &mut self.camera_size,
            camera.camera_rgba,
            camera.src_full_w,
            camera.src_full_h,
        );
    }

    /// Use a borrowed `GL_TEXTURE_EXTERNAL_OES` as the camera source (zero-copy)
    /// instead of the uploaded 2D texture. `uv` is the row-major canonical uv
    /// transform (unit-quad → sensor uv, encoding upright/crop/flip). Lazily
    /// builds the external-OES program; if that fails (no extension) the source
    /// is left unchanged. Cleared with [`clear_camera_external`](Self::clear_camera_external).
    pub fn set_camera_external(&mut self, id: u32, uv: [f32; 9]) {
        if self.ext.is_none() {
            self.ext = self.build_ext_program();
        }
        if self.ext.is_some() {
            self.camera_external = Some((id, uv));
        }
    }

    pub fn clear_camera_external(&mut self) {
        self.camera_external = None;
    }

    fn build_ext_program(&self) -> Option<ExtProgram> {
        let gl = &self.gl;
        let program = match link_program_frag(gl, EXT_FRAG_SRC) {
            Ok(p) => p,
            Err(e) => {
                log::error!("GlesRenderer: external-OES program link failed: {e:?}");
                return None;
            }
        };
        unsafe {
            Some(ExtProgram {
                u_transform: gl.get_uniform_location(program, "u_transform")?,
                u_uv_xform: gl.get_uniform_location(program, "u_uv_xform")?,
                u_tex: gl.get_uniform_location(program, "u_tex")?,
                a_pos: gl.get_attrib_location(program, "a_pos")?,
                program,
            })
        }
    }

    /// Render the borrowed external camera (canonical transform, no overlays)
    /// into the owned FBO at `w×h` and read it back as top-down grayscale — the
    /// tracker's input, replacing the CPU map + downscale. Returns `None` if no
    /// external camera source is set.
    pub fn read_camera_gray(&mut self, w: u32, h: u32) -> Option<Vec<u8>> {
        let (id, uv) = self.camera_external?;
        let id = NonZeroU32::new(id)?;
        self.ext.as_ref()?;
        self.ensure_fbo(w, h);
        let cam_transform = mat3_mul(
            &ndc_from_viewport(w as f32, h as f32),
            &scale(w as f32, h as f32),
        );
        let mut rgba = vec![0u8; (w as usize) * (h as usize) * 4];
        unsafe {
            let gl = &self.gl;
            let e = self.ext.as_ref().expect("ext program present");
            gl.bind_framebuffer(glow::FRAMEBUFFER, self.fbo);
            gl.viewport(0, 0, w as i32, h as i32);
            gl.disable(glow::BLEND);
            gl.use_program(Some(e.program));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.quad_vbo));
            gl.enable_vertex_attrib_array(e.a_pos);
            gl.vertex_attrib_pointer_f32(e.a_pos, 2, glow::FLOAT, false, 0, 0);
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(TEXTURE_EXTERNAL_OES, Some(glow::NativeTexture(id)));
            gl.uniform_1_i32(Some(&e.u_tex), 0);
            gl.uniform_matrix_3_f32_slice(
                Some(&e.u_transform),
                false,
                &to_column_major(&cam_transform),
            );
            gl.uniform_matrix_3_f32_slice(Some(&e.u_uv_xform), false, &to_column_major(&uv));
            gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
            gl.read_pixels(
                0,
                0,
                w as i32,
                h as i32,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelPackData::Slice(Some(&mut rgba)),
            );
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        }
        // glReadPixels is bottom-up; the tracker wants top-down. Convert
        // RGBA→luma (Rec.601-ish integer weights) and flip rows in one pass.
        let stride = w as usize;
        let mut gray = vec![0u8; stride * h as usize];
        for y in 0..h as usize {
            let src = (h as usize - 1 - y) * stride * 4;
            let dst = y * stride;
            for x in 0..stride {
                let p = src + x * 4;
                let (r, g, b) = (rgba[p] as u32, rgba[p + 1] as u32, rgba[p + 2] as u32);
                gray[dst + x] = ((r * 77 + g * 150 + b * 29) >> 8) as u8;
            }
        }
        Some(gray)
    }

    /// Render the composite into whatever framebuffer is currently bound;
    /// the caller owns FBO binding, viewport, and buffer swap. No
    /// readback — this is the production present path. Assumes the camera
    /// frame was already uploaded via [`upload_camera_tex`](Self::upload_camera_tex)
    /// for this frame.
    ///
    /// `display_xform` maps dst-pixel coords (top-left origin, y-down,
    /// `dst_w`×`dst_h`) to clip space. The caller folds surface size +
    /// display rotation + FILL_CENTER scale into it — the GL equivalent
    /// of the old Canvas `drawMatrix`. For an unrotated surface-sized
    /// blit it is just `ndc_from_viewport(surface_w, surface_h)` times the
    /// dst→surface fit.
    pub fn present(&mut self, input: &CompositeInput<'_>, display_xform: &[f32; 9]) {
        self.draw(input, display_xform);
    }

    /// Composite the (already-uploaded) camera texture + overlays into the
    /// bound framebuffer. The camera upload is the caller's responsibility
    /// (via [`upload_camera_tex`](Self::upload_camera_tex)); overlay uploads
    /// stay here because they depend on the per-anchor overlay items.
    fn draw(&mut self, input: &CompositeInput<'_>, dst_to_clip: &[f32; 9]) {
        let dst_w = input.dst_w as f32;
        let dst_h = input.dst_h as f32;
        // Fast-clear so tiled GPUs skip loading the previous framebuffer; the
        // opaque camera quad then covers it.
        unsafe {
            let gl = &self.gl;
            gl.clear_color(0.0, 0.0, 0.0, 1.0);
            gl.clear(glow::COLOR_BUFFER_BIT);
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.quad_vbo));
            gl.active_texture(glow::TEXTURE0);
            gl.disable(glow::BLEND);
        }

        // Camera passthrough (opaque base). Either a borrowed external-OES
        // texture (zero-copy, sampled through the canonical uv transform that
        // carries upright/crop/flip) or the uploaded 2D texture sampled through
        // the sensor-crop uv.
        let cam_transform = mat3_mul(dst_to_clip, &scale(dst_w, dst_h));
        match self.camera_external {
            Some((id, uv)) if self.ext.is_some() => unsafe {
                let gl = &self.gl;
                let e = self.ext.as_ref().expect("ext program present");
                gl.use_program(Some(e.program));
                gl.enable_vertex_attrib_array(e.a_pos);
                gl.vertex_attrib_pointer_f32(e.a_pos, 2, glow::FLOAT, false, 0, 0);
                gl.uniform_1_i32(Some(&e.u_tex), 0);
                if let Some(id) = NonZeroU32::new(id) {
                    gl.bind_texture(TEXTURE_EXTERNAL_OES, Some(glow::NativeTexture(id)));
                }
                gl.uniform_matrix_3_f32_slice(
                    Some(&e.u_transform),
                    false,
                    &to_column_major(&cam_transform),
                );
                gl.uniform_matrix_3_f32_slice(Some(&e.u_uv_xform), false, &to_column_major(&uv));
                gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
            },
            _ => unsafe {
                let gl = &self.gl;
                gl.use_program(Some(self.program));
                gl.enable_vertex_attrib_array(self.a_pos);
                gl.vertex_attrib_pointer_f32(self.a_pos, 2, glow::FLOAT, false, 0, 0);
                gl.uniform_1_i32(Some(&self.u_tex), 0);
                gl.bind_texture(glow::TEXTURE_2D, Some(self.camera_tex));
                let cam_uv = [
                    dst_w / input.src_full_w as f32,
                    0.0,
                    input.src_offset_x as f32 / input.src_full_w as f32,
                    0.0,
                    dst_h / input.src_full_h as f32,
                    input.src_offset_y as f32 / input.src_full_h as f32,
                    0.0,
                    0.0,
                    1.0,
                ];
                gl.uniform_matrix_3_f32_slice(
                    Some(&self.u_transform),
                    false,
                    &to_column_major(&cam_transform),
                );
                gl.uniform_matrix_3_f32_slice(
                    Some(&self.u_uv_xform),
                    false,
                    &to_column_major(&cam_uv),
                );
                gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
            },
        }

        // Overlays: source-over with straight (non-premultiplied) alpha,
        // matching the CPU blend. Always the 2D program. Each item warped by the
        // same H the CPU path uses, composed with its surface origin and size.
        unsafe {
            let gl = &self.gl;
            gl.use_program(Some(self.program));
            gl.enable_vertex_attrib_array(self.a_pos);
            gl.vertex_attrib_pointer_f32(self.a_pos, 2, glow::FLOAT, false, 0, 0);
            gl.uniform_1_i32(Some(&self.u_tex), 0);
            gl.enable(glow::BLEND);
            gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);
            let identity_uv = to_column_major(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
            for item in input.items {
                if item.bitmap_rgba.is_empty() || item.bitmap_width == 0 || item.bitmap_height == 0
                {
                    continue;
                }
                // Skip the upload while the overlay bitmap is unchanged
                // (same `Vec` ptr+len) — it only changes when the OCR
                // worker rebuilds the canvas on acquire/refresh.
                let id = (item.bitmap_rgba.as_ptr() as usize, item.bitmap_rgba.len());
                if self.overlay_id == Some(id) {
                    gl.bind_texture(glow::TEXTURE_2D, Some(self.overlay_tex));
                } else {
                    upload_tex(
                        &self.gl,
                        self.overlay_tex,
                        &mut self.overlay_size,
                        item.bitmap_rgba,
                        item.bitmap_width,
                        item.bitmap_height,
                    );
                    self.overlay_id = Some(id);
                }
                let to_surface =
                    translate(item.bitmap_origin_surface_x, item.bitmap_origin_surface_y);
                let bitmap_to_viewport = mat3_mul(input.h_surface_to_viewport, &to_surface);
                let sized = mat3_mul(
                    &bitmap_to_viewport,
                    &scale(item.bitmap_width as f32, item.bitmap_height as f32),
                );
                let transform = mat3_mul(dst_to_clip, &sized);
                gl.uniform_matrix_3_f32_slice(
                    Some(&self.u_transform),
                    false,
                    &to_column_major(&transform),
                );
                gl.uniform_matrix_3_f32_slice(Some(&self.u_uv_xform), false, &identity_uv);
                gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
            }
        }
    }

    /// (Re)create the readback FBO + color texture to match `w*h`.
    /// FBO incompleteness with an RGBA8 attachment is a driver/setup
    /// invariant, not a runtime input error, so it panics loudly.
    fn ensure_fbo(&mut self, w: u32, h: u32) {
        if self.fbo_size == Some((w, h)) {
            return;
        }
        let gl = &self.gl;
        unsafe {
            if let Some(t) = self.fbo_tex.take() {
                gl.delete_texture(t);
            }
            let tex = new_texture(gl).expect("create fbo texture");
            gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGBA as i32,
                w as i32,
                h as i32,
                0,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(None),
            );
            let fbo = match self.fbo {
                Some(f) => f,
                None => {
                    let f = gl.create_framebuffer().expect("create framebuffer");
                    self.fbo = Some(f);
                    f
                }
            };
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
            gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(tex),
                0,
            );
            let status = gl.check_framebuffer_status(glow::FRAMEBUFFER);
            assert_eq!(
                status,
                glow::FRAMEBUFFER_COMPLETE,
                "readback FBO incomplete: status=0x{status:x}"
            );
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            self.fbo_tex = Some(tex);
            self.fbo_size = Some((w, h));
        }
    }
}

impl GlesRenderer {
    /// Render into the owned FBO with an explicit dst→clip transform and
    /// read the result back (row-flipped to top-down). [`composite`] is
    /// the `ndc_from_viewport` special case; exposed so the display-xform
    /// path `present` uses can be exercised under readback in tests.
    ///
    /// [`composite`]: Renderer::composite
    pub fn render_to_buffer(
        &mut self,
        input: &CompositeInput<'_>,
        dst_to_clip: &[f32; 9],
        out: &mut [u8],
    ) -> Result<(), CompositeError> {
        self.upload_camera_tex(&CameraFrame {
            camera_rgba: input.camera_rgba,
            src_full_w: input.src_full_w,
            src_full_h: input.src_full_h,
        });
        self.render_to_buffer_prepared(input, dst_to_clip, out)
    }

    /// [`render_to_buffer`](Self::render_to_buffer) without the camera
    /// upload — assumes the camera texture is already resident. The
    /// readback counterpart of [`present`](Self::present) for the split
    /// (upload-then-draw) target path.
    fn render_to_buffer_prepared(
        &mut self,
        input: &CompositeInput<'_>,
        dst_to_clip: &[f32; 9],
        out: &mut [u8],
    ) -> Result<(), CompositeError> {
        validate_sizes(input, out)?;
        self.ensure_fbo(input.dst_w, input.dst_h);
        let (w, h) = (input.dst_w, input.dst_h);
        let mut flipped = vec![0u8; out.len()];
        unsafe {
            let gl = &self.gl;
            gl.bind_framebuffer(glow::FRAMEBUFFER, self.fbo);
            gl.viewport(0, 0, w as i32, h as i32);
        }
        // `draw` clears + renders; it needs `&mut self`, so it can't run
        // inside the `&self.gl` borrow above.
        self.draw(input, dst_to_clip);
        unsafe {
            self.gl.read_pixels(
                0,
                0,
                w as i32,
                h as i32,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelPackData::Slice(Some(&mut flipped)),
            );
            self.gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        }
        // glReadPixels is bottom-up; the CPU contract is top-down.
        let stride = (w as usize) * 4;
        for y in 0..h as usize {
            let src = (h as usize - 1 - y) * stride;
            let dst = y * stride;
            out[dst..dst + stride].copy_from_slice(&flipped[src..src + stride]);
        }
        Ok(())
    }
}

impl Renderer for GlesRenderer {
    fn composite(
        &mut self,
        input: &CompositeInput<'_>,
        out: &mut [u8],
    ) -> Result<(), CompositeError> {
        let ndc = ndc_from_viewport(input.dst_w as f32, input.dst_h as f32);
        self.render_to_buffer(input, &ndc, out)
    }
}

impl Drop for GlesRenderer {
    fn drop(&mut self) {
        unsafe {
            self.gl.delete_program(self.program);
            self.gl.delete_buffer(self.quad_vbo);
            self.gl.delete_texture(self.camera_tex);
            self.gl.delete_texture(self.overlay_tex);
            if let Some(fbo) = self.fbo {
                self.gl.delete_framebuffer(fbo);
            }
            if let Some(t) = self.fbo_tex {
                self.gl.delete_texture(t);
            }
        }
    }
}

/// GPU production target: presents into the framebuffer the caller bound
/// (window surface / scene-graph FBO). No readback — the display consumes
/// the result directly. `process_frame` must be called on the thread that
/// owns the GL context. `display_xform` is the dst→clip transform (see
/// [`GlesRenderer::present`]).
pub struct PresentTarget<'a> {
    pub renderer: &'a mut GlesRenderer,
    pub display_xform: [f32; 9],
}

impl ComposeTarget for PresentTarget<'_> {
    fn upload_camera(&mut self, camera: &CameraFrame<'_>) -> Result<(), CompositeError> {
        self.renderer.upload_camera_tex(camera);
        Ok(())
    }

    fn draw(&mut self, input: &CompositeInput<'_>) -> Result<(), CompositeError> {
        self.renderer.present(input, &self.display_xform);
        Ok(())
    }
}

/// GPU readback target: renders into the renderer's own FBO and copies
/// the result into a CPU slice. Lets the GPU path drive `process_frame`
/// while still producing the same `&mut [u8]` the CPU path does — used by
/// tests and the pre-present migration step, not the hot path.
pub struct ReadbackTarget<'a> {
    pub renderer: &'a mut GlesRenderer,
    pub dst: &'a mut [u8],
}

impl ComposeTarget for ReadbackTarget<'_> {
    fn upload_camera(&mut self, camera: &CameraFrame<'_>) -> Result<(), CompositeError> {
        self.renderer.upload_camera_tex(camera);
        Ok(())
    }

    fn draw(&mut self, input: &CompositeInput<'_>) -> Result<(), CompositeError> {
        let ndc = ndc_from_viewport(input.dst_w as f32, input.dst_h as f32);
        self.renderer
            .render_to_buffer_prepared(input, &ndc, self.dst)
    }
}

/// Mirrors the size/bounds checks of
/// [`crate::live_compositor::composite_frame_into_cropped`] so both
/// backends reject identical inputs identically.
fn validate_sizes(input: &CompositeInput<'_>, out: &[u8]) -> Result<(), CompositeError> {
    let dst_bytes = (input.dst_w as usize)
        .checked_mul(input.dst_h as usize)
        .and_then(|n| n.checked_mul(4))
        .ok_or(CompositeError::DstBufferSize)?;
    if out.len() != dst_bytes {
        return Err(CompositeError::DstBufferSize);
    }
    let src_full_bytes = (input.src_full_w as usize)
        .checked_mul(input.src_full_h as usize)
        .and_then(|n| n.checked_mul(4))
        .ok_or(CompositeError::SrcBufferSize)?;
    if input.camera_rgba.len() != src_full_bytes {
        return Err(CompositeError::SrcBufferSize);
    }
    if input.src_offset_x.saturating_add(input.dst_w) > input.src_full_w
        || input.src_offset_y.saturating_add(input.dst_h) > input.src_full_h
    {
        return Err(CompositeError::SrcBufferSize);
    }
    Ok(())
}

fn new_texture(gl: &glow::Context) -> Result<glow::Texture, GlError> {
    unsafe {
        let tex = gl.create_texture().map_err(GlError::ProgramLink)?;
        gl.bind_texture(glow::TEXTURE_2D, Some(tex));
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_MIN_FILTER,
            glow::LINEAR as i32,
        );
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_MAG_FILTER,
            glow::LINEAR as i32,
        );
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_WRAP_S,
            glow::CLAMP_TO_EDGE as i32,
        );
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_WRAP_T,
            glow::CLAMP_TO_EDGE as i32,
        );
        Ok(tex)
    }
}

/// Bind `tex` and upload `data`. Reuses the existing GL storage with
/// `glTexSubImage2D` when the size is unchanged (the common per-frame
/// case), only reallocating via `glTexImage2D` when the size differs.
fn upload_tex(
    gl: &glow::Context,
    tex: glow::Texture,
    cached_size: &mut Option<(u32, u32)>,
    data: &[u8],
    w: u32,
    h: u32,
) {
    unsafe {
        gl.bind_texture(glow::TEXTURE_2D, Some(tex));
        if *cached_size == Some((w, h)) {
            gl.tex_sub_image_2d(
                glow::TEXTURE_2D,
                0,
                0,
                0,
                w as i32,
                h as i32,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(Some(data)),
            );
        } else {
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGBA as i32,
                w as i32,
                h as i32,
                0,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(Some(data)),
            );
            *cached_size = Some((w, h));
        }
    }
}

fn link_program(gl: &glow::Context) -> Result<glow::Program, GlError> {
    link_program_frag(gl, FRAG_SRC)
}

fn link_program_frag(gl: &glow::Context, frag_src: &str) -> Result<glow::Program, GlError> {
    unsafe {
        let program = gl.create_program().map_err(GlError::ProgramLink)?;
        let shaders = [
            compile_shader(gl, glow::VERTEX_SHADER, VERT_SRC)?,
            compile_shader(gl, glow::FRAGMENT_SHADER, frag_src)?,
        ];
        for s in shaders {
            gl.attach_shader(program, s);
        }
        gl.link_program(program);
        if !gl.get_program_link_status(program) {
            return Err(GlError::ProgramLink(gl.get_program_info_log(program)));
        }
        for s in shaders {
            gl.detach_shader(program, s);
            gl.delete_shader(s);
        }
        Ok(program)
    }
}

fn compile_shader(gl: &glow::Context, kind: u32, src: &str) -> Result<glow::Shader, GlError> {
    unsafe {
        let shader = gl.create_shader(kind).map_err(GlError::ShaderCompile)?;
        gl.shader_source(shader, src);
        gl.compile_shader(shader);
        if !gl.get_shader_compile_status(shader) {
            return Err(GlError::ShaderCompile(gl.get_shader_info_log(shader)));
        }
        Ok(shader)
    }
}
