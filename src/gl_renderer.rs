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
#[cfg(feature = "planar-tracker")]
use crate::live_session::OverlayDrawList;

#[derive(Debug)]
pub enum GlError {
    ShaderCompile(String),
    ProgramLink(String),
    MissingLocation(&'static str),
}

/// What [`GlesRenderer::present`] draws into the bound framebuffer.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum PresentContent {
    /// Opaque camera passthrough with the overlays composited on top — the
    /// live-camera path. Clears to opaque black under the camera quad.
    #[default]
    CameraAndOverlay,
    /// Only the overlays, over a transparent (alpha 0) clear, with
    /// premultiplied-alpha output. For a translucent window that floats over
    /// live content the renderer does not own (the MediaProjection
    /// screen-translate overlay): the camera passthrough quad is skipped so
    /// the real screen behind the window shows through. The tracker gray/RGBA
    /// readbacks still sample the external source, so OCR is unaffected.
    OverlayOnly,
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
uniform float u_overlay_alpha;
varying vec2 v_uv;
void main() {
    vec4 c = texture2D(u_tex, v_uv);
    // Parametric overlay opacity (1.0 for camera passthrough). Scaling alpha
    // pre-blend is premultiply-correct: the source-over blend then yields an
    // effective coverage of (texel alpha × u_overlay_alpha).
    c.a *= u_overlay_alpha;
    gl_FragColor = c;
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

/// Like [`EXT_FRAG_SRC`] but outputs luminance (Rec. 601) instead of colour,
/// for the per-frame tracker readback into an R8 framebuffer. Writing luma to
/// all channels keeps it correct whether the attachment is R8 or RGBA.
const EXT_LUMA_FRAG_SRC: &str = r#"#version 100
#extension GL_OES_EGL_image_external : require
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
uniform samplerExternalOES u_tex;
varying vec2 v_uv;
void main() {
    vec3 c = texture2D(u_tex, v_uv).rgb;
    float y = dot(c, vec3(0.299, 0.587, 0.114));
    gl_FragColor = vec4(y, y, y, 1.0);
}
"#;

/// Pill vertex shader: the unit quad spans one oriented pill. `a_pos` (0..1) is
/// passed through as the local uv so the fragment shader can evaluate the rounded
/// rect SDF in pill-local pixel space; `u_transform` maps the unit quad to clip.
const PILL_VERT_SRC: &str = r#"#version 100
attribute vec2 a_pos;
uniform mat3 u_transform;
varying vec2 v_local;
void main() {
    v_local = a_pos;
    vec3 clip = u_transform * vec3(a_pos, 1.0);
    gl_Position = vec4(clip.xy, 0.0, clip.z);
}
"#;

/// Pill fragment shader: rounded-rect SDF coverage, the GPU port of
/// `planar_engine::fill_oriented_rect_blended`. `u_half` is the pill half-extent
/// in texels, `u_radius` the corner radius (texels); coverage `= clamp(0.5 - sdf,
/// 0, 1)` with a 1px feather. Output is opaque-coverage (rgb = color, a =
/// coverage) so overlapping pills union flat in the pill-layer FBO.
const PILL_FRAG_SRC: &str = r#"#version 100
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
uniform vec4 u_color;
uniform vec2 u_half;
uniform float u_radius;
varying vec2 v_local;
void main() {
    // Local pixel coords centred on the pill (v_local 0..1 → [-half, half]).
    vec2 p = (v_local - 0.5) * 2.0 * u_half;
    vec2 q = abs(p) - (u_half - vec2(u_radius));
    float outside = length(max(q, 0.0));
    float inside = min(max(q.x, q.y), 0.0);
    float sdf = outside + inside - u_radius;
    float coverage = clamp(0.5 - sdf, 0.0, 1.0);
    if (coverage <= 0.0) {
        discard;
    }
    gl_FragColor = vec4(u_color.rgb, u_color.a * coverage);
}
"#;

/// Glyph fragment shader (shares [`VERT_SRC`]): samples an R8 coverage atlas and
/// outputs straight alpha (fg color × coverage × text_alpha). Each quad covers one
/// glyph; the vertex transform positions and rotates it in overlay-texel space.
const GLYPH_FRAG_SRC: &str = r#"#version 100
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
uniform sampler2D u_atlas;
uniform vec4 u_color;
uniform float u_text_alpha;
varying vec2 v_uv;
void main() {
    float a = texture2D(u_atlas, v_uv).r;
    gl_FragColor = vec4(u_color.rgb, a * u_text_alpha);
}
"#;

/// `GL_TEXTURE_EXTERNAL_OES` (not in glow's constant set).
const TEXTURE_EXTERNAL_OES: u32 = 0x8D65;
/// `GL_R8` sized internal format (GLES3) for the single-channel gray FBO.
const R8: u32 = 0x8229;

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

/// Like [`ndc_from_viewport`] but **without** the y flip: top-left-origin pixel
/// coords map to clip with `y` increasing downward landing at texture `v` = 0 for
/// the top row. Used when rendering *into* the overlay FBO so the baked texture
/// is stored top-row-at-`v=0`, matching the CPU-uploaded overlay convention the
/// present path already samples.
fn ndc_from_viewport_no_flip(w: f32, h: f32) -> [f32; 9] {
    [2.0 / w, 0.0, -1.0, 0.0, 2.0 / h, -1.0, 0.0, 0.0, 1.0]
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
    u_overlay_alpha: glow::UniformLocation,
    a_pos: u32,
    /// Parametric overlay opacity for the 2D program, applied to overlay draws
    /// (forced 1.0 for the camera passthrough). Set once per renderer via
    /// [`set_overlay_alpha`](Self::set_overlay_alpha); the screen path uses it
    /// to control overlay opacity independently of the camera path.
    overlay_alpha: f32,
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
    /// Lazily-built external-OES program that outputs luminance, for the
    /// single-channel tracker-gray readback.
    ext_luma: Option<ExtProgram>,
    /// R8 FBO for the gray readback (separate from the RGBA `fbo`).
    gray_fbo: Option<glow::Framebuffer>,
    gray_fbo_tex: Option<glow::Texture>,
    gray_fbo_size: Option<(u32, u32)>,
    /// `(external texture id, canonical uv transform row-major)` when the
    /// camera source is a borrowed `GL_TEXTURE_EXTERNAL_OES` (zero-copy) rather
    /// than the uploaded 2D `camera_tex`. The uv transform carries the
    /// upright/crop/flip from sensor space into the canonical frame.
    camera_external: Option<(u32, [f32; 9])>,
    /// Whether [`present`](Self::present) composites the camera + overlays or
    /// only the overlays over transparent. Set once for the renderer's
    /// lifetime via [`set_present_content`](Self::set_present_content).
    present_content: PresentContent,
    /// Lazily-built rounded-rect pill program (the GPU overlay compositor).
    pill: Option<PillProgram>,
    /// Lazily-built glyph atlas program (R8 coverage × fg color → straight alpha).
    /// Shares [`VERT_SRC`]; only the fragment shader differs from the 2D program.
    glyph_prog: Option<GlyphProgram>,
    /// Overlay-compositor FBO: the baked pills+glyphs overlay texture that the
    /// present then warps (camera) / blits (screen). `(fbo, tex, w, h)`.
    overlay_fbo: Option<(glow::Framebuffer, glow::Texture, u32, u32)>,
    /// Pill-layer FBO: opaque rounded pills are unioned here, then composited
    /// into `overlay_fbo` at the pill opacity. `(fbo, tex, w, h)`.
    pill_fbo: Option<(glow::Framebuffer, glow::Texture, u32, u32)>,
    /// R8 glyph coverage atlas with shelf packing. Grows until overflow, then
    /// clears and re-packs the current frame's masks.
    glyph_atlas: Option<GlyphAtlas>,
}

/// The rounded-rect pill program (shares nothing with the 2D program — its own
/// SDF fragment shader). Built lazily on first overlay composite.
struct PillProgram {
    program: glow::Program,
    u_transform: glow::UniformLocation,
    u_color: glow::UniformLocation,
    u_half: glow::UniformLocation,
    u_radius: glow::UniformLocation,
    a_pos: u32,
}

/// The glyph program (shares [`VERT_SRC`]; [`GLYPH_FRAG_SRC`] fragment). Built
/// lazily on first overlay composite.
struct GlyphProgram {
    program: glow::Program,
    u_transform: glow::UniformLocation,
    u_uv_xform: glow::UniformLocation,
    u_atlas: glow::UniformLocation,
    u_color: glow::UniformLocation,
    u_text_alpha: glow::UniformLocation,
    a_pos: u32,
}

/// R8 glyph coverage atlas with a shelf packer. Grows to `max_side`×`max_side`;
/// on overflow clears all slots and re-packs the current frame's glyphs.
struct GlyphAtlas {
    texture: glow::Texture,
    width: u32,
    height: u32,
    /// `(atlas_x, atlas_y)` for glyphs already uploaded.
    slots: std::collections::HashMap<crate::image_render::GlyphKey, (u32, u32)>,
    shelf_x: u32,
    shelf_y: u32,
    shelf_h: u32,
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
            let u_overlay_alpha = gl
                .get_uniform_location(program, "u_overlay_alpha")
                .ok_or(GlError::MissingLocation("u_overlay_alpha"))?;
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
                u_overlay_alpha,
                a_pos,
                overlay_alpha: 1.0,
                quad_vbo,
                camera_tex,
                overlay_tex,
                camera_size: None,
                overlay_size: None,
                overlay_id: None,
                fbo: None,
                fbo_tex: None,
                fbo_size: None,
                ext_luma: None,
                gray_fbo: None,
                gray_fbo_tex: None,
                gray_fbo_size: None,
                ext: None,
                camera_external: None,
                present_content: PresentContent::default(),
                pill: None,
                glyph_prog: None,
                overlay_fbo: None,
                pill_fbo: None,
                glyph_atlas: None,
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

    /// Choose what [`present`](Self::present) draws — camera + overlays, or
    /// overlays only over a transparent clear (see [`PresentContent`]). The
    /// camera/screen source is independent of this; it governs only the
    /// final present composite, not the tracker readbacks.
    pub fn set_present_content(&mut self, content: PresentContent) {
        self.present_content = content;
    }

    /// Set the overlay opacity multiplier (0..1) applied to overlay draws.
    /// `1.0` (default) = the overlay's own alpha unchanged. The camera path
    /// leaves it at 1.0; the screen path can lower it independently of the
    /// (touch-capped) window alpha.
    pub fn set_overlay_alpha(&mut self, alpha: f32) {
        self.overlay_alpha = alpha.clamp(0.0, 1.0);
    }

    fn build_ext_program(&self) -> Option<ExtProgram> {
        self.build_ext_program_frag(EXT_FRAG_SRC)
    }

    fn build_ext_program_frag(&self, frag_src: &str) -> Option<ExtProgram> {
        let gl = &self.gl;
        let program = match link_program_frag(gl, frag_src) {
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

    /// Build the rounded-rect pill program if not already built.
    fn ensure_pill_program(&mut self) -> bool {
        if self.pill.is_some() {
            return true;
        }
        let gl = &self.gl;
        let program = match link_program_vert_frag(gl, PILL_VERT_SRC, PILL_FRAG_SRC) {
            Ok(p) => p,
            Err(e) => {
                log::error!("GlesRenderer: pill program link failed: {e:?}");
                return false;
            }
        };
        unsafe {
            let pill = (|| {
                Some(PillProgram {
                    u_transform: gl.get_uniform_location(program, "u_transform")?,
                    u_color: gl.get_uniform_location(program, "u_color")?,
                    u_half: gl.get_uniform_location(program, "u_half")?,
                    u_radius: gl.get_uniform_location(program, "u_radius")?,
                    a_pos: gl.get_attrib_location(program, "a_pos")?,
                    program,
                })
            })();
            self.pill = pill;
        }
        self.pill.is_some()
    }

    /// Build the glyph atlas program if not already built.
    fn ensure_glyph_program(&mut self) -> bool {
        if self.glyph_prog.is_some() {
            return true;
        }
        let gl = &self.gl;
        let program = match link_program_vert_frag(gl, VERT_SRC, GLYPH_FRAG_SRC) {
            Ok(p) => p,
            Err(e) => {
                log::error!("GlesRenderer: glyph program link failed: {e:?}");
                return false;
            }
        };
        unsafe {
            let prog = (|| {
                Some(GlyphProgram {
                    u_transform: gl.get_uniform_location(program, "u_transform")?,
                    u_uv_xform: gl.get_uniform_location(program, "u_uv_xform")?,
                    u_atlas: gl.get_uniform_location(program, "u_atlas")?,
                    u_color: gl.get_uniform_location(program, "u_color")?,
                    u_text_alpha: gl.get_uniform_location(program, "u_text_alpha")?,
                    a_pos: gl.get_attrib_location(program, "a_pos")?,
                    program,
                })
            })();
            self.glyph_prog = prog;
        }
        self.glyph_prog.is_some()
    }

    /// Ensure the R8 glyph atlas texture is allocated (1024×1024 fixed). Built once;
    /// when it overflows (repack needed) `upload_new_glyphs` clears and re-packs.
    fn ensure_glyph_atlas(&mut self) -> bool {
        if self.glyph_atlas.is_some() {
            return true;
        }
        const ATLAS_W: u32 = 1024;
        const ATLAS_H: u32 = 1024;
        let gl = &self.gl;
        let texture = match new_texture(gl) {
            Ok(t) => t,
            Err(_) => return false,
        };
        unsafe {
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            // Use NEAREST for coverage masks — subpixel sampling of a 1-byte mask
            // through a linear filter can soften coverage in GLES2 mediump float.
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MIN_FILTER,
                glow::NEAREST as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MAG_FILTER,
                glow::NEAREST as i32,
            );
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                R8 as i32,
                ATLAS_W as i32,
                ATLAS_H as i32,
                0,
                glow::RED,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(None),
            );
        }
        self.glyph_atlas = Some(GlyphAtlas {
            texture,
            width: ATLAS_W,
            height: ATLAS_H,
            slots: std::collections::HashMap::new(),
            shelf_x: 0,
            shelf_y: 0,
            shelf_h: 0,
        });
        true
    }

    /// Ensure the slot holds an RGBA8 FBO + texture at least `w×h`, **grow-only**:
    /// reused as-is when already big enough, reallocated only when the request
    /// exceeds the current allocation, never shrunk. The overlay content size
    /// changes every acquire; growing to the running max (≈ display size after the
    /// first near-full overlay) means a reallocation happens a couple of times
    /// early and then never again — vs. a per-acquire realloc of two ~18 MB
    /// textures, which was the dominant first-present cost. Returns
    /// `(fbo, tex, alloc_w, alloc_h)`; callers render into the `[0..w]×[0..h]`
    /// sub-rect and sample it back via the alloc dims.
    fn ensure_rgba_fbo(
        slot: &mut Option<(glow::Framebuffer, glow::Texture, u32, u32)>,
        gl: &glow::Context,
        w: u32,
        h: u32,
    ) -> (glow::Framebuffer, glow::Texture, u32, u32) {
        if let Some((fbo, tex, sw, sh)) = *slot {
            if sw >= w && sh >= h {
                return (fbo, tex, sw, sh);
            }
            unsafe {
                gl.delete_framebuffer(fbo);
                gl.delete_texture(tex);
            }
        }
        // Grow to the max of the request and any prior allocation so the buffer
        // only ever grows.
        let (w, h) = match *slot {
            Some((_, _, sw, sh)) => (w.max(sw), h.max(sh)),
            None => (w, h),
        };
        unsafe {
            let tex = new_texture(gl).expect("create overlay fbo texture");
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
            let fbo = gl.create_framebuffer().expect("create overlay framebuffer");
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
                "overlay FBO incomplete: status=0x{status:x}"
            );
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            *slot = Some((fbo, tex, w, h));
            (fbo, tex, w, h)
        }
    }

    /// (Re)create the R8 gray readback FBO + texture to match `w*h`.
    fn ensure_gray_fbo(&mut self, w: u32, h: u32) {
        if self.gray_fbo_size == Some((w, h)) {
            return;
        }
        let gl = &self.gl;
        unsafe {
            if let Some(t) = self.gray_fbo_tex.take() {
                gl.delete_texture(t);
            }
            let tex = new_texture(gl).expect("create gray fbo texture");
            gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                R8 as i32,
                w as i32,
                h as i32,
                0,
                glow::RED,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(None),
            );
            let fbo = match self.gray_fbo {
                Some(f) => f,
                None => {
                    let f = gl.create_framebuffer().expect("create gray framebuffer");
                    self.gray_fbo = Some(f);
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
                "gray readback FBO incomplete: status=0x{status:x}"
            );
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            self.gray_fbo_tex = Some(tex);
            self.gray_fbo_size = Some((w, h));
        }
    }

    /// Render the borrowed external camera as **luminance** into the R8 FBO at
    /// `w×h` and read it back top-down — the per-frame tracker gray, produced on
    /// the GPU so the readback transfers one channel and the CPU skips the
    /// RGBA→luma pass. `dst_to_clip` must match [`read_camera_rgba`] / the
    /// present so the gray is oriented exactly like the displayed frame.
    pub fn read_camera_gray(&mut self, w: u32, h: u32, dst_to_clip: &[f32; 9]) -> Option<Vec<u8>> {
        let (id, uv) = self.camera_external?;
        let id = NonZeroU32::new(id)?;
        if self.ext_luma.is_none() {
            self.ext_luma = self.build_ext_program_frag(EXT_LUMA_FRAG_SRC);
        }
        self.ext_luma.as_ref()?;
        self.ensure_gray_fbo(w, h);
        let cam_transform = mat3_mul(dst_to_clip, &scale(w as f32, h as f32));
        let mut flipped = vec![0u8; (w as usize) * (h as usize)];
        unsafe {
            let gl = &self.gl;
            let e = self.ext_luma.as_ref().expect("ext_luma program present");
            gl.bind_framebuffer(glow::FRAMEBUFFER, self.gray_fbo);
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
            // GL_RED is 1 byte/px; the default GL_PACK_ALIGNMENT of 4 pads each
            // row up to a 4-byte boundary, overrunning a tightly-packed w*h
            // buffer when w isn't a multiple of 4. Pack tight.
            gl.pixel_store_i32(glow::PACK_ALIGNMENT, 1);
            gl.read_pixels(
                0,
                0,
                w as i32,
                h as i32,
                glow::RED,
                glow::UNSIGNED_BYTE,
                glow::PixelPackData::Slice(Some(&mut flipped)),
            );
            gl.bind_texture(TEXTURE_EXTERNAL_OES, None);
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        }
        // glReadPixels is bottom-up; the pipeline wants top-down. Flip rows.
        let stride = w as usize;
        let mut gray = vec![0u8; flipped.len()];
        for y in 0..h as usize {
            let src = (h as usize - 1 - y) * stride;
            let dst = y * stride;
            gray[dst..dst + stride].copy_from_slice(&flipped[src..src + stride]);
        }
        Some(gray)
    }

    /// Render the borrowed external camera (canonical transform, no overlays)
    /// into the owned FBO at `w×h` and read it back as **top-down RGBA** — the
    /// canonical frame the pipeline's `LiveFrame` carries (replacing the CPU
    /// `map` + `transform_frame`). `None` if no external camera source is set.
    ///
    /// `dst_to_clip` must be the **same** transform used to present, so the
    /// frame the OCR sees is exactly what's displayed (no divergent orientation).
    pub fn read_camera_rgba(&mut self, w: u32, h: u32, dst_to_clip: &[f32; 9]) -> Option<Vec<u8>> {
        let (id, uv) = self.camera_external?;
        let id = NonZeroU32::new(id)?;
        self.ext.as_ref()?;
        self.ensure_fbo(w, h);
        let cam_transform = mat3_mul(dst_to_clip, &scale(w as f32, h as f32));
        let mut flipped = vec![0u8; (w as usize) * (h as usize) * 4];
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
                glow::PixelPackData::Slice(Some(&mut flipped)),
            );
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        }
        // glReadPixels is bottom-up; the pipeline wants top-down. Flip rows.
        let stride = (w as usize) * 4;
        let mut rgba = vec![0u8; flipped.len()];
        for y in 0..h as usize {
            let src = (h as usize - 1 - y) * stride;
            let dst = y * stride;
            rgba[dst..dst + stride].copy_from_slice(&flipped[src..src + stride]);
        }
        Some(rgba)
    }

    /// Bind a framebuffer + viewport on the renderer's own GL context. Pass
    /// `fbo_id == 0` for the default framebuffer (EGL window surface). Lets
    /// callers that don't own a separate `glow::Context` (the Android JNI
    /// shim) re-target the present output after [`read_camera_gray`] /
    /// [`read_camera_rgba`] leave their own readback FBO bound.
    pub fn bind_present_framebuffer(&self, fbo_id: u32, width: i32, height: i32) {
        unsafe {
            let fbo = NonZeroU32::new(fbo_id).map(glow::NativeFramebuffer);
            self.gl.bind_framebuffer(glow::FRAMEBUFFER, fbo);
            self.gl.viewport(0, 0, width, height);
        }
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

    /// Bake the overlay draw list into [`Self::overlay_fbo`]'s straight-alpha
    /// texture on the GPU. Two passes: (A) rounded SDF pills unioned opaque into
    /// the pill-layer FBO, then (B) that layer composited at `pill_alpha` + glyph
    /// quads sampled from the R8 atlas src-over at `text_alpha` into the overlay FBO.
    /// New glyph masks are uploaded to the atlas before the draw; the atlas clears
    /// and re-packs from scratch when it overflows. Returns false on failure.
    /// Must run on the GL thread; leaves no framebuffer bound.
    #[cfg(feature = "planar-tracker")]
    pub fn render_overlay_to_texture(
        &mut self,
        dl: &OverlayDrawList,
        pill_alpha: f32,
        text_alpha: f32,
    ) -> bool {
        if dl.bitmap_w == 0 || dl.bitmap_h == 0 {
            return false;
        }
        if !self.ensure_pill_program() || !self.ensure_glyph_program() || !self.ensure_glyph_atlas()
        {
            return false;
        }
        let (bw, bh) = (dl.bitmap_w, dl.bitmap_h);
        let os = dl.oversample.max(1e-3);
        // Render into the FBO without a y flip so the baked texture stores the top
        // row at v=0 (the convention the present path samples for the CPU canvas).
        let ndc = ndc_from_viewport_no_flip(bw as f32, bh as f32);

        // Upload any new glyph masks to the R8 atlas before the draw pass.
        let t_upload = std::time::Instant::now();
        let new_glyphs = self.upload_new_glyphs(&dl.glyphs.masks);
        let upload_ms = t_upload.elapsed().as_secs_f64() * 1000.0;

        // FBO (re)alloc: grow-only so the per-acquire size change only pays once.
        let t_alloc = std::time::Instant::now();
        let (pill_fbo, pill_tex, pa_w, pa_h) =
            Self::ensure_rgba_fbo(&mut self.pill_fbo, &self.gl, bw, bh);
        let (overlay_fbo, _overlay_tex, oa_w, oa_h) =
            Self::ensure_rgba_fbo(&mut self.overlay_fbo, &self.gl, bw, bh);
        let alloc_ms = t_alloc.elapsed().as_secs_f64() * 1000.0;
        let pill_uv_sx = bw as f32 / pa_w as f32;
        let pill_uv_sy = bh as f32 / pa_h as f32;
        let _ = (oa_w, oa_h);
        let t_passes = std::time::Instant::now();

        unsafe {
            let gl = &self.gl;
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.quad_vbo));
            gl.active_texture(glow::TEXTURE0);
            gl.disable(glow::CULL_FACE);
            gl.disable(glow::DEPTH_TEST);
            gl.disable(glow::SCISSOR_TEST);
            gl.enable(glow::BLEND);
            gl.blend_equation(glow::FUNC_ADD);
            gl.blend_func_separate(
                glow::SRC_ALPHA,
                glow::ONE_MINUS_SRC_ALPHA,
                glow::ONE,
                glow::ONE_MINUS_SRC_ALPHA,
            );

            // --- Pass A: opaque rounded pills unioned into the pill layer. ---
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(pill_fbo));
            gl.viewport(0, 0, bw as i32, bh as i32);
            gl.clear_color(0.0, 0.0, 0.0, 0.0);
            gl.clear(glow::COLOR_BUFFER_BIT);
            let pill = self.pill.as_ref().expect("pill program present");
            gl.use_program(Some(pill.program));
            gl.enable_vertex_attrib_array(pill.a_pos);
            gl.vertex_attrib_pointer_f32(pill.a_pos, 2, glow::FLOAT, false, 0, 0);
            for p in &dl.pills {
                let r = &p.rect;
                let cos = r.angle_radians.cos();
                let sin = r.angle_radians.sin();
                let cx = (r.cx - dl.origin_x) * os;
                let cy = (r.cy - dl.origin_y) * os;
                let hw = r.width * os * 0.5;
                let hh = r.height * os * 0.5;
                if hw <= 0.0 || hh <= 0.0 {
                    continue;
                }
                let radius = (hw.min(hh) * 0.5).min(12.0).max(0.0);
                let (tx, ty) = (cos, sin);
                let (px, py) = (-sin, cos);
                let c0x = cx - hw * tx - hh * px;
                let c0y = cy - hw * ty - hh * py;
                let quad_to_texel = [
                    2.0 * hw * tx,
                    2.0 * hh * px,
                    c0x,
                    2.0 * hw * ty,
                    2.0 * hh * py,
                    c0y,
                    0.0,
                    0.0,
                    1.0,
                ];
                let transform = mat3_mul(&ndc, &quad_to_texel);
                gl.uniform_matrix_3_f32_slice(
                    Some(&pill.u_transform),
                    false,
                    &to_column_major(&transform),
                );
                gl.uniform_4_f32_slice(
                    Some(&pill.u_color),
                    &[
                        p.color[0] as f32 / 255.0,
                        p.color[1] as f32 / 255.0,
                        p.color[2] as f32 / 255.0,
                        1.0,
                    ],
                );
                gl.uniform_2_f32_slice(Some(&pill.u_half), &[hw, hh]);
                gl.uniform_1_f32(Some(&pill.u_radius), radius);
                gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
            }

            // --- Pass B: pill layer at pill_alpha, then glyph quads at text_alpha. ---
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(overlay_fbo));
            gl.viewport(0, 0, bw as i32, bh as i32);
            gl.clear_color(0.0, 0.0, 0.0, 0.0);
            gl.clear(glow::COLOR_BUFFER_BIT);

            // B1: pill layer composited at pill_alpha.
            let pill_uv = to_column_major(&scale(pill_uv_sx, pill_uv_sy));
            let full = mat3_mul(&ndc, &scale(bw as f32, bh as f32));
            gl.use_program(Some(self.program));
            gl.enable_vertex_attrib_array(self.a_pos);
            gl.vertex_attrib_pointer_f32(self.a_pos, 2, glow::FLOAT, false, 0, 0);
            gl.uniform_1_i32(Some(&self.u_tex), 0);
            gl.uniform_1_f32(Some(&self.u_overlay_alpha), pill_alpha.clamp(0.0, 1.0));
            gl.uniform_matrix_3_f32_slice(Some(&self.u_transform), false, &to_column_major(&full));
            gl.uniform_matrix_3_f32_slice(Some(&self.u_uv_xform), false, &pill_uv);
            gl.bind_texture(glow::TEXTURE_2D, Some(pill_tex));
            gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);

            // B2: glyph quads from the R8 atlas, one draw call per instance.
            // Each quad is rotated in overlay-texel space by the glyph's line angle,
            // placed at (pen + rotate(left, top)), and UV-mapped to its atlas slot.
            let gp = self.glyph_prog.as_ref().expect("glyph program present");
            let atlas = self.glyph_atlas.as_ref().expect("atlas present");
            gl.use_program(Some(gp.program));
            gl.enable_vertex_attrib_array(gp.a_pos);
            gl.vertex_attrib_pointer_f32(gp.a_pos, 2, glow::FLOAT, false, 0, 0);
            gl.uniform_1_i32(Some(&gp.u_atlas), 0);
            gl.uniform_1_f32(Some(&gp.u_text_alpha), text_alpha.clamp(0.0, 1.0));
            gl.bind_texture(glow::TEXTURE_2D, Some(atlas.texture));
            let (aw, ah) = (atlas.width as f32, atlas.height as f32);
            for inst in &dl.glyphs.instances {
                let Some(&(ax, ay)) = atlas.slots.get(&inst.key) else {
                    continue;
                };
                let Some(mask) = dl.glyphs.masks.get(&inst.key) else {
                    continue;
                };
                let (w, h) = (mask.w as f32, mask.h as f32);
                let (left, top) = (mask.left as f32, mask.top as f32);
                // Unit quad → overlay-texel: rotate (left,top)+(u*w,v*h) by (cos,sin).
                // x(u,v) = pen_x + (left+u*w)*cos - (top+v*h)*sin
                // y(u,v) = pen_y + (left+u*w)*sin + (top+v*h)*cos
                let local_to_texel = [
                    w * inst.cos,
                    -h * inst.sin,
                    inst.pen_x + left * inst.cos - top * inst.sin,
                    w * inst.sin,
                    h * inst.cos,
                    inst.pen_y + left * inst.sin + top * inst.cos,
                    0.0,
                    0.0,
                    1.0,
                ];
                let transform = mat3_mul(&ndc, &local_to_texel);
                let u0 = ax as f32 / aw;
                let v0 = ay as f32 / ah;
                let uv_xform = [w / aw, 0.0, u0, 0.0, h / ah, v0, 0.0, 0.0, 1.0];
                gl.uniform_matrix_3_f32_slice(
                    Some(&gp.u_transform),
                    false,
                    &to_column_major(&transform),
                );
                gl.uniform_matrix_3_f32_slice(
                    Some(&gp.u_uv_xform),
                    false,
                    &to_column_major(&uv_xform),
                );
                gl.uniform_4_f32_slice(
                    Some(&gp.u_color),
                    &[
                        inst.color[0] as f32 / 255.0,
                        inst.color[1] as f32 / 255.0,
                        inst.color[2] as f32 / 255.0,
                        inst.color[3] as f32 / 255.0,
                    ],
                );
                gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
            }

            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        }
        let passes_ms = t_passes.elapsed().as_secs_f64() * 1000.0;
        log::info!(
            "[overlay-bake] {bw}x{bh} (alloc {pa_w}x{pa_h}) \
             upload={upload_ms:.1}ms({new_glyphs} new glyphs/{} inst) \
             alloc={alloc_ms:.1}ms passes={passes_ms:.1}ms",
            dl.glyphs.instances.len()
        );
        true
    }

    /// Upload glyph masks not yet in the atlas. If a mask doesn't fit on the current
    /// shelf, a new shelf is started; if the atlas overflows entirely, all slots are
    /// cleared and the current frame's masks are re-uploaded from scratch. Returns
    /// the count of newly uploaded masks.
    #[cfg(feature = "planar-tracker")]
    fn upload_new_glyphs(
        &mut self,
        masks: &std::collections::HashMap<
            crate::image_render::GlyphKey,
            crate::image_render::GlyphMaskData,
        >,
    ) -> usize {
        let atlas = match self.glyph_atlas.as_mut() {
            Some(a) => a,
            None => return 0,
        };
        // Collect masks that are not yet in the atlas.
        let missing: Vec<&crate::image_render::GlyphMaskData> = masks
            .values()
            .filter(|m| !atlas.slots.contains_key(&m.key))
            .collect();
        if missing.is_empty() {
            return 0;
        }
        // Try to place each missing mask. If any fails, clear the atlas and retry the
        // full current-frame mask set (which always fits if no single mask > atlas size).
        let mut placed: Vec<(&crate::image_render::GlyphMaskData, u32, u32)> = Vec::new();
        let mut overflow = false;
        for m in &missing {
            if m.w == 0 || m.h == 0 {
                continue;
            }
            match Self::atlas_place(atlas, m.w, m.h) {
                Some((x, y)) => placed.push((m, x, y)),
                None => {
                    overflow = true;
                    break;
                }
            }
        }
        if overflow {
            // Clear and re-pack all current-frame masks from scratch.
            atlas.slots.clear();
            atlas.shelf_x = 0;
            atlas.shelf_y = 0;
            atlas.shelf_h = 0;
            placed.clear();
            for m in masks.values() {
                if m.w == 0 || m.h == 0 {
                    continue;
                }
                if let Some((x, y)) = Self::atlas_place(atlas, m.w, m.h) {
                    placed.push((m, x, y));
                }
            }
        }
        let gl = &self.gl;
        let atlas = self.glyph_atlas.as_mut().expect("atlas present");
        let count = placed.len();
        unsafe {
            gl.bind_texture(glow::TEXTURE_2D, Some(atlas.texture));
            for (m, ax, ay) in &placed {
                gl.tex_sub_image_2d(
                    glow::TEXTURE_2D,
                    0,
                    *ax as i32,
                    *ay as i32,
                    m.w as i32,
                    m.h as i32,
                    glow::RED,
                    glow::UNSIGNED_BYTE,
                    glow::PixelUnpackData::Slice(Some(&m.cov)),
                );
                atlas.slots.insert(m.key, (*ax, *ay));
            }
        }
        count
    }

    /// Shelf-packer: try to place a `w×h` rect in the atlas. Advances `shelf_x`
    /// on success; starts a new shelf if the current one is full. Returns `None` when
    /// the entire atlas is exhausted (caller should clear and retry).
    fn atlas_place(atlas: &mut GlyphAtlas, w: u32, h: u32) -> Option<(u32, u32)> {
        if atlas.shelf_x + w > atlas.width {
            // New shelf.
            let new_y = atlas.shelf_y + atlas.shelf_h;
            if new_y + h > atlas.height {
                return None;
            }
            atlas.shelf_y = new_y;
            atlas.shelf_x = 0;
            atlas.shelf_h = 0;
        }
        let x = atlas.shelf_x;
        let y = atlas.shelf_y;
        atlas.shelf_x += w;
        atlas.shelf_h = atlas.shelf_h.max(h);
        Some((x, y))
    }

    /// Present the GPU-baked overlay texture ([`Self::render_overlay_to_texture`])
    /// straight into the bound EGL window surface (the screen `TextureView`): one
    /// axis-aligned quad over a transparent clear, no homography. The texture is
    /// straight-alpha; the window surface wants premultiplied (SurfaceFlinger), so
    /// the global `overlay_alpha` is applied and the blend leaves premultiplied
    /// output. `dl` supplies the same geometry the bake used. Returns false when
    /// there's no baked overlay. The caller binds the window surface + swaps.
    #[cfg(feature = "planar-tracker")]
    pub fn present_screen_overlay_fbo(
        &mut self,
        dl: &OverlayDrawList,
        canonical_w: u32,
        canonical_h: u32,
        surface_w: u32,
        surface_h: u32,
    ) -> bool {
        let Some((_, overlay_tex, alloc_w, alloc_h)) = self.overlay_fbo else {
            return false;
        };
        if alloc_w == 0 || alloc_h == 0 {
            return false;
        }
        // The overlay FBO is grow-only, so the baked content occupies only the
        // `[0..bitmap_w]×[0..bitmap_h]` sub-rect of the (larger) texture. The
        // footprint covers the used content's surface size; the uv samples just
        // that sub-rect.
        let (used_w, used_h) = (dl.bitmap_w.max(1), dl.bitmap_h.max(1));
        self.bind_present_framebuffer(0, surface_w as i32, surface_h as i32);
        let os = dl.oversample.max(1e-3);
        let sx = surface_w as f32 / canonical_w.max(1) as f32;
        let sy = surface_h as f32 / canonical_h.max(1) as f32;
        let inv_os = 1.0 / os;
        let footprint = scale(used_w as f32 * inv_os * sx, used_h as f32 * inv_os * sy);
        let to_surface = translate(dl.origin_x * sx, dl.origin_y * sy);
        let ndc = ndc_from_viewport(surface_w as f32, surface_h as f32);
        let transform = mat3_mul(&ndc, &mat3_mul(&to_surface, &footprint));
        let uv_sub = to_column_major(&scale(
            used_w as f32 / alloc_w as f32,
            used_h as f32 / alloc_h as f32,
        ));
        unsafe {
            let gl = &self.gl;
            gl.clear_color(0.0, 0.0, 0.0, 0.0);
            gl.clear(glow::COLOR_BUFFER_BIT);
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.quad_vbo));
            gl.active_texture(glow::TEXTURE0);
            gl.use_program(Some(self.program));
            gl.enable_vertex_attrib_array(self.a_pos);
            gl.vertex_attrib_pointer_f32(self.a_pos, 2, glow::FLOAT, false, 0, 0);
            gl.uniform_1_i32(Some(&self.u_tex), 0);
            gl.uniform_1_f32(Some(&self.u_overlay_alpha), self.overlay_alpha);
            gl.bind_texture(TEXTURE_EXTERNAL_OES, None);
            gl.disable(glow::CULL_FACE);
            gl.disable(glow::DEPTH_TEST);
            gl.disable(glow::SCISSOR_TEST);
            gl.enable(glow::BLEND);
            gl.blend_equation(glow::FUNC_ADD);
            gl.blend_func_separate(
                glow::SRC_ALPHA,
                glow::ONE_MINUS_SRC_ALPHA,
                glow::ONE,
                glow::ONE_MINUS_SRC_ALPHA,
            );
            gl.uniform_matrix_3_f32_slice(
                Some(&self.u_transform),
                false,
                &to_column_major(&transform),
            );
            gl.uniform_matrix_3_f32_slice(Some(&self.u_uv_xform), false, &uv_sub);
            gl.bind_texture(glow::TEXTURE_2D, Some(overlay_tex));
            gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
        }
        true
    }

    /// Composite the (already-uploaded) camera texture + overlays into the
    /// bound framebuffer. The camera upload is the caller's responsibility
    /// (via [`upload_camera_tex`](Self::upload_camera_tex)); overlay uploads
    /// stay here because they depend on the per-anchor overlay items.
    /// Draw the opaque camera passthrough quad: the base layer beneath the
    /// overlays in [`PresentContent::CameraAndOverlay`]. Either the borrowed
    /// external-OES texture (zero-copy, sampled through the canonical uv
    /// transform that carries upright/crop/flip) or the uploaded 2D texture
    /// sampled through the sensor-crop uv.
    fn draw_camera_passthrough(&self, input: &CompositeInput<'_>, dst_to_clip: &[f32; 9]) {
        let dst_w = input.dst_w as f32;
        let dst_h = input.dst_h as f32;
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
                gl.uniform_1_f32(Some(&self.u_overlay_alpha), 1.0);
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
    }

    fn draw(&mut self, input: &CompositeInput<'_>, dst_to_clip: &[f32; 9]) {
        let overlay_only = self.present_content == PresentContent::OverlayOnly;
        // Fast-clear so tiled GPUs skip loading the previous framebuffer. The
        // camera path clears to opaque black under the camera quad; overlay-only
        // clears transparent so the glyphs composite over the live content
        // behind a translucent window.
        unsafe {
            let gl = &self.gl;
            let clear_a = if overlay_only { 0.0 } else { 1.0 };
            gl.clear_color(0.0, 0.0, 0.0, clear_a);
            gl.clear(glow::COLOR_BUFFER_BIT);
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.quad_vbo));
            gl.active_texture(glow::TEXTURE0);
            gl.disable(glow::BLEND);
        }
        if !overlay_only {
            self.draw_camera_passthrough(input, dst_to_clip);
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
            gl.uniform_1_f32(Some(&self.u_overlay_alpha), self.overlay_alpha);
            // The external-camera pass left a GL_TEXTURE_EXTERNAL_OES bound on
            // unit 0. Adreno/Mali return nothing when a sampler2D reads a unit
            // that still has an external texture bound, so clear it before
            // sampling the 2D overlay texture.
            if self.camera_external.is_some() {
                gl.bind_texture(TEXTURE_EXTERNAL_OES, None);
            }
            // The scene graph hands us its pipeline state; depth test in
            // particular is left enabled and rejects the overlay quad (same z=0
            // as the camera quad under GL_LESS). Establish our own 2D state.
            gl.disable(glow::CULL_FACE);
            gl.disable(glow::DEPTH_TEST);
            gl.disable(glow::SCISSOR_TEST);
            gl.enable(glow::BLEND);
            gl.blend_equation(glow::FUNC_ADD);
            // Overlay-only output goes to a translucent window, so the alpha
            // channel matters: accumulate it with (ONE, ONE_MINUS_SRC_ALPHA)
            // while RGB stays source-over, leaving the framebuffer premultiplied
            // (rgb already ×alpha, alpha = coverage) the way SurfaceFlinger
            // blends a layer. The opaque camera path keeps the plain blend.
            if overlay_only {
                gl.blend_func_separate(
                    glow::SRC_ALPHA,
                    glow::ONE_MINUS_SRC_ALPHA,
                    glow::ONE,
                    glow::ONE_MINUS_SRC_ALPHA,
                );
            } else {
                gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);
            }
            // On the external path `read_camera_rgba` row-flips the readback to
            // give the pipeline a top-down canonical frame, so overlays are
            // authored vertically mirrored relative to the (non-flipped) on-screen
            // camera. Undo that flip in canonical space here. The CPU path uploads
            // an already-top-down frame, so its overlays need no flip.
            let overlay_to_clip = if self.camera_external.is_some() {
                let flip_y = [1.0, 0.0, 0.0, 0.0, -1.0, input.dst_h as f32, 0.0, 0.0, 1.0];
                mat3_mul(dst_to_clip, &flip_y)
            } else {
                *dst_to_clip
            };
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
                // The texture carries `bitmap_width × bitmap_height` texels but
                // only covers `.../oversample` surface units; scaling the quad to
                // the surface footprint lets the sampler read the denser texture
                // across a smaller area, so the warp upscales it less.
                let inv_os = 1.0 / item.oversample.max(1e-3);
                let sized = mat3_mul(
                    &bitmap_to_viewport,
                    &scale(
                        item.bitmap_width as f32 * inv_os,
                        item.bitmap_height as f32 * inv_os,
                    ),
                );
                let transform = mat3_mul(&overlay_to_clip, &sized);
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
            if let Some(p) = self.pill.take() {
                self.gl.delete_program(p.program);
            }
            if let Some(p) = self.glyph_prog.take() {
                self.gl.delete_program(p.program);
            }
            if let Some(a) = self.glyph_atlas.take() {
                self.gl.delete_texture(a.texture);
            }
            for (fbo, tex, _, _) in [self.overlay_fbo.take(), self.pill_fbo.take()]
                .into_iter()
                .flatten()
            {
                self.gl.delete_framebuffer(fbo);
                self.gl.delete_texture(tex);
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

/// Like [`PresentTarget`] but the camera is a borrowed external-OES texture
/// (set via [`GlesRenderer::set_camera_external`] before `process_frame`), so
/// `upload_camera` is a no-op — there's no CPU camera buffer to upload. `draw`
/// presents the external camera + overlays into the bound framebuffer.
pub struct ExternalPresentTarget<'a> {
    pub renderer: &'a mut GlesRenderer,
    pub display_xform: [f32; 9],
}

impl ComposeTarget for ExternalPresentTarget<'_> {
    fn upload_camera(&mut self, _camera: &CameraFrame<'_>) -> Result<(), CompositeError> {
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
    link_program_vert_frag(gl, VERT_SRC, frag_src)
}

fn link_program_vert_frag(
    gl: &glow::Context,
    vert_src: &str,
    frag_src: &str,
) -> Result<glow::Program, GlError> {
    unsafe {
        let program = gl.create_program().map_err(GlError::ProgramLink)?;
        let shaders = [
            compile_shader(gl, glow::VERTEX_SHADER, vert_src)?,
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
