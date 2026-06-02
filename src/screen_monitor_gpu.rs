//! GPU readback for the per-box screen monitor (the "Wiring" the classifier in
//! [`crate::screen_monitor`] consumes): given the captured frame — which on
//! MediaProjection / accessibility-screenshot includes our own opaque pills — a
//! shader samples the 50%-alpha pinhole positions, recovers the screen content
//! underneath (`screen_est = 2·raw − pill`), and reads back one luma byte per
//! lattice point.
//!
//! This is the piece that makes the monitor work on real captures: we never see
//! the text under a pill directly, only through the holes, so the recovery is
//! done where the frame already lives — as a texture on the GPU — and only the
//! small lattice grid is read back, not the whole frame.
//!
//! Recovery only (no per-box reduction yet): it returns the full recovered
//! lattice in `Lattice::points()` order so the existing CPU classifier can run on
//! it unchanged. The per-box atomic reduction (GLES 3.1 compute) is a later
//! optimization that would replace the readback with a handful of per-box scores.

use std::ffi::c_void;

use glow::HasContext;

use crate::ocr::OrientedRect;
use crate::screen_monitor::Lattice;

/// `GL_R8` sized internal format (not in glow's constant set).
const R8: u32 = 0x8229;
/// Max pills the recovery shader handles in one pass (uniform array bound).
/// Matches the device recovery (`gl_renderer::RecProgram`, `REC_MAX_PILLS = 64`)
/// so the host stand-in monitors the same number of boxes as prod.
const MAX_PILLS: usize = 64;

/// Vertex shader: a single full-screen triangle from `gl_VertexID`, no buffers.
const PROBE_VERT_SRC: &str = r#"#version 300 es
void main() {
    float x = (gl_VertexID == 2) ? 3.0 : -1.0;
    float y = (gl_VertexID == 1) ? 3.0 : -1.0;
    gl_Position = vec4(x, y, 0.0, 1.0);
}
"#;

/// Fragment shader: each output texel is one lattice point. Sample the captured
/// frame at that point; if a pill covers it, invert the 50% blend to recover the
/// screen underneath, else pass the raw value through.
const PROBE_FRAG_SRC: &str = r#"#version 300 es
precision highp float;
uniform sampler2D u_captured;
uniform vec2 u_tex_size;   // captured texture (w, h)
uniform float u_spacing;   // lattice pitch
uniform float u_origin;    // first-point centre offset (spacing/2)
uniform float u_pill_luma; // overlay pill luma (0..255)
uniform int u_pill_count;
uniform vec4 u_pills[64];   // (cx, cy, half_w, half_h) in canonical coords
out vec4 frag;
void main() {
    float col = floor(gl_FragCoord.x);
    float row = floor(gl_FragCoord.y);
    float wx = u_origin + col * u_spacing;
    float wy = u_origin + row * u_spacing;
    // Sample the CENTRE of the hole's texel, not its corner: at even lattice
    // spacing wx is an integer (a texel boundary) and `wx/size` rounds to the
    // neighbouring opaque-pill texel, reading a constant. floor()+0.5 lands dead
    // on the hole pixel (the device recovery centres for the same reason).
    vec2 uv = (floor(vec2(wx, wy)) + 0.5) / u_tex_size;
    float raw = texture(u_captured, uv).r * 255.0;
    float rec = raw;
    for (int i = 0; i < u_pill_count; i++) {
        vec4 p = u_pills[i];
        if (abs(wx - p.x) <= p.z && abs(wy - p.y) <= p.w) {
            rec = clamp(2.0 * raw - u_pill_luma, 0.0, 255.0);
            break;
        }
    }
    frag = vec4(rec / 255.0, 0.0, 0.0, 1.0);
}
"#;

/// Axis-aligned footprint of a pill, in canonical coords. The recovery only needs
/// "is this lattice point occluded by a pill," so an oriented pill is reduced to
/// its enclosing box (a slight over-cover at the corners, harmless since corner
/// holes outside the real box aren't in any monitored set).
#[derive(Debug, Clone, Copy)]
pub struct PillRegion {
    pub cx: f32,
    pub cy: f32,
    pub half_w: f32,
    pub half_h: f32,
}

impl PillRegion {
    pub fn from_oriented(r: &OrientedRect) -> Self {
        let (c, s) = (r.angle_radians.cos().abs(), r.angle_radians.sin().abs());
        let hw = r.width * 0.5;
        let hh = r.height * 0.5;
        PillRegion {
            cx: r.cx,
            cy: r.cy,
            half_w: hw * c + hh * s,
            half_h: hw * s + hh * c,
        }
    }
}

struct Sized2d {
    tex: glow::Texture,
    w: u32,
    h: u32,
}

pub struct LatticeProbe {
    gl: glow::Context,
    program: glow::Program,
    vao: glow::VertexArray,
    u_captured: glow::UniformLocation,
    u_tex_size: glow::UniformLocation,
    u_spacing: glow::UniformLocation,
    u_origin: glow::UniformLocation,
    u_pill_luma: glow::UniformLocation,
    u_pill_count: glow::UniformLocation,
    u_pills: glow::UniformLocation,
    captured: Option<Sized2d>,
    target: Option<Sized2d>,
    fbo: glow::Framebuffer,
}

impl LatticeProbe {
    /// Build against a current GL context via its proc loader (same shape as
    /// `GlesRenderer::new`). Requires a GLES 3.0+ / GL 3.0+ context.
    pub fn new<F>(loader: F) -> Result<Self, String>
    where
        F: FnMut(&str) -> *const c_void,
    {
        let gl = unsafe { glow::Context::from_loader_function(loader) };
        unsafe {
            let program = link_program(&gl, PROBE_VERT_SRC, PROBE_FRAG_SRC)?;
            let vao = gl
                .create_vertex_array()
                .map_err(|e| format!("create vao: {e}"))?;
            let uniform = |name: &str| {
                gl.get_uniform_location(program, name)
                    .ok_or_else(|| format!("missing uniform {name}"))
            };
            let u_captured = uniform("u_captured")?;
            let u_tex_size = uniform("u_tex_size")?;
            let u_spacing = uniform("u_spacing")?;
            let u_origin = uniform("u_origin")?;
            let u_pill_luma = uniform("u_pill_luma")?;
            let u_pill_count = uniform("u_pill_count")?;
            let u_pills = uniform("u_pills[0]")?;
            let fbo = gl
                .create_framebuffer()
                .map_err(|e| format!("create fbo: {e}"))?;
            Ok(LatticeProbe {
                gl,
                program,
                vao,
                u_captured,
                u_tex_size,
                u_spacing,
                u_origin,
                u_pill_luma,
                u_pill_count,
                u_pills,
                captured: None,
                target: None,
                fbo,
            })
        }
    }

    /// Recover `screen_est` at every lattice point of `captured` (an `w×h` single
    /// channel luma frame that already contains our overlay). `pill_luma` is the
    /// pill colour we drew; `pills` are the occluding footprints. Returns one byte
    /// per lattice point in `Lattice::points()` order.
    pub fn recover(
        &mut self,
        captured_luma: &[u8],
        w: u32,
        h: u32,
        lat: &Lattice,
        pill_luma: u8,
        pills: &[PillRegion],
    ) -> Vec<u8> {
        assert_eq!(captured_luma.len(), (w * h) as usize, "captured size");
        let (cols, rows) = (lat.cols(), lat.rows());
        let n = pills.len().min(MAX_PILLS);
        let mut flat = Vec::with_capacity(n * 4);
        for p in &pills[..n] {
            flat.extend_from_slice(&[p.cx, p.cy, p.half_w, p.half_h]);
        }
        let gl = &self.gl;
        let mut out = vec![0u8; (cols * rows) as usize];
        unsafe {
            gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
            let captured = ensure_r8(gl, &mut self.captured, w, h, Some(captured_luma));
            let target = ensure_r8(gl, &mut self.target, cols, rows, None);

            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.fbo));
            gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(target),
                0,
            );
            gl.viewport(0, 0, cols as i32, rows as i32);
            gl.use_program(Some(self.program));
            gl.bind_vertex_array(Some(self.vao));
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(captured));
            gl.uniform_1_i32(Some(&self.u_captured), 0);
            gl.uniform_2_f32(Some(&self.u_tex_size), w as f32, h as f32);
            gl.uniform_1_f32(Some(&self.u_spacing), lat.spacing() as f32);
            gl.uniform_1_f32(Some(&self.u_origin), lat.origin());
            gl.uniform_1_f32(Some(&self.u_pill_luma), pill_luma as f32);
            gl.uniform_1_i32(Some(&self.u_pill_count), n as i32);
            if !flat.is_empty() {
                gl.uniform_4_f32_slice(Some(&self.u_pills), &flat);
            }
            gl.draw_arrays(glow::TRIANGLES, 0, 3);

            gl.pixel_store_i32(glow::PACK_ALIGNMENT, 1);
            gl.read_pixels(
                0,
                0,
                cols as i32,
                rows as i32,
                glow::RED,
                glow::UNSIGNED_BYTE,
                glow::PixelPackData::Slice(Some(&mut out)),
            );
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        }
        out
    }
}

/// (Re)create an `R8` texture sized `w×h`, optionally uploading `data`. Reused
/// across calls when the size is unchanged.
unsafe fn ensure_r8(
    gl: &glow::Context,
    slot: &mut Option<Sized2d>,
    w: u32,
    h: u32,
    data: Option<&[u8]>,
) -> glow::Texture {
    let reuse = matches!(slot, Some(s) if s.w == w && s.h == h);
    if !reuse {
        if let Some(s) = slot.take() {
            unsafe { gl.delete_texture(s.tex) };
        }
        let tex = unsafe { gl.create_texture() }.expect("create texture");
        unsafe {
            gl.bind_texture(glow::TEXTURE_2D, Some(tex));
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
        }
        *slot = Some(Sized2d { tex, w, h });
    }
    let tex = slot.as_ref().expect("slot filled").tex;
    unsafe {
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
            glow::PixelUnpackData::Slice(data),
        );
    }
    tex
}

fn link_program(gl: &glow::Context, vert: &str, frag: &str) -> Result<glow::Program, String> {
    unsafe {
        let program = gl
            .create_program()
            .map_err(|e| format!("create program: {e}"))?;
        let vs = compile(gl, glow::VERTEX_SHADER, vert)?;
        let fs = compile(gl, glow::FRAGMENT_SHADER, frag)?;
        gl.attach_shader(program, vs);
        gl.attach_shader(program, fs);
        gl.link_program(program);
        if !gl.get_program_link_status(program) {
            return Err(gl.get_program_info_log(program));
        }
        gl.detach_shader(program, vs);
        gl.detach_shader(program, fs);
        gl.delete_shader(vs);
        gl.delete_shader(fs);
        Ok(program)
    }
}

fn compile(gl: &glow::Context, kind: u32, src: &str) -> Result<glow::Shader, String> {
    unsafe {
        let shader = gl
            .create_shader(kind)
            .map_err(|e| format!("create shader: {e}"))?;
        gl.shader_source(shader, src);
        gl.compile_shader(shader);
        if !gl.get_shader_compile_status(shader) {
            return Err(gl.get_shader_info_log(shader));
        }
        Ok(shader)
    }
}
