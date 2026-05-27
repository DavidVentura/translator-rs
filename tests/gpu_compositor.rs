//! Validates the GLES `GlesRenderer` against the CPU `CpuRenderer`
//! reference by driving both through the identical `CompositeInput` and
//! comparing the readback. Runs headless via a surfaceless EGL GLES2
//! context (Mesa llvmpipe in CI). Comparisons are tolerance-based, not
//! bit-exact: GPU bilinear/rounding differs from the integer CPU path by
//! sub-pixel amounts, so we assert structure + geometry within ~1px +
//! loose error bounds, plus run-to-run determinism.

#![cfg(feature = "gpu")]

use khronos_egl as egl;
use translator::gl_renderer::GlesRenderer;
use translator::live_compositor::{CompositeInput, CpuRenderer, OverlayItem, Renderer};

const PLATFORM_SURFACELESS_MESA: egl::Enum = 0x31DD;

/// Stands up a surfaceless GLES2 context and returns a `GlesRenderer`.
/// The `DynamicInstance` and context are leaked intentionally: they must
/// outlive every GL call, and the process exits at test end anyway.
fn make_gpu_renderer() -> GlesRenderer {
    let lib = unsafe { egl::DynamicInstance::<egl::EGL1_5>::load_required() }
        .expect("load libEGL (install Mesa/llvmpipe for headless GL)");
    let lib: &'static egl::DynamicInstance<egl::EGL1_5> = Box::leak(Box::new(lib));

    let display = unsafe {
        lib.get_platform_display(
            PLATFORM_SURFACELESS_MESA,
            egl::DEFAULT_DISPLAY,
            &[egl::ATTRIB_NONE],
        )
    }
    .expect("get surfaceless platform display");
    lib.initialize(display).expect("eglInitialize");
    lib.bind_api(egl::OPENGL_ES_API).expect("bind GLES API");

    let config = lib
        .choose_first_config(
            display,
            &[
                egl::SURFACE_TYPE,
                egl::PBUFFER_BIT,
                egl::RENDERABLE_TYPE,
                egl::OPENGL_ES2_BIT,
                egl::RED_SIZE,
                8,
                egl::GREEN_SIZE,
                8,
                egl::BLUE_SIZE,
                8,
                egl::ALPHA_SIZE,
                8,
                egl::NONE,
            ],
        )
        .expect("choose_config")
        .expect("a matching EGL config");

    let ctx = lib
        .create_context(
            display,
            config,
            None,
            &[egl::CONTEXT_CLIENT_VERSION, 2, egl::NONE],
        )
        .expect("create GLES2 context");
    lib.make_current(display, None, None, Some(ctx))
        .expect("make surfaceless context current");

    GlesRenderer::new(|name| {
        lib.get_proc_address(name)
            .map(|p| p as *const std::ffi::c_void)
            .unwrap_or(std::ptr::null())
    })
    .expect("build GlesRenderer")
}

fn gradient_camera(w: u32, h: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            v.push((x.wrapping_mul(4) % 256) as u8);
            v.push((y.wrapping_mul(4) % 256) as u8);
            v.push((x.wrapping_add(y).wrapping_mul(2) % 256) as u8);
            v.push(255);
        }
    }
    v
}

/// Opaque colored square centered in a fully-transparent bitmap, so the
/// overlay has both blended (transparent) and replaced (opaque) regions.
fn bordered_overlay(w: u32, h: u32, rgb: [u8; 3], border: u32) -> Vec<u8> {
    let mut v = vec![0u8; (w * h * 4) as usize];
    for y in border..h - border {
        for x in border..w - border {
            let i = ((y * w + x) * 4) as usize;
            v[i] = rgb[0];
            v[i + 1] = rgb[1];
            v[i + 2] = rgb[2];
            v[i + 3] = 255;
        }
    }
    v
}

struct Metrics {
    mae: f64,
    max_err: u8,
    pct_within_tol: f64,
}

/// RGB-only comparison (CPU forces output alpha to 255 while the GPU
/// leaves the blended alpha, and alpha is irrelevant to the displayed
/// frame).
fn compare_rgb(a: &[u8], b: &[u8], tol: u8) -> Metrics {
    assert_eq!(a.len(), b.len());
    let mut sum = 0u64;
    let mut max_err = 0u8;
    let mut within = 0u64;
    let mut count = 0u64;
    for (pa, pb) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        let mut pixel_ok = true;
        for c in 0..3 {
            let e = pa[c].abs_diff(pb[c]);
            sum += e as u64;
            max_err = max_err.max(e);
            if e > tol {
                pixel_ok = false;
            }
            count += 1;
        }
        if pixel_ok {
            within += 1;
        }
    }
    Metrics {
        mae: sum as f64 / count as f64,
        max_err,
        pct_within_tol: within as f64 / (a.len() / 4) as f64 * 100.0,
    }
}

fn run_both(input: &CompositeInput<'_>, gpu: &mut GlesRenderer) -> (Vec<u8>, Vec<u8>) {
    let bytes = (input.dst_w * input.dst_h * 4) as usize;
    let mut cpu_out = vec![0u8; bytes];
    let mut gpu_out = vec![0u8; bytes];
    CpuRenderer.composite(input, &mut cpu_out).unwrap();
    gpu.composite(input, &mut gpu_out).unwrap();
    (cpu_out, gpu_out)
}

const FULL_W: u32 = 64;
const FULL_H: u32 = 64;
const DST_W: u32 = 48;
const DST_H: u32 = 48;
const OFF_X: u32 = 8;
const OFF_Y: u32 = 8;
const IDENTITY_H: [f32; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

#[test]
fn gpu_passthrough_crop_matches_cpu() {
    // No overlay: exercises the camera crop + uv math + y-flip. A 1:1
    // blit should reproduce the CPU crop almost exactly.
    let mut gpu = make_gpu_renderer();
    let cam = gradient_camera(FULL_W, FULL_H);
    let input = CompositeInput {
        dst_w: DST_W,
        dst_h: DST_H,
        camera_rgba: &cam,
        src_full_w: FULL_W,
        src_full_h: FULL_H,
        src_offset_x: OFF_X,
        src_offset_y: OFF_Y,
        h_surface_to_viewport: &IDENTITY_H,
        items: &[],
    };
    let (cpu, gpu_out) = run_both(&input, &mut gpu);
    let m = compare_rgb(&cpu, &gpu_out, 1);
    eprintln!(
        "[passthrough] mae={:.3} max_err={} pct_within_1={:.2}",
        m.mae, m.max_err, m.pct_within_tol
    );
    assert!(
        m.max_err <= 2,
        "passthrough max channel error {}",
        m.max_err
    );
    assert!(m.mae < 0.2, "passthrough mae {}", m.mae);
}

#[test]
fn gpu_overlay_identity_h_matches_cpu() {
    // Axis-aligned overlay (no perspective): sampling is near-exact, so
    // the GPU should track the CPU blend tightly.
    let mut gpu = make_gpu_renderer();
    let cam = gradient_camera(FULL_W, FULL_H);
    let overlay = bordered_overlay(16, 16, [220, 20, 30], 3);
    let item = OverlayItem {
        bitmap_rgba: &overlay,
        bitmap_width: 16,
        bitmap_height: 16,
        bitmap_origin_surface_x: 12.0,
        bitmap_origin_surface_y: 12.0,
        row_extents: &[],
    };
    let input = CompositeInput {
        dst_w: DST_W,
        dst_h: DST_H,
        camera_rgba: &cam,
        src_full_w: FULL_W,
        src_full_h: FULL_H,
        src_offset_x: OFF_X,
        src_offset_y: OFF_Y,
        h_surface_to_viewport: &IDENTITY_H,
        items: std::slice::from_ref(&item),
    };
    let (cpu, gpu_out) = run_both(&input, &mut gpu);
    let m = compare_rgb(&cpu, &gpu_out, 2);
    eprintln!(
        "[identity overlay] mae={:.3} max_err={} pct_within_2={:.2}",
        m.mae, m.max_err, m.pct_within_tol
    );
    // The opaque interior must land: somewhere the GPU shows the overlay red.
    assert!(
        gpu_out
            .chunks_exact(4)
            .any(|p| p[0] > 180 && p[1] < 80 && p[2] < 80),
        "overlay color absent in GPU output"
    );
    assert!(
        m.pct_within_tol > 98.0,
        "identity overlay agreement {:.2}%",
        m.pct_within_tol
    );
}

#[test]
fn gpu_overlay_perspective_h_matches_cpu() {
    // Perspective warp: sub-pixel and edge differences are expected, so
    // bound the bulk error loosely and require most pixels to agree.
    let mut gpu = make_gpu_renderer();
    let cam = gradient_camera(FULL_W, FULL_H);
    let overlay = bordered_overlay(20, 20, [40, 210, 90], 4);
    let item = OverlayItem {
        bitmap_rgba: &overlay,
        bitmap_width: 20,
        bitmap_height: 20,
        bitmap_origin_surface_x: 10.0,
        bitmap_origin_surface_y: 8.0,
        row_extents: &[],
    };
    let h = [1.05, -0.03, 1.5, 0.02, 0.97, -0.7, 1.0e-3, -5.0e-4, 1.0];
    let input = CompositeInput {
        dst_w: DST_W,
        dst_h: DST_H,
        camera_rgba: &cam,
        src_full_w: FULL_W,
        src_full_h: FULL_H,
        src_offset_x: OFF_X,
        src_offset_y: OFF_Y,
        h_surface_to_viewport: &h,
        items: std::slice::from_ref(&item),
    };
    let (cpu, gpu_out) = run_both(&input, &mut gpu);
    let m = compare_rgb(&cpu, &gpu_out, 8);
    eprintln!(
        "[perspective overlay] mae={:.3} max_err={} pct_within_8={:.2}",
        m.mae, m.max_err, m.pct_within_tol
    );
    assert!(m.mae < 3.0, "perspective mae {:.3}", m.mae);
    assert!(
        m.pct_within_tol > 95.0,
        "perspective agreement {:.2}% (edge band should be the only disagreement)",
        m.pct_within_tol
    );
}

#[test]
fn gpu_render_is_deterministic() {
    let mut gpu = make_gpu_renderer();
    let cam = gradient_camera(FULL_W, FULL_H);
    let overlay = bordered_overlay(18, 18, [10, 120, 240], 2);
    let item = OverlayItem {
        bitmap_rgba: &overlay,
        bitmap_width: 18,
        bitmap_height: 18,
        bitmap_origin_surface_x: 9.0,
        bitmap_origin_surface_y: 11.0,
        row_extents: &[],
    };
    let h = [1.02, -0.01, 0.8, 0.015, 0.99, -0.4, 6.0e-4, -3.0e-4, 1.0];
    let input = CompositeInput {
        dst_w: DST_W,
        dst_h: DST_H,
        camera_rgba: &cam,
        src_full_w: FULL_W,
        src_full_h: FULL_H,
        src_offset_x: OFF_X,
        src_offset_y: OFF_Y,
        h_surface_to_viewport: &h,
        items: std::slice::from_ref(&item),
    };
    let bytes = (DST_W * DST_H * 4) as usize;
    let mut a = vec![0u8; bytes];
    let mut b = vec![0u8; bytes];
    gpu.composite(&input, &mut a).unwrap();
    gpu.composite(&input, &mut b).unwrap();
    assert_eq!(a, b, "GPU composite is not deterministic for fixed input");
}

#[test]
fn gpu_display_transform_rotates_output() {
    // The display-xform path `present` uses: a 180° dst→clip transform
    // must produce the base composite reversed in both axes. Validates
    // that the transform plumbed into `draw` actually orients the output.
    let mut gpu = make_gpu_renderer();
    let cam = gradient_camera(FULL_W, FULL_H);
    let overlay = bordered_overlay(16, 16, [220, 20, 30], 3);
    let item = OverlayItem {
        bitmap_rgba: &overlay,
        bitmap_width: 16,
        bitmap_height: 16,
        bitmap_origin_surface_x: 12.0,
        bitmap_origin_surface_y: 8.0,
        row_extents: &[],
    };
    let input = CompositeInput {
        dst_w: DST_W,
        dst_h: DST_H,
        camera_rgba: &cam,
        src_full_w: FULL_W,
        src_full_h: FULL_H,
        src_offset_x: OFF_X,
        src_offset_y: OFF_Y,
        h_surface_to_viewport: &IDENTITY_H,
        items: std::slice::from_ref(&item),
    };
    let (w, h) = (DST_W as f32, DST_H as f32);
    let ndc = [2.0 / w, 0.0, -1.0, 0.0, -2.0 / h, 1.0, 0.0, 0.0, 1.0];
    // 180°: flip both clip axes of `ndc`.
    let rot180 = [-2.0 / w, 0.0, 1.0, 0.0, 2.0 / h, -1.0, 0.0, 0.0, 1.0];

    let bytes = (DST_W * DST_H * 4) as usize;
    let mut base = vec![0u8; bytes];
    let mut rotated = vec![0u8; bytes];
    gpu.render_to_buffer(&input, &ndc, &mut base).unwrap();
    gpu.render_to_buffer(&input, &rot180, &mut rotated).unwrap();

    let stride = (DST_W * 4) as usize;
    let mut mism = 0u64;
    for y in 0..DST_H as usize {
        for x in 0..DST_W as usize {
            let r = y * stride + x * 4;
            let b = (DST_H as usize - 1 - y) * stride + (DST_W as usize - 1 - x) * 4;
            for c in 0..3 {
                if rotated[r + c].abs_diff(base[b + c]) > 4 {
                    mism += 1;
                }
            }
        }
    }
    let pct = mism as f64 / (DST_W * DST_H * 3) as f64 * 100.0;
    eprintln!("[display xform 180] mismatch {pct:.2}%");
    assert!(
        pct < 3.0,
        "180° transform should reverse the base composite; {pct:.2}% mismatch"
    );
}
