//! Validates the GPU "Wiring" of the screen monitor: an opaque pill with a
//! 50%-alpha pinhole grid is composited over a synthetic source, and the
//! [`LatticeProbe`] shader reads the holes back and recovers the screen content
//! underneath (`screen_est = 2·raw − pill`). Then the recovered samples drive the
//! real [`ScreenMonitor`] classifier through the two cases the design targets.
//!
//! Runs headless on a surfaceless GLES3 context (Mesa llvmpipe), like
//! `gpu_compositor`.

#![cfg(feature = "gpu")]

use khronos_egl as egl;
use translator::ocr::OrientedRect;
use translator::screen_monitor::{FrameClassification, Lattice, MonitorConfig, ScreenMonitor};
use translator::screen_monitor_gpu::{LatticeProbe, PillRegion};

const PLATFORM_SURFACELESS_MESA: egl::Enum = 0x31DD;

const W: u32 = 120;
const H: u32 = 90;
const SPACING: f32 = 5.0;
const PILL_LUMA: u8 = 0; // opaque black pill, matching the screen overlay
const BG: u8 = 230; // band background between strokes (high contrast, like a page)
const INK: u8 = 20; // dark text strokes

fn make_probe() -> LatticeProbe {
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
    .expect("surfaceless display");
    lib.initialize(display).expect("eglInitialize");
    lib.bind_api(egl::OPENGL_ES_API).expect("bind GLES");
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
                egl::NONE,
            ],
        )
        .expect("choose_config")
        .expect("a matching config");
    let ctx = lib
        .create_context(
            display,
            config,
            None,
            &[egl::CONTEXT_CLIENT_VERSION, 3, egl::NONE],
        )
        .expect("create GLES3 context");
    lib.make_current(display, None, None, Some(ctx))
        .expect("make current");
    LatticeProbe::new(|name| {
        lib.get_proc_address(name)
            .map(|p| p as *const std::ffi::c_void)
            .unwrap_or(std::ptr::null())
    })
    .expect("build LatticeProbe")
}

/// The monitored band: covers rows ~[60,80] across the full width.
fn band_rect() -> OrientedRect {
    OrientedRect {
        cx: W as f32 / 2.0,
        cy: 70.0,
        width: W as f32,
        height: 20.0,
        angle_radians: 0.0,
    }
}

fn pill() -> PillRegion {
    PillRegion::from_oriented(&band_rect())
}

fn in_band(_x: u32, y: u32) -> bool {
    (60..80).contains(&y)
}

/// A source luma frame: striped dark text in the band, plus an optional override
/// closure to mutate pixels (background jitter, erased text, …).
fn source_frame(strokes: bool) -> Vec<u8> {
    let mut v = vec![BG; (W * H) as usize];
    for y in 0..H {
        for x in 0..W {
            if in_band(x, y) {
                // 4px dark ink / 12px light gap = bg-dominated "letters" (the median
                // sits on the background, so ink holes carry the weight).
                let ink = strokes && (x % 16) < 4;
                v[(y * W + x) as usize] = if ink { INK } else { BG };
            }
        }
    }
    v
}

/// Composite our opaque pill + 50%-alpha hole grid over `source`, producing the
/// frame a MediaProjection capture would hand us. Inside the pill everything is
/// the opaque pill colour except the lattice hole pixels, which are a half blend
/// of pill and screen.
fn composite_capture(source: &[u8], lat: &Lattice, pill: &PillRegion) -> Vec<u8> {
    let mut cap = source.to_vec();
    let inside =
        |x: f32, y: f32| (x - pill.cx).abs() <= pill.half_w && (y - pill.cy).abs() <= pill.half_h;
    for y in 0..H {
        for x in 0..W {
            if inside(x as f32 + 0.5, y as f32 + 0.5) {
                cap[(y * W + x) as usize] = PILL_LUMA;
            }
        }
    }
    for p in lat.points() {
        if !inside(p.x, p.y) {
            continue;
        }
        let (px, py) = (p.x as u32, p.y as u32);
        let s = source[(py * W + px) as usize] as u16;
        // round(0.5*pill + 0.5*screen)
        cap[(py * W + px) as usize] = ((PILL_LUMA as u16 + s + 1) / 2) as u8;
    }
    cap
}

fn sample_source(source: &[u8], lat: &Lattice) -> Vec<u8> {
    lat.points()
        .iter()
        .map(|p| source[((p.y as u32) * W + (p.x as u32)) as usize])
        .collect()
}

#[test]
fn shader_recovers_screen_under_the_pill() {
    let mut probe = make_probe();
    let lat = Lattice::build(W, H, SPACING);
    let pill = pill();
    let source = source_frame(true);
    let cap = composite_capture(&source, &lat, &pill);

    let recovered = probe.recover(&cap, W, H, &lat, PILL_LUMA, &[pill]);
    assert_eq!(recovered.len(), lat.len());

    // Recovery must return the screen content (≈ source) at every lattice point —
    // both under the pill (via the 2·raw−pill inversion) and outside it (raw).
    let want = sample_source(&source, &lat);
    let max_err = recovered
        .iter()
        .zip(&want)
        .map(|(r, w)| (r[0] as i32 - *w as i32).unsigned_abs())
        .max()
        .unwrap();
    eprintln!("recovery max error: {max_err}");
    assert!(
        max_err <= 2,
        "recovered screen_est diverged: max_err={max_err}"
    );
}

#[test]
fn recovered_samples_drive_the_classifier() {
    let mut probe = make_probe();
    let lat = Lattice::build(W, H, SPACING);
    let pill = pill();
    let cfg = MonitorConfig {
        warmup_frames: 4,
        hard_threshold: 110,
        // The synthetic band is sparse ink (~25% of holes), so erasing it flips ~25%;
        // keep the trip fraction below that.
        hard_frac: 0.15,
        scroll_frac: 0.7,
        scroll_min_boxes: 2,
    };

    // Baseline: recover from the clean (text present) capture.
    let base_src = source_frame(true);
    let baseline = probe.recover(
        &composite_capture(&base_src, &lat, &pill),
        W,
        H,
        &lat,
        PILL_LUMA,
        &[pill],
    );

    let holes = lat.holes_in_rect(&band_rect());
    let mut mon = ScreenMonitor::new(lat, cfg);
    mon.set_box(1, holes.clone(), &baseline);

    // Warmup + background jitter between the letters (the strokes hold), all
    // routed through the GPU recovery. Must stay Quiet.
    for f in 0..6u32 {
        let mut src = base_src.clone();
        for y in 60..80 {
            for x in 0..W {
                if (x % 16) >= 4 {
                    // a non-stroke (background) column: jitter it
                    // Moderate, temporally-coherent background motion (Δ frame-to-frame
                    // < hard_threshold) — a playing video, not a full-range strobe.
                    src[(y * W + x) as usize] = if (f + x) % 2 == 0 { 180 } else { 255 };
                }
            }
        }
        let rec = probe.recover(
            &composite_capture(&src, mon.lattice(), &pill),
            W,
            H,
            mon.lattice(),
            PILL_LUMA,
            &[pill],
        );
        assert_eq!(
            mon.observe(&rec),
            FrameClassification::Quiet,
            "frame {f}: background jitter under the pill must not trip the box"
        );
    }

    // Text erased (subtitle advanced to blank): the stroke holes now read
    // background → the box must trip, through the same GPU recovery path.
    let changed_src = source_frame(false);
    let rec = probe.recover(
        &composite_capture(&changed_src, mon.lattice(), &pill),
        W,
        H,
        mon.lattice(),
        PILL_LUMA,
        &[pill],
    );
    match mon.observe(&rec) {
        FrameClassification::BoxesChanged(ids) => assert!(ids.contains(&1), "{ids:?}"),
        other => panic!("expected the box to trip, got {other:?}"),
    }
}
