//! Plane-pose recovery from a planar-scene homography.
//!
//! Given `H: anchor → view` from the planar tracker and the camera's
//! intrinsics `K`, this module factors `H` into `(R, t, n)` — the
//! camera rotation, scaled translation, and plane normal — and helps
//! pick the physically valid candidate via gravity / surface-kind
//! priors.
//!
//! Phase 1 scope (per FUTURE_ANCHOR_RECTIFICATION.md): library only.
//! No pipeline wiring. The public surface is:
//!
//! - `compose_homography(R, t_over_d, n, K) -> H` — forward direction,
//!   used by tests to build synthetic warps from known ground truth.
//! - `decompose_homography(H, K) -> Vec<PlanePose>` — Faugeras's
//!   analytical decomposition; up to 4 candidates.
//! - `disambiguate(candidates, gravity, kind) -> Option<PlanePose>` —
//!   pick the right one with a gravity prior + surface-kind hint.
//! - `decomposition_residual` / `is_pose_plausible` — sanity helpers.
//!
//! All matrices are row-major `[f32; 9]`, matching `homography.rs`.

use crate::homography::{invert, mat3_mul};
use crate::imu_prior::CameraIntrinsics;

/// One decomposition candidate. The relationship is
/// `H = K · (R + t_over_d · nᵀ) · K⁻¹`.
///
/// - `rotation`: camera1 → camera2 rotation, row-major 3×3.
/// - `t_over_d`: camera translation expressed in source-camera frame,
///   divided by the source-camera-to-plane distance `d`. Unitless.
/// - `normal`: unit plane normal in source-camera frame, pointing
///   toward the camera (i.e. `normal · z_axis < 0` for a plane in
///   front of the camera, since +z is the optical axis).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlanePose {
    pub rotation: [f32; 9],
    pub t_over_d: [f32; 3],
    pub normal: [f32; 3],
}

/// Hint for disambiguation: what shape of surface are we trying to
/// rectify? Used together with a gravity prior.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceKind {
    /// Roughly horizontal (book on a table, magazine on a desk).
    /// Normal aligns with gravity.
    Horizontal,
    /// Roughly vertical (sign, poster, monitor). Normal is
    /// perpendicular to gravity.
    Vertical,
    /// No orientation prior; rely on "normal toward camera" only.
    Unknown,
}

// ---------------------------------------------------------------------
// Forward: build H from a known plane pose. Used by tests.
// ---------------------------------------------------------------------

/// Construct the 3×3 image-space homography corresponding to the given
/// plane pose and intrinsics. `H = K · (R + t_over_d · nᵀ) · K⁻¹`.
pub fn compose_homography(pose: &PlanePose, k: &CameraIntrinsics) -> [f32; 9] {
    let outer = outer_product(&pose.t_over_d, &pose.normal);
    let mut middle = pose.rotation;
    for i in 0..9 {
        middle[i] += outer[i];
    }
    let k_mat = k.k();
    let k_inv = k.k_inv();
    let tmp = mat3_mul(&k_mat, &middle);
    mat3_mul(&tmp, &k_inv)
}

// ---------------------------------------------------------------------
// Decomposition (Faugeras / Hartley-Zisserman Algorithm 15.1).
// ---------------------------------------------------------------------

/// Decompose `H` into the up-to-4 physically-valid `(R, t/d, n)`
/// candidates. Filters out solutions with negative depth or normal
/// pointing away from the camera. Returns an empty vec if `H` is
/// degenerate (pure rotation, or singular).
pub fn decompose_homography(h: &[f32; 9], k: &CameraIntrinsics) -> Vec<PlanePose> {
    let k_mat = k.k();
    let k_inv = k.k_inv();
    let tmp = mat3_mul(&k_inv, h);
    let h_norm_raw = mat3_mul(&tmp, &k_mat);

    let (_u, sigma, v) = match svd3(&h_norm_raw) {
        Some(decomp) => decomp,
        None => return Vec::new(),
    };
    let h_norm = scale_mat3(&h_norm_raw, 1.0 / sigma[1]);
    let s = [sigma[0] / sigma[1], 1.0, sigma[2] / sigma[1]];

    if (s[0] - s[2]).abs() < 1e-6 {
        return Vec::new();
    }

    let v1 = col(&v, 0);
    let v2 = col(&v, 1);
    let v3 = col(&v, 2);

    let s1_sq_minus_s3_sq = s[0] * s[0] - s[2] * s[2];
    let s1_sq_minus_s2_sq = (s[0] * s[0] - 1.0).max(0.0);
    let s2_sq_minus_s3_sq = (1.0 - s[2] * s[2]).max(0.0);

    let denom = s1_sq_minus_s3_sq.sqrt();
    let a = s2_sq_minus_s3_sq.sqrt();
    let b = s1_sq_minus_s2_sq.sqrt();

    let u1 = unit_combo(&v1, a / denom, &v3, b / denom);
    let u2 = unit_combo(&v1, a / denom, &v3, -b / denom);

    let mut out = Vec::with_capacity(8);
    for &u_choice in &[&u1, &u2] {
        let big_u = cols_to_mat([&v2, u_choice, &cross(&v2, u_choice)]);
        let hv2 = mat3_vec(&h_norm, &v2);
        let hu = mat3_vec(&h_norm, u_choice);
        let big_w = cols_to_mat([&hv2, &hu, &cross(&hv2, &hu)]);
        let big_u_t = transpose3(&big_u);
        let r = mat3_mul(&big_w, &big_u_t);
        let n = cross(&v2, u_choice);

        let r_n = mat3_vec(&r, &n);
        let mut t = [0.0; 3];
        for i in 0..3 {
            let hn_i = h_norm[i * 3] * n[0] + h_norm[i * 3 + 1] * n[1] + h_norm[i * 3 + 2] * n[2];
            t[i] = hn_i - r_n[i];
        }

        for sign in &[1.0_f32, -1.0_f32] {
            let signed_n = scale_vec3(&n, *sign);
            let signed_t = scale_vec3(&t, *sign);
            if signed_n[2] >= 0.0 {
                continue;
            }
            out.push(PlanePose {
                rotation: r,
                t_over_d: signed_t,
                normal: signed_n,
            });
        }
    }
    out
}

/// Squared Frobenius error between `H` and the homography rebuilt
/// from `pose` and `k`. Used as a cheap sanity check.
pub fn decomposition_residual(h: &[f32; 9], pose: &PlanePose, k: &CameraIntrinsics) -> f32 {
    let rebuilt = compose_homography(pose, k);
    let h_scale = frobenius(h).max(1e-9);
    let rebuilt_scaled = scale_mat3(&rebuilt, frobenius(h) / frobenius(&rebuilt).max(1e-9));
    let mut acc = 0.0_f32;
    for i in 0..9 {
        let d = h[i] - rebuilt_scaled[i];
        acc += d * d;
    }
    acc.sqrt() / h_scale
}

/// Build the image-space homography that resamples the source camera
/// image into a fronto-parallel canonical view of the plane defined
/// by `pose.normal`. Apply this to a source pixel via `project(H_rect,
/// x, y)` to find the rectified-frame pixel where it should land.
///
/// The construction:
/// 1. The source camera sees a 3D point on the plane at depth
///    `d / (n · ray_dir)`, where `ray_dir = K⁻¹·[u,v,1]ᵀ`. So source
///    pixel `(u,v)` corresponds to the plane point
///    `d · ray_dir / (n · ray_dir)`.
/// 2. A virtual fronto-parallel camera looking along `-n` sees the
///    plane axis-aligned. Choose its in-plane basis `(b1, b2)` and
///    project the 3D point onto that basis to get rectified pixel
///    coords.
///
/// The resulting H_rect is `K_virtual · [b1, b2, -n]ᵀ · K⁻¹` (up
/// to scale). `out_scale_px` sets the pixel density of the rectified
/// image (pixels per unit at the source's depth).
///
/// `out_centre_px` places the rectified-frame origin in output
/// pixel coords; typically set this to half the desired output
/// width/height so the centre of the source's text region lands
/// near the rectified-frame centre.
///
/// Returns `None` if `pose.normal` is degenerate (zero, or aligned
/// with the world-up basis we use to construct the in-plane axes).
pub fn rectification_matrix(
    pose: &PlanePose,
    k: &CameraIntrinsics,
    out_scale_px: f32,
    out_centre_px: (f32, f32),
) -> Option<[f32; 9]> {
    let n = normalize3(&pose.normal);
    if (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).abs() < 0.5 {
        return None;
    }
    // Pick an in-plane basis. The two basis vectors `b1`, `b2` span
    // the plane (both perpendicular to n) and define which way is
    // "up" / "right" in the rectified image. Use a stable
    // construction that doesn't collapse for typical normals.
    //
    // Anchor "up" against world-Y (camera's +y, image-down) projected
    // onto the plane. If `n` is nearly parallel to +y (the plane is
    // viewed edge-on, near-degenerate), fall back to anchoring
    // against world-X.
    let reference = if n[1].abs() < 0.9 {
        [0.0_f32, 1.0, 0.0]
    } else {
        [1.0_f32, 0.0, 0.0]
    };
    let b2_raw = [
        reference[0] - reference[0] * 0.0 - n[0] * dot3(&reference, &n),
        reference[1] - n[1] * dot3(&reference, &n),
        reference[2] - n[2] * dot3(&reference, &n),
    ];
    let b2 = normalize3(&b2_raw);
    if dot3(&b2, &b2) < 0.5 {
        return None;
    }
    let b1 = cross(&b2, &n);

    // Build [b1, b2, -n] as rows of the world→plane-frame rotation.
    // Source pixel (u,v) → ray d = K⁻¹·[u,v,1].
    // Plane point P = d / (n·d) (set plane at depth 1).
    // Rectified coords (X, Y) = (b1·P, b2·P) · out_scale_px + centre.
    // Equivalently, build H so that H · [u,v,1]ᵀ ~ [X, Y, 1] up to
    // scale. The trick: substitute P = d/(n·d), so
    //   X = (b1·d) / (n·d) · scale + cx
    //   Y = (b2·d) / (n·d) · scale + cy
    //   1 = (n·d) / (n·d)
    // which means H_rect = M · K⁻¹, with M's rows:
    //   [scale·b1ᵀ + cx·nᵀ;  scale·b2ᵀ + cy·nᵀ;  nᵀ].
    let (cx, cy) = out_centre_px;
    let m = [
        out_scale_px * b1[0] - cx * n[0],
        out_scale_px * b1[1] - cx * n[1],
        out_scale_px * b1[2] - cx * n[2],
        out_scale_px * b2[0] - cy * n[0],
        out_scale_px * b2[1] - cy * n[1],
        out_scale_px * b2[2] - cy * n[2],
        -n[0],
        -n[1],
        -n[2],
    ];
    Some(mat3_mul(&m, &k.k_inv()))
}

fn dot3(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Inverse-sample `src` (an RGBA buffer, row-major, 4 bytes per pixel)
/// into a `dst_w × dst_h` rectified RGBA. For each destination pixel,
/// projects through `h_dst_to_src` to find the source pixel and
/// bilinear-samples. Pixels whose source coords land outside the
/// source bounds are written as transparent black.
///
/// Caller computes `h_dst_to_src = invert(H_rect)` where `H_rect`
/// is the forward "source pixel → rectified pixel" homography from
/// [`rectification_matrix`]. Doing the inverse caller-side keeps
/// this function pure pixel math, decoupled from the homography
/// builder.
pub fn resample_rgba_through(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    h_dst_to_src: &[f32; 9],
) -> Vec<u8> {
    let mut dst = vec![0u8; (dst_w as usize) * (dst_h as usize) * 4];
    if src.len() < (src_w as usize) * (src_h as usize) * 4 || dst_w == 0 || dst_h == 0 {
        return dst;
    }
    let src_stride = (src_w as usize) * 4;
    let src_w_max = src_w as f32 - 1.0;
    let src_h_max = src_h as f32 - 1.0;
    for dy in 0..dst_h {
        for dx in 0..dst_w {
            let u = dx as f32 + 0.5;
            let v = dy as f32 + 0.5;
            let denom = h_dst_to_src[6] * u + h_dst_to_src[7] * v + h_dst_to_src[8];
            if denom.abs() < 1e-6 {
                continue;
            }
            let sx = (h_dst_to_src[0] * u + h_dst_to_src[1] * v + h_dst_to_src[2]) / denom;
            let sy = (h_dst_to_src[3] * u + h_dst_to_src[4] * v + h_dst_to_src[5]) / denom;
            let sxc = sx - 0.5;
            let syc = sy - 0.5;
            if sxc < 0.0 || syc < 0.0 || sxc > src_w_max || syc > src_h_max {
                continue;
            }
            let x0 = sxc.floor() as i32;
            let y0 = syc.floor() as i32;
            let x1 = (x0 + 1).min(src_w as i32 - 1);
            let y1 = (y0 + 1).min(src_h as i32 - 1);
            let fx = sxc - x0 as f32;
            let fy = syc - y0 as f32;
            let w00 = (1.0 - fx) * (1.0 - fy);
            let w01 = fx * (1.0 - fy);
            let w10 = (1.0 - fx) * fy;
            let w11 = fx * fy;
            let i00 = (y0 as usize) * src_stride + (x0 as usize) * 4;
            let i01 = (y0 as usize) * src_stride + (x1 as usize) * 4;
            let i10 = (y1 as usize) * src_stride + (x0 as usize) * 4;
            let i11 = (y1 as usize) * src_stride + (x1 as usize) * 4;
            let dst_idx = ((dy as usize) * (dst_w as usize) + (dx as usize)) * 4;
            for c in 0..4 {
                let v = w00 * src[i00 + c] as f32
                    + w01 * src[i01 + c] as f32
                    + w10 * src[i10 + c] as f32
                    + w11 * src[i11 + c] as f32;
                dst[dst_idx + c] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    dst
}

/// True if the candidate pose passes basic physical-validity checks.
pub fn is_pose_plausible(pose: &PlanePose, h: &[f32; 9], k: &CameraIntrinsics) -> bool {
    if pose.normal[2] >= 0.0 {
        return false;
    }
    if !is_rotation_matrix(&pose.rotation) {
        return false;
    }
    decomposition_residual(h, pose, k) < 0.05
}

// ---------------------------------------------------------------------
// Disambiguation.
// ---------------------------------------------------------------------

/// Result of multi-frame disambiguation: the picked pose plus the
/// **mean per-frame disagreement** of the chosen normal with the
/// closest matching normal in every other frame (in radians, ≥ 0).
/// A small value means cross-frame consistency was good; a large
/// value means the frames disagreed and the rectification is
/// suspect. Sanity-check this against a threshold before committing
/// to a rectification.
#[derive(Clone, Copy, Debug)]
pub struct DisambiguationResult {
    pub pose: PlanePose,
    pub mean_disagreement_radians: f32,
}

/// Multi-frame disambiguation: pick the candidate whose normal is
/// most consistent across multiple frames' decompositions of the
/// same plane. The plane normal in the anchor's coordinate system is
/// **fixed** (the anchor frame doesn't move; the plane is fixed in
/// world) — so across frames, the *true* candidate's normal is the
/// same; the twisted-pair's normal differs from frame to frame.
/// Voting on consistency is robust.
///
/// `per_frame` is `[frame_candidates_0, frame_candidates_1, ...]`,
/// each typically with 2 entries. Returns the frame-0 candidate
/// (i.e. caller's reference frame) whose normal best matches a
/// candidate in every other frame, plus the mean disagreement angle
/// to the matched normals. `None` if any frame produced no
/// candidates.
///
/// This is the **primary** disambiguator; the single-frame
/// `disambiguate_with_priors` below is a fallback for when only one
/// frame is available, and is best-effort.
pub fn disambiguate_across_frames(per_frame: &[Vec<PlanePose>]) -> Option<DisambiguationResult> {
    if per_frame.is_empty() {
        return None;
    }
    let frame0 = per_frame.first()?;
    if frame0.is_empty() {
        return None;
    }
    if per_frame.iter().any(|f| f.is_empty()) {
        return None;
    }
    let other_count = per_frame.len() - 1;
    let mut best: Option<(PlanePose, f32)> = None;
    for c0 in frame0 {
        let mut total_disagreement = 0.0_f32;
        for other in &per_frame[1..] {
            let mut min_angle = f32::INFINITY;
            for c_other in other {
                let dot = c0.normal[0] * c_other.normal[0]
                    + c0.normal[1] * c_other.normal[1]
                    + c0.normal[2] * c_other.normal[2];
                let angle = dot.clamp(-1.0, 1.0).acos();
                if angle < min_angle {
                    min_angle = angle;
                }
            }
            total_disagreement += min_angle;
        }
        match best {
            None => best = Some((*c0, total_disagreement)),
            Some((_, s)) if total_disagreement < s => best = Some((*c0, total_disagreement)),
            _ => {}
        }
    }
    best.map(|(pose, total)| DisambiguationResult {
        pose,
        mean_disagreement_radians: if other_count == 0 {
            0.0
        } else {
            total / other_count as f32
        },
    })
}

/// Single-frame fallback. Uses a weighted combination of rotation
/// magnitude and kind/gravity fit. Best-effort: in some pose
/// geometries the wrong candidate has smaller rotation, in others
/// better gravity alignment, so single-frame success is not
/// guaranteed (~5-10% failure rate on broad random poses). Use
/// `disambiguate_across_frames` whenever ≥ 2 frames are available.
const ROTATION_WEIGHT: f32 = 2.0;

pub fn disambiguate_with_priors(
    candidates: &[PlanePose],
    gravity_camera: Option<[f32; 3]>,
    kind: SurfaceKind,
) -> Option<PlanePose> {
    if candidates.is_empty() {
        return None;
    }
    if candidates.len() == 1 {
        return Some(candidates[0]);
    }
    let g_opt = match (gravity_camera, kind) {
        (Some(g), k) if k != SurfaceKind::Unknown => Some(normalize3(&g)),
        _ => None,
    };
    let mut best: Option<(PlanePose, f32)> = None;
    for &cand in candidates {
        let rot = rotation_angle(&cand.rotation);
        let kind_cost = match (g_opt, kind) {
            (Some(g), SurfaceKind::Horizontal) => {
                let dot = cand.normal[0] * g[0] + cand.normal[1] * g[1] + cand.normal[2] * g[2];
                1.0 - dot.abs()
            }
            (Some(g), SurfaceKind::Vertical) => {
                let dot = cand.normal[0] * g[0] + cand.normal[1] * g[1] + cand.normal[2] * g[2];
                dot.abs()
            }
            _ => 0.0,
        };
        let cost = kind_cost + ROTATION_WEIGHT * rot;
        match best {
            None => best = Some((cand, cost)),
            Some((_, s)) if cost < s => best = Some((cand, cost)),
            _ => {}
        }
    }
    best.map(|(p, _)| p)
}

fn rotation_angle(r: &[f32; 9]) -> f32 {
    let trace = r[0] + r[4] + r[8];
    ((trace - 1.0) / 2.0).clamp(-1.0, 1.0).acos()
}

// ---------------------------------------------------------------------
// 3×3 matrix utilities (row-major).
// ---------------------------------------------------------------------

fn outer_product(a: &[f32; 3], b: &[f32; 3]) -> [f32; 9] {
    [
        a[0] * b[0],
        a[0] * b[1],
        a[0] * b[2],
        a[1] * b[0],
        a[1] * b[1],
        a[1] * b[2],
        a[2] * b[0],
        a[2] * b[1],
        a[2] * b[2],
    ]
}

fn mat3_vec(m: &[f32; 9], v: &[f32; 3]) -> [f32; 3] {
    [
        m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
        m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
        m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
    ]
}

fn transpose3(m: &[f32; 9]) -> [f32; 9] {
    [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]]
}

fn scale_mat3(m: &[f32; 9], s: f32) -> [f32; 9] {
    let mut out = *m;
    for v in &mut out {
        *v *= s;
    }
    out
}

fn scale_vec3(v: &[f32; 3], s: f32) -> [f32; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

fn col(m: &[f32; 9], j: usize) -> [f32; 3] {
    [m[j], m[3 + j], m[6 + j]]
}

fn cols_to_mat(cols: [&[f32; 3]; 3]) -> [f32; 9] {
    [
        cols[0][0], cols[1][0], cols[2][0], cols[0][1], cols[1][1], cols[2][1], cols[0][2],
        cols[1][2], cols[2][2],
    ]
}

fn cross(a: &[f32; 3], b: &[f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize3(v: &[f32; 3]) -> [f32; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if n < 1e-12 {
        return [0.0, 0.0, 0.0];
    }
    [v[0] / n, v[1] / n, v[2] / n]
}

fn unit_combo(v1: &[f32; 3], a: f32, v3: &[f32; 3], b: f32) -> [f32; 3] {
    let raw = [
        a * v1[0] + b * v3[0],
        a * v1[1] + b * v3[1],
        a * v1[2] + b * v3[2],
    ];
    normalize3(&raw)
}

fn frobenius(m: &[f32; 9]) -> f32 {
    let mut acc = 0.0;
    for &x in m {
        acc += x * x;
    }
    acc.sqrt()
}

fn det3(m: &[f32; 9]) -> f32 {
    m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6])
}

fn is_rotation_matrix(m: &[f32; 9]) -> bool {
    if (det3(m) - 1.0).abs() > 1e-2 {
        return false;
    }
    let mt = transpose3(m);
    let prod = mat3_mul(&mt, m);
    let id = [1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let mut err = 0.0_f32;
    for i in 0..9 {
        err += (prod[i] - id[i]).powi(2);
    }
    err.sqrt() < 1e-2
}

// ---------------------------------------------------------------------
// SVD of a 3×3 matrix.
//
// We compute it via symmetric eigendecomposition of `Aᵀ·A` (Jacobi
// sweeps). Returns (U, σ, V) with `A = U · diag(σ) · Vᵀ` and singular
// values sorted descending.
// ---------------------------------------------------------------------

fn svd3(a: &[f32; 9]) -> Option<([f32; 9], [f32; 3], [f32; 9])> {
    let at_a = mat3_mul(&transpose3(a), a);
    let (eigvals, eigvecs) = jacobi_eig3(&at_a);

    let mut idx = [0usize, 1, 2];
    idx.sort_by(|&i, &j| eigvals[j].partial_cmp(&eigvals[i]).unwrap());
    let sigma = [
        eigvals[idx[0]].max(0.0).sqrt(),
        eigvals[idx[1]].max(0.0).sqrt(),
        eigvals[idx[2]].max(0.0).sqrt(),
    ];
    let mut v = [0.0_f32; 9];
    for (k, &source_idx) in idx.iter().enumerate() {
        let col_v = col(&eigvecs, source_idx);
        v[k] = col_v[0];
        v[3 + k] = col_v[1];
        v[6 + k] = col_v[2];
    }

    if sigma[0] < 1e-9 {
        return None;
    }

    let mut u = [0.0_f32; 9];
    for k in 0..3 {
        let v_col = col(&v, k);
        let av = mat3_vec(a, &v_col);
        if sigma[k] > 1e-9 {
            for i in 0..3 {
                u[i * 3 + k] = av[i] / sigma[k];
            }
        } else {
            let col0 = col(&u, 0);
            let col1 = col(&u, 1);
            let fallback = cross(&col0, &col1);
            let fallback = normalize3(&fallback);
            for i in 0..3 {
                u[i * 3 + k] = fallback[i];
            }
        }
    }

    if det3(&u) * det3(&v) < 0.0 {
        for i in 0..3 {
            u[i * 3 + 2] = -u[i * 3 + 2];
        }
    }

    Some((u, sigma, v))
}

/// Symmetric-eigendecompose a 3×3 matrix via Jacobi sweeps. Returns
/// (eigenvalues, eigenvectors-as-columns).
fn jacobi_eig3(m: &[f32; 9]) -> ([f32; 3], [f32; 9]) {
    let mut a = *m;
    let mut v = [1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    for _ in 0..50 {
        let (p, q) = max_off_diag(&a);
        let apq = a[p * 3 + q];
        if apq.abs() < 1e-9 {
            break;
        }
        let app = a[p * 3 + p];
        let aqq = a[q * 3 + q];
        let theta = (aqq - app) / (2.0 * apq);
        let t = if theta >= 0.0 {
            1.0 / (theta + (1.0 + theta * theta).sqrt())
        } else {
            1.0 / (theta - (1.0 + theta * theta).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;

        for i in 0..3 {
            let aip = a[i * 3 + p];
            let aiq = a[i * 3 + q];
            a[i * 3 + p] = c * aip - s * aiq;
            a[i * 3 + q] = s * aip + c * aiq;
        }
        for j in 0..3 {
            let apj = a[p * 3 + j];
            let aqj = a[q * 3 + j];
            a[p * 3 + j] = c * apj - s * aqj;
            a[q * 3 + j] = s * apj + c * aqj;
        }
        for i in 0..3 {
            let vip = v[i * 3 + p];
            let viq = v[i * 3 + q];
            v[i * 3 + p] = c * vip - s * viq;
            v[i * 3 + q] = s * vip + c * viq;
        }
    }
    let eigvals = [a[0], a[4], a[8]];
    (eigvals, v)
}

fn max_off_diag(a: &[f32; 9]) -> (usize, usize) {
    let candidates = [(0, 1), (0, 2), (1, 2)];
    let mut best = (0, 1);
    let mut best_val = a[1].abs();
    for &(p, q) in &candidates[1..] {
        let v = a[p * 3 + q].abs();
        if v > best_val {
            best_val = v;
            best = (p, q);
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn k_test() -> CameraIntrinsics {
        CameraIntrinsics {
            fx: 600.0,
            fy: 600.0,
            cx: 320.0,
            cy: 240.0,
        }
    }

    fn axis_angle_rotation(axis: [f32; 3], angle: f32) -> [f32; 9] {
        let a = normalize3(&axis);
        let c = angle.cos();
        let s = angle.sin();
        let one_minus_c = 1.0 - c;
        [
            c + a[0] * a[0] * one_minus_c,
            a[0] * a[1] * one_minus_c - a[2] * s,
            a[0] * a[2] * one_minus_c + a[1] * s,
            a[1] * a[0] * one_minus_c + a[2] * s,
            c + a[1] * a[1] * one_minus_c,
            a[1] * a[2] * one_minus_c - a[0] * s,
            a[2] * a[0] * one_minus_c - a[1] * s,
            a[2] * a[1] * one_minus_c + a[0] * s,
            c + a[2] * a[2] * one_minus_c,
        ]
    }

    fn angle_between(a: &[f32; 3], b: &[f32; 3]) -> f32 {
        let a_n = normalize3(a);
        let b_n = normalize3(b);
        let dot = (a_n[0] * b_n[0] + a_n[1] * b_n[1] + a_n[2] * b_n[2]).clamp(-1.0, 1.0);
        dot.acos()
    }

    fn rotation_angle_between(r1: &[f32; 9], r2: &[f32; 9]) -> f32 {
        let r1t = transpose3(r1);
        let delta = mat3_mul(&r1t, r2);
        let trace = delta[0] + delta[4] + delta[8];
        ((trace - 1.0) / 2.0).clamp(-1.0, 1.0).acos()
    }

    #[test]
    fn compose_then_decompose_recovers_inputs() {
        let k = k_test();
        let r = axis_angle_rotation([0.0, 1.0, 0.0], 8.0_f32.to_radians());
        let n = normalize3(&[0.1, 0.2, -1.0]);
        let t_over_d = [0.05, 0.02, 0.10];
        let pose = PlanePose {
            rotation: r,
            t_over_d,
            normal: n,
        };
        let h = compose_homography(&pose, &k);
        let candidates = decompose_homography(&h, &k);
        assert!(
            !candidates.is_empty(),
            "decomposition returned no candidates"
        );

        let mut best_normal_err = f32::INFINITY;
        let mut best_rot_err = f32::INFINITY;
        for c in &candidates {
            let ne = angle_between(&c.normal, &n);
            let re = rotation_angle_between(&c.rotation, &r);
            if ne < best_normal_err {
                best_normal_err = ne;
            }
            if re < best_rot_err {
                best_rot_err = re;
            }
        }
        assert!(
            best_normal_err < 1.0_f32.to_radians(),
            "normal err {} deg",
            best_normal_err.to_degrees()
        );
        assert!(
            best_rot_err < 1.0_f32.to_radians(),
            "rotation err {} deg",
            best_rot_err.to_degrees()
        );
    }

    #[test]
    fn pure_rotation_returns_empty() {
        let k = k_test();
        let r = axis_angle_rotation([0.0, 1.0, 0.0], 5.0_f32.to_radians());
        let pose = PlanePose {
            rotation: r,
            t_over_d: [0.0, 0.0, 0.0],
            normal: [0.0, 0.0, -1.0],
        };
        let h = compose_homography(&pose, &k);
        let candidates = decompose_homography(&h, &k);
        assert!(
            candidates.is_empty(),
            "pure rotation should produce no plane candidates"
        );
    }

    #[test]
    fn disambiguate_vertical_with_gravity() {
        let k = k_test();
        let r = axis_angle_rotation([1.0, 0.0, 0.0], 5.0_f32.to_radians());
        let n_true = normalize3(&[0.0, 0.0, -1.0]);
        let t_over_d = [0.04, 0.0, 0.06];
        let pose = PlanePose {
            rotation: r,
            t_over_d,
            normal: n_true,
        };
        let h = compose_homography(&pose, &k);
        let candidates = decompose_homography(&h, &k);
        let gravity = [0.0, 1.0, 0.0];
        let picked =
            disambiguate_with_priors(&candidates, Some(gravity), SurfaceKind::Vertical).unwrap();
        assert!(
            angle_between(&picked.normal, &n_true) < 1.0_f32.to_radians(),
            "picked normal off"
        );
    }

    #[test]
    fn disambiguate_horizontal_with_gravity() {
        let k = k_test();
        let r = axis_angle_rotation([1.0, 0.0, 0.0], 30.0_f32.to_radians());
        let n_true = normalize3(&[0.0, -1.0, -0.3]);
        let t_over_d = [0.02, 0.05, 0.03];
        let pose = PlanePose {
            rotation: r,
            t_over_d,
            normal: n_true,
        };
        let h = compose_homography(&pose, &k);
        let candidates = decompose_homography(&h, &k);
        let gravity = [0.0, -1.0, -0.3];
        let picked =
            disambiguate_with_priors(&candidates, Some(gravity), SurfaceKind::Horizontal).unwrap();
        assert!(
            angle_between(&picked.normal, &n_true) < 2.0_f32.to_radians(),
            "horizontal normal off: {}°",
            angle_between(&picked.normal, &n_true).to_degrees()
        );
    }

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) as f32) / (u32::MAX as f32)
        }
        fn signed(&mut self) -> f32 {
            self.next() * 2.0 - 1.0
        }
    }

    /// "Vertical wall" pose: surface normal strictly perpendicular to
    /// gravity. The wall may tilt slightly forward/back (~5°) but not
    /// "lean" sideways. Gravity is camera +y (phone roughly upright).
    fn random_vertical_pose(rng: &mut Lcg) -> PlanePose {
        let axis = [rng.signed() * 0.3, rng.signed(), rng.signed() * 0.3];
        let angle = rng.signed() * 15.0_f32.to_radians();
        let r = axis_angle_rotation(axis, angle);
        // n[1] ∈ [-0.08, 0.08] → wall tilt ≤ ~5° from strictly vertical.
        let n = normalize3(&[rng.signed() * 0.2, rng.signed() * 0.08, -1.0]);
        let t = [
            rng.signed() * 0.1,
            rng.signed() * 0.1,
            rng.next() * 0.1 + 0.02,
        ];
        PlanePose {
            rotation: r,
            t_over_d: t,
            normal: n,
        }
    }

    /// "Horizontal desk" pose: surface normal aligned with gravity
    /// (large positive Y component in camera frame after camera tilts
    /// down to look at the desk). Gravity vector matches the camera's
    /// tilt-down angle.
    fn random_horizontal_pose(rng: &mut Lcg) -> PlanePose {
        let axis = [rng.signed(), rng.signed() * 0.3, rng.signed() * 0.3];
        let angle = rng.signed() * 15.0_f32.to_radians();
        let r = axis_angle_rotation(axis, angle);
        // Strongly down-and-toward-camera normal; minor x-variation.
        let n = normalize3(&[rng.signed() * 0.05, -1.0, -rng.next() * 0.2 - 0.1]);
        let t = [
            rng.signed() * 0.1,
            rng.signed() * 0.1,
            rng.next() * 0.08 + 0.02,
        ];
        PlanePose {
            rotation: r,
            t_over_d: t,
            normal: n,
        }
    }

    /// Build a second-frame pose that observes the *same* plane (same
    /// `n`) from a different camera pose. Multi-frame disambiguation
    /// then picks the consistent normal.
    fn second_frame_for(plane_normal: [f32; 3], rng: &mut Lcg) -> PlanePose {
        let axis = [rng.signed(), rng.signed(), rng.signed()];
        let angle = (rng.signed() * 10.0_f32 + 5.0).to_radians();
        let r = axis_angle_rotation(axis, angle);
        let t = [
            rng.signed() * 0.15,
            rng.signed() * 0.15,
            rng.next() * 0.1 + 0.03,
        ];
        PlanePose {
            rotation: r,
            t_over_d: t,
            normal: plane_normal,
        }
    }

    #[test]
    fn random_vertical_poses_two_frame_disambiguation() {
        let k = k_test();
        let mut rng = Lcg(0xDEADBEEF);
        let mut failures = 0;
        let mut max_err = 0.0_f32;
        for _ in 0..100 {
            let pose_f0 = random_vertical_pose(&mut rng);
            let pose_f1 = second_frame_for(pose_f0.normal, &mut rng);
            let h0 = compose_homography(&pose_f0, &k);
            let h1 = compose_homography(&pose_f1, &k);
            let c0 = decompose_homography(&h0, &k);
            let c1 = decompose_homography(&h1, &k);
            if c0.is_empty() || c1.is_empty() {
                failures += 1;
                continue;
            }
            let picked = disambiguate_across_frames(&[c0, c1]);
            match picked {
                Some(r) => {
                    let err = angle_between(&r.pose.normal, &pose_f0.normal);
                    if err > max_err {
                        max_err = err;
                    }
                    if err > 1.0_f32.to_radians() {
                        failures += 1;
                    }
                }
                None => failures += 1,
            }
        }
        assert!(
            failures == 0,
            "{} failures over 100 random vertical poses (max err {}°)",
            failures,
            max_err.to_degrees()
        );
    }

    #[test]
    fn random_horizontal_poses_two_frame_disambiguation() {
        let k = k_test();
        let mut rng = Lcg(0xC0FFEE);
        let mut failures = 0;
        let mut max_err = 0.0_f32;
        for _ in 0..100 {
            let pose_f0 = random_horizontal_pose(&mut rng);
            let pose_f1 = second_frame_for(pose_f0.normal, &mut rng);
            let h0 = compose_homography(&pose_f0, &k);
            let h1 = compose_homography(&pose_f1, &k);
            let c0 = decompose_homography(&h0, &k);
            let c1 = decompose_homography(&h1, &k);
            if c0.is_empty() || c1.is_empty() {
                failures += 1;
                continue;
            }
            let picked = disambiguate_across_frames(&[c0, c1]);
            match picked {
                Some(r) => {
                    let err = angle_between(&r.pose.normal, &pose_f0.normal);
                    if err > max_err {
                        max_err = err;
                    }
                    if err > 1.0_f32.to_radians() {
                        failures += 1;
                    }
                }
                None => failures += 1,
            }
        }
        assert!(
            failures == 0,
            "{} failures over 100 random horizontal poses (max err {}°)",
            failures,
            max_err.to_degrees()
        );
    }

    /// Two-frame noise tolerance: inject Gaussian noise on each
    /// frame's point correspondences, re-fit H per frame, decompose
    /// both, disambiguate by cross-frame consistency. This is the
    /// production-shape path. The printed table informs Phase 3
    /// averaging-window sizing.
    #[test]
    fn noise_tolerance_table_two_frame() {
        use crate::homography::fit_homography;
        let k = k_test();
        // Shared plane: same `normal`, two different camera poses.
        let pose_f0 = PlanePose {
            rotation: axis_angle_rotation([0.0, 1.0, 0.2], 20.0_f32.to_radians()),
            t_over_d: [0.06, 0.03, 0.07],
            normal: normalize3(&[0.15, 0.10, -1.0]),
        };
        let pose_f1 = PlanePose {
            rotation: axis_angle_rotation([0.5, 1.0, 0.0], 8.0_f32.to_radians()),
            t_over_d: [-0.04, 0.05, 0.10],
            normal: pose_f0.normal,
        };
        let h0_true = compose_homography(&pose_f0, &k);
        let h1_true = compose_homography(&pose_f1, &k);

        let srcs: Vec<(f32, f32)> = vec![
            (100.0, 100.0),
            (540.0, 100.0),
            (540.0, 380.0),
            (100.0, 380.0),
        ];
        let project_with = |h: &[f32; 9]| -> Vec<(f32, f32)> {
            srcs.iter()
                .map(|&(x, y)| {
                    let denom = h[6] * x + h[7] * y + h[8];
                    (
                        (h[0] * x + h[1] * y + h[2]) / denom,
                        (h[3] * x + h[4] * y + h[5]) / denom,
                    )
                })
                .collect()
        };
        let t0 = project_with(&h0_true);
        let t1 = project_with(&h1_true);

        let noise_levels = [0.0_f32, 1.0, 3.0, 5.0];
        let mut rng = Lcg(0xBADF00D);
        let mut table = Vec::new();
        for &noise_px in &noise_levels {
            let trials = 50;
            let mut sum_err = 0.0_f32;
            let mut max_err = 0.0_f32;
            let mut successes = 0;
            for _ in 0..trials {
                let mut perturb = |tgts: &[(f32, f32)]| -> Vec<(f32, f32, f32, f32)> {
                    srcs.iter()
                        .zip(tgts.iter())
                        .map(|(&(sx, sy), &(tx, ty))| {
                            (
                                sx,
                                sy,
                                tx + rng.signed() * noise_px,
                                ty + rng.signed() * noise_px,
                            )
                        })
                        .collect()
                };
                let (Some(h0_noisy), Some(h1_noisy)) =
                    (fit_homography(&perturb(&t0)), fit_homography(&perturb(&t1)))
                else {
                    continue;
                };
                let c0 = decompose_homography(&h0_noisy, &k);
                let c1 = decompose_homography(&h1_noisy, &k);
                if c0.is_empty() || c1.is_empty() {
                    continue;
                }
                if let Some(r) = disambiguate_across_frames(&[c0, c1]) {
                    let err = angle_between(&r.pose.normal, &pose_f0.normal);
                    sum_err += err;
                    if err > max_err {
                        max_err = err;
                    }
                    successes += 1;
                }
            }
            let mean_err = if successes > 0 {
                (sum_err / successes as f32).to_degrees()
            } else {
                f32::NAN
            };
            table.push((noise_px, mean_err, max_err.to_degrees(), successes));
        }
        println!("\n2-frame noise (px) | mean err (°) | max err (°) | recoveries");
        println!("-------------------+--------------+-------------+-----------");
        for (n, mean, max, ok) in &table {
            println!(
                "{:>16.1}   | {:>11.2}  | {:>10.2}  | {:>4}/50",
                n, mean, max, ok
            );
        }
        let (_, mean0, _, _) = table[0];
        assert!(
            mean0 < 0.5,
            "zero-noise mean error {} ° is not near zero",
            mean0
        );
    }

    /// A rectangular patch on a tilted plane, projected through the
    /// source camera, then through `H_rect`, should land on an
    /// axis-aligned rectangle in the rectified image.
    #[test]
    fn rectification_axis_aligns_plane_rectangle() {
        use crate::homography::project;
        let k = k_test();
        let alpha = 30.0_f32.to_radians();
        let n = normalize3(&[0.0, alpha.sin(), -alpha.cos()]);
        let pose = PlanePose {
            rotation: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            t_over_d: [0.0, 0.0, 0.0],
            normal: n,
        };
        let b1 = normalize3(&cross(&[0.0, 1.0, 0.0], &n));
        let b2 = normalize3(&cross(&n, &b1));
        let plane_centre = scale_vec3(&n, -1.0);
        let size = 0.25;
        let corners_3d: Vec<[f32; 3]> = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
            .iter()
            .map(|&(u, v)| {
                [
                    plane_centre[0] + size * u * b1[0] + size * v * b2[0],
                    plane_centre[1] + size * u * b1[1] + size * v * b2[1],
                    plane_centre[2] + size * u * b1[2] + size * v * b2[2],
                ]
            })
            .collect();
        let source_pixels: Vec<(f32, f32)> = corners_3d
            .iter()
            .map(|p| {
                let x = k.fx * p[0] / p[2] + k.cx;
                let y = k.fy * p[1] / p[2] + k.cy;
                (x, y)
            })
            .collect();
        let h_rect = rectification_matrix(&pose, &k, 200.0, (320.0, 240.0)).unwrap();
        let rectified: Vec<(f32, f32)> = source_pixels
            .iter()
            .map(|&(x, y)| project(&h_rect, x, y).unwrap())
            .collect();
        let mut by_y = rectified.clone();
        by_y.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let tol = 0.5;
        assert!(
            (by_y[0].1 - by_y[1].1).abs() < tol,
            "top y mismatch: {:?}",
            by_y
        );
        assert!(
            (by_y[2].1 - by_y[3].1).abs() < tol,
            "bottom y mismatch: {:?}",
            by_y
        );
        let mut by_x = rectified.clone();
        by_x.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        assert!(
            (by_x[0].0 - by_x[1].0).abs() < tol,
            "left x mismatch: {:?}",
            by_x
        );
        assert!(
            (by_x[2].0 - by_x[3].0).abs() < tol,
            "right x mismatch: {:?}",
            by_x
        );
    }

    /// Rectifying a fronto-parallel plane should produce a pure
    /// similarity (no perspective skew, just scale + translation).
    #[test]
    fn rectification_of_fronto_parallel_is_similarity() {
        use crate::homography::project;
        let k = k_test();
        let pose = PlanePose {
            rotation: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            t_over_d: [0.0, 0.0, 0.0],
            normal: [0.0, 0.0, -1.0],
        };
        let h_rect = rectification_matrix(&pose, &k, 200.0, (320.0, 240.0)).unwrap();
        let pixels = [
            (160.0, 120.0),
            (480.0, 120.0),
            (480.0, 360.0),
            (160.0, 360.0),
        ];
        let mapped: Vec<(f32, f32)> = pixels
            .iter()
            .map(|&(x, y)| project(&h_rect, x, y).unwrap())
            .collect();
        let top_dy = (mapped[0].1 - mapped[1].1).abs();
        let bot_dy = (mapped[2].1 - mapped[3].1).abs();
        let left_dx = (mapped[0].0 - mapped[3].0).abs();
        let right_dx = (mapped[1].0 - mapped[2].0).abs();
        assert!(top_dy < 0.5 && bot_dy < 0.5, "y not preserved");
        assert!(left_dx < 0.5 && right_dx < 0.5, "x not preserved");
    }

    #[test]
    fn residual_is_small_for_correct_pose() {
        let k = k_test();
        let r = axis_angle_rotation([0.0, 1.0, 0.5], 12.0_f32.to_radians());
        let n = normalize3(&[0.05, 0.1, -1.0]);
        let pose = PlanePose {
            rotation: r,
            t_over_d: [0.07, 0.03, 0.08],
            normal: n,
        };
        let h = compose_homography(&pose, &k);
        let res = decomposition_residual(&h, &pose, &k);
        assert!(res < 1e-4, "residual too large: {}", res);
    }

    fn synthetic_rgba(w: u32, h: u32) -> Vec<u8> {
        let mut out = vec![0u8; (w as usize) * (h as usize) * 4];
        for y in 0..h {
            for x in 0..w {
                let i = ((y as usize) * (w as usize) + (x as usize)) * 4;
                out[i] = (x % 256) as u8;
                out[i + 1] = (y % 256) as u8;
                out[i + 2] = ((x ^ y) % 256) as u8;
                out[i + 3] = 0xFF;
            }
        }
        out
    }

    #[test]
    fn resample_identity_recovers_source() {
        let w = 32u32;
        let h = 24u32;
        let src = synthetic_rgba(w, h);
        let identity = [1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let dst = resample_rgba_through(&src, w, h, w, h, &identity);
        assert_eq!(dst.len(), src.len());
        // Bilinear at half-pixel centres reproduces the input exactly
        // when the map is identity.
        for i in 0..src.len() {
            assert!(
                (dst[i] as i32 - src[i] as i32).abs() <= 1,
                "pixel {} differs: dst={} src={}",
                i,
                dst[i],
                src[i],
            );
        }
    }

    #[test]
    fn resample_outside_bounds_is_zero() {
        let src = synthetic_rgba(16, 16);
        // h_dst_to_src that maps every destination pixel to (-1000, -1000) → outside.
        // Translate the destination origin far off the source.
        let translate = [1.0_f32, 0.0, -1000.0, 0.0, 1.0, -1000.0, 0.0, 0.0, 1.0];
        let dst = resample_rgba_through(&src, 16, 16, 16, 16, &translate);
        assert!(dst.iter().all(|&b| b == 0), "expected all-zero output");
    }

    #[test]
    fn resample_horizontal_flip_swaps_columns() {
        let w = 8u32;
        let h = 4u32;
        let src = synthetic_rgba(w, h);
        // Inverse map dst→src that horizontally flips: dst pixel x
        // samples src pixel (w - x).
        let flip = [-1.0_f32, 0.0, w as f32, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let dst = resample_rgba_through(&src, w, h, w, h, &flip);
        // dst column x should equal src column (w-1-x), modulo
        // half-pixel-centre bilinear rounding (≤ 1 byte off).
        for y in 0..h {
            for x in 0..w {
                let di = ((y as usize) * (w as usize) + (x as usize)) * 4;
                let si = ((y as usize) * (w as usize) + ((w - 1 - x) as usize)) * 4;
                for c in 0..3 {
                    assert!(
                        (dst[di + c] as i32 - src[si + c] as i32).abs() <= 1,
                        "flip mismatch at ({}, {}) channel {}: dst={} src={}",
                        x,
                        y,
                        c,
                        dst[di + c],
                        src[si + c],
                    );
                }
            }
        }
    }
}
