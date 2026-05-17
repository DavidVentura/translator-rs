//! Homography fitting and 3x3 matrix utilities shared between the legacy
//! per-region tracker and the planar surface tracker.
//!
//! Convention: a homography is a row-major `[f32; 9]` such that
//! `(qx, qy, 1) ~ H * (px, py, 1)` in homogeneous coordinates — divide
//! the projected vector by its `w` component to recover the destination
//! pixel.

/// Direct Linear Transform homography fit with Hartley point normalization.
/// Given correspondences `(px, py) -> (qx, qy)`, returns a row-major 3x3
/// matrix `H` such that `(qx, qy, 1) ~ H * (px, py, 1)` in homogeneous
/// coordinates (i.e. the projected point divided by w gives `(qx, qy)`).
///
/// Uses normal equations on the 2N x 8 system that fixes `h33 = 1`. Each
/// correspondence contributes two rows:
///
///   [px, py, 1, 0, 0, 0, -px*qx, -py*qx]   = qx
///   [0, 0, 0, px, py, 1, -px*qy, -py*qy]   = qy
///
/// Returns `None` if the system is too ill-conditioned (e.g. correspondences
/// are colinear or all clustered at one point).
pub fn fit_homography(pairs: &[(f32, f32, f32, f32)]) -> Option<[f32; 9]> {
    if pairs.len() < 4 {
        return None;
    }
    let n = pairs.len() as f32;
    let (mut mpx, mut mpy, mut mqx, mut mqy) = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
    for &(px, py, qx, qy) in pairs {
        mpx += px;
        mpy += py;
        mqx += qx;
        mqy += qy;
    }
    mpx /= n;
    mpy /= n;
    mqx /= n;
    mqy /= n;
    let (mut sp, mut sq) = (0.0_f32, 0.0_f32);
    for &(px, py, qx, qy) in pairs {
        let dpx = px - mpx;
        let dpy = py - mpy;
        let dqx = qx - mqx;
        let dqy = qy - mqy;
        sp += (dpx * dpx + dpy * dpy).sqrt();
        sq += (dqx * dqx + dqy * dqy).sqrt();
    }
    sp /= n;
    sq /= n;
    if sp <= 1e-3 || sq <= 1e-3 {
        return None;
    }
    let kp = (2.0_f32).sqrt() / sp;
    let kq = (2.0_f32).sqrt() / sq;

    let mut ata = [[0.0_f64; 8]; 8];
    let mut atb = [0.0_f64; 8];
    for &(px, py, qx, qy) in pairs {
        let px_n = (px - mpx) * kp;
        let py_n = (py - mpy) * kp;
        let qx_n = (qx - mqx) * kq;
        let qy_n = (qy - mqy) * kq;
        let r1 = [
            px_n as f64,
            py_n as f64,
            1.0,
            0.0,
            0.0,
            0.0,
            (-px_n * qx_n) as f64,
            (-py_n * qx_n) as f64,
        ];
        let r2 = [
            0.0,
            0.0,
            0.0,
            px_n as f64,
            py_n as f64,
            1.0,
            (-px_n * qy_n) as f64,
            (-py_n * qy_n) as f64,
        ];
        for i in 0..8 {
            for j in 0..8 {
                ata[i][j] += r1[i] * r1[j] + r2[i] * r2[j];
            }
            atb[i] += r1[i] * qx_n as f64 + r2[i] * qy_n as f64;
        }
    }
    let h_norm = solve_8x8(ata, atb)?;
    let h_normalised = [
        h_norm[0] as f32,
        h_norm[1] as f32,
        h_norm[2] as f32,
        h_norm[3] as f32,
        h_norm[4] as f32,
        h_norm[5] as f32,
        h_norm[6] as f32,
        h_norm[7] as f32,
        1.0_f32,
    ];
    let tp = [kp, 0.0, -kp * mpx, 0.0, kp, -kp * mpy, 0.0, 0.0, 1.0];
    let tq_inv = [1.0 / kq, 0.0, mqx, 0.0, 1.0 / kq, mqy, 0.0, 0.0, 1.0];
    let h1 = mat3_mul(&h_normalised, &tp);
    let h2 = mat3_mul(&tq_inv, &h1);
    if !h2.iter().all(|v| v.is_finite()) {
        return None;
    }
    Some(h2)
}

pub fn mat3_mul(a: &[f32; 9], b: &[f32; 9]) -> [f32; 9] {
    let mut out = [0.0_f32; 9];
    for i in 0..3 {
        for j in 0..3 {
            let mut s = 0.0_f32;
            for k in 0..3 {
                s += a[i * 3 + k] * b[k * 3 + j];
            }
            out[i * 3 + j] = s;
        }
    }
    out
}

/// Project a 2D point through a row-major 3x3 homography.
/// Returns `None` if the projected homogeneous `w` is degenerate.
pub fn project(h: &[f32; 9], x: f32, y: f32) -> Option<(f32, f32)> {
    let qx = h[0] * x + h[1] * y + h[2];
    let qy = h[3] * x + h[4] * y + h[5];
    let qw = h[6] * x + h[7] * y + h[8];
    if qw.abs() < 1e-9 || !qw.is_finite() {
        return None;
    }
    Some((qx / qw, qy / qw))
}

/// Inverse of a row-major 3x3. Returns `None` if singular.
pub fn invert(h: &[f32; 9]) -> Option<[f32; 9]> {
    let a = h[0] as f64;
    let b = h[1] as f64;
    let c = h[2] as f64;
    let d = h[3] as f64;
    let e = h[4] as f64;
    let f = h[5] as f64;
    let g = h[6] as f64;
    let hh = h[7] as f64;
    let i = h[8] as f64;
    let det = a * (e * i - f * hh) - b * (d * i - f * g) + c * (d * hh - e * g);
    if det.abs() < 1e-12 || !det.is_finite() {
        return None;
    }
    let inv_det = 1.0 / det;
    let m = [
        ((e * i - f * hh) * inv_det) as f32,
        ((c * hh - b * i) * inv_det) as f32,
        ((b * f - c * e) * inv_det) as f32,
        ((f * g - d * i) * inv_det) as f32,
        ((a * i - c * g) * inv_det) as f32,
        ((c * d - a * f) * inv_det) as f32,
        ((d * hh - e * g) * inv_det) as f32,
        ((b * g - a * hh) * inv_det) as f32,
        ((a * e - b * d) * inv_det) as f32,
    ];
    if !m.iter().all(|v| v.is_finite()) {
        return None;
    }
    Some(m)
}

/// Gauss-Jordan elimination with partial pivoting on the 8x8 system
/// `A * x = b`. Returns `None` if the matrix is near-singular.
fn solve_8x8(mut a: [[f64; 8]; 8], mut b: [f64; 8]) -> Option<[f64; 8]> {
    for col in 0..8 {
        let mut piv_row = col;
        let mut piv_abs = a[col][col].abs();
        for r in (col + 1)..8 {
            let v = a[r][col].abs();
            if v > piv_abs {
                piv_abs = v;
                piv_row = r;
            }
        }
        if piv_abs < 1e-9 {
            return None;
        }
        if piv_row != col {
            a.swap(col, piv_row);
            b.swap(col, piv_row);
        }
        let inv = 1.0 / a[col][col];
        for j in 0..8 {
            a[col][j] *= inv;
        }
        b[col] *= inv;
        for r in 0..8 {
            if r == col {
                continue;
            }
            let factor = a[r][col];
            if factor == 0.0 {
                continue;
            }
            for j in 0..8 {
                a[r][j] -= factor * a[col][j];
            }
            b[r] -= factor * b[col];
        }
    }
    Some(b)
}
