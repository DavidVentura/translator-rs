//! IMU-based homography prior for the planar tracker.
//!
//! When the camera rotates between two frames, every pixel of a
//! planar surface (at infinity, or far enough that translation
//! effects are negligible) moves by a homography
//!
//!   H_imu = K · R_cam · K^-1
//!
//! where K is the camera intrinsics matrix and R_cam is the rotation
//! delta expressed in camera frame. The IMU gives us the rotation
//! delta in *device* frame, so we sandwich by a device→camera basis
//! transform M.
//!
//! Combining with the last successful `canonical → frame` homography
//! gives us a *prediction* of the new canonical→frame H, which the
//! tracker can use as a RANSAC seed. This is much faster and more
//! robust than random sampling, especially when visual matches are
//! sparse (fast pan, motion blur).
//!
//! All functions are pure — they take rotation matrices and intrinsics
//! and return matrices. The Kotlin side feeds IMU rotation matrices
//! plus camera intrinsics; the engine combines them with its own
//! tracked last-H state to produce the per-frame prior.

use crate::homography::mat3_mul;

/// Pinhole camera intrinsics in pixels. Standard Android `CameraX` /
/// `Camera2` calibration shape — `fx`, `fy` focal lengths and
/// `cx`, `cy` principal-point offsets, all in the same pixel space as
/// the image we're processing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraIntrinsics {
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
}

impl CameraIntrinsics {
    /// Row-major K matrix.
    pub fn k(&self) -> [f32; 9] {
        [self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0]
    }

    /// Row-major K^-1. Closed-form because K is upper-triangular with
    /// 1 in the bottom-right.
    pub fn k_inv(&self) -> [f32; 9] {
        [
            1.0 / self.fx,
            0.0,
            -self.cx / self.fx,
            0.0,
            1.0 / self.fy,
            -self.cy / self.fy,
            0.0,
            0.0,
            1.0,
        ]
    }
}

/// Body-frame transform that maps a world-fixed vector's body-frame
/// coordinates at time `prev` to its body-frame coordinates at time
/// `curr`. With the IMU's convention that `R(t)` rotates body→world
/// (equivalently, `R(t)^T · v_world = v_body(t)`):
///
///   v_body(curr) = R_curr^T · v_world = R_curr^T · R_prev · v_body(prev)
///
/// so the delta is `R_curr^T · R_prev`. NOT `R_curr · R_prev^T` — that
/// expression is the inverse for the single-axis case, which silently
/// flipped the IMU prior direction and made it *anti-helpful* under
/// fast rotations (RANSAC tolerated it because the visual inliers
/// outvoted the bad seed, but the seed was pulling RANSAC away from
/// the right answer instead of toward it).
pub fn rotation_delta(r_prev: &[f32; 9], r_curr: &[f32; 9]) -> [f32; 9] {
    let r_curr_t = transpose3(r_curr);
    mat3_mul(&r_curr_t, r_prev)
}

/// Build the device→camera basis-change matrix `M`, parameterised by
/// `sensor_orientation_degrees` (= CameraX `ImageInfo.rotationDegrees`,
/// which equals the camera characteristic `SENSOR_ORIENTATION` when
/// the device is held in its natural portrait orientation — the
/// dominant case for live OCR).
///
/// Two effects compose:
/// 1. **Sign flip** `diag(1, -1, -1)`: image-space Y points "down"
///    where device-frame Y points "up"; camera-frame Z points "into
///    the scene" while device-frame Z points "out of the screen
///    toward the user". This was the entire transform before
///    parameterisation.
/// 2. **Rotation by sensor orientation** around the camera Z axis:
///    on a typical back camera with `SENSOR_ORIENTATION = 90`, the
///    sensor is mounted rotated 90° clockwise relative to the
///    device's natural orientation. A device-Y yaw therefore appears
///    as motion along the sensor's *long* axis (sensor X) in the
///    image, not the short axis. Without this rotation the prior
///    predicts feature motion on the wrong axis; slow motion hides
///    it (the offset is small enough to stay inside the guided-match
///    window), but fast pans send the predicted window dozens of
///    pixels off the true feature location and the matcher
///    collapses (observed: 187 → 9 inliers in one frame).
///
/// At `sensor_orientation_degrees = 0` the result equals the legacy
/// `diag(1, -1, -1)`.
fn device_to_camera_matrix(sensor_orientation_degrees: i32) -> [f32; 9] {
    // Sign: -theta. Confirmed empirically on a back-camera portrait
    // phone (SENSOR_ORIENTATION=90) — with +theta the rotation
    // prior pulled features on the correct axis but in the wrong
    // direction, so horizontal pans landed the guided window on the
    // opposite side of the actual feature. Slow motion hid it
    // (offset fell within the window slack); fast pans put the
    // window dozens of pixels away → matcher collapse.
    let theta = -(sensor_orientation_degrees as f32).to_radians();
    let c = theta.cos();
    let s = theta.sin();
    let r_z = [c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0];
    let m_flip = [1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0];
    // M = R_z · M_flip — apply the sign flip first (taking us into
    // a "natural-sensor-orientation camera frame"), then rotate
    // around camera Z by the sensor mount angle.
    mat3_mul(&r_z, &m_flip)
}

/// Re-express a rotation matrix from device frame to camera frame.
/// `R_cam = M · R_dev · M^T` where M comes from
/// [`device_to_camera_matrix`].
pub fn device_to_camera(r_dev: &[f32; 9], sensor_orientation_degrees: i32) -> [f32; 9] {
    let m = device_to_camera_matrix(sensor_orientation_degrees);
    let m_t = transpose3(&m);
    let temp = mat3_mul(&m, r_dev);
    mat3_mul(&temp, &m_t)
}

/// Transform a 3-vector (e.g. acceleration) from device frame to
/// camera frame: `v_cam = M · v_dev`. Same `M` as
/// [`device_to_camera`] — kept here so callers don't have to
/// rebuild M or reach into private helpers.
pub fn device_to_camera_vec(v_dev: [f32; 3], sensor_orientation_degrees: i32) -> [f32; 3] {
    let m = device_to_camera_matrix(sensor_orientation_degrees);
    [
        m[0] * v_dev[0] + m[1] * v_dev[1] + m[2] * v_dev[2],
        m[3] * v_dev[0] + m[4] * v_dev[1] + m[5] * v_dev[2],
        m[6] * v_dev[0] + m[7] * v_dev[1] + m[8] * v_dev[2],
    ]
}

/// Image-space homography corresponding to a pure camera rotation
/// (planar-at-infinity approximation). For OCR-relevant surfaces a
/// few cm to a few m away from the camera, this is accurate to
/// sub-pixel for normal hand motion.
pub fn homography_from_rotation(intrinsics: &CameraIntrinsics, r_cam: &[f32; 9]) -> [f32; 9] {
    let k = intrinsics.k();
    let k_inv = intrinsics.k_inv();
    let kr = mat3_mul(&k, r_cam);
    mat3_mul(&kr, &k_inv)
}

/// Compose `H_canonical_to_current = delta · H_canonical_to_previous`.
/// `delta` is the per-frame image-space homography from previous frame
/// to current frame (typically the output of `homography_from_rotation`
/// using the IMU rotation delta).
pub fn compose_prior(canonical_to_previous: &[f32; 9], delta: &[f32; 9]) -> [f32; 9] {
    mat3_mul(delta, canonical_to_previous)
}

/// End-to-end convenience: from raw device-frame IMU rotations and
/// intrinsics, predict the new `canonical → current_frame` homography
/// given the last successful one. Result is suitable as a RANSAC
/// seed.
pub fn predict_canonical_to_current(
    intrinsics: &CameraIntrinsics,
    r_prev_dev: &[f32; 9],
    r_curr_dev: &[f32; 9],
    canonical_to_previous: &[f32; 9],
    sensor_orientation_degrees: i32,
) -> [f32; 9] {
    let r_delta_dev = rotation_delta(r_prev_dev, r_curr_dev);
    let r_delta_cam = device_to_camera(&r_delta_dev, sensor_orientation_degrees);
    let imu_delta_h = homography_from_rotation(intrinsics, &r_delta_cam);
    compose_prior(canonical_to_previous, &imu_delta_h)
}

fn transpose3(m: &[f32; 9]) -> [f32; 9] {
    [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::homography::project;

    const IDENTITY: [f32; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    fn normalize_h(h: &[f32; 9]) -> [f32; 9] {
        let s = h[8];
        let mut out = [0.0; 9];
        for i in 0..9 {
            out[i] = h[i] / s;
        }
        out
    }

    #[test]
    fn k_times_k_inv_is_identity() {
        let intr = CameraIntrinsics {
            fx: 864.0,
            fy: 866.3,
            cx: 480.0,
            cy: 640.0,
        };
        let result = mat3_mul(&intr.k(), &intr.k_inv());
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx_eq(result[i * 3 + j], expected, 1e-4),
                    "K·K^-1 wrong at ({}, {}): {}",
                    i,
                    j,
                    result[i * 3 + j]
                );
            }
        }
    }

    #[test]
    fn identity_rotation_yields_identity_homography() {
        let intr = CameraIntrinsics {
            fx: 864.0,
            fy: 866.3,
            cx: 480.0,
            cy: 640.0,
        };
        let h = predict_canonical_to_current(&intr, &IDENTITY, &IDENTITY, &IDENTITY, 0);
        let h_n = normalize_h(&h);
        assert!(approx_eq(h_n[0], 1.0, 1e-3));
        assert!(approx_eq(h_n[4], 1.0, 1e-3));
        assert!(approx_eq(h_n[8], 1.0, 1e-3));
        for &off_diag in &[h_n[1], h_n[2], h_n[3], h_n[5], h_n[6], h_n[7]] {
            assert!(
                off_diag.abs() < 1e-3,
                "off-diagonal {} should be 0",
                off_diag
            );
        }
    }

    #[test]
    fn small_yaw_shifts_image_centre_horizontally() {
        // Pure camera-frame yaw (rotation about camera Y axis). For a
        // point at the image centre projected from infinity, the
        // pixel shift is approximately fx · tan(yaw) along x.
        let intr = CameraIntrinsics {
            fx: 1000.0,
            fy: 1000.0,
            cx: 480.0,
            cy: 640.0,
        };
        let yaw = 0.05_f32; // ~2.9°
        // R_cam = rotation about Y axis (camera frame).
        let (s, c) = yaw.sin_cos();
        let r_cam = [c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c];
        let h = homography_from_rotation(&intr, &r_cam);

        // Project image centre.
        let (px, py) = project(&h, intr.cx, intr.cy).expect("not degenerate");
        let dx = px - intr.cx;
        let dy = py - intr.cy;
        let expected_dx = intr.fx * yaw.tan();
        assert!(
            (dx - expected_dx).abs() < 1.0,
            "yaw shift x: got {}, expected ≈ {}",
            dx,
            expected_dx
        );
        assert!(dy.abs() < 0.5, "yaw shouldn't shift y; got {}", dy);
    }

    #[test]
    fn device_to_camera_is_involution_at_sensor_orient_zero() {
        // Only at sensor_orientation=0 does M = diag(1, -1, -1) which
        // is its own inverse. With a non-zero rotation the matrix
        // isn't an involution any more.
        let r = [0.99, -0.05, 0.03, 0.05, 0.99, -0.02, -0.03, 0.02, 0.99];
        let once = device_to_camera(&r, 0);
        let twice = device_to_camera(&once, 0);
        for i in 0..9 {
            assert!(
                approx_eq(twice[i], r[i], 1e-6),
                "involution failed at {}: {} vs {}",
                i,
                twice[i],
                r[i]
            );
        }
    }

    #[test]
    fn device_to_camera_at_zero_matches_legacy_diag() {
        // Legacy behaviour: M = diag(1, -1, -1). Applied as
        // M · R · M^T, entry (i, j) is multiplied by sign(i) · sign(j)
        // with signs [+, -, -].
        let r = [0.99, -0.05, 0.03, 0.05, 0.99, -0.02, -0.03, 0.02, 0.99];
        let got = device_to_camera(&r, 0);
        let signs = [1.0_f32, -1.0, -1.0];
        for i in 0..3 {
            for j in 0..3 {
                let expected = signs[i] * signs[j] * r[i * 3 + j];
                assert!(
                    approx_eq(got[i * 3 + j], expected, 1e-6),
                    "({}, {}): {} vs {}",
                    i,
                    j,
                    got[i * 3 + j],
                    expected
                );
            }
        }
    }

    #[test]
    fn device_to_camera_vec_at_zero_matches_legacy_sign_flip() {
        // Legacy translation prior used `[a[0], -a[1], -a[2]]` —
        // diag(1, -1, -1) applied as a vector transform.
        let v = [0.5_f32, 1.5, -2.0];
        let got = device_to_camera_vec(v, 0);
        assert!(approx_eq(got[0], 0.5, 1e-6));
        assert!(approx_eq(got[1], -1.5, 1e-6));
        assert!(approx_eq(got[2], 2.0, 1e-6));
    }

    #[test]
    fn rotation_delta_maps_body_coords_of_world_fixed_vector() {
        // For a world-fixed vector v_world with v_body(t) = R(t)^T · v_world,
        // the body-frame transform from `prev` to `curr` is
        // R_curr^T · R_prev = rotation_delta(prev, curr). Verify by checking
        // `delta · R_prev^T == R_curr^T`.
        let r_prev = [0.9962, -0.0872, 0.0, 0.0872, 0.9962, 0.0, 0.0, 0.0, 1.0]; // ~5° about Z
        let r_curr = [0.9848, -0.1736, 0.0, 0.1736, 0.9848, 0.0, 0.0, 0.0, 1.0]; // ~10° about Z
        let delta = rotation_delta(&r_prev, &r_curr);
        let composed = mat3_mul(&delta, &transpose3(&r_prev));
        let expected = transpose3(&r_curr);
        for i in 0..9 {
            assert!(
                approx_eq(composed[i], expected[i], 1e-3),
                "delta · R_prev^T != R_curr^T at {}: {} vs {}",
                i,
                composed[i],
                expected[i]
            );
        }
    }

    #[test]
    fn prior_composes_with_previous_canonical_h() {
        // Setup: canonical→prev is a known translation; IMU delta is
        // a small yaw; prior should equal yaw_H · translation_H.
        let intr = CameraIntrinsics {
            fx: 1000.0,
            fy: 1000.0,
            cx: 480.0,
            cy: 640.0,
        };
        let canonical_to_prev = [1.0, 0.0, 20.0, 0.0, 1.0, -15.0, 0.0, 0.0, 1.0];
        let r_prev_dev = IDENTITY;
        let yaw_dev = 0.03_f32;
        let (s, c) = yaw_dev.sin_cos();
        let r_curr_dev = [c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c];

        let predicted =
            predict_canonical_to_current(&intr, &r_prev_dev, &r_curr_dev, &canonical_to_prev, 0);

        // Manual compose: device yaw → camera yaw via sandwich → H_imu
        // → H_imu · canonical_to_prev.
        let r_delta_dev = rotation_delta(&r_prev_dev, &r_curr_dev);
        let r_delta_cam = device_to_camera(&r_delta_dev, 0);
        let h_imu = homography_from_rotation(&intr, &r_delta_cam);
        let expected = mat3_mul(&h_imu, &canonical_to_prev);
        for i in 0..9 {
            assert!(
                approx_eq(predicted[i], expected[i], 1e-4),
                "predicted[{}] = {} vs expected {}",
                i,
                predicted[i],
                expected[i]
            );
        }
    }

    #[test]
    fn imu_predicted_h_matches_visual_projection_for_known_rotation() {
        // The acid test: the camera physically rotates by `r_cam`
        // (camera's orientation relative to baseline). For a world-fixed
        // point seen at image position p_prev, its new image position
        // p_curr is governed by H = K · r_cam^T · K^-1 — because the
        // body→body transform for a world-fixed vector is the transpose
        // of the body-orientation matrix. The IMU path, fed the
        // corresponding device-frame rotation, should produce the same H.
        let intr = CameraIntrinsics {
            fx: 864.0,
            fy: 866.3,
            cx: 480.0,
            cy: 640.0,
        };
        let yaw_cam = 0.04_f32;
        let (s, c) = yaw_cam.sin_cos();
        let r_cam = [c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c];
        let r_cam_t = transpose3(&r_cam);
        let h_visual = homography_from_rotation(&intr, &r_cam_t);

        // Device rotation that yields this camera rotation: at
        // sensor_orientation=0, M = M^T = M^-1, so R_dev = M · R_cam · M.
        let r_dev = device_to_camera(&r_cam, 0);
        let h_predicted = predict_canonical_to_current(&intr, &IDENTITY, &r_dev, &IDENTITY, 0);

        // Compare at several image points.
        for &(x, y) in &[
            (50.0, 50.0),
            (480.0, 640.0),
            (900.0, 100.0),
            (100.0, 1200.0),
        ] {
            let (vx, vy) = project(&h_visual, x, y).unwrap();
            let (px, py) = project(&h_predicted, x, y).unwrap();
            let err = ((vx - px).powi(2) + (vy - py).powi(2)).sqrt();
            assert!(
                err < 0.5,
                "point ({}, {}): visual=({}, {}), predicted=({}, {}), err={}",
                x,
                y,
                vx,
                vy,
                px,
                py,
                err
            );
        }
    }
}
