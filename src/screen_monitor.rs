//! Per-box screen-monitoring core (the v2 "Monitoring" logic from
//! `SCREEN_CHANGE_DETECTION.md`).
//!
//! Pure classification over a global pinhole lattice. Each frame the GPU
//! reduction recovers the screen content under our overlay (`screen_est`) at a
//! fixed lattice of points and hands this module one byte per point; this module
//! decides whether the view scrolled, which boxes' text changed, or nothing
//! happened. No IO and no GPU live here — only the math that turns lattice
//! samples + resident box state into a [`FrameClassification`].
//!
//! The signal is defined relative to each box's text strokes, not raw pixels
//! ("measure the text, not the pixels"). A glyph stroke is opaque, so a hole that
//! sits on it stays put while the background behind the box moves, and only moves
//! when the text itself changes. Which holes are "on a stroke" is decided by
//! behaviour (low temporal variance) intersected with appearance (the binarize
//! bootstrap), because the background can mimic ink's appearance but not its
//! stability — see the three-layer mask in the spec.

use crate::ocr::OrientedRect;

/// A point in the global sampling lattice, in canonical (overlay) coords.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LatticePoint {
    pub x: f32,
    pub y: f32,
}

/// The fixed per-frame sampling lattice for a canonical `w×h` frame. Built once
/// per capture geometry; every frame supplies exactly one `screen_est` byte per
/// point, in `points()` order.
pub struct Lattice {
    points: Vec<LatticePoint>,
}

impl Lattice {
    /// A regular `spacing`-pitch grid centred in each cell. Alternating-row /
    /// staggered hole patterns are a rendering concern (hole visibility); the
    /// classifier only needs the point positions, so a plain grid is enough here.
    pub fn build(canon_w: u32, canon_h: u32, spacing: u32) -> Self {
        let spacing = spacing.max(1);
        let half = spacing as f32 * 0.5;
        let mut points = Vec::new();
        let mut y = half;
        while y < canon_h as f32 {
            let mut x = half;
            while x < canon_w as f32 {
                points.push(LatticePoint { x, y });
                x += spacing as f32;
            }
            y += spacing as f32;
        }
        Lattice { points }
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    pub fn points(&self) -> &[LatticePoint] {
        &self.points
    }

    /// Indices of the lattice points inside oriented rect `r`. Callers use this to
    /// assign a box its hole subset and to align an externally-computed bootstrap
    /// mask to the same order.
    pub fn holes_in_rect(&self, r: &OrientedRect) -> Vec<usize> {
        self.points
            .iter()
            .enumerate()
            .filter(|(_, p)| point_in_rect(p.x, p.y, r))
            .map(|(i, _)| i)
            .collect()
    }
}

fn point_in_rect(px: f32, py: f32, r: &OrientedRect) -> bool {
    let dx = px - r.cx;
    let dy = py - r.cy;
    let cos = r.angle_radians.cos();
    let sin = r.angle_radians.sin();
    let lx = (dx * cos + dy * sin).abs();
    let ly = (-dx * sin + dy * cos).abs();
    lx <= r.width * 0.5 && ly <= r.height * 0.5
}

/// Tunables for the classifier. Distances are in `screen_est` luma units (0..255).
#[derive(Debug, Clone, Copy)]
pub struct MonitorConfig {
    /// Frames of stable observation after an acquire before the per-hole variance
    /// mask is frozen. While warming up, the bootstrap (binarize) mask is used.
    pub warmup_frames: u32,
    /// A hole whose observed luma variance is below this is treated as stable
    /// (candidate glyph); above it the hole tracks the moving background.
    pub glyph_var_threshold: f32,
    /// Per-hole `|current − baseline|` luma delta that counts the hole as changed.
    pub change_threshold: u8,
    /// Fraction of a box's glyph holes that must change for the box to trip.
    /// Spatial-coherence guard: a few ambiguous holes never trip a box alone.
    pub box_coherence_frac: f32,
    /// A box needs at least this many glyph holes before we'll judge it at all.
    pub min_glyph_holes: usize,
    /// Fraction of *all* judged boxes' glyph holes changing at once that reads as a
    /// scroll (the anchored text itself moved, not just the background).
    pub scroll_frac: f32,
    /// Scroll needs at least this many judgeable boxes — a single box can't be told
    /// apart from an in-place text change, so we don't try.
    pub scroll_min_boxes: usize,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            warmup_frames: 6,
            glyph_var_threshold: 50.0,
            change_threshold: 40,
            box_coherence_frac: 0.4,
            min_glyph_holes: 4,
            scroll_frac: 0.7,
            scroll_min_boxes: 2,
        }
    }
}

/// What the monitor concludes for one frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrameClassification {
    /// Nothing of interest changed — leave the overlay as-is.
    Quiet,
    /// The whole view translated (anchored text moved together) — hide + full
    /// re-acquire.
    Scroll,
    /// These boxes' underlying text changed — batched targeted re-OCR.
    BoxesChanged(Vec<u64>),
}

/// Welford running mean/variance for one lattice hole. Accumulated only during a
/// box's warmup window so a later text change can't corrupt the frozen mask.
#[derive(Clone, Default)]
struct HoleStats {
    count: u32,
    mean: f32,
    m2: f32,
}

impl HoleStats {
    fn observe(&mut self, x: f32) {
        self.count += 1;
        let delta = x - self.mean;
        self.mean += delta / self.count as f32;
        self.m2 += delta * (x - self.mean);
    }

    fn variance(&self) -> f32 {
        if self.count < 2 {
            return 0.0;
        }
        self.m2 / self.count as f32
    }
}

struct BoxMonitor {
    id: u64,
    /// Lattice indices inside this box's region.
    holes: Vec<usize>,
    /// `screen_est` at the last acquire, parallel to `holes`. The change baseline.
    baseline: Vec<u8>,
    /// Appearance mask from binarizing the clean crop, parallel to `holes`. Used
    /// until the variance mask is frozen.
    bootstrap_glyph: Vec<bool>,
    /// Frozen behaviour∩appearance mask, parallel to `holes`. `None` during warmup.
    learned_glyph: Option<Vec<bool>>,
    /// Frames observed since the baseline was set.
    frames: u32,
}

impl BoxMonitor {
    fn effective_glyph(&self) -> &[bool] {
        match &self.learned_glyph {
            Some(m) => m,
            None => &self.bootstrap_glyph,
        }
    }

    fn deviation(&self, samples: &[u8], cfg: &MonitorConfig) -> BoxDeviation {
        let mask = self.effective_glyph();
        let mut glyph_holes = 0usize;
        let mut deviating = 0usize;
        for (k, &hi) in self.holes.iter().enumerate() {
            if !mask[k] {
                continue;
            }
            glyph_holes += 1;
            let delta = (samples[hi] as i32 - self.baseline[k] as i32).unsigned_abs();
            if delta > cfg.change_threshold as u32 {
                deviating += 1;
            }
        }
        BoxDeviation {
            id: self.id,
            glyph_holes,
            deviating,
        }
    }
}

struct BoxDeviation {
    id: u64,
    glyph_holes: usize,
    deviating: usize,
}

impl BoxDeviation {
    fn judgeable(&self, cfg: &MonitorConfig) -> bool {
        self.glyph_holes >= cfg.min_glyph_holes
    }

    fn changed(&self, cfg: &MonitorConfig) -> bool {
        self.judgeable(cfg)
            && self.deviating as f32 >= cfg.box_coherence_frac * self.glyph_holes as f32
    }
}

/// Holds the resident box state + per-hole stats and classifies each frame.
/// Stateful shell around the pure [`classify`] core: `observe` updates the
/// running stats, then defers the decision to `classify`.
pub struct ScreenMonitor {
    lattice: Lattice,
    stats: Vec<HoleStats>,
    boxes: Vec<BoxMonitor>,
    cfg: MonitorConfig,
}

impl ScreenMonitor {
    pub fn new(lattice: Lattice, cfg: MonitorConfig) -> Self {
        let stats = vec![HoleStats::default(); lattice.len()];
        ScreenMonitor {
            lattice,
            stats,
            boxes: Vec::new(),
            cfg,
        }
    }

    pub fn lattice(&self) -> &Lattice {
        &self.lattice
    }

    /// Register or replace a box after an acquire. `holes` and `bootstrap_glyph`
    /// must be the same length and in the same order (the caller gets `holes` from
    /// [`Lattice::holes_in_rect`] and samples its binarized crop at those points);
    /// `clean_samples` is the full-lattice `screen_est` of the clean read, from
    /// which the per-hole baseline is snapshotted. Resets the variance for this
    /// box's holes so the glyph mask is relearned for the new text.
    pub fn set_box(
        &mut self,
        id: u64,
        holes: Vec<usize>,
        bootstrap_glyph: Vec<bool>,
        clean_samples: &[u8],
    ) {
        assert_eq!(
            holes.len(),
            bootstrap_glyph.len(),
            "holes and bootstrap_glyph must align"
        );
        assert_eq!(
            clean_samples.len(),
            self.lattice.len(),
            "clean_samples must cover the whole lattice"
        );
        let baseline = holes.iter().map(|&h| clean_samples[h]).collect();
        for &h in &holes {
            self.stats[h] = HoleStats::default();
        }
        let monitor = BoxMonitor {
            id,
            holes,
            baseline,
            bootstrap_glyph,
            learned_glyph: None,
            frames: 0,
        };
        match self.boxes.iter_mut().find(|b| b.id == id) {
            Some(slot) => *slot = monitor,
            None => self.boxes.push(monitor),
        }
    }

    pub fn remove_box(&mut self, id: u64) {
        self.boxes.retain(|b| b.id != id);
    }

    pub fn box_ids(&self) -> Vec<u64> {
        self.boxes.iter().map(|b| b.id).collect()
    }

    /// One monitoring frame. `samples` is the recovered `screen_est`, one byte per
    /// lattice point in `points()` order. Updates per-hole variance, freezes any
    /// box whose warmup just elapsed, then classifies.
    pub fn observe(&mut self, samples: &[u8]) -> FrameClassification {
        assert_eq!(
            samples.len(),
            self.lattice.len(),
            "samples must cover the whole lattice"
        );
        for (stat, &s) in self.stats.iter_mut().zip(samples) {
            stat.observe(s as f32);
        }
        // Disjoint field borrows: variance is read off `stats` while `boxes` is
        // mutated. The intersection (appearance ∩ stability) drops both video
        // misread as ink (high variance) and stable non-ink like bar gaps (not in
        // the bootstrap).
        let stats = &self.stats;
        let cfg = &self.cfg;
        for b in self.boxes.iter_mut() {
            b.frames += 1;
            if b.learned_glyph.is_none() && b.frames >= cfg.warmup_frames {
                let learned = b
                    .holes
                    .iter()
                    .zip(&b.bootstrap_glyph)
                    .map(|(&h, &boot)| boot && stats[h].variance() < cfg.glyph_var_threshold)
                    .collect();
                b.learned_glyph = Some(learned);
            }
        }
        let devs: Vec<BoxDeviation> = self
            .boxes
            .iter()
            .map(|b| b.deviation(samples, cfg))
            .collect();
        classify(&devs, cfg)
    }
}

/// Pure decision from the per-box deviations. Scroll is tested first (anchored
/// text across many boxes all moving at once), then per-box change.
fn classify(devs: &[BoxDeviation], cfg: &MonitorConfig) -> FrameClassification {
    let judged: Vec<&BoxDeviation> = devs.iter().filter(|d| d.judgeable(cfg)).collect();
    let total_glyph: usize = judged.iter().map(|d| d.glyph_holes).sum();
    let total_dev: usize = judged.iter().map(|d| d.deviating).sum();
    if judged.len() >= cfg.scroll_min_boxes
        && total_glyph > 0
        && total_dev as f32 >= cfg.scroll_frac * total_glyph as f32
    {
        return FrameClassification::Scroll;
    }
    let changed: Vec<u64> = devs
        .iter()
        .filter(|d| d.changed(cfg))
        .map(|d| d.id)
        .collect();
    if changed.is_empty() {
        FrameClassification::Quiet
    } else {
        FrameClassification::BoxesChanged(changed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GLYPH: u8 = 230; // opaque white stroke
    const BG: u8 = 100; // resting background luma

    fn rect(cx: f32, cy: f32, w: f32, h: f32) -> OrientedRect {
        OrientedRect {
            cx,
            cy,
            width: w,
            height: h,
            angle_radians: 0.0,
        }
    }

    /// Full-lattice sample buffer at `default`, with the listed (index, value)
    /// overrides applied.
    fn samples(len: usize, default: u8, overrides: &[(usize, u8)]) -> Vec<u8> {
        let mut v = vec![default; len];
        for &(i, val) in overrides {
            v[i] = val;
        }
        v
    }

    fn cfg() -> MonitorConfig {
        MonitorConfig {
            warmup_frames: 4,
            glyph_var_threshold: 50.0,
            change_threshold: 40,
            box_coherence_frac: 0.4,
            min_glyph_holes: 3,
            scroll_frac: 0.7,
            scroll_min_boxes: 2,
        }
    }

    #[test]
    fn lattice_grid_and_membership() {
        let lat = Lattice::build(100, 100, 10);
        assert_eq!(lat.len(), 100); // 10×10 centred points
        // A band covering y∈[70,90], x∈[10,90].
        let band = rect(50.0, 80.0, 80.0, 20.0);
        let holes = lat.holes_in_rect(&band);
        // x ∈ {15..85} = 8 columns, y ∈ {75,85} = 2 rows.
        assert_eq!(holes.len(), 16);
        for &h in &holes {
            let p = lat.points()[h];
            assert!(p.x >= 10.0 && p.x <= 90.0 && p.y >= 70.0 && p.y <= 90.0);
        }
    }

    #[test]
    fn hole_variance_separates_stable_from_moving() {
        let mut stable = HoleStats::default();
        let mut moving = HoleStats::default();
        for f in 0..8 {
            stable.observe(GLYPH as f32);
            moving.observe(if f % 2 == 0 { 60.0 } else { 200.0 });
        }
        assert!(stable.variance() < 1.0);
        assert!(moving.variance() > 50.0);
    }

    /// Split a box's holes into a glyph half and a background half. Returns
    /// `(all_holes, glyph_holes, video_holes, bootstrap)` where `bootstrap` is the
    /// glyph mask aligned to `all_holes`.
    fn split_box(
        lat: &Lattice,
        r: &OrientedRect,
    ) -> (Vec<usize>, Vec<usize>, Vec<usize>, Vec<bool>) {
        let all = lat.holes_in_rect(r);
        let half = all.len() / 2;
        let glyph: Vec<usize> = all[..half].to_vec();
        let video: Vec<usize> = all[half..].to_vec();
        let bootstrap: Vec<bool> = (0..all.len()).map(|k| k < half).collect();
        (all, glyph, video, bootstrap)
    }

    fn clean_frame(lat: &Lattice, glyph_holes: &[usize]) -> Vec<u8> {
        samples(
            lat.len(),
            BG,
            &glyph_holes.iter().map(|&h| (h, GLYPH)).collect::<Vec<_>>(),
        )
    }

    #[test]
    fn subtitle_text_change_trips_only_its_box() {
        let lat = Lattice::build(100, 100, 10);
        let sub = rect(50.0, 80.0, 80.0, 20.0);
        let (all, glyph, video, boot) = split_box(&lat, &sub);
        let mut mon = ScreenMonitor::new(lat, cfg());
        let clean = clean_frame(mon.lattice(), &glyph);
        mon.set_box(1, all, boot, &clean);

        // A square moving outside the box, plus video flicker on the box's
        // background holes — text held steady. Always Quiet.
        for f in 0..8 {
            let mut ovr: Vec<(usize, u8)> = glyph.iter().map(|&h| (h, GLYPH)).collect();
            for (j, &h) in video.iter().enumerate() {
                ovr.push((h, if (f + j) % 2 == 0 { 70 } else { 150 }));
            }
            ovr.push((0, if f % 2 == 0 { 30 } else { 220 })); // square, far from box
            let s = samples(mon.lattice().len(), BG, &ovr);
            assert_eq!(
                mon.observe(&s),
                FrameClassification::Quiet,
                "frame {f}: only background/square moved"
            );
        }

        // Subtitle advances: the glyph holes now read background — large delta on
        // every glyph hole.
        let changed = samples(
            mon.lattice().len(),
            BG,
            &glyph.iter().map(|&h| (h, 50)).collect::<Vec<_>>(),
        );
        assert_eq!(
            mon.observe(&changed),
            FrameClassification::BoxesChanged(vec![1])
        );
    }

    #[test]
    fn moving_background_never_trips_a_still_subtitle() {
        let lat = Lattice::build(100, 100, 10);
        let sub = rect(50.0, 80.0, 80.0, 20.0);
        let (all, glyph, video, boot) = split_box(&lat, &sub);
        let mut mon = ScreenMonitor::new(lat, cfg());
        let clean = clean_frame(mon.lattice(), &glyph);
        mon.set_box(1, all, boot, &clean);

        for f in 0..30 {
            let mut ovr: Vec<(usize, u8)> = glyph.iter().map(|&h| (h, GLYPH)).collect();
            for (j, &h) in video.iter().enumerate() {
                // Wide-swinging background, never the text.
                ovr.push((h, ((f * 37 + j * 53) % 256) as u8));
            }
            let s = samples(mon.lattice().len(), BG, &ovr);
            assert_eq!(mon.observe(&s), FrameClassification::Quiet, "frame {f}");
        }
    }

    #[test]
    fn learned_mask_drops_video_holes_the_bootstrap_misread_as_ink() {
        // Bootstrap marks ALL of the box's holes as glyph, but half of them are
        // actually moving background. After warmup, variance must drop those, so a
        // pure-background frame is Quiet (it would have tripped on the bootstrap).
        let lat = Lattice::build(100, 100, 10);
        let sub = rect(50.0, 80.0, 80.0, 20.0);
        let all = lat.holes_in_rect(&sub);
        let half = all.len() / 2;
        let real_glyph: Vec<usize> = all[..half].to_vec();
        let video: Vec<usize> = all[half..].to_vec();
        let boot = vec![true; all.len()]; // appearance says everything is ink

        let mut mon = ScreenMonitor::new(lat, cfg());
        let clean = clean_frame(mon.lattice(), &real_glyph); // video holes baseline = BG
        mon.set_box(1, all, boot, &clean);

        // Warm up: real glyph steady at GLYPH, video holes swing hard.
        for f in 0..cfg().warmup_frames as usize {
            let mut ovr: Vec<(usize, u8)> = real_glyph.iter().map(|&h| (h, GLYPH)).collect();
            for (j, &h) in video.iter().enumerate() {
                ovr.push((h, ((f * 41 + j * 17) % 256) as u8));
            }
            let s = samples(mon.lattice().len(), BG, &ovr);
            mon.observe(&s);
        }

        // Post-warmup background-only swing: the learned mask has excluded the
        // video holes, so the steady real glyph keeps it Quiet.
        for f in 0..10 {
            let mut ovr: Vec<(usize, u8)> = real_glyph.iter().map(|&h| (h, GLYPH)).collect();
            for (j, &h) in video.iter().enumerate() {
                ovr.push((h, ((f * 91 + j * 7) % 256) as u8));
            }
            let s = samples(mon.lattice().len(), BG, &ovr);
            assert_eq!(mon.observe(&s), FrameClassification::Quiet, "frame {f}");
        }
    }

    #[test]
    fn all_boxes_moving_together_is_a_scroll() {
        let lat = Lattice::build(100, 100, 10);
        let b1 = rect(50.0, 30.0, 80.0, 20.0);
        let b2 = rect(50.0, 70.0, 80.0, 20.0);
        let (all1, g1, _v1, boot1) = split_box(&lat, &b1);
        let (all2, g2, _v2, boot2) = split_box(&lat, &b2);
        let mut mon = ScreenMonitor::new(lat, cfg());
        let mut clean = clean_frame(mon.lattice(), &g1);
        for &h in &g2 {
            clean[h] = GLYPH;
        }
        mon.set_box(1, all1, boot1, &clean);
        mon.set_box(2, all2, boot2, &clean);

        // Hold steady through warmup so the glyph mask freezes on stable frames.
        for _ in 0..cfg().warmup_frames {
            assert_eq!(mon.observe(&clean), FrameClassification::Quiet);
        }

        // Everything translates: every glyph hole in both boxes now reads
        // background.
        let mut scrolled = vec![BG; mon.lattice().len()];
        for &h in g1.iter().chain(&g2) {
            scrolled[h] = 40;
        }
        assert_eq!(mon.observe(&scrolled), FrameClassification::Scroll);
    }

    #[test]
    fn one_box_changing_among_many_is_not_a_scroll() {
        let lat = Lattice::build(100, 100, 10);
        let b1 = rect(50.0, 30.0, 80.0, 20.0);
        let b2 = rect(50.0, 70.0, 80.0, 20.0);
        let (all1, g1, _v1, boot1) = split_box(&lat, &b1);
        let (all2, g2, _v2, boot2) = split_box(&lat, &b2);
        let mut mon = ScreenMonitor::new(lat, cfg());
        let mut clean = clean_frame(mon.lattice(), &g1);
        for &h in &g2 {
            clean[h] = GLYPH;
        }
        mon.set_box(1, all1, boot1, &clean);
        mon.set_box(2, all2, boot2, &clean);
        mon.observe(&clean);

        // Only box 1's text changes; box 2 holds.
        let mut frame = clean.clone();
        for &h in &g1 {
            frame[h] = 40;
        }
        assert_eq!(
            mon.observe(&frame),
            FrameClassification::BoxesChanged(vec![1])
        );
    }

    #[test]
    fn re_acquire_resets_baseline_and_relearns() {
        let lat = Lattice::build(100, 100, 10);
        let sub = rect(50.0, 80.0, 80.0, 20.0);
        let (all, glyph, _video, boot) = split_box(&lat, &sub);
        let mut mon = ScreenMonitor::new(lat, cfg());

        let clean_a = clean_frame(mon.lattice(), &glyph);
        mon.set_box(1, all.clone(), boot.clone(), &clean_a);
        mon.observe(&clean_a);

        // New text after the targeted re-OCR: new baseline where glyph holes read
        // a different stroke value. Observing that same frame is then Quiet.
        let clean_b = samples(
            mon.lattice().len(),
            BG,
            &glyph.iter().map(|&h| (h, 50)).collect::<Vec<_>>(),
        );
        mon.set_box(1, all, boot, &clean_b);
        assert_eq!(mon.observe(&clean_b), FrameClassification::Quiet);
    }
}
