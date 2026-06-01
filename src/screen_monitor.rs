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
    cols: u32,
    rows: u32,
    spacing: u32,
    /// Centre offset of the first point: `spacing / 2`. A point's canonical
    /// position is `(origin + col*spacing, origin + row*spacing)`.
    origin: f32,
}

impl Lattice {
    /// A regular `spacing`-pitch grid centred in each cell. Alternating-row /
    /// staggered hole patterns are a rendering concern (hole visibility); the
    /// classifier only needs the point positions, so a plain grid is enough here.
    /// Points are emitted row-major (y outer, x inner), so index `row*cols + col`
    /// — the same order the GPU readback grid is read back in.
    pub fn build(canon_w: u32, canon_h: u32, spacing: u32) -> Self {
        let spacing = spacing.max(1);
        let half = spacing as f32 * 0.5;
        let cols = (canon_w.saturating_sub(1) / spacing) + 1;
        let rows = (canon_h.saturating_sub(1) / spacing) + 1;
        let mut points = Vec::with_capacity((cols * rows) as usize);
        for row in 0..rows {
            let y = half + (row * spacing) as f32;
            for col in 0..cols {
                let x = half + (col * spacing) as f32;
                points.push(LatticePoint { x, y });
            }
        }
        Lattice {
            points,
            cols,
            rows,
            spacing,
            origin: half,
        }
    }

    /// Grid `(cols, rows)` for a `canon_w×canon_h` frame at `spacing`, without
    /// allocating the points — the same counts [`build`](Self::build) produces, so
    /// the GPU readback can be sized to match before a `Lattice` exists.
    pub fn dims(canon_w: u32, canon_h: u32, spacing: u32) -> (u32, u32) {
        let spacing = spacing.max(1);
        let cols = (canon_w.saturating_sub(1) / spacing) + 1;
        let rows = (canon_h.saturating_sub(1) / spacing) + 1;
        (cols, rows)
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

    pub fn cols(&self) -> u32 {
        self.cols
    }

    pub fn rows(&self) -> u32 {
        self.rows
    }

    pub fn spacing(&self) -> u32 {
        self.spacing
    }

    pub fn origin(&self) -> f32 {
        self.origin
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
    /// A box needs at least this many stable ink holes to be judged on its text.
    /// A box that can't reach this (low-contrast / saturated recovery, no stable
    /// strokes) falls back to the gross whole-box test below, so it can never
    /// become an immortal pill that blocks re-OCR of its region.
    pub min_glyph_holes: usize,
    /// Fallback for a box with too few stable ink holes: fraction of *all* its
    /// holes that must change vs the baseline to trip it. Higher than
    /// [`box_coherence_frac`] so partial background motion is tolerated and only a
    /// wholesale change (scroll / navigation / app-switch) clears it.
    pub gross_change_frac: f32,
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
            gross_change_frac: 0.7,
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
        // Not judged until warmup completes: a freshly-added box's baseline is
        // captured before our overlay has settled into the captured mirror, so
        // trusting it early reads the overlay appearing as a "change" (self-trip).
        if self.learned_glyph.is_none() {
            return BoxDeviation {
                id: self.id,
                glyph_holes: 0,
                deviating: 0,
                gross: false,
                max_delta: 0,
            };
        }
        let mask = self.effective_glyph();
        let mut ink_holes = 0usize;
        let mut ink_dev = 0usize;
        let mut all_dev = 0usize;
        let mut max_delta = 0u32;
        for (k, &hi) in self.holes.iter().enumerate() {
            let delta = (samples[hi] as i32 - self.baseline[k] as i32).unsigned_abs();
            max_delta = max_delta.max(delta);
            let changed = delta > cfg.change_threshold as u32;
            if changed {
                all_dev += 1;
            }
            if mask[k] {
                ink_holes += 1;
                if changed {
                    ink_dev += 1;
                }
            }
        }
        if ink_holes >= cfg.min_glyph_holes {
            BoxDeviation {
                id: self.id,
                glyph_holes: ink_holes,
                deviating: ink_dev,
                gross: false,
                max_delta,
            }
        } else {
            // No stable text strokes to track (low-contrast / saturated recovery):
            // fall back to a whole-box change test so the box can still be cleared
            // and never becomes an immortal pill blocking re-OCR of its region.
            BoxDeviation {
                id: self.id,
                glyph_holes: self.holes.len(),
                deviating: all_dev,
                gross: true,
                max_delta,
            }
        }
    }
}

struct BoxDeviation {
    id: u64,
    glyph_holes: usize,
    deviating: usize,
    /// Judged on all holes (no stable ink strokes) rather than on the glyph mask.
    gross: bool,
    /// Largest single-hole `|current − baseline|` over the box, for debugging: ~0
    /// means the box reads a constant (recovery isn't tracking the screen there);
    /// moderate-but-under-threshold means the per-hole threshold is too high.
    max_delta: u32,
}

impl BoxDeviation {
    fn judgeable(&self, cfg: &MonitorConfig) -> bool {
        self.glyph_holes >= cfg.min_glyph_holes
    }

    fn changed(&self, cfg: &MonitorConfig) -> bool {
        let frac = if self.gross {
            cfg.gross_change_frac
        } else {
            cfg.box_coherence_frac
        };
        self.judgeable(cfg) && self.deviating as f32 >= frac * self.glyph_holes as f32
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
    /// Diagnostic summary of the last `observe`: (judgeable boxes, total glyph
    /// holes across them, total deviating). If glyph holes collapse to ~0 after
    /// warmup, the variance mask is dropping everything (recovery too noisy).
    last_stats: (usize, usize, usize),
    /// The judged box with the highest deviating fraction last observe:
    /// (id, glyph_holes, deviating). Lets us see a subtitle box's own fraction
    /// rather than the totals (which dilute it across stable boxes).
    last_top: Option<(u64, usize, usize)>,
    /// Per-box `(id, monitored_holes, deviating, gross, max_delta)` from the last
    /// observe, for debug logging joined with each block's text.
    last_devs: Vec<(u64, usize, usize, bool, u32)>,
}

impl ScreenMonitor {
    pub fn new(lattice: Lattice, cfg: MonitorConfig) -> Self {
        let stats = vec![HoleStats::default(); lattice.len()];
        ScreenMonitor {
            lattice,
            stats,
            boxes: Vec::new(),
            cfg,
            last_stats: (0, 0, 0),
            last_top: None,
            last_devs: Vec::new(),
        }
    }

    /// Per-box `(id, monitored_holes, deviating, gross, max_delta)` from last observe.
    pub fn debug_boxes(&self) -> &[(u64, usize, usize, bool, u32)] {
        &self.last_devs
    }

    /// (judgeable boxes, total glyph holes, total deviating) from the last observe.
    pub fn debug_last_stats(&self) -> (usize, usize, usize) {
        self.last_stats
    }

    /// The judged box with the highest deviating fraction last observe:
    /// (id, glyph_holes, deviating).
    pub fn debug_top_box(&self) -> Option<(u64, usize, usize)> {
        self.last_top
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

    /// Drop all boxes (the resident overlay was replaced); per-hole stats are left
    /// to be reset by the next `set_box`.
    pub fn clear_boxes(&mut self) {
        self.boxes.clear();
    }

    /// Mark `mask[h] = true` for every lattice hole any box covers (our overlay's
    /// footprint). Used to exclude our own pills from the global-motion signal so
    /// the overlay's own redraw isn't read as screen motion.
    pub fn fill_covered(&self, mask: &mut [bool]) {
        for b in &self.boxes {
            for &h in &b.holes {
                if let Some(m) = mask.get_mut(h) {
                    *m = true;
                }
            }
        }
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
                // Re-baseline to the now-settled frame: the initial baseline was
                // captured before our overlay reached the captured mirror, so it
                // held source where the overlay now sits. Snapshot current so a
                // stable overlay reads as no-change.
                for (k, &h) in b.holes.iter().enumerate() {
                    b.baseline[k] = samples[h];
                }
            }
        }
        let devs: Vec<BoxDeviation> = self
            .boxes
            .iter()
            .map(|b| b.deviation(samples, cfg))
            .collect();
        let judged = devs.iter().filter(|d| d.judgeable(cfg)).count();
        let total_glyph: usize = devs.iter().map(|d| d.glyph_holes).sum();
        let total_dev: usize = devs.iter().map(|d| d.deviating).sum();
        self.last_stats = (judged, total_glyph, total_dev);
        self.last_top = devs
            .iter()
            .filter(|d| d.judgeable(cfg))
            .max_by(|a, b| {
                let fa = a.deviating as f32 / a.glyph_holes.max(1) as f32;
                let fb = b.deviating as f32 / b.glyph_holes.max(1) as f32;
                fa.total_cmp(&fb)
            })
            .map(|d| (d.id, d.glyph_holes, d.deviating));
        self.last_devs = devs
            .iter()
            .map(|d| (d.id, d.glyph_holes, d.deviating, d.gross, d.max_delta))
            .collect();
        classify(&devs, cfg)
    }
}

/// Pure decision from the per-box deviations. Scroll is tested first (anchored
/// text across many boxes all moving at once), then per-box change.
fn classify(devs: &[BoxDeviation], cfg: &MonitorConfig) -> FrameClassification {
    // Scroll is an ink-hole signal (anchored strokes moving together); gross boxes
    // have no strokes to anchor, so they only ever contribute via per-box `changed`
    // (which the pipeline escalates to a drop-all when enough boxes trip at once).
    let judged: Vec<&BoxDeviation> = devs
        .iter()
        .filter(|d| d.judgeable(cfg) && !d.gross)
        .collect();
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
            gross_change_frac: 0.7,
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
        // Warm up so the boxes are judged (and re-baselined to the stable frame).
        for _ in 0..cfg().warmup_frames + 1 {
            mon.observe(&clean);
        }

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

    #[test]
    fn box_without_ink_holes_still_clears_on_a_wholesale_change() {
        // A box whose recovered content has no stable strokes (bootstrap marks
        // nothing as ink) learns zero glyph holes. Without the gross-change
        // fallback it could never trip → an immortal pill that blocks its region.
        // With it: stays put while unchanged, tolerates partial motion, and is
        // cleared by a wholesale change (scroll / app-switch).
        let lat = Lattice::build(100, 100, 10);
        let region = rect(50.0, 50.0, 80.0, 80.0);
        let all = lat.holes_in_rect(&region);
        assert!(all.len() >= 8, "need enough holes to exercise fractions");
        let boot = vec![false; all.len()]; // appearance says nothing is ink
        let mut mon = ScreenMonitor::new(lat, cfg());
        let clean = vec![BG; mon.lattice().len()];
        mon.set_box(1, all.clone(), boot, &clean);

        // Warm up on the stable clean frame: must never trip during warmup.
        for _ in 0..cfg().warmup_frames + 1 {
            assert_eq!(mon.observe(&clean), FrameClassification::Quiet);
        }

        // Same content: stays put (the translated label must persist).
        assert_eq!(mon.observe(&clean), FrameClassification::Quiet);

        // Partial change (a third of the holes) — below gross_change_frac: tolerated.
        let mut partial = clean.clone();
        for &h in all.iter().take(all.len() / 3) {
            partial[h] = 240;
        }
        assert_eq!(mon.observe(&partial), FrameClassification::Quiet);

        // Wholesale change (every hole) — a scene swap: clear the box.
        let mut swapped = vec![BG; mon.lattice().len()];
        for &h in &all {
            swapped[h] = 240;
        }
        assert_eq!(
            mon.observe(&swapped),
            FrameClassification::BoxesChanged(vec![1])
        );
    }
}
