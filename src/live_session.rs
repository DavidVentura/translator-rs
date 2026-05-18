//! Cross-platform live-translate session state.
//!
//! Holds the orchestration state that's shared between the Android
//! `LivePlanarTracker` (uniffi-exposed) and the `surface_sim`
//! desktop binary. The goal is **one source of truth** for the
//! pipeline: tracking → detection → surface-map update → recognition
//! → translation → overlay update. Without this, every new feature
//! (re-OCR triggers, pixel stripes, etc.) has to be implemented
//! twice and the two implementations drift.
//!
//! This module starts small: it owns the persistent `SurfaceMap`
//! and is grown in subsequent phases to encompass the engine,
//! overlay store, matting cache, and the orchestration methods
//! themselves. See FUTURE_SURFACE_MAP.md for the bigger plan.

use std::sync::Mutex;

use crate::ocr::OrientedRect;
use crate::surface_map::{AddResult, SurfaceLineId, SurfaceLineObservation, SurfaceMap};

/// One rasterized overlay item resident across composite calls. The
/// caller hashes the source content (strips + texts + language) and
/// only re-rasterizes items whose hash changed, so dense pages with
/// stable content stay cheap to render.
#[derive(Clone)]
pub struct OverlayItem {
    pub id: u64,
    /// RGBA bitmap in canonical (surface) coords.
    pub bitmap: Vec<u8>,
    pub width: u32,
    pub height: u32,
    /// Where the bitmap's top-left sits in surface coords. The
    /// compositor warps from this origin through the per-frame H.
    pub surface_origin_x: f32,
    pub surface_origin_y: f32,
    /// Hash of (strips + display text + language). Used to skip
    /// re-raster when content is unchanged across acquires.
    pub content_hash: u64,
}

/// Lifetime-bound state shared between platform wrappers (Android
/// `LivePlanarTracker`, desktop `surface_sim`). One instance per
/// active session; cleared on reset (tap-to-focus, language change).
pub struct LiveSession {
    /// Per-physical-line storage across acquires. The `same_line`
    /// predicate's geometric matching makes this safe across
    /// scene changes — stale entries from a different scene stay
    /// inert (no false merges) until `clear()` is called.
    pub surface_map: Mutex<SurfaceMap>,
    /// Resident rasterized overlays, keyed by block id. Populated
    /// by `upsert_overlay_item` and consumed by the compositor.
    pub overlay_items: Mutex<Vec<OverlayItem>>,
}

impl LiveSession {
    pub fn new() -> Self {
        Self {
            surface_map: Mutex::new(SurfaceMap::new()),
            overlay_items: Mutex::new(Vec::new()),
        }
    }

    /// Drop all session state. Caller invokes on tap-to-focus,
    /// language change, or any other coarse-grained reset signal.
    pub fn clear(&self) {
        if let Ok(mut map) = self.surface_map.lock() {
            map.clear();
        }
        if let Ok(mut items) = self.overlay_items.lock() {
            items.clear();
        }
    }

    pub fn clear_overlays(&self) {
        if let Ok(mut items) = self.overlay_items.lock() {
            items.clear();
        }
    }

    /// Feed a batch of detections (in surface coords) into the
    /// surface map and return per-detection outcomes the caller
    /// uses to (a) decide which detections need recognition, and
    /// (b) push rec results back via [`Self::ingest_rec`].
    ///
    /// `source_language` is used as the default for newly-created
    /// lines; existing lines keep their previously-recorded
    /// language unless the observation carries a non-empty value.
    ///
    /// Replaces the open-coded `for entry in entries: map.add_or_merge`
    /// loops in `uniffi_catalog::run_acquire_pipeline` and the
    /// simulator's `run_detection_cycle`.
    pub fn observe_detections(
        &self,
        detections: &[OrientedRect],
        source_language: &str,
    ) -> Vec<DetectionOutcome> {
        let mut out = Vec::with_capacity(detections.len());
        let mut map = match self.surface_map.lock() {
            Ok(m) => m,
            Err(_) => {
                return detections
                    .iter()
                    .map(|_| DetectionOutcome::poisoned())
                    .collect();
            }
        };
        for d in detections {
            let obs = SurfaceLineObservation {
                bbox: d.clone(),
                source_text: String::new(),
                translated_text: String::new(),
                source_language: source_language.to_string(),
            };
            let res = map.add_or_merge(obs);
            let needs_rec = res.needs_rec();
            let line_id = res.id();
            let kind = AddResultKind::from(&res);
            let cached_source_text = if !needs_rec {
                map.get(line_id)
                    .map(|l| l.source_text.clone())
                    .unwrap_or_default()
            } else {
                String::new()
            };
            let cached_source_language = if !needs_rec {
                map.get(line_id)
                    .map(|l| l.source_language.clone())
                    .unwrap_or_default()
            } else {
                String::new()
            };
            out.push(DetectionOutcome {
                line_id,
                kind,
                needs_rec,
                cached_source_text,
                cached_source_language,
            });
        }
        out
    }

    /// Push a single recognized line back into the map: store the
    /// text and language, and snapshot the line's current u-extent
    /// as "rec just saw up to here" so future observations that
    /// extend past it trigger re-recognition.
    pub fn ingest_rec(&self, line_id: SurfaceLineId, source_text: &str, source_language: &str) {
        if let Ok(mut map) = self.surface_map.lock() {
            if let Some(line) = map.get_mut(line_id) {
                line.source_text = source_text.to_string();
                if !source_language.is_empty() {
                    line.source_language = source_language.to_string();
                }
                line.record_rec_extent();
            }
        }
    }

    /// Push translated text back into a set of lines (all
    /// recipients receive the same string — caller has already
    /// performed block-level translation across the joined source
    /// strings).
    pub fn ingest_translation(&self, line_ids: &[SurfaceLineId], translated: &str) {
        if let Ok(mut map) = self.surface_map.lock() {
            for &id in line_ids {
                if let Some(line) = map.get_mut(id) {
                    line.translated_text = translated.to_string();
                }
            }
        }
    }
}

/// Per-detection result from [`LiveSession::observe_detections`].
#[derive(Clone, Debug)]
pub struct DetectionOutcome {
    pub line_id: SurfaceLineId,
    pub kind: AddResultKind,
    /// True when caller should run recognition for this line. False
    /// for cache hits (`MergedUnchanged` on a line with text).
    pub needs_rec: bool,
    /// Cached source text from a prior rec, when `!needs_rec`.
    /// Empty otherwise.
    pub cached_source_text: String,
    /// Cached source language from a prior rec, when `!needs_rec`.
    /// Empty otherwise.
    pub cached_source_language: String,
}

impl DetectionOutcome {
    fn poisoned() -> Self {
        Self {
            line_id: 0,
            kind: AddResultKind::Created,
            needs_rec: true,
            cached_source_text: String::new(),
            cached_source_language: String::new(),
        }
    }
}

/// Human-readable variant of `AddResult` for diagnostics + visual
/// color coding in the simulator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AddResultKind {
    Created,
    MergedAndExtended,
    MergedUnchanged,
}

impl From<&AddResult> for AddResultKind {
    fn from(r: &AddResult) -> Self {
        match r {
            AddResult::Created(_) => AddResultKind::Created,
            AddResult::MergedAndExtended(_) => AddResultKind::MergedAndExtended,
            AddResult::MergedUnchanged(_) => AddResultKind::MergedUnchanged,
        }
    }
}

impl Default for LiveSession {
    fn default() -> Self {
        Self::new()
    }
}
