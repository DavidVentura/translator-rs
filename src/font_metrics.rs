//! Per-font glyph-advance measurement for accurate text wrapping, plus the
//! supporting metadata the PDF writer needs to embed the same font in the
//! output (phase 2).

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

#[derive(Debug, Clone, Copy)]
pub struct GlyphInfo {
    /// Glyph ID inside the font.
    pub gid: u16,
    /// Horizontal advance in font units (i.e. needs `font_size / units_per_em`
    /// scaling to become PDF points).
    pub advance: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FontFileKind {
    TrueType,
    OpenTypeCff,
}

/// Salient `head`/`hhea`/`OS/2` fields needed to build the PDF
/// `/FontDescriptor` dict.
#[derive(Debug, Clone)]
pub struct FontDescriptorInfo {
    pub postscript_name: String,
    pub italic: bool,
    pub bold: bool,
    pub monospace: bool,
    pub italic_angle: f32,
    pub ascent: i16,
    pub descent: i16,
    pub cap_height: i16,
    pub bbox: (i16, i16, i16, i16),
    pub kind: FontFileKind,
}

/// How to convert a stretch of characters into a width in PDF points.
///
/// - [`FontMetrics::Approx`] — legacy `chars × font_size × em_advance`. Used
///   when no font is available (Standard-14 / no provider).
/// - [`FontMetrics::Real`] — built from a parsed TTF/OTF/TTC. Each char's
///   advance comes from the font's `hmtx` table; `bytes` carries the parsed
///   file so the writer can stream it into a `/FontFile2` later without
///   re-reading the disk.
#[derive(Clone)]
pub enum FontMetrics {
    Approx {
        em_advance: f32,
    },
    Real {
        units_per_em: u16,
        glyphs: HashMap<char, GlyphInfo>,
        /// Glyph 0 (`.notdef`), used for codepoints with no cmap entry.
        fallback: GlyphInfo,
        /// Source font bytes — shared via `Arc` so cloning per-block doesn't
        /// duplicate large CJK files.
        bytes: Arc<Vec<u8>>,
        ttc_index: u32,
        descriptor: FontDescriptorInfo,
    },
}

#[derive(Debug)]
pub enum FontMetricsError {
    Io(std::io::Error),
    Parse(String),
}

impl std::fmt::Display for FontMetricsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io: {e}"),
            Self::Parse(msg) => write!(f, "parse: {msg}"),
        }
    }
}

impl std::error::Error for FontMetricsError {}

impl From<std::io::Error> for FontMetricsError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl FontMetrics {
    pub fn approx(em_advance: f32) -> Self {
        Self::Approx { em_advance }
    }

    /// Parse a `.ttf` / `.otf` / `.ttc` and pre-resolve glyph info for
    /// codepoints in `text`. Subsequent characters not in `text` look up via
    /// `.notdef`.
    pub fn from_file_for_text(
        path: impl AsRef<Path>,
        ttc_index: u32,
        text: &str,
    ) -> Result<Self, FontMetricsError> {
        let data = fs::read(path)?;
        Self::from_bytes_for_text(&data, ttc_index, text)
    }

    pub fn from_bytes_for_text(
        data: &[u8],
        ttc_index: u32,
        text: &str,
    ) -> Result<Self, FontMetricsError> {
        let bytes = Arc::new(data.to_vec());
        Self::from_arc_for_text(bytes, ttc_index, text)
    }

    pub fn from_arc_for_text(
        bytes: Arc<Vec<u8>>,
        ttc_index: u32,
        text: &str,
    ) -> Result<Self, FontMetricsError> {
        let face = ttf_parser::Face::parse(bytes.as_slice(), ttc_index)
            .map_err(|e| FontMetricsError::Parse(format!("{e:?}")))?;
        let units_per_em = face.units_per_em();

        let mut glyphs: HashMap<char, GlyphInfo> = HashMap::new();
        for ch in text.chars() {
            if glyphs.contains_key(&ch) {
                continue;
            }
            if let Some(gid) = face.glyph_index(ch) {
                let advance = face.glyph_hor_advance(gid).unwrap_or(0);
                glyphs.insert(
                    ch,
                    GlyphInfo {
                        gid: gid.0,
                        advance,
                    },
                );
            }
        }

        let notdef = ttf_parser::GlyphId(0);
        let fallback = GlyphInfo {
            gid: 0,
            advance: face
                .glyph_hor_advance(notdef)
                .unwrap_or((units_per_em / 2).max(1)),
        };

        let descriptor = build_descriptor(&face, bytes.as_slice());

        Ok(Self::Real {
            units_per_em,
            glyphs,
            fallback,
            bytes,
            ttc_index,
            descriptor,
        })
    }

    /// `(bytes, ttc_index, descriptor)` when this is a real font; used by
    /// the writer to embed the font in the output PDF.
    pub fn embedding_source(&self) -> Option<(Arc<Vec<u8>>, u32, &FontDescriptorInfo)> {
        match self {
            Self::Real {
                bytes,
                ttc_index,
                descriptor,
                ..
            } => Some((bytes.clone(), *ttc_index, descriptor)),
            Self::Approx { .. } => None,
        }
    }

    /// `(gid, advance)` for hex-Tj emission with a Type-0 CID-keyed font.
    /// Falls back to `.notdef` for chars outside the font's cmap.
    pub fn glyph_for(&self, c: char) -> Option<GlyphInfo> {
        match self {
            Self::Real {
                glyphs, fallback, ..
            } => Some(glyphs.get(&c).copied().unwrap_or(*fallback)),
            Self::Approx { .. } => None,
        }
    }

    pub fn units_per_em(&self) -> Option<u16> {
        match self {
            Self::Real { units_per_em, .. } => Some(*units_per_em),
            Self::Approx { .. } => None,
        }
    }

    /// Width of `text` rendered at `font_size` in PDF points.
    pub fn measure(&self, text: &str, font_size: f32) -> f32 {
        match self {
            Self::Approx { em_advance } => text.chars().count() as f32 * font_size * em_advance,
            Self::Real {
                units_per_em,
                glyphs,
                fallback,
                ..
            } => {
                let scale = font_size / *units_per_em as f32;
                let total_units: u32 = text
                    .chars()
                    .map(|c| {
                        let g = glyphs.get(&c).copied().unwrap_or(*fallback);
                        g.advance as u32
                    })
                    .sum();
                total_units as f32 * scale
            }
        }
    }
}

fn build_descriptor(face: &ttf_parser::Face<'_>, raw: &[u8]) -> FontDescriptorInfo {
    let bbox = face.global_bounding_box();
    let postscript_name = face
        .names()
        .into_iter()
        .find_map(|name| {
            if name.name_id == ttf_parser::name_id::POST_SCRIPT_NAME {
                name.to_string()
            } else {
                None
            }
        })
        .unwrap_or_else(|| "Embedded".to_string());

    let kind = if raw.starts_with(b"OTTO") || face.tables().cff.is_some() {
        FontFileKind::OpenTypeCff
    } else {
        FontFileKind::TrueType
    };

    FontDescriptorInfo {
        postscript_name,
        italic: face.is_italic(),
        bold: face.is_bold(),
        monospace: face.is_monospaced(),
        italic_angle: face.italic_angle(),
        ascent: face.ascender(),
        descent: face.descender(),
        cap_height: face.capital_height().unwrap_or(face.ascender()),
        bbox: (bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max),
        kind,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn approx_matches_legacy_formula() {
        let m = FontMetrics::approx(0.5);
        assert!((m.measure("abcdefghij", 12.0) - 60.0).abs() < 0.001);
    }

    #[test]
    fn approx_empty_string_zero() {
        let m = FontMetrics::approx(0.5);
        assert_eq!(m.measure("", 12.0), 0.0);
    }
}
