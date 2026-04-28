//! Font lookup interface for PDF output.
//!
//! The translator crate is platform-agnostic: it does not know about
//! `/system/fonts`, `fonts.xml`, `AFontMatcher`, fontconfig, CoreText, etc.
//! Consumer applications (Android, native Linux, …) own that knowledge and
//! expose it through this trait.
//!
//! The PDF writer asks for a font when it has decided what to render and what
//! style is needed. The provider returns a filesystem path to a TrueType /
//! OpenType file that covers the requested language + style; the writer reads
//! it for glyph metrics (real wrap widths) and embeds a subset in the output.

use std::path::PathBuf;

/// What the writer is asking for.
///
/// `language` is the BCP-47 tag of the *translated* text (e.g. `"en"`,
/// `"ja"`, `"zh-Hans"`, `"ar"`). Most fonts that cover a language already
/// include the Latin block, so a single font per block is enough — we don't
/// build per-codepoint fallback chains here.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FontRequest {
    pub language: String,
    pub bold: bool,
    pub italic: bool,
    pub monospace: bool,
}

/// Path + sub-font index. For `.ttf` / `.otf` set `ttc_index = 0`; for
/// `.ttc` collections (e.g. `NotoSansCJK-Regular.ttc`) the platform's font
/// API tells you which index inside the collection covers the requested
/// language (Android's `AFont_getCollectionIndex()`, fontconfig's `index`
/// property, the `index="N"` attribute in `fonts.xml`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FontHandle {
    pub path: PathBuf,
    pub ttc_index: u32,
}

impl FontHandle {
    pub fn new(path: impl Into<PathBuf>, ttc_index: u32) -> Self {
        Self {
            path: path.into(),
            ttc_index,
        }
    }
}

impl<P: Into<PathBuf>> From<P> for FontHandle {
    /// Convenience for the common single-font case where `ttc_index` is 0.
    fn from(path: P) -> Self {
        Self::new(path, 0)
    }
}

/// Resolves a [`FontRequest`] to a font file on disk.
///
/// Returning `None` means "no preference / unsupported"; the writer falls
/// back to the PDF Standard-14 path (Helvetica / Courier), which only works
/// for Latin-1 / WinAnsi.
pub trait FontProvider {
    fn locate(&self, request: &FontRequest) -> Option<FontHandle>;
}

/// Blanket impl so callers can pass a closure when a one-liner suffices,
/// e.g. integration tests: `&|_req| Some(FontHandle::from("/usr/share/.../X.ttf"))`.
impl<F> FontProvider for F
where
    F: Fn(&FontRequest) -> Option<FontHandle>,
{
    fn locate(&self, request: &FontRequest) -> Option<FontHandle> {
        self(request)
    }
}

/// Always returns `None`. Use when font discovery isn't wired up yet — the
/// writer keeps its current Standard-14 behavior.
pub struct NoFontProvider;

impl FontProvider for NoFontProvider {
    fn locate(&self, _request: &FontRequest) -> Option<FontHandle> {
        None
    }
}
