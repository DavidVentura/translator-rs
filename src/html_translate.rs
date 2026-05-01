//! HTML translation pipeline: parse on Rust side with html5ever, translate
//! plain text + alignment via slimt, write translated content back into the
//! same DOM nodes it came from, then serialise.
//!
//! The DOM tree is sacred: structural elements (`<p>`, `<a>`, `<button>`, …),
//! their attributes (`href`, `data-*`, `class`), and document order are never
//! mutated. Only the contents of each text leaf are replaced.
//!
//! Text leaves are grouped into "scopes" by their nearest non-inline ancestor
//! so the model gets full sentence context within a paragraph / list item /
//! heading rather than per-leaf snippets. Inline elements (`<em>`, `<b>`,
//! `<span>`, `<a>`, …) do not break a scope. Token alignments returned by
//! slimt route each translated character back to the source leaf its source
//! character belonged to.

use std::cell::RefCell;
use std::rc::Rc;

use html5ever::driver::ParseOpts;
use html5ever::serialize::{SerializeOpts, TraversalScope, serialize};
use html5ever::tendril::TendrilSink;
use html5ever::{QualName, local_name, ns, parse_fragment};
use markup5ever_rcdom::{Handle, NodeData, RcDom, SerializableHandle};

use crate::translate::{TokenAlignment, TranslationWithAlignment};

/// Elements whose subtree must be left verbatim — neither extracted as
/// translation source nor mutated. Two flavours of reason:
///
/// 1. Non-prose: `script`, `style`, `noscript`, `template`, `iframe`,
///    `textarea`, `svg`, `canvas`, `object`, `embed`. Translating their
///    contents is meaningless or actively breaks the page.
///
/// 2. Author-as-source-text intent: `code`, `pre`, `kbd`, `samp`, `var`,
///    `math`. The author marked these because the literal characters
///    matter (a function name, a key sequence, a sample value, an
///    equation). Running them through a translation model corrupts them.
///
/// Skip is a hard scope boundary: we flush the in-progress scope at entry,
/// don't walk into the subtree (so its text leaves never become
/// translation sources), and the next sibling starts a fresh scope. The
/// DOM tree itself stays untouched because we only mutate text leaves we
/// extracted, and we extracted none from inside.
fn is_skipped_subtree(local_name: &str) -> bool {
    matches!(
        local_name,
        "script"
            | "style"
            | "noscript"
            | "template"
            | "iframe"
            | "textarea"
            | "svg"
            | "canvas"
            | "object"
            | "embed"
            | "code"
            | "pre"
            | "kbd"
            | "samp"
            | "var"
            | "math"
    )
}

/// Author-level "do not translate" signals on an arbitrary element:
/// `translate="no"`, `class="notranslate"` (Google's de-facto convention),
/// and `contenteditable` (anything the user types into is not ours to
/// rewrite). Same skip semantics as `is_skipped_subtree`.
fn has_skip_attr(attrs: &[html5ever::Attribute]) -> bool {
    for a in attrs {
        let name: &str = &a.name.local;
        let value: &str = &a.value;
        if name == "translate" && value.eq_ignore_ascii_case("no") {
            return true;
        }
        if name == "contenteditable" {
            return true;
        }
        if name == "class" && value.split_ascii_whitespace().any(|c| c == "notranslate") {
            return true;
        }
    }
    false
}

/// Inline elements do not break a translation scope. Anything not in this
/// set is treated as block-level (its boundaries split scopes). The list
/// matches slimt's historical `inline_tags` set so behaviour with the new
/// pipeline stays close to the old C++ HTML mode.
fn is_inline(local_name: &str) -> bool {
    matches!(
        local_name,
        "a" | "abbr"
            | "b"
            | "bdi"
            | "bdo"
            | "br"
            | "cite"
            | "code"
            | "data"
            | "del"
            | "dfn"
            | "em"
            | "font"
            | "i"
            | "img"
            | "ins"
            | "kbd"
            | "label"
            | "mark"
            | "math"
            | "output"
            | "q"
            | "ruby"
            | "rb"
            | "rp"
            | "rt"
            | "s"
            | "samp"
            | "small"
            | "span"
            | "strong"
            | "sub"
            | "sup"
            | "time"
            | "tt"
            | "u"
            | "var"
            | "wbr"
    )
}

/// One contiguous run of text leaves whose nearest non-inline ancestor is the
/// same. The flat `text` is what we send to slimt; `leaf_char_ends` lets us
/// look up which leaf any source-text character belongs to.
struct Scope {
    leaves: Vec<Handle>,
    text: String,
    /// `leaf_char_ends[i]` is the cumulative char count *after* leaf `i`,
    /// so leaf `i` covers char range `leaf_char_ends[i-1]..leaf_char_ends[i]`
    /// (with an implicit 0 before index 0).
    leaf_char_ends: Vec<usize>,
}

struct ParsedFragment {
    dom: RcDom,
    scopes: Vec<Scope>,
    /// For each scope, its index in the flat `scope_texts` list passed to
    /// the translator. `None` means the scope is empty/whitespace-only and
    /// should be left untouched.
    translation_idx: Vec<Option<usize>>,
}

/// Result of `prepare` — owns the parsed DOMs and the flat list of scope
/// texts that the caller must translate (with alignment) before calling
/// `finish`.
pub struct PreparedHtml {
    fragments: Vec<ParsedFragment>,
}

/// Parse fragments and extract the per-scope plain text the caller should
/// pass to the translator. Returns the prepared state plus the texts in
/// flat order; the caller batches them in one `translate_with_alignment`
/// call and feeds the responses back via `finish`.
pub fn prepare(fragments: &[String]) -> (PreparedHtml, Vec<String>) {
    let mut all_texts = Vec::new();
    let mut parsed = Vec::with_capacity(fragments.len());
    for fragment in fragments {
        let dom = parse_fragment_dom(fragment);
        let scopes = collect_scopes(&dom);
        let mut translation_idx = Vec::with_capacity(scopes.len());
        for scope in &scopes {
            if scope.text.trim().is_empty() {
                translation_idx.push(None);
            } else {
                translation_idx.push(Some(all_texts.len()));
                all_texts.push(scope.text.clone());
            }
        }
        parsed.push(ParsedFragment {
            dom,
            scopes,
            translation_idx,
        });
    }
    (PreparedHtml { fragments: parsed }, all_texts)
}

/// Convenience: parse, translate via the supplied closure, reassemble. The
/// closure receives the per-scope flat texts and must return one
/// `TranslationWithAlignment` per input in the same order.
pub fn translate_html_with<F, E>(fragments: &[String], translate: F) -> Result<Vec<String>, E>
where
    F: FnOnce(&[String]) -> Result<Vec<TranslationWithAlignment>, E>,
{
    let (prepared, scope_texts) = prepare(fragments);
    let translations = if scope_texts.is_empty() {
        Vec::new()
    } else {
        translate(&scope_texts)?
    };
    Ok(finish(prepared, &translations))
}

/// Apply translation results back onto each parsed DOM and serialise to
/// HTML strings. `translations.len()` must equal the count returned by
/// `prepare` (the second tuple element).
pub fn finish(prepared: PreparedHtml, translations: &[TranslationWithAlignment]) -> Vec<String> {
    let mut out = Vec::with_capacity(prepared.fragments.len());
    for fragment in prepared.fragments {
        for (scope, idx) in fragment.scopes.iter().zip(fragment.translation_idx.iter()) {
            let Some(idx) = idx else { continue };
            let translation = &translations[*idx];
            apply_scope(scope, &translation.translated_text, &translation.alignments);
        }
        out.push(serialize_fragment_dom(&fragment.dom));
    }
    out
}

fn parse_fragment_dom(input: &str) -> RcDom {
    parse_fragment(
        RcDom::default(),
        ParseOpts::default(),
        QualName::new(None, ns!(html), local_name!("body")),
        Vec::new(),
        false,
    )
    .one(input)
}

fn collect_scopes(dom: &RcDom) -> Vec<Scope> {
    let mut state = ScopeCollector {
        scopes: Vec::new(),
        current: Vec::new(),
    };
    // `parse_fragment` wraps the input in an implicit context element under
    // `dom.document`. We walk that wrapper's children — i.e. just the user's
    // markup — and treat the wrapper itself as a scope-breaking root so a
    // leading text leaf still lives in its own scope cleanly.
    let document_children = dom.document.children.borrow();
    for child in document_children.iter() {
        state.walk(child);
    }
    state.flush();
    state.scopes
}

struct ScopeCollector {
    scopes: Vec<Scope>,
    current: Vec<Handle>,
}

impl ScopeCollector {
    fn flush(&mut self) {
        if self.current.is_empty() {
            return;
        }
        let leaves = std::mem::take(&mut self.current);
        let mut text = String::new();
        let mut leaf_char_ends = Vec::with_capacity(leaves.len());
        let mut total = 0usize;
        let mut prev_text_end_char: Option<char> = None;
        for leaf in &leaves {
            if let NodeData::Text { contents } = &leaf.data {
                let s = contents.borrow();
                let s_str: &str = &s;
                // Inline tags between leaves (e.g. `cat<br>eats`) leave no
                // whitespace in either neighbouring leaf, so naive concat
                // would feed `cateats` to the model. If neither side already
                // has whitespace at the boundary, inject a single space —
                // and grow this leaf's char range to include it so the
                // synthetic char is owned by the trailing leaf.
                let starts_ws = s_str
                    .chars()
                    .next()
                    .map(char::is_whitespace)
                    .unwrap_or(true);
                let prev_ends_ws = prev_text_end_char.map(char::is_whitespace).unwrap_or(true);
                if !prev_ends_ws && !starts_ws {
                    text.push(' ');
                    total += 1;
                }
                text.push_str(s_str);
                total += s_str.chars().count();
                prev_text_end_char = s_str.chars().next_back().or(prev_text_end_char);
            }
            leaf_char_ends.push(total);
        }
        self.scopes.push(Scope {
            leaves,
            text,
            leaf_char_ends,
        });
    }

    fn walk(&mut self, node: &Handle) {
        match &node.data {
            NodeData::Text { .. } => {
                self.current.push(node.clone());
            }
            NodeData::Element { name, attrs, .. } => {
                let local: &str = &name.local;
                let inline = is_inline(local);
                let skipped = is_skipped_subtree(local) || has_skip_attr(&attrs.borrow());
                if !inline {
                    self.flush();
                }
                if !skipped {
                    let children = node.children.borrow().clone();
                    for child in &children {
                        self.walk(child);
                    }
                }
                if !inline {
                    self.flush();
                }
            }
            // Document, Doctype, Comment, ProcessingInstruction: walk children
            // (Document only) but otherwise contribute no text.
            NodeData::Document => {
                let children = node.children.borrow().clone();
                for child in &children {
                    self.walk(child);
                }
            }
            _ => {}
        }
    }
}

fn apply_scope(scope: &Scope, translated: &str, alignments: &[TokenAlignment]) {
    if scope.leaves.is_empty() {
        return;
    }
    let tgt_chars: Vec<char> = translated.chars().collect();
    let tgt_len = tgt_chars.len();
    let src_len = scope.text.chars().count();

    let mut per_char_leaf: Vec<Option<usize>> = vec![None; tgt_len];

    // Walk alignments in target-text order, alternating between gap regions
    // (inter-token whitespace, untranslated runs) and aligned regions. Each
    // gap maps a `[prev_tgt_end, this_tgt_begin)` slice of target text to a
    // `[prev_src_end, this_src_begin)` slice of source text — we use the
    // *source* slice's leaf membership to assign whitespace correctly. That
    // matters for inline tags like `<b>my name</b>`: the trailing space
    // belongs to whichever leaf held the corresponding space in the source.
    let mut sorted_aligns: Vec<&TokenAlignment> = alignments.iter().collect();
    sorted_aligns.sort_by_key(|a| (a.tgt_begin, a.tgt_end));

    let mut prev_tgt_end = 0usize;
    let mut prev_src_end = 0usize;
    for align in &sorted_aligns {
        let tgt_b = (align.tgt_begin as usize).min(tgt_len);
        let tgt_e = (align.tgt_end as usize).min(tgt_len);
        let src_b = (align.src_begin as usize).min(src_len);
        let src_e = (align.src_end as usize).min(src_len);

        if tgt_b > prev_tgt_end {
            fill_gap(
                &mut per_char_leaf,
                scope,
                prev_tgt_end,
                tgt_b,
                prev_src_end,
                src_b,
            );
        }
        let mid = if src_e > src_b {
            (src_b + src_e) / 2
        } else {
            src_b
        };
        let leaf_idx = leaf_for_src_char(scope, mid);
        for slot in &mut per_char_leaf[tgt_b..tgt_e] {
            *slot = Some(leaf_idx);
        }
        prev_tgt_end = tgt_e.max(prev_tgt_end);
        prev_src_end = src_e.max(prev_src_end);
    }
    if prev_tgt_end < tgt_len {
        fill_gap(
            &mut per_char_leaf,
            scope,
            prev_tgt_end,
            tgt_len,
            prev_src_end,
            src_len,
        );
    }

    // Group target chars into per-leaf strings, in target text order. A leaf
    // with no chars assigned ends up empty — its element stays in the DOM
    // but its text content is cleared (e.g. a word the model dropped).
    let mut per_leaf_text: Vec<String> = vec![String::new(); scope.leaves.len()];
    for (c_idx, ch) in tgt_chars.iter().enumerate() {
        let leaf_idx = per_char_leaf[c_idx].unwrap_or(0);
        per_leaf_text[leaf_idx].push(*ch);
    }

    // Write back into the DOM. Each leaf's `contents` is a `RefCell<StrTendril>`
    // we can mutate in place — the DOM tree itself is unchanged.
    for (leaf, text) in scope.leaves.iter().zip(per_leaf_text) {
        if let NodeData::Text { contents } = &leaf.data {
            *contents.borrow_mut() = text.into();
        }
    }
}

fn fill_gap(
    per_char_leaf: &mut [Option<usize>],
    scope: &Scope,
    tgt_begin: usize,
    tgt_end: usize,
    src_begin: usize,
    src_end: usize,
) {
    let tgt_len = tgt_end - tgt_begin;
    if tgt_len == 0 {
        return;
    }
    let src_len = src_end.saturating_sub(src_begin);
    if src_len == 0 {
        // Empty source gap: collapse to whichever leaf the source position
        // sits in (or the previous leaf if we're at the very end).
        let probe = src_begin.min(scope.text.chars().count().saturating_sub(1));
        let leaf = leaf_for_src_char(scope, probe);
        for slot in &mut per_char_leaf[tgt_begin..tgt_end] {
            *slot = Some(leaf);
        }
        return;
    }
    // Distribute target gap chars proportionally across the source gap so a
    // wider source gap (multiple leaves) splits naturally between them.
    for c in 0..tgt_len {
        let frac = (c as f64 + 0.5) / tgt_len as f64;
        let src_pos = src_begin + ((src_len as f64) * frac).floor() as usize;
        let src_pos = src_pos.min(src_end.saturating_sub(1));
        per_char_leaf[tgt_begin + c] = Some(leaf_for_src_char(scope, src_pos));
    }
}

fn leaf_for_src_char(scope: &Scope, src_char: usize) -> usize {
    // leaf_char_ends is non-decreasing; return the first leaf whose end is
    // strictly greater than `src_char`. If src_char is past the end (e.g. an
    // alignment edge), pin to the last leaf.
    for (i, &end) in scope.leaf_char_ends.iter().enumerate() {
        if src_char < end {
            return i;
        }
    }
    scope.leaves.len().saturating_sub(1)
}

fn serialize_fragment_dom(dom: &RcDom) -> String {
    // `parse_fragment` puts everything under one synthetic context element
    // (a `<html>`-named wrapper) sitting directly under `dom.document`. We
    // serialise that wrapper with `ChildrenOnly` so the wrapper itself is
    // not emitted — only the user's original markup, with translated text.
    let document_children = dom.document.children.borrow();
    let Some(root) = document_children.first().cloned() else {
        return String::new();
    };
    drop(document_children);

    let serializable = SerializableHandle::from(root);
    let mut buf: Vec<u8> = Vec::new();
    serialize(
        &mut buf,
        &serializable,
        SerializeOpts {
            traversal_scope: TraversalScope::ChildrenOnly(None),
            ..Default::default()
        },
    )
    .expect("html serialize must succeed for an in-memory tree");
    String::from_utf8(buf).expect("html5ever emits UTF-8")
}

// Borrow-check helper: silence dead-code warnings on Rc/RefCell imports
// when this module is built standalone for tests.
#[allow(dead_code)]
fn _types_in_scope(_: &Rc<RefCell<()>>) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::translate::{TokenAlignment, TranslationWithAlignment};

    /// Build an alignment that says "source char range [sb,se) maps to
    /// target char range [tb,te)". Used by the synthetic-translation tests
    /// below to exercise the reassembly logic without loading a model.
    fn align(sb: u64, se: u64, tb: u64, te: u64) -> TokenAlignment {
        TokenAlignment {
            src_begin: sb,
            src_end: se,
            tgt_begin: tb,
            tgt_end: te,
        }
    }

    #[test]
    fn flat_text_no_tags_round_trips() {
        let (prepared, texts) = prepare(&["Hello world".to_string()]);
        assert_eq!(texts, vec!["Hello world".to_string()]);
        let translations = vec![TranslationWithAlignment {
            source_text: "Hello world".into(),
            translated_text: "Hola mundo".into(),
            alignments: vec![align(0, 5, 0, 4), align(6, 11, 5, 10)],
        }];
        let out = finish(prepared, &translations);
        assert_eq!(out, vec!["Hola mundo".to_string()]);
    }

    #[test]
    fn inline_tag_keeps_translation_in_one_scope() {
        // <p>hi <b>my name</b> is david</p> — three text leaves but one scope
        // because <b> is inline. The model receives "hi my name is david" as
        // one string and the alignment routes each translated word back to
        // the leaf its source word belonged to.
        let input = "<p>hi <b>my name</b> is david</p>".to_string();
        let (prepared, texts) = prepare(&[input]);
        assert_eq!(
            texts,
            vec!["hi my name is david".to_string()],
            "inline <b> must not break the scope"
        );

        // Synthetic translation: "hola mi nombre es david". Alignments:
        //   "hola"   ← src "hi"     (leaf 0, chars 0..2)    → tgt 0..4
        //   "mi"     ← src "my"     (leaf 1, chars 3..5)    → tgt 5..7
        //   "nombre" ← src "name"   (leaf 1, chars 6..10)   → tgt 8..14
        //   "es"     ← src "is"     (leaf 2, chars 11..13)  → tgt 15..17
        //   "david"  ← src "david"  (leaf 2, chars 14..19)  → tgt 18..23
        let translations = vec![TranslationWithAlignment {
            source_text: "hi my name is david".into(),
            translated_text: "hola mi nombre es david".into(),
            alignments: vec![
                align(0, 2, 0, 4),
                align(3, 5, 5, 7),
                align(6, 10, 8, 14),
                align(11, 13, 15, 17),
                align(14, 19, 18, 23),
            ],
        }];
        let out = finish(prepared, &translations);
        assert_eq!(out.len(), 1);
        // `<b>` survives, attribute-free elements survive, and the words
        // for "my name" land inside the <b>...</b> the same way "my name"
        // did in the source. (Inter-word whitespace inherits the previous
        // leaf, so the leading "hola " trailing space goes with leaf 0.)
        assert!(out[0].starts_with("<p>hola "), "got: {}", out[0]);
        assert!(
            out[0].contains("<b>mi nombre</b>"),
            "<b> must wrap the translation of 'my name': {}",
            out[0]
        );
        assert!(out[0].ends_with(" es david</p>"), "got: {}", out[0]);
    }

    #[test]
    fn attributes_and_void_tags_are_preserved_verbatim() {
        // href, data-*, class must round-trip exactly. <br> has no text node,
        // so it just stays in place — we never touch structure, only text.
        let input = "<a href=\"https://example.com\" data-id=\"42\" class=\"link\">Click <em>here</em><br>to continue</a>".to_string();
        let (prepared, texts) = prepare(&[input]);
        assert_eq!(texts.len(), 1);
        // <a> is inline, <em> is inline, <br> is inline+void → one scope
        // with three text leaves: "Click ", "here", "to continue". The
        // collector injects a synthetic space between "here" and "to" so
        // the model sees a properly tokenisable sentence; the synthetic
        // char is owned by the trailing leaf for alignment purposes.
        assert_eq!(texts[0], "Click here to continue");

        let translations = vec![TranslationWithAlignment {
            source_text: texts[0].clone(),
            translated_text: "Haz clic aquí para continuar".into(),
            alignments: vec![
                align(0, 5, 0, 8),     // "Click" → "Haz clic" → leaf 0
                align(6, 10, 9, 13),   // "here"  → "aquí"     → leaf 1
                align(11, 22, 14, 28), // "to continue" → "para continuar" → leaf 2
            ],
        }];
        let out = finish(prepared, &translations);
        assert_eq!(out.len(), 1);
        assert!(
            out[0].contains("href=\"https://example.com\""),
            "href must be preserved verbatim: {}",
            out[0]
        );
        assert!(
            out[0].contains("data-id=\"42\""),
            "data-id must be preserved verbatim: {}",
            out[0]
        );
        assert!(
            out[0].contains("class=\"link\""),
            "class must be preserved verbatim: {}",
            out[0]
        );
        assert!(
            out[0].contains("<br>"),
            "void <br> tag must be preserved: {}",
            out[0]
        );
        assert!(
            out[0].contains("<em>aquí</em>"),
            "<em> wrapping must be preserved with translated content: {}",
            out[0]
        );
    }

    #[test]
    fn block_level_elements_form_separate_scopes() {
        // Two <p> blocks → two scopes → two translation calls (the model
        // never sees them as one sentence — block boundaries split context,
        // matching how slimt's old HTML mode also inserted \n\n at block
        // boundaries).
        let input = "<div><p>The cat</p><p>eats fish</p></div>".to_string();
        let (_prepared, texts) = prepare(&[input]);
        assert_eq!(texts, vec!["The cat".to_string(), "eats fish".to_string()]);
    }

    #[test]
    fn malformed_html_does_not_panic() {
        // html5ever recovers from mismatched tags rather than aborting.
        // The XHScanner-based slimt HTML mode used to crash on this; our
        // pipeline must always produce *some* translatable output.
        let input = "<p>open <b>but never closed</p>".to_string();
        let (prepared, texts) = prepare(&[input]);
        assert_eq!(texts.len(), 1);
        assert_eq!(texts[0], "open but never closed");
        // Empty alignments + identity translation: the parser tolerates the
        // mismatch and the round-trip should at minimum keep the text content.
        let translations = vec![TranslationWithAlignment {
            source_text: texts[0].clone(),
            translated_text: texts[0].clone(),
            alignments: vec![align(0, 21, 0, 21)],
        }];
        let out = finish(prepared, &translations);
        assert!(out[0].contains("open"), "got: {}", out[0]);
        assert!(out[0].contains("but never closed"), "got: {}", out[0]);
    }

    #[test]
    fn empty_and_whitespace_fragments_skip_translation() {
        let (prepared, texts) = prepare(&[
            "".to_string(),
            "   ".to_string(),
            "<p>real content</p>".to_string(),
        ]);
        assert_eq!(
            texts,
            vec!["real content".to_string()],
            "only non-empty scopes should be sent to the translator"
        );
        let translations = vec![TranslationWithAlignment {
            source_text: "real content".into(),
            translated_text: "contenido real".into(),
            alignments: vec![align(0, 12, 0, 14)],
        }];
        let out = finish(prepared, &translations);
        assert_eq!(out.len(), 3);
        assert!(out[2].contains("contenido real"), "got: {}", out[2]);
    }

    #[test]
    fn inline_code_kbd_samp_var_subtrees_are_not_translated() {
        // Inline code-ish elements inside a paragraph used to leak their
        // text into the surrounding scope, so a sentence like
        // "the console.log call" would get its function name machine-
        // translated. The skip keeps those subtrees opaque while the
        // surrounding sentence stays one scope so the model sees the
        // intended phrasing.
        let input =
            r#"<p>Call <code>console.log</code> then press <kbd>Enter</kbd>.</p>"#.to_string();
        let (prepared, texts) = prepare(&[input]);
        assert_eq!(
            texts,
            vec!["Call  then press .".to_string()],
            "inline skipped subtrees keep the surrounding scope intact"
        );
        let translations = vec![TranslationWithAlignment {
            source_text: "Call  then press .".into(),
            translated_text: "Llama luego pulsa.".into(),
            alignments: vec![align(0, 18, 0, 18)],
        }];
        let out = finish(prepared, &translations);
        assert!(
            out[0].contains("<code>console.log</code>"),
            "code text must round-trip verbatim: {}",
            out[0]
        );
        assert!(
            out[0].contains("<kbd>Enter</kbd>"),
            "kbd text must round-trip verbatim: {}",
            out[0]
        );
    }

    #[test]
    fn pre_block_inside_structural_keeps_text_verbatim() {
        // A <pre> preserves whitespace and is by author intent literal.
        // Even when wrapped in a structural element that JS picks as a
        // unit, Rust must not extract the pre's text leaves.
        let input = "<div>before<pre>x = 1\ny = 2</pre>after</div>".to_string();
        let (prepared, texts) = prepare(&[input]);
        assert_eq!(
            texts,
            vec!["before".to_string(), "after".to_string()],
            "pre subtree must not be a translation source"
        );
        let translations = vec![
            TranslationWithAlignment {
                source_text: "before".into(),
                translated_text: "antes".into(),
                alignments: vec![align(0, 6, 0, 5)],
            },
            TranslationWithAlignment {
                source_text: "after".into(),
                translated_text: "después".into(),
                alignments: vec![align(0, 5, 0, 7)],
            },
        ];
        let out = finish(prepared, &translations);
        assert!(
            out[0].contains("<pre>x = 1\ny = 2</pre>"),
            "pre content must round-trip verbatim: {}",
            out[0]
        );
    }

    #[test]
    fn translate_no_attribute_is_honored() {
        // <span translate="no"> is the standards-track way to mark a
        // brand name or term as untranslatable. The span is inline so
        // the surrounding sentence stays one scope.
        let input = r#"<p>Buy a <span translate="no">Acme Widget</span> today</p>"#.to_string();
        let (_prepared, texts) = prepare(&[input]);
        assert_eq!(
            texts,
            vec!["Buy a  today".to_string()],
            "translate=no on an inline element keeps scope continuity"
        );
    }

    #[test]
    fn notranslate_class_is_honored() {
        // Google's de-facto convention: class="notranslate" on any
        // element opts that subtree out of translation.
        let input =
            r##"<p>Send to <a class="notranslate" href="#">support@example.com</a> now</p>"##
                .to_string();
        let (_prepared, texts) = prepare(&[input]);
        assert_eq!(
            texts,
            vec!["Send to  now".to_string()],
            "class=notranslate on an inline element keeps scope continuity"
        );
    }

    #[test]
    fn contenteditable_subtree_is_skipped() {
        // User-editable regions are not ours to rewrite — translating
        // them would clobber whatever the user typed.
        let input =
            r#"<div>label <span contenteditable="true">user typed text</span></div>"#.to_string();
        let (_prepared, texts) = prepare(&[input]);
        assert_eq!(
            texts,
            vec!["label ".to_string()],
            "contenteditable subtree must be skipped"
        );
    }

    #[test]
    fn svg_and_canvas_subtrees_are_skipped() {
        // SVG <text> children would otherwise be extracted; we leave them
        // alone because translating chart labels is rarely what the user
        // wants and we have no way of knowing. SVG is block-level so it
        // does break the surrounding scope.
        let input = r#"<div>chart: <svg><text>2024</text></svg></div>"#.to_string();
        let (_prepared, texts) = prepare(&[input]);
        assert_eq!(
            texts,
            vec!["chart: ".to_string()],
            "svg subtree must be skipped"
        );
    }

    #[test]
    fn user_reported_create_extension_stays_literal() {
        // Reproduces the user-reported bug: SQL-like content inside an
        // inline <code> got translated as "CREAR EXTENSIÓN <extension>".
        // The whole <code> subtree must be opaque.
        let input =
            "<p>When a user runs <code>CREATE EXTENSION &lt;extension&gt;</code>, the server will:</p>"
                .to_string();
        let (prepared, texts) = prepare(&[input]);
        assert_eq!(
            texts,
            vec!["When a user runs , the server will:".to_string()],
            "code subtree must not contribute to scope text"
        );
        let translations = vec![TranslationWithAlignment {
            source_text: "When a user runs , the server will:".into(),
            translated_text: "Cuando un usuario ejecuta , el servidor:".into(),
            alignments: vec![align(0, 35, 0, 41)],
        }];
        let out = finish(prepared, &translations);
        assert!(
            out[0].contains("<code>CREATE EXTENSION &lt;extension&gt;</code>"),
            "code content must round-trip verbatim: {}",
            out[0]
        );
    }

    #[test]
    fn script_and_style_subtrees_are_skipped() {
        // Defensive: even if a <p> somehow gets a <script> child shipped
        // to us, we must not feed its source code to the model. <script>
        // is block-level (parser hoists it anyway) so it splits scopes.
        let input = r#"<p>hello <script>alert(1)</script> world</p>"#.to_string();
        let (_prepared, texts) = prepare(&[input]);
        assert_eq!(
            texts,
            vec!["hello ".to_string(), " world".to_string()],
            "script subtree must be skipped"
        );
    }
}
