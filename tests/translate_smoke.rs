//! End-to-end smoke test for the slimt-backed BergamotEngine.
//!
//! Skipped (passes with no work) when the bergamot Spanish/English assets
//! aren't installed on the test host.

use std::path::PathBuf;

use translator::TranslationMode;
use translator::bergamot::{BergamotEngine, ModelPaths};

fn asset_root() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let root = PathBuf::from(home).join(".local/share/dev.davidv.translator/bin");
    root.is_dir().then_some(root)
}

fn paths_for(root: &PathBuf, pair: &str) -> ModelPaths {
    ModelPaths {
        model: root.join(format!("model.{pair}.intgemm.alphas.bin")),
        vocabulary: root.join(format!("vocab.{pair}.spm")),
        shortlist: root.join(format!("lex.50.50.{pair}.s2t.bin")),
    }
}

#[test]
fn translates_spanish_to_english() {
    let Some(root) = asset_root() else {
        eprintln!("translator assets not installed; skipping");
        return;
    };
    let paths = paths_for(&root, "esen");
    if !paths.model.exists() || !paths.vocabulary.exists() || !paths.shortlist.exists() {
        eprintln!("esen assets missing; skipping");
        return;
    }

    let mut engine = BergamotEngine::new();
    engine
        .load_model_into_cache(&paths, "es-en")
        .expect("load esen");

    let inputs = vec!["El gato come pescado.".to_string()];
    let outs = engine
        .translate_multiple(&inputs, "es-en", TranslationMode::PlainText)
        .expect("translate");

    assert_eq!(outs.len(), 1);
    let translation = outs[0].to_lowercase();
    assert!(
        translation.contains("cat") && translation.contains("fish"),
        "unexpected translation: {}",
        outs[0]
    );
}

#[test]
fn translates_html_en_to_es_preserving_tags() {
    let Some(root) = asset_root() else {
        eprintln!("translator assets not installed; skipping");
        return;
    };
    let paths = paths_for(&root, "enes");
    if !paths.model.exists() || !paths.vocabulary.exists() || !paths.shortlist.exists() {
        eprintln!("enes assets missing; skipping");
        return;
    }

    let mut engine = BergamotEngine::new();
    engine
        .load_model_into_cache(&paths, "en-es")
        .expect("load enes");

    let inputs = vec![
        "<p>The <b>cat</b> eats <i>fish</i>.</p>".to_string(),
        "<a href=\"https://example.com\">Click <em>here</em></a> to continue.".to_string(),
    ];
    let outs = engine
        .translate_multiple(&inputs, "en-es", TranslationMode::Html)
        .expect("translate html");

    assert_eq!(outs.len(), 2);
    eprintln!("[html_en_to_es] outs: {:?}", outs);

    let first = &outs[0];
    assert!(
        first.contains("<p>") && first.contains("</p>"),
        "<p> tags not preserved: {first}"
    );
    assert!(
        first.contains("<b>") && first.contains("</b>"),
        "<b> tags not preserved: {first}"
    );
    assert!(
        first.contains("<i>") && first.contains("</i>"),
        "<i> tags not preserved: {first}"
    );
    let lower = first.to_lowercase();
    assert!(
        lower.contains("gato") && lower.contains("pescado") || lower.contains("pez"),
        "expected Spanish translation of cat/fish: {first}"
    );

    let second = &outs[1];
    assert!(
        second.contains("href=\"https://example.com\""),
        "anchor href not preserved verbatim: {second}"
    );
    assert!(
        second.contains("<a") && second.contains("</a>"),
        "<a> tags not preserved: {second}"
    );
    assert!(
        second.contains("<em>") && second.contains("</em>"),
        "<em> tags not preserved: {second}"
    );
}

#[test]
fn html_passes_through_data_attributes_verbatim() {
    let Some(root) = asset_root() else {
        eprintln!("translator assets not installed; skipping");
        return;
    };
    let paths = paths_for(&root, "enes");
    if !paths.model.exists() {
        eprintln!("enes assets missing; skipping");
        return;
    }

    let mut engine = BergamotEngine::new();
    engine
        .load_model_into_cache(&paths, "en-es")
        .expect("load enes");

    let inputs = vec![
        "<span data-test=\"this content should pass through\">The cat eats fish.</span>"
            .to_string(),
        "<div data-id=\"42\" data-track=\"hello world\" class=\"foo\">Hello world!</div>"
            .to_string(),
    ];
    let outs = engine
        .translate_multiple(&inputs, "en-es", TranslationMode::Html)
        .expect("translate html");

    assert_eq!(outs.len(), 2);
    eprintln!("[html_data_attrs] outs: {:?}", outs);

    let first = &outs[0];
    assert!(
        first.contains("data-test=\"this content should pass through\""),
        "data-test attribute mangled: {first}"
    );
    let lower = first.to_lowercase();
    assert!(
        lower.contains("gato"),
        "expected 'gato' inside the span: {first}"
    );

    let second = &outs[1];
    assert!(
        second.contains("data-id=\"42\""),
        "data-id attribute mangled: {second}"
    );
    assert!(
        second.contains("data-track=\"hello world\""),
        "data-track attribute (incl. value) mangled: {second}"
    );
    assert!(
        second.contains("class=\"foo\""),
        "class attribute mangled: {second}"
    );
}

#[test]
fn html_mode_treats_tags_as_markup_not_text() {
    // Same input, once as plain-text and once as HTML. In HTML mode the tags
    // should not appear inside translated phrases as literal words; in plain
    // mode they typically would.
    let Some(root) = asset_root() else {
        eprintln!("translator assets not installed; skipping");
        return;
    };
    let paths = paths_for(&root, "enes");
    if !paths.model.exists() {
        eprintln!("enes assets missing; skipping");
        return;
    }

    let mut engine = BergamotEngine::new();
    engine
        .load_model_into_cache(&paths, "en-es")
        .expect("load enes");

    let input = "<p>Hello <b>world</b>!</p>".to_string();
    let html_out = engine
        .translate_multiple(&[input.clone()], "en-es", TranslationMode::Html)
        .expect("translate html");
    assert_eq!(html_out.len(), 1);

    let html_translation = &html_out[0];
    eprintln!("[html_mode] html_out: {html_translation}");
    assert!(
        html_translation.contains("<p>") && html_translation.contains("<b>"),
        "expected <p> and <b> in HTML output: {html_translation}"
    );
    // The translated word for "world" should land inside the <b>...</b>
    // wrapper that surrounded "world" in the source.
    let lower = html_translation.to_lowercase();
    assert!(
        lower.contains("hola") || lower.contains("saludos"),
        "expected Spanish greeting: {html_translation}"
    );
    assert!(
        lower.contains("mundo"),
        "expected 'mundo' (Spanish for 'world') in: {html_translation}"
    );
}

#[test]
fn swapping_between_loaded_models_uses_the_right_one() {
    // Load both directions, fire interleaved translate calls, and check that
    // each call routes to the right model rather than re-using whichever was
    // last loaded.
    let Some(root) = asset_root() else {
        eprintln!("translator assets not installed; skipping");
        return;
    };
    let esen = paths_for(&root, "esen");
    let enes = paths_for(&root, "enes");
    if !esen.model.exists() || !enes.model.exists() {
        eprintln!("esen or enes assets missing; skipping");
        return;
    }

    let mut engine = BergamotEngine::new();
    engine
        .load_model_into_cache(&esen, "es-en")
        .expect("load esen");
    engine
        .load_model_into_cache(&enes, "en-es")
        .expect("load enes");

    let pairs = [
        ("es-en", "El gato come pescado.", "cat"),
        ("en-es", "The cat eats fish.", "gato"),
        ("es-en", "Hola mundo.", "world"),
        ("en-es", "Hello world.", "mundo"),
        ("es-en", "El gato come pescado.", "fish"),
    ];
    for (key, src, expected_word) in pairs {
        let outs = engine
            .translate_multiple(&[src.to_string()], key, TranslationMode::PlainText)
            .expect("translate");
        assert_eq!(outs.len(), 1);
        let lower = outs[0].to_lowercase();
        assert!(
            lower.contains(expected_word),
            "key {key} on src {src:?} expected '{expected_word}' in output, got: {}",
            outs[0]
        );
    }
}

#[test]
fn evict_makes_key_unusable_then_reload_works_again() {
    let Some(root) = asset_root() else {
        eprintln!("translator assets not installed; skipping");
        return;
    };
    let paths = paths_for(&root, "esen");
    if !paths.model.exists() {
        eprintln!("esen assets missing; skipping");
        return;
    }

    let mut engine = BergamotEngine::new();
    engine.load_model_into_cache(&paths, "es-en").expect("load");

    let outs = engine
        .translate_multiple(
            &["El gato come pescado.".to_string()],
            "es-en",
            TranslationMode::PlainText,
        )
        .expect("translate before evict");
    assert!(outs[0].to_lowercase().contains("cat"));

    engine.evict("es-en");

    let err = engine
        .translate_multiple(
            &["El gato come pescado.".to_string()],
            "es-en",
            TranslationMode::PlainText,
        )
        .expect_err("translate after evict must error");
    assert!(
        err.contains("not loaded") || err.contains("es-en"),
        "unexpected error after evict: {err}"
    );

    // Reload and translate again — the cache should be repopulatable.
    engine
        .load_model_into_cache(&paths, "es-en")
        .expect("reload");
    let outs = engine
        .translate_multiple(
            &["El gato come pescado.".to_string()],
            "es-en",
            TranslationMode::PlainText,
        )
        .expect("translate after reload");
    assert!(outs[0].to_lowercase().contains("cat"));
}

#[test]
fn evict_unmaps_the_model_file() {
    // Verify slimt actually releases the mmap when we drop the SlimtModel.
    // Inspect /proc/self/maps for the model file path before/after evict.
    let Some(root) = asset_root() else {
        eprintln!("translator assets not installed; skipping");
        return;
    };
    let paths = paths_for(&root, "esen");
    if !paths.model.exists() {
        eprintln!("esen assets missing; skipping");
        return;
    }

    let model_path = paths.model.canonicalize().expect("canonicalize");
    let model_str = model_path.to_str().expect("utf8");

    let count_mappings = || -> usize {
        let maps = std::fs::read_to_string("/proc/self/maps").expect("read /proc/self/maps");
        maps.lines().filter(|line| line.contains(model_str)).count()
    };

    assert_eq!(
        count_mappings(),
        0,
        "model file unexpectedly mapped before load"
    );

    let mut engine = BergamotEngine::new();
    engine.load_model_into_cache(&paths, "es-en").expect("load");

    let mapped_after_load = count_mappings();
    eprintln!("[evict_unmaps] mappings after load: {mapped_after_load}");
    assert!(
        mapped_after_load >= 1,
        "expected the model file to be mmap'd after load"
    );

    // Run a translation so the slimt async batcher is fully spun up — if any
    // worker thread held a transient reference, this exercises that path.
    let _ = engine
        .translate_multiple(
            &["El gato come pescado.".to_string()],
            "es-en",
            TranslationMode::PlainText,
        )
        .expect("translate");

    engine.evict("es-en");

    let mapped_after_evict = count_mappings();
    eprintln!("[evict_unmaps] mappings after evict: {mapped_after_evict}");
    assert_eq!(
        mapped_after_evict, 0,
        "model file is still mapped after evict — eviction did not release mmap"
    );
}

#[test]
fn many_load_evict_cycles_do_not_leak_mappings() {
    // Hammer load + evict to make sure we don't accumulate stale mmaps.
    let Some(root) = asset_root() else {
        eprintln!("translator assets not installed; skipping");
        return;
    };
    let paths = paths_for(&root, "esen");
    if !paths.model.exists() {
        eprintln!("esen assets missing; skipping");
        return;
    }

    let model_path = paths.model.canonicalize().expect("canonicalize");
    let model_str = model_path.to_str().expect("utf8");

    let count_mappings = || -> usize {
        let maps = std::fs::read_to_string("/proc/self/maps").expect("read maps");
        maps.lines().filter(|line| line.contains(model_str)).count()
    };

    let mut engine = BergamotEngine::new();
    for _ in 0..8 {
        engine.load_model_into_cache(&paths, "es-en").expect("load");
        let _ = engine
            .translate_multiple(
                &["El gato come pescado.".to_string()],
                "es-en",
                TranslationMode::PlainText,
            )
            .expect("translate");
        engine.evict("es-en");
    }

    let mappings = count_mappings();
    eprintln!("[load_evict_cycles] mappings after 8 cycles: {mappings}");
    assert_eq!(
        mappings, 0,
        "load+evict loop leaked {mappings} mmap(s) of the model"
    );
}

#[test]
fn two_loaded_models_each_account_for_exactly_one_mapping() {
    // With models A and B both loaded, we expect exactly two mmap regions for
    // the model files (one each), and the count must not grow as we translate
    // with them in alternation. Evicting A must drop A's mapping but leave B
    // alone.
    let Some(root) = asset_root() else {
        eprintln!("translator assets not installed; skipping");
        return;
    };
    let esen = paths_for(&root, "esen");
    let enes = paths_for(&root, "enes");
    if !esen.model.exists() || !enes.model.exists() {
        eprintln!("esen or enes assets missing; skipping");
        return;
    }

    let esen_path = esen.model.canonicalize().expect("canonicalize esen");
    let enes_path = enes.model.canonicalize().expect("canonicalize enes");

    let count_for = |needle: &str| -> usize {
        std::fs::read_to_string("/proc/self/maps")
            .expect("read maps")
            .lines()
            .filter(|l| l.contains(needle))
            .count()
    };
    let esen_count = || count_for(esen_path.to_str().unwrap());
    let enes_count = || count_for(enes_path.to_str().unwrap());

    let mut engine = BergamotEngine::new();
    engine
        .load_model_into_cache(&esen, "es-en")
        .expect("load esen");
    engine
        .load_model_into_cache(&enes, "en-es")
        .expect("load enes");

    assert_eq!(esen_count(), 1, "esen must be mapped once after load");
    assert_eq!(enes_count(), 1, "enes must be mapped once after load");

    // Hammer back-and-forth translations to confirm neither model accrues
    // extra mappings while the other is in use.
    for _ in 0..6 {
        let _ = engine
            .translate_multiple(
                &["El gato come pescado.".to_string()],
                "es-en",
                TranslationMode::PlainText,
            )
            .expect("es->en");
        let _ = engine
            .translate_multiple(
                &["The cat eats fish.".to_string()],
                "en-es",
                TranslationMode::PlainText,
            )
            .expect("en->es");
        assert_eq!(esen_count(), 1, "esen must stay at 1 mapping during use");
        assert_eq!(enes_count(), 1, "enes must stay at 1 mapping during use");
    }

    // Evict only one; the other must remain fully usable.
    engine.evict("es-en");
    assert_eq!(esen_count(), 0, "esen must be unmapped after evict");
    assert_eq!(enes_count(), 1, "enes must still be mapped");

    let outs = engine
        .translate_multiple(
            &["The cat eats fish.".to_string()],
            "en-es",
            TranslationMode::PlainText,
        )
        .expect("en->es after sibling evict");
    assert!(outs[0].to_lowercase().contains("gato"));
    assert_eq!(enes_count(), 1, "enes must still hold one mapping");
}

#[test]
fn translate_with_alignment_returns_aligned_tokens() {
    let Some(root) = asset_root() else {
        eprintln!("translator assets not installed; skipping");
        return;
    };
    let paths = paths_for(&root, "esen");
    if !paths.model.exists() {
        eprintln!("esen assets missing; skipping");
        return;
    }

    let mut engine = BergamotEngine::new();
    engine
        .load_model_into_cache(&paths, "es-en")
        .expect("load esen");

    let inputs = vec!["El gato come pescado.".to_string()];
    let outs = engine
        .translate_multiple_with_alignment(&inputs, "es-en")
        .expect("translate with alignment");

    assert_eq!(outs.len(), 1);
    assert!(!outs[0].translated_text.is_empty());
    assert!(
        !outs[0].alignments.is_empty(),
        "expected at least one token alignment"
    );
    for a in &outs[0].alignments {
        assert!(a.src_begin <= a.src_end);
        assert!(a.tgt_begin <= a.tgt_end);
    }
}
