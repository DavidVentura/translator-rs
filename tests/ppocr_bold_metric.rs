//! Per-line bold detection from the ink text-metrics, end to end on a real page.
//!
//! `files/letter.jpg` is a photographed Dutch letter with two bold section
//! headings ("Wat moet u doen?", "Wanneer krijgt u bericht?") amid regular body
//! text. `LineMetrics::is_bold` (stroke-core width ÷ x-height) should flag exactly
//! those headings and leave the body regular.
//!
//! Skips when the bucket models are missing.

#![cfg(feature = "ppocr")]

use std::path::{Path, PathBuf};

use translator::PpocrScript;
use translator::ppocr::{PpocrEngine, PpocrProfile, PpocrRecognizerSpec};
use translator::text_metrics::measure_line;

const MODEL_DIR: &str = "/home/david/AndroidStudioProjects/bucket/ocr/1/PP-OCRv5";
const IMAGE_PATH: &str = "files/letter.jpg";

fn model_paths() -> Option<(PathBuf, PathBuf, PathBuf, PathBuf)> {
    let dir = Path::new(MODEL_DIR);
    let det = dir.join("PP-OCRv5_mobile_det.mnn");
    let rec = dir.join("latin_PP-OCRv5_mobile_rec_infer.mnn");
    let keys = dir.join("latin_PP-OCRv5_keys.txt");
    let ink = dir.join("ink.mnn");
    [&det, &rec, &keys, &ink]
        .iter()
        .all(|p| p.exists())
        .then_some((det, rec, keys, ink))
}

#[test]
fn bold_headings_detected_body_stays_regular() {
    let _ = env_logger::builder().is_test(true).try_init();

    let Some((det, rec, keys, ink)) = model_paths() else {
        eprintln!("PPOCR bucket models missing under {MODEL_DIR}; skipping");
        return;
    };
    let Ok(image) = image::open(IMAGE_PATH) else {
        eprintln!("{IMAGE_PATH} missing; skipping");
        return;
    };

    let spec = PpocrRecognizerSpec {
        script: PpocrScript::Latin,
        model_path: rec,
        keys_path: keys,
    };
    let engine = PpocrEngine::load(&det, None, None, vec![spec], 1, Some(ink.as_path()))
        .expect("load ppocr engine");

    let boxes = engine
        .detect_only_image(&image, PpocrProfile::Still)
        .expect("detect");
    let gray = image.to_luma8();
    let scripts = vec![PpocrScript::Latin; boxes.len()];
    let lines = engine
        .recognize_text_in_boxes_image(&image, &gray, &boxes, &scripts, PpocrProfile::Still, None)
        .expect("recognize");
    let masks = engine.ink_masks(&image, &boxes);

    // (lowercased text, is_bold) per recognised line that produced a matte band.
    let labelled: Vec<(String, bool)> = boxes
        .iter()
        .zip(lines)
        .enumerate()
        .filter_map(|(i, (b, line))| {
            let mask = masks.get(i)?.as_ref()?;
            let m = measure_line(mask, b.oriented_box.width, b.oriented_box.height)?;
            Some((line.text.to_lowercase(), m.is_bold()))
        })
        .collect();

    let bold_of = |needle: &str| -> bool {
        labelled
            .iter()
            .find(|(t, _)| t.contains(needle))
            .unwrap_or_else(|| panic!("line containing {needle:?} not recognised"))
            .1
    };

    // The two section headings are bold.
    assert!(
        bold_of("wat moet u doen"),
        "'Wat moet u doen?' should be bold"
    );
    assert!(
        bold_of("wanneer krijgt u bericht"),
        "'Wanneer krijgt u bericht?' should be bold",
    );
    // Representative body lines are regular.
    assert!(!bold_of("geef uw rekeningnummer"), "body line read as bold");
    assert!(
        !bold_of("u hebt eerder een brief"),
        "body line read as bold"
    );
    assert!(
        !bold_of("nadat wij uw rekeningnummer"),
        "body line read as bold"
    );

    // And bold is the exception, not the rule — guard against a drift that
    // flags everything.
    let bold_count = labelled.iter().filter(|(_, b)| *b).count();
    assert!(
        bold_count <= 4,
        "expected only the headings bold, got {bold_count} of {}",
        labelled.len(),
    );
}
