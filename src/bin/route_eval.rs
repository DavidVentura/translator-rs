//! Script-routing eval: for each page, route every detected line to a recognizer the way
//! still-image auto-detect does (PULC classify → `route_ppocr_predictions`), and score the
//! choice against what the line actually contains.
//!
//! A line's label comes from reading it with each non-Latin recognizer the manifest lists for its
//! page: the line needs that recognizer when its output has at least two letters of that script,
//! and is a Latin line otherwise. Each routed line then counts as
//!   * correct   — routed to the recognizer it needs,
//!   * degraded  — a Latin line read by a non-Latin recognizer (they all read Latin, less well),
//!   * destroyed — a non-Latin line read by a recognizer that can't read its script.
//!
//! The PULC classifier's unclassified-line count is reported too: those lines are routed by
//! guesswork, so it is the number a better classifier input should drive down.
//!
//! A degraded line also reports its character error rate against the Latin recognizer's reading,
//! so "read by the wrong recognizer" and "read wrong" can be told apart. `ROUTE_EVAL_VERBOSE=1`
//! prints every line, not just the misrouted ones.
//!
//!   cargo run --release --features planar-tracker --bin route_eval -- \
//!       <install_dir> <index.json> <manifest.tsv...>
//!
//! `install_dir` is an app install (its `ppocr/` models); each manifest line is
//! `<image>\t<comma-separated recognizer slugs>`, image paths relative to the manifest.

use std::path::{Path, PathBuf};

use translator::TranslatorSession;
use translator::catalog::{FsPackInstallChecker, PpocrScript};
use translator::coords::Quadrant;
use translator::live_session::LiveOcrHost;
use translator::ocr_runtime::{
    build_still_image, detect_boxes_from_still, route_ppocr_predictions,
};
use translator::ppocr::PpocrProfile;
use translator::script::Script;

const MAX_IMAGE_SIZE: u32 = 1400;
const MIN_SCRIPT_LETTERS: usize = 2;

const RECOGNIZER_SCRIPTS: &[PpocrScript] = &[
    PpocrScript::Arabic,
    PpocrScript::Cj,
    PpocrScript::Cyrillic,
    PpocrScript::Devanagari,
    PpocrScript::El,
    PpocrScript::Eslav,
    PpocrScript::Georgian,
    PpocrScript::Hebrew,
    PpocrScript::Indic,
    PpocrScript::Korean,
    PpocrScript::Latin,
    PpocrScript::Ta,
    PpocrScript::Te,
    PpocrScript::Th,
];

/// The Unicode scripts a recognizer exists to read, beyond Latin.
fn reads(recognizer: PpocrScript, script: Script) -> bool {
    match recognizer {
        PpocrScript::Cyrillic | PpocrScript::Eslav => script == Script::Cyrillic,
        PpocrScript::Arabic => script == Script::Arabic,
        PpocrScript::Devanagari => script == Script::Devanagari,
        PpocrScript::Cj => matches!(script, Script::Han | Script::Hiragana | Script::Katakana),
        PpocrScript::El => script == Script::Greek,
        PpocrScript::Georgian => script == Script::Georgian,
        PpocrScript::Hebrew => script == Script::Hebrew,
        PpocrScript::Korean => script == Script::Hangul,
        PpocrScript::Th => script == Script::Thai,
        PpocrScript::Ta => script == Script::Tamil,
        PpocrScript::Te => script == Script::Telugu,
        PpocrScript::Indic => matches!(
            script,
            Script::Bengali | Script::Gujarati | Script::Kannada | Script::Malayalam
        ),
        PpocrScript::Latin => false,
    }
}

fn same_family(a: PpocrScript, b: PpocrScript) -> bool {
    let cyrillic = |s| matches!(s, PpocrScript::Cyrillic | PpocrScript::Eslav);
    a == b || (cyrillic(a) && cyrillic(b))
}

#[derive(Default)]
struct Tally {
    correct: usize,
    degraded: usize,
    destroyed: usize,
    /// Edit distance and reference length summed over degraded lines.
    degraded_edits: usize,
    degraded_chars: usize,
    /// Lines PULC gave no confident class, which routing then has to guess.
    unclassified: usize,
}

impl Tally {
    fn add(&mut self, other: &Tally) {
        self.correct += other.correct;
        self.degraded += other.degraded;
        self.destroyed += other.destroyed;
        self.degraded_edits += other.degraded_edits;
        self.degraded_chars += other.degraded_chars;
        self.unclassified += other.unclassified;
    }

    fn degraded_cer(&self) -> f32 {
        self.degraded_edits as f32 / self.degraded_chars.max(1) as f32
    }
}

fn edit_distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut diagonal = row[0];
        row[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            let above = row[j + 1];
            row[j + 1] = (above + 1)
                .min(row[j] + 1)
                .min(diagonal + usize::from(ca != cb));
            diagonal = above;
        }
    }
    row[b.len()]
}

fn parse_manifest(path: &Path) -> Vec<(PathBuf, Vec<PpocrScript>)> {
    let dir = path.parent().expect("manifest has a parent dir");
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            let (image, scripts) = line.split_once('\t').unwrap_or((line, ""));
            let scripts = scripts
                .split(',')
                .filter(|s| !s.is_empty())
                .map(|slug| {
                    *RECOGNIZER_SCRIPTS
                        .iter()
                        .find(|s| s.as_slug() == slug)
                        .unwrap_or_else(|| panic!("unknown recognizer slug {slug:?}"))
                })
                .collect();
            (dir.join(image), scripts)
        })
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        eprintln!("usage: route_eval <install_dir> <index.json> <manifest.tsv...>");
        std::process::exit(2);
    }
    let install = &args[0];
    let index = std::fs::read_to_string(&args[1]).expect("read index");
    let session = TranslatorSession::open(
        &index,
        None,
        install.clone(),
        &FsPackInstallChecker::new(install),
    )
    .expect("open session");
    let ppocr = session.ppocr_engine().expect("ppocr engine");
    let installed: Vec<PpocrScript> = ppocr.installed_scripts().collect();
    println!(
        "installed recognizers: {:?}",
        installed.iter().map(|s| s.as_slug()).collect::<Vec<_>>()
    );

    let verbose = std::env::var_os("ROUTE_EVAL_VERBOSE").is_some();
    let mut total = Tally::default();
    for manifest in &args[2..] {
        for (image_path, page_scripts) in parse_manifest(Path::new(manifest)) {
            let rgba = image::open(&image_path)
                .unwrap_or_else(|e| panic!("open {}: {e}", image_path.display()))
                .to_rgba8();
            let (w, h) = rgba.dimensions();
            let still = build_still_image(rgba.as_raw(), w, h, MAX_IMAGE_SIZE).expect("still");
            let boxes = detect_boxes_from_still(&ppocr, &still, w, h).expect("detect");
            let predictions = ppocr
                .classify_text_boxes_image(&still.rgb, &boxes, Some(Quadrant::R0))
                .expect("classify");
            let routed = route_ppocr_predictions(&ppocr, &predictions, &boxes).expect("route");

            let readings: Vec<(PpocrScript, Vec<String>)> = page_scripts
                .iter()
                .map(|&script| {
                    let lines = ppocr
                        .recognize_text_in_boxes_image(
                            &still.rgb,
                            &boxes,
                            &vec![script; boxes.len()],
                            PpocrProfile::Still,
                            Some(Quadrant::R0),
                        )
                        .expect("recognize");
                    (script, lines.into_iter().map(|l| l.text).collect())
                })
                .collect();
            let latin_lines = ppocr
                .recognize_text_in_boxes_image(
                    &still.rgb,
                    &boxes,
                    &vec![PpocrScript::Latin; boxes.len()],
                    PpocrProfile::Still,
                    Some(Quadrant::R0),
                )
                .expect("recognize latin");

            let routed_texts: Vec<String> = ppocr
                .recognize_text_in_boxes_image(
                    &still.rgb,
                    &boxes,
                    &routed,
                    PpocrProfile::Still,
                    Some(Quadrant::R0),
                )
                .expect("recognize routed")
                .into_iter()
                .map(|l| l.text)
                .collect();

            let mut tally = Tally {
                unclassified: predictions.iter().filter(|p| p.is_none()).count(),
                ..Tally::default()
            };
            let mut failures = Vec::new();
            for (i, &route) in routed.iter().enumerate() {
                let label = readings
                    .iter()
                    .map(|(script, texts)| {
                        let letters = texts[i]
                            .chars()
                            .filter(|&c| reads(*script, Script::of_char(c)))
                            .count();
                        (*script, letters)
                    })
                    .filter(|(_, letters)| *letters >= MIN_SCRIPT_LETTERS)
                    .max_by_key(|(_, letters)| *letters)
                    .map(|(script, _)| script)
                    .unwrap_or(PpocrScript::Latin);
                let class = predictions[i]
                    .as_ref()
                    .map(|p| p.class.name())
                    .unwrap_or("-");
                let text = readings
                    .iter()
                    .find(|(s, _)| *s == label)
                    .map(|(_, t)| t[i].as_str())
                    .unwrap_or(latin_lines[i].text.as_str());
                if same_family(route, label) {
                    tally.correct += 1;
                    if verbose {
                        failures.push(format!(
                            "  ok        pulc={class:9} routed={:8} {:?}",
                            route.as_slug(),
                            routed_texts[i]
                        ));
                    }
                } else if label == PpocrScript::Latin {
                    let reference = latin_lines[i].text.as_str();
                    let edits = edit_distance(&routed_texts[i], reference);
                    tally.degraded += 1;
                    tally.degraded_edits += edits;
                    tally.degraded_chars += reference.chars().count();
                    failures.push(format!(
                        "  degraded  pulc={class:9} routed={:8} edits={edits:2} {:?} (latin: {reference:?})",
                        route.as_slug(),
                        routed_texts[i]
                    ));
                } else {
                    tally.destroyed += 1;
                    failures.push(format!(
                        "  DESTROYED pulc={class:9} routed={:8} needs={:8} {text:?}",
                        route.as_slug(),
                        label.as_slug()
                    ));
                }
            }
            println!(
                "{:28} lines={:3} unclassified={:3} correct={:3} degraded={:3} (cer {:.3}) destroyed={:3}",
                image_path.file_name().unwrap().to_string_lossy(),
                routed.len(),
                tally.unclassified,
                tally.correct,
                tally.degraded,
                tally.degraded_cer(),
                tally.destroyed,
            );
            for f in failures {
                println!("{f}");
            }
            total.add(&tally);
        }
    }
    println!(
        "TOTAL unclassified={} correct={} degraded={} (cer {:.3}) destroyed={}",
        total.unclassified,
        total.correct,
        total.degraded,
        total.degraded_cer(),
        total.destroyed
    );
}
