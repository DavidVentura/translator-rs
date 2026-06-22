#[cfg(feature = "uniffi")]
uniffi::setup_scaffolding!();

pub mod api;
pub mod catalog;
pub mod coords;
pub mod homography;
pub mod homography_ekf;
pub mod language;
pub mod ocr;
pub mod script;
pub mod script_normalize;
pub mod settings;
pub mod tts;

#[cfg(feature = "raster")]
pub mod color_matting;
#[cfg(feature = "raster")]
pub mod live_frame;
#[cfg(feature = "raster")]
pub mod text_metrics;

pub use settings::BackgroundMode;
