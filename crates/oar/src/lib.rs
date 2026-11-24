//! The Rust OCR engine linked into `anytomd._native`: local PP-OCR recognition through oar-ocr, on ONNX Runtime.
//! `tiny` is embedded: the PP-OCRv6 tiny detector with the PP-OCRv5 mobile recognizer of one script.
//! `small` and `medium` are the multilingual PP-OCRv6 sets, loaded from a directory.

use std::io::Cursor;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use image::ImageReader;
use oar_ocr::core::{ModelSource, config::OrtSessionConfig};
use oar_ocr::domain::tasks::TextDetectionConfig;
use oar_ocr::oarocr::{OAROCRBuilder, PpOcrV6Size, ReadingOrder};
use oar_ocr::prelude::OAROCR;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

const TINY_DETECTION: &[u8] = include_bytes!("../models/pp-ocrv6_tiny_det.onnx");
const LATIN_RECOGNITION: &[u8] = include_bytes!("../models/latin_pp-ocrv5_mobile_rec.onnx");
const LATIN_DICTIONARY: &str = include_str!("../models/ppocrv5_latin_dict.txt");
const CYRILLIC_RECOGNITION: &[u8] = include_bytes!("../models/cyrillic_pp-ocrv5_mobile_rec.onnx");
const CYRILLIC_DICTIONARY: &str = include_str!("../models/ppocrv5_cyrillic_dict.txt");
const DOCUMENT_ORIENTATION: &[u8] = include_bytes!("../models/pp-lcnet_x1_0_doc_ori.onnx");

// A 400 Mpx frame costs 1.2 GB decoded: decompression bombs are refused by pixels before decoding, typed as an error.
const MAX_PIXELS: u64 = 100_000_000;

// The engine moves between threads: the Python object is one per cached spec, calls release the GIL.
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<OAROCR>();
};

// PRIVATE

fn label_of(size: PpOcrV6Size) -> &'static str {
    match size {
        PpOcrV6Size::Tiny => "tiny",
        PpOcrV6Size::Small => "small",
        _ => "medium",
    }
}

// The embedded recognizer of a script: the v6 recognizers read every script with one CJK-heavy dictionary,
// which turns Latin symbols into ideographs and Cyrillic into Latin look-alikes; one v5 model per script does not.
fn recognizer_of(script: &str) -> PyResult<(&'static [u8], &'static str)> {
    match script {
        "latin" => Ok((LATIN_RECOGNITION, LATIN_DICTIONARY)),
        "cyrillic" => Ok((CYRILLIC_RECOGNITION, CYRILLIC_DICTIONARY)),
        other => Err(PyValueError::new_err(format!(
            "unknown script: {other:?}, one of [\"cyrillic\", \"latin\"]"
        ))),
    }
}

fn names_of(size: PpOcrV6Size) -> (&'static str, &'static str, &'static str) {
    match size {
        PpOcrV6Size::Small => (
            "pp-ocrv6_small_det.onnx",
            "pp-ocrv6_small_rec.onnx",
            "ppocrv6_dict.txt",
        ),
        _ => (
            "pp-ocrv6_medium_det.onnx",
            "pp-ocrv6_medium_rec.onnx",
            "ppocrv6_dict.txt",
        ),
    }
}

// The official PP-OCRv6 detection defaults (upstream DBPostProcess): tiny keeps a lower box threshold.
fn detection_of(size: PpOcrV6Size) -> TextDetectionConfig {
    TextDetectionConfig {
        score_threshold: 0.2,
        box_threshold: if matches!(size, PpOcrV6Size::Tiny) {
            0.4
        } else {
            0.45
        },
        unclip_ratio: 1.4,
        max_candidates: 3000,
        ..TextDetectionConfig::default()
    }
}

fn failure<E: std::fmt::Display>(error: E) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

fn decode(bytes: &[u8]) -> PyResult<image::RgbImage> {
    let peek = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(failure)?;
    let (width, height) = peek.into_dimensions().map_err(failure)?;
    if u64::from(width) * u64::from(height) > MAX_PIXELS {
        return Err(PyValueError::new_err(format!(
            "image too large: {width}x{height} pixels"
        )));
    }
    ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(failure)?
        .decode()
        .map(|decoded| decoded.to_rgb8())
        .map_err(|error| PyValueError::new_err(format!("image could not be decoded: {error}")))
}

// ============================================================

// PUBLIC

/// A PP-OCR recognizer: `recognize` returns the page's text and its mean confidence.
/// One ONNX session serializes its pipeline, so the engine holds a small pool and calls round-robin over it.
#[pyclass(frozen, module = "anytomd._native")]
pub struct OarEngine {
    sessions: Vec<OAROCR>,
    next: AtomicUsize,
    size: String,
    source: String,
}

#[pymethods]
impl OarEngine {
    /// Build the pipeline `pool` times. `tiny` and the orientation model are embedded, `script` picks tiny's
    /// recognizer; `small` and `medium` read every script and need `model_dir`.
    #[new]
    #[pyo3(signature = (*, size = "tiny", script = "latin", columns = true, orientation = true, model_dir = None, threads = None, pool = 1))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        size: &str,
        script: &str,
        columns: bool,
        orientation: bool,
        model_dir: Option<&str>,
        threads: Option<usize>,
        pool: usize,
    ) -> PyResult<Self> {
        let size = match size {
            "tiny" => PpOcrV6Size::Tiny,
            "small" => PpOcrV6Size::Small,
            "medium" => PpOcrV6Size::Medium,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown model size: {other:?}"
                )));
            }
        };
        let pool = pool.max(1);
        let mut sessions = Vec::with_capacity(pool);
        for _ in 0..pool {
            let mut builder = match (size, model_dir) {
                (PpOcrV6Size::Tiny, _) => {
                    let (recognition, dictionary) = recognizer_of(script)?;
                    OAROCRBuilder::new(TINY_DETECTION, recognition, "embedded")
                        .character_dict_content(dictionary)
                }
                (_, Some(dir)) => {
                    let (detection, recognition, dictionary) = names_of(size);
                    let dir = PathBuf::from(dir);
                    OAROCRBuilder::new(
                        dir.join(detection),
                        dir.join(recognition),
                        dir.join(dictionary),
                    )
                }
                (_, None) => {
                    return Err(PyValueError::new_err(
                        "only tiny models are embedded: pass model_dir",
                    ));
                }
            };
            if orientation {
                builder = builder.with_document_image_orientation_classification(
                    ModelSource::Memory(DOCUMENT_ORIENTATION.into()),
                );
            }
            let mut builder = builder
                .text_detection_config(detection_of(size))
                .reading_order(if columns {
                    ReadingOrder::TextColumns
                } else {
                    ReadingOrder::Standard
                });
            if let Some(threads) = threads {
                builder = builder.ort_session(OrtSessionConfig::new().with_intra_threads(threads));
            }
            sessions.push(builder.build().map_err(failure)?);
        }
        Ok(Self {
            sessions,
            next: AtomicUsize::new(0),
            size: label_of(size).to_string(),
            source: match size {
                PpOcrV6Size::Tiny => format!("embedded/{script}"),
                _ => model_dir.unwrap_or_default().to_string(),
            },
        })
    }

    #[getter]
    fn size(&self) -> &str {
        &self.size
    }

    #[getter]
    fn source(&self) -> &str {
        &self.source
    }

    #[getter]
    fn pool(&self) -> usize {
        self.sessions.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "<OarEngine size={} source={} pool={}>",
            self.size,
            self.source,
            self.sessions.len()
        )
    }

    /// The page's text, rows joined: one space inside a row, one newline between rows.
    fn recognize<'py>(&self, py: Python<'py>, image: &[u8]) -> PyResult<(String, Option<f32>)> {
        let frame = decode(image)?;
        let engine =
            &self.sessions[self.next.fetch_add(1, Ordering::Relaxed) % self.sessions.len()];
        // A panic inside oar-ocr or ORT reaches Python as a typed RuntimeError, never as a BaseException.
        py.detach(|| {
            catch_unwind(AssertUnwindSafe(|| {
                let page = engine
                    .predict_with_rows(vec![frame])
                    .map_err(failure)?
                    .into_iter()
                    .next()
                    .ok_or_else(|| PyRuntimeError::new_err("the engine returned no page"))?;
                Ok((
                    page.page.text_by_rows(&page.row_ids).map_err(failure)?,
                    page.page.average_confidence(),
                ))
            }))
            .unwrap_or_else(|_| Err(PyRuntimeError::new_err("the OCR engine panicked")))
        })
    }
}
