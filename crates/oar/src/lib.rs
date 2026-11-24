//! The optional `anytomd[oar]` engine: local PP-OCRv6 recognition in Rust through oar-ocr,
//! on ONNX Runtime. Models load from a directory (disk, offline) or, for tiny, from the wheel.

use std::io::Cursor;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use image::ImageReader;
use oar_ocr::core::{ModelSource, config::OrtSessionConfig};
use oar_ocr::domain::tasks::TextDetectionConfig;
use oar_ocr::oarocr::{OAROCRBuilder, PpOcrV6Size, ReadingOrder};
use oar_ocr::prelude::{OAROCR, TextRegion};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

const TINY_DETECTION: &[u8] = include_bytes!("../models/pp-ocrv6_tiny_det.onnx");
const TINY_RECOGNITION: &[u8] = include_bytes!("../models/pp-ocrv6_tiny_rec.onnx");
const TINY_DICTIONARY: &str = include_str!("../models/ppocrv6_tiny_dict.txt");
const DOCUMENT_ORIENTATION: &[u8] = include_bytes!("../models/pp-lcnet_x1_0_doc_ori.onnx");
const ORIENTATION_NAME: &str = "pp-lcnet_x1_0_doc_ori.onnx";

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

fn names_of(size: PpOcrV6Size) -> (&'static str, &'static str, &'static str) {
    match size {
        PpOcrV6Size::Tiny => (
            "pp-ocrv6_tiny_det.onnx",
            "pp-ocrv6_tiny_rec.onnx",
            "ppocrv6_tiny_dict.txt",
        ),
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

fn bounds(region: &TextRegion) -> Option<(u32, u32, u32, u32)> {
    let mut points = region.bounding_box.points.iter();
    let first = points.next()?;
    let mut box_bounds = (first.x, first.y, first.x, first.y);
    for point in points {
        box_bounds = (
            box_bounds.0.min(point.x),
            box_bounds.1.min(point.y),
            box_bounds.2.max(point.x),
            box_bounds.3.max(point.y),
        );
    }
    Some((
        box_bounds.0.max(0.) as u32,
        box_bounds.1.max(0.) as u32,
        box_bounds.2 as u32,
        box_bounds.3 as u32,
    ))
}

// One recognized line: text, confidence, and the box as x0, y0, x1, y1.
type Line = (String, f32, u32, u32, u32, u32);

fn line_of(region: &TextRegion) -> Option<Line> {
    let text = region.text.as_ref()?;
    let bounds = bounds(region)?;
    Some((
        text.to_string(),
        region.confidence.unwrap_or(0.0),
        bounds.0,
        bounds.1,
        bounds.2,
        bounds.3,
    ))
}

// ============================================================

// PUBLIC

/// A PP-OCRv6 recognizer: `recognize` returns the page's text and its mean confidence.
/// One ONNX session serializes its pipeline, so the engine holds a small pool and calls round-robin over it.
#[pyclass(frozen)]
pub struct OarEngine {
    sessions: Vec<OAROCR>,
    next: AtomicUsize,
    size: String,
    source: String,
}

#[pymethods]
impl OarEngine {
    /// Build the pipeline `pool` times. Only `tiny` and orientation models are embedded: other sizes need `model_dir`.
    #[new]
    #[pyo3(signature = (*, size = "tiny", columns = false, orientation = true, model_dir = None, threads = None, pool = 1))]
    fn new(
        size: &str,
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
            let builder = match model_dir {
                Some(dir) => {
                    let (detection, recognition, dictionary) = names_of(size);
                    let dir = PathBuf::from(dir);
                    OAROCRBuilder::new(
                        dir.join(detection),
                        dir.join(recognition),
                        dir.join(dictionary),
                    )
                }
                None if matches!(size, PpOcrV6Size::Tiny) => {
                    OAROCRBuilder::new(TINY_DETECTION, TINY_RECOGNITION, "embedded")
                        .character_dict_content(TINY_DICTIONARY)
                }
                None => {
                    return Err(PyValueError::new_err(
                        "only tiny models are embedded: pass model_dir",
                    ));
                }
            };
            let mut builder = builder;
            if orientation {
                // The classifier is optional on disk and embedded otherwise: a missing file means no orientation pass.
                let classifier = match model_dir {
                    Some(dir) => {
                        let path = PathBuf::from(dir).join(ORIENTATION_NAME);
                        path.is_file().then_some(ModelSource::Path(path))
                    }
                    None => Some(ModelSource::Memory(DOCUMENT_ORIENTATION.into())),
                };
                if let Some(source) = classifier {
                    builder = builder.with_document_image_orientation_classification(source);
                }
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
            source: model_dir.unwrap_or("embedded").to_string(),
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
        py.detach(|| {
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
        })
    }

    /// Every recognized line: text, confidence and box, in geometric detection order.
    fn lines<'py>(&self, py: Python<'py>, image: &[u8]) -> PyResult<Vec<Line>> {
        let frame = decode(image)?;
        let engine =
            &self.sessions[self.next.fetch_add(1, Ordering::Relaxed) % self.sessions.len()];
        let pages = py.detach(|| engine.predict(vec![frame]).map_err(failure))?;
        Ok(pages
            .into_iter()
            .flat_map(|page| {
                page.text_regions
                    .iter()
                    .filter_map(line_of)
                    .collect::<Vec<_>>()
            })
            .collect())
    }
}

#[pymodule]
fn _oar(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<OarEngine>()?;
    Ok(())
}
