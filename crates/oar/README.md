# anytomd-oar

Local Rust OCR engine for [anytomd](https://pypi.org/project/anytomd/): PP-OCRv6 on ONNX Runtime, via
[oar-ocr](https://crates.io/crates/oar-ocr). Installed as the `anytomd[oar]` extra; used by
`ad.OcrSpec(backend="oar")` or automatically when it is the only engine present.

- `size="tiny"` needs nothing: the PP-OCRv6 tiny models and the document-orientation model are compiled into the wheel.
- `size="small"` and `size="medium"` load three files each from `ANYTOMD_OCR_MODEL_DIR`. Nothing is ever downloaded at run time.
- Apache-2.0 throughout (code and PaddleOCR model weights).
