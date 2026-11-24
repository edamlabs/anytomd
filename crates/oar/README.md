# anytomd-oar

Local Rust OCR engine for [anytomd](https://pypi.org/project/anytomd/): PP-OCR on ONNX Runtime, via
[oar-ocr](https://crates.io/crates/oar-ocr). Installed as the `anytomd[oar]` extra; used by
`ad.OcrSpec(backend="oar")` or automatically when it is the only engine present.

- `size="tiny"` needs nothing: the PP-OCRv6 tiny detector, the PP-OCRv5 Latin and Cyrillic recognizers and the
  document-orientation model are compiled into the wheel; `OarEngine(script=...)` picks the recognizer.
- `size="small"` and `size="medium"` load three files each from `ANYTOMD_OCR_MODEL_DIR`. Nothing is ever downloaded at run time.
- Apache-2.0 throughout (code and PaddleOCR model weights).
