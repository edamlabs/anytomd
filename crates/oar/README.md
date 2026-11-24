# crates/oar

The OCR engine of [anytomd](https://pypi.org/project/anytomd/), linked into `anytomd._native`: PP-OCR on ONNX Runtime
through [oar-ocr](https://crates.io/crates/oar-ocr). Not published on its own; `ad.OcrSpec(backend="oar")` or `auto`
reach it.

- `size="tiny"` needs nothing: the PP-OCRv6 tiny detector, the PP-OCRv5 Latin and Cyrillic recognizers and the
  document-orientation model are compiled in; `script` picks the recognizer. `make models` fetches them, sha-pinned.
- `size="small"` and `size="medium"` load three files each from `ANYTOMD_OCR_MODEL_DIR`. Nothing is ever downloaded at
  run time.
- Not built for Intel macOS: ONNX Runtime publishes no `x86_64-apple-darwin` binaries.
- Apache-2.0 code and PaddleOCR weights; ONNX Runtime is MIT (see `THIRD_PARTY_LICENSES.md`).
