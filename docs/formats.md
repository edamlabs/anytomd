# Formats

`convert` picks the format from the file's first bytes and its extension. For bytes, you name it: `format="docx"`.

## Documents

| Format | Extensions | Out |
|---|---|---|
| Word | `.docx` | Headings, paragraphs, bold/italic, lists, tables, links, images, comments |
| PowerPoint | `.pptx` | One `## Slide N` per slide, tables, speaker notes, images, comments |
| Excel | `.xlsx` `.xls` | One `## Sheet` per sheet, as a table |
| PDF | `.pdf` | Every page, in reading order. **Not native — needs `anytomd[pdf]`** |

## Web and data

| Format | Extensions | Out |
|---|---|---|
| HTML | `.html` `.htm` | Headings, tables, lists, links, code blocks; `<title>` becomes `document.title` |
| CSV | `.csv` | One table |
| Jupyter | `.ipynb` | Markdown cells as-is, code cells fenced |
| JSON | `.json` | Pretty-printed, fenced |
| XML | `.xml` | Pretty-printed, fenced |

## Text and code

| Format | Extensions | Out |
|---|---|---|
| Text | `.txt` `.md` `.rst` `.log` `.yaml` `.toml` `.ini` … | As-is; UTF-8, UTF-16 and Windows-1252 detected |
| Code | `.py` `.rs` `.js` `.ts` `.go` `.java` `.c` … | Fenced, with its language |

## Images

`.png` `.jpg` `.gif` `.webp` `.bmp` `.tiff` `.svg` `.heic` `.avif`, alone or inside a DOCX, PPTX or XLSX.

| Installed | An image with text | An image without text |
|---|---|---|
| nothing | its text | `""` |
| `[vlm]` | its text | `Image description: A red square on white.` |

Intel macOS has no ONNX Runtime build, so its wheel carries no Rust OCR: there, `[ocr]` reads the text. AVIF, HEIC and
SVG are not decoded by the Rust engine: the image keeps its alt text and the document carries an `ocr_failed` warning.

OCR always goes first: it is local, free and exact. The vision model is asked only when OCR finds no letters, and its answer is always labeled `Image description:`, so you can tell it from text that was really in the image.

```python
import anytomd as ad

ad.convert("invoice.png").markdown   # e.g. 'Invoice 2026\nAmount due: 1.240,00 EUR'
ad.convert("cat.png").markdown       # e.g. 'Image description: A tabby cat asleep on a wooden chair.'
```

Inside documents, the same rule fills each image's alt text, on one line:

```markdown
![Invoice 2026 Amount due: 1.240,00 EUR](image1.png)
![Image description: A tabby cat asleep on a wooden chair.](image2.png)
```

When neither finds anything, the author's own alt text stays: `![Sales chart](chart.png)`.

Vision needs a model → [VisionSpec](api.md#visionspec). For OCR, `True` takes the language and DPI from the
environment; pass an [OcrSpec](api.md#ocrspec) to choose them per call — `ad.OcrSpec(language="spa", dpi=300)` reads a
Spanish form, `ocr=False` turns the pass off. `auto` uses the built-in Rust `oar` engine when it reads the language, liteparse otherwise.

## OCR engines

Three local engines, one switch: `ocr=ad.OcrSpec(backend=...)`. The columns are the measured numbers of
`tools/ocr_bench.py` (baseline in `docs/ocr_fidelity.json`, enforced by `make ocr`).

| Engine | Install | Offline from first use | Citroën sheet, 28 key values | p50 |
|---|---|---|---|---|
| `tesseract` | `[ocr]` | only with the language file on disk | 18/28 (spa, 300 dpi) | 145 ms |
| `ocria-nano` | `[ocr]` | only after pulling the model | 21/28 | 63 ms |
| `oar` | built in | yes: the tiny models ship inside the wheel | 26/28 (tiny) · 25/28 (small) | 66 ms (tiny) |

Tesseract downloads `tessdata_best/<lang>.traineddata` from GitHub on first use. On an offline machine, keep the files
yourself and point at them (`tessdata_path`, or `TESSDATA_PREFIX`); `anytomd.ocr_ready()` lists exactly what is still
missing without touching the network. ocria-nano is pulled the same way:

```bash
python -c "from liteparse import models; models.pull('ocria-nano')"
```

Its weights carry a modified OpenRAIL-M license with use restrictions: measured and documented here, not the
recommendation.

oar is Rust (PP-OCR on ONNX Runtime through oar-ocr, linked into `anytomd._native`; Apache-2.0 weights). `size="tiny"` needs nothing: the PP-OCRv6
tiny detector and one PP-OCRv5 recognizer per script are compiled into the wheel, and `language` picks the script —
Latin for `eng`, `spa`, `fra`, `deu` and 29 more, Cyrillic for `rus`, `ukr`, `bel`, `bul`, `srp`, `mkd`, `kaz`; with
`"spa+eng"` the first language decides. A language no embedded recognizer reads (`chi_sim`) sends `auto` to liteparse;
`small` and `medium` are the multilingual PP-OCRv6 sets, three files each in `ANYTOMD_OCR_MODEL_DIR` — never downloaded
at run time. The tiny set is accuracy first: about 0.2 s per scanned page on all cores, about 1 s on one
(`threads=1`). oar reads each row across, so a form keeps each label beside its value (`columns=False` returns
detection order instead). Prose set in two columns comes out interleaved row by row in either mode.

The numbers come from `tools/ocr_bench.py` against the committed corpus; the baseline is `docs/ocr_fidelity.json`,
enforced by `make ocr`.

## PDF

The engine leaves PDF out on purpose: LLMs read it natively. To convert it here:

```bash
pip install "anytomd[pdf]"
```

```python
document = ad.convert("report.pdf")
document.markdown   # every page, joined
document.pages      # 12
```

Text layers are read directly; scanned pages go through OCR (`ocr=False` skips it). Without the extra:

```python
ad.convert("report.pdf")
# MissingExtraError: PDF needs the 'pdf' extra: pip install 'anytomd[pdf]'
```

## Text recovery

Every supported format is generated from one and the same 103-word passage, then converted back and scored on how
many of those words come out. It is a comparison between formats, not an absolute benchmark: the passage is fixed and
committed, so the bars show where each format's text lands relative to the others.

![Text recovered per format](assets/fidelity.svg)

Text and structured formats recover the passage in full. The gap is the OCR row: a rendered image is read back through
the OCR engine, and some words survive imperfectly. The SVG row is zero — the engine treats an image as pixels and does
not read vector `<text>`, so nothing comes back until the binding can rasterize it.

The measurement lives in `tools/fidelity.py`; `docs/fidelity.json` is the committed baseline. Re-run and compare with:

```bash
make fidelity            # re-measure, fail on any regression against the baseline
make fidelity ARGS=--update   # rewrite the baseline and the chart
```

## Not supported

Plain ZIP archives, EPUB, binaries → `UnknownFormatError`.
