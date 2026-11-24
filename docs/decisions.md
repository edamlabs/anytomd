# Decisions

### 2026-11-24 · `auto` prefers the Rust engine; `full` stays without it

**Decision.** When `anytomd-oar` is importable and reads the language, the default OCR path is oar (tiny set,
models inside the wheel); otherwise liteparse decides (Tesseract, ocria-nano, its own order). The `oar` extra does
not enter `full`.

**Evidence.** Bench over the corpus with the Citroën sheet, truth corrected against the scan: oar-tiny 26/28, against
tesseract-spa-300 18/28 and ocria-nano 21/28; corpus totals 134 (default through oar) against 123 (ocria-nano), and
the wheel works offline from the first use, which neither liteparse engine does without a download. `full` without
oar: the wheel weighs 34 MB per platform, and scanned PDFs go through liteparse anyway. Revisit `full` if oar takes
scanned PDFs.

**Rejected.** oar as the only engine (PDF, DOCX images still need liteparse); ocria as the default (modified
OpenRAIL-M license with use restrictions, needs a first pull); keeping the liteparse-only order.

### 2026-11-24 · One engine, several ONNX sessions: the pool

**Decision.** `OarEngine` holds `pool` OAROCR sessions and dispatches calls round-robin. The backend uses `pool=4`
for tiny, `1` for small/medium.

**Evidence.** One shared session does not scale: four Python threads finish in 1.60 s, while four separate engines
do it in 0.68 s — each ONNX session serializes its own pipeline. With the pool, four threads at one intra-op thread
reached 3.42× over serial with the v6 tiny recognizer, and 2.2–2.4× with the Latin v5 one, bound by
memory bandwidth (P-13 gates at 2.0: a held GIL gives ~1×). A small/medium session costs around 150 MB, so its pool stays at 1.

**Rejected.** An engine per thread-local (doubles the memory and the cache key); parallelism only through ORT
intra-op threads (the pipeline stages outside the session remain serialized).

### 2026-11-24 · Decompression bombs refused by pixels, before decoding

**Decision.** The Rust crate reads the dimensions from the header and refuses images over 100 Mpx before decoding,
as a typed `ValueError`; the backend turns it into an `ocr_failed` warning.

**Evidence.** A 20000×20000 JPEG of 4.7 MB costs 1.2 GB decoded: the child process peaks at 2739 MB of RSS; with
the guard, 170 MB, with the same warning in the document. It is what the P-18 gate (< 1.5 GB over the set of
malformed payloads) measures.

**Rejected.** Rescaling large images (it requires decoding them first); relying on PIL's `Image.MAX_IMAGE_PIXELS`
(Python never touches the pixels: the decoding is in Rust).

### 2026-11-24 · oar tiny reads by script: one PP-OCRv5 recognizer per alphabet

**Decision.** The embedded tiny set is the PP-OCRv6 tiny detector with the PP-OCRv5 mobile Latin and Cyrillic
recognizers. `OcrSpec.language` (a Tesseract code) maps to the script through a bank (`OCR_SCRIPTS`); the first
language of `"spa+eng"` decides. A language with no embedded recognizer makes `auto` fall back to liteparse and the
explicit `oar` backend raise. This reverses the same-day decision "No language→script mapping", whose evidence was
wrong.

**Evidence.** Mini bench on the Citroën sheet with its truth corrected (9 of 28 fields had been mis-transcribed):
v6 tiny 21/28, v6 small 25/28 at 31 MB, v6 medium 25/28 at 139 MB, v6 tiny det + latin v5 rec 26/28 at 9.9 MB, ocrs
12/28. The v6 recognizers carry a CJK dictionary: `Y*B***` came out as `Y安B开研部`, and Cyrillic came out as
Latin look-alikes (tiny) or was dropped (small, medium); the earlier "rus 2/2" counted only digits. The v5 Cyrillic
recognizer reads the C4 sentence exactly. Cost: about 13× the CPU of v6 tiny — 1.07 s per 420 px sheet on one
intra-op thread, 0.22 s on all cores; wheel 34 MB (cap 40). Report: `.ergon/reports/plans/bench.md`.

**Rejected.** v6 tiny (fast, 21/28, ideographs in Latin text); v6 small as default (31 MB, one field fewer, same CJK
fault); upscaling by default (±1 field, sign depends on the model); ocrs (12/28).

### 2026-10-26 · The project moves to EDamLabs

**Decision.** `damvolkov/anytomd` is transferred to `edamlabs/anytomd`; documentation moves to
`edamlabs.github.io/anytomd`. The PyPI name stays `anytomd`.

**Evidence.** EDamLabs gathers the open-source bindings and services under one organisation, one GitHub App
(`eself-maintenance`), one set of provider settings (`SM_*`) and one maintenance engine (esteward). GitHub keeps
redirects from the old URLs; a PyPI project cannot be renamed, so the distribution name is kept.

**Rejected.** A new PyPI name (breaks every existing install for a cosmetic gain).

### 2026-10-26 · Routine updates merge themselves; health moves to esteward

**Decision.** Renovate merges Python tooling, GitHub actions, lock file maintenance, the agent's CLI pin and
runtime patch releases through GitHub auto-merge after every required check; the engine, PyO3 and tokio never
merge themselves. `health.yml` and `tools/health.py` are removed: esteward watches the dependencies daily
(`fleet/anytomd.toml`).

**Evidence.** The same policy runs on etrafilatura; esteward covers everything `health.yml` checked, plus PEP 792
status, license changes, OSV advisories and action pins.

**Rejected.** A human merging every green routine PR; two reports of the same facts.

### 2026-10-26 · Documentation built with Zensical

**Decision.** The site builds with Zensical (`zensical build --strict --clean`); `mkdocs.yml` stays the
configuration, read natively.

**Evidence.** Material for MkDocs is in maintenance until 2026-11-05 and MkDocs 2.0 drops plugins and theme
overrides; the existing site builds unchanged with Zensical 0.0.71.

**Rejected.** Pinning MkDocs 1.x on a framework leaving maintenance.
