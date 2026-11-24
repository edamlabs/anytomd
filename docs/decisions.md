# Decisions

### 2026-11-24 · `auto` prefers the Rust engine; `full` stays without it

**Decision.** When `anytomd-oar` is importable, the default OCR path is oar (PP-OCRv6 tiny, models inside the
wheel); when it is not, liteparse decides (Tesseract, ocria-nano, its own order). The `oar` extra does not enter
`full`.

**Evidence.** Bench over the corpus with the Citroën sheet: oar-tiny 17/28 at p50 23 ms, against tesseract-spa-300
15/28 at p50 155 ms and ocria-nano 17/28 at p50 64 ms; the corpus totals are equal or better, and the wheel works
offline from the first use, which neither liteparse engine does without a download. `full` without oar: the wheel
weighs 23 MB per platform and adds no fidelity over what `full` already installs, and scanned PDFs go through
liteparse anyway.

**Rejected.** oar as the only engine (PDF, DOCX images still need liteparse); ocria as the default (modified
OpenRAIL-M license with use restrictions, needs a first pull); keeping the liteparse-only order.

### 2026-11-24 · One engine, several ONNX sessions: the pool

**Decision.** `OarEngine` holds `pool` OAROCR sessions and dispatches calls round-robin. The backend uses `pool=4`
for tiny, `1` for small/medium.

**Evidence.** One shared session does not scale: four Python threads finish in 1.60 s, while four separate engines
do it in 0.68 s — each ONNX session serializes its own pipeline. With the pool, four threads at one intra-op thread
reach a 3.42× speedup over serial (P-13). A small/medium session costs around 150 MB, so its pool stays at 1.

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

### 2026-11-24 · No language→script mapping

**Decision.** The oar engine does not map `language` (a Tesseract code) to a script: PP-OCRv6 recognizes Latin,
Cyrillic and Chinese with one model. `language` only affects which languages `ocr_ready()` checks in Tesseract.

**Evidence.** The rus corpus reads 2/2 without a mapping (P-10 as measured); the mapping would add state with no
demonstrated benefit. What the step actually needed was model loading from disk, and it is proven with oar-small.

**Rejected.** A per-backend language→script table; exposing `script` in the spec (nothing in the house uses it).

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
