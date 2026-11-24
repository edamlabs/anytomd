"""OCR quality, latency and footprint per engine over the `tests/resources/ocr` corpus.

`make bench-ocr` re-measures and compares against the committed baseline `docs/ocr_fidelity.json`;
`--update` rewrites it. One engine is a named way to ask anytomd for text from an image; one item is
a fixture plus its truth. C1 (the real Citroen sheet) is scored when its image is present locally —
it is never committed.
"""

import argparse
import importlib.util
import json
import os
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import anytomd as ad

ROOT: Final = Path(__file__).resolve().parent.parent
CORPUS: Final = ROOT / "tests" / "resources" / "ocr"
CAR_SHEET: Final = ROOT / "tests" / "resources" / "media" / "car_sheet.png"
BASELINE: Final = ROOT / "docs" / "ocr_fidelity.json"
FIELDS_TOLERANCE: Final = 2
CER_TOLERANCE: Final = 0.01


@dataclass(frozen=True, slots=True)
class Item:
    """One scored fixture: what to read, what must appear, whether per-character scoring applies."""

    id: str
    file: str | None
    truth: str
    text_scored: bool = True
    timed: bool = True

    def present(self) -> bool:
        return self.file is not None and (CAR_SHEET if self.file == "car_sheet.png" else CORPUS / self.file).exists()


ITEMS: Final[tuple[Item, ...]] = (
    ### C1 truth text is a paraphrase of the scan, not a transcription: only its 28 fields are scored.
    Item("C1", "car_sheet.png", "c1_sheet.truth.json", text_scored=False),
    Item("C2", "c2_sheet.jpg", "c2_sheet.truth.json"),
    Item("C3", "c3_screenshot.png", "c3_screenshot.truth.json"),
    Item("C4-spa", "c4_spa.png", "c4_languages.truth.json", text_scored=False),
    Item("C4-fra", "c4_fra.png", "c4_languages.truth.json", text_scored=False),
    Item("C4-deu", "c4_deu.png", "c4_languages.truth.json", text_scored=False),
    Item("C4-por", "c4_por.png", "c4_languages.truth.json", text_scored=False),
    Item("C4-rus", "c4_rus.png", "c4_languages.truth.json", text_scored=False),
    Item("C5", "c5_receipt.jpg", "c5_receipt.truth.json"),
    Item("C6a", "c6_rot90.png", "c6_orientation.truth.json"),
    Item("C6b", "c6_rot180.png", "c6_orientation.truth.json"),
    Item("C10", "c10_small.png", "c10_small.truth.json"),
    Item("C8", "c8_album50.docx", "c8_album50.truth.json", text_scored=False, timed=False),
    Item("C9", "c9_scan10.pdf", "c9_scan10.truth.json", text_scored=False, timed=False),
)


def _oar_engines() -> dict[str, ad.OcrSpec]:
    """The oar arms only when the wheel is importable; medium is measured out (71 hits at p50 386 ms)."""
    if importlib.util.find_spec("anytomd_oar") is None:
        return {}
    os.environ.setdefault("ANYTOMD_OCR_MODEL_DIR", str(ROOT / "crates" / "oar" / "models"))
    return {
        "oar-tiny": ad.OcrSpec(backend="oar"),
        "oar-rows": ad.OcrSpec(backend="oar", columns=False),
        "oar-small": ad.OcrSpec(backend="oar", size="small"),
    }


ENGINES: Final[dict[str, ad.OcrSpec | bool]] = {
    "default": True,
    "tesseract-eng-150": ad.OcrSpec(language="eng", dpi=150, backend="tesseract"),
    "tesseract-spa-300": ad.OcrSpec(language="spa", backend="tesseract"),
    "ocria-nano": ad.OcrSpec(backend="ocria-nano"),
    **_oar_engines(),
}


def _normalize(text: str) -> str:
    return "".join(character for character in text.upper() if not character.isspace())


def _missed(output: str, fields: dict[str, str]) -> tuple[str, ...]:
    haystack = _normalize(output)
    return tuple(name for name, value in fields.items() if _normalize(value) not in haystack)


def _cer(output: str, expected: str) -> float:
    """Character error rate: edit distance over the expected length, after case and whitespace folding."""
    left, right = _normalize(output), _normalize(expected)
    if not right:
        return 0.0 if not left else 1.0
    row = list(range(len(right) + 1))
    for index, first in enumerate(left, start=1):
        previous, row[0] = row[0], index
        for position, second in enumerate(right, start=1):
            previous, row[position] = (
                row[position],
                min(row[position] + 1, row[position - 1] + 1, previous + (first != second)),
            )
    return row[-1] / len(right)


##### RUN #####


def _path_of(item: Item) -> Path:
    return CAR_SHEET if item.file == "car_sheet.png" else CORPUS / str(item.file)


def _convert(item: Item, *, ocr: ad.OcrSpec | bool) -> str:
    return ad.convert(_path_of(item).read_bytes(), format=_path_of(item).suffix.removeprefix("."), ocr=ocr).text


def _percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))]


def _score_item(item: Item, *, ocr: ad.OcrSpec | bool, repeats: int) -> dict:
    truth = json.loads((CORPUS / item.truth).read_text())
    timings = [0.0]
    if item.timed:
        timings = []
        for _ in range(repeats):
            started = time.perf_counter()
            output = _convert(item, ocr=ocr)
            timings.append((time.perf_counter() - started) * 1000)
    else:
        output = _convert(item, ocr=ocr)
    missed = _missed(output, truth["fields"])
    return {
        "hits": len(truth["fields"]) - len(missed),
        "of": len(truth["fields"]),
        "missing": list(missed),
        "median_ms": round(sorted(timings)[len(timings) // 2], 1) if item.timed else None,
        **({"cer": round(_cer(output, truth["text"]), 4)} if item.text_scored else {}),
    }


def _oar_skips(item: Item, ocr: object) -> bool:
    ### oar recognizes images; the C8/C9 embedded route hands images to the PDF/DOCX reader, which stays liteparse.
    return isinstance(ocr, ad.OcrSpec) and ocr.backend == "oar" and str(item.file).endswith((".docx", ".pdf"))


def measure(engines: dict[str, ad.OcrSpec | bool], repeats: int) -> dict[str, dict]:
    """Per engine: one entry per present item, pooled latencies, peak RSS of this process."""
    scores: dict[str, dict] = {}
    for engine_id, ocr in engines.items():
        per_item = {
            item.id: _score_item(item, ocr=ocr, repeats=repeats)
            for item in ITEMS
            if (item.file is None or item.present()) and not _oar_skips(item, ocr)
        }
        medians = [entry["median_ms"] for entry in per_item.values() if entry["median_ms"] is not None]
        scores[engine_id] = {
            "per_item": per_item,
            "hits_total": sum(entry["hits"] for entry in per_item.values()),
            "p50_ms": round(_percentile(medians, 0.5), 1) if medians else None,
            "p95_ms": round(_percentile(medians, 0.95), 1) if medians else None,
            "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024,
        }
    return scores


def compare(scores: dict[str, dict], baseline: dict[str, dict]) -> tuple[str, ...]:
    """Regression lines: two field hits lost or a point of CER gained against the baseline fails the run."""
    regressions: list[str] = []
    for engine_id, now in scores.items():
        before = baseline.get(engine_id)
        for item_id, entry in now["per_item"].items():
            old = before.get("per_item", {}).get(item_id) if before else None
            if old is None:
                continue
            if entry["hits"] < old["hits"] - FIELDS_TOLERANCE:
                regressions.append(f"{engine_id}/{item_id}: hits {old['hits']} -> {entry['hits']}")
            if "cer" in entry and "cer" in old and entry["cer"] > old["cer"] + CER_TOLERANCE:
                regressions.append(f"{engine_id}/{item_id}: cer {old['cer']:.3f} -> {entry['cer']:.3f}")
    return tuple(regressions)


def report(scores: dict[str, dict]) -> None:
    for engine_id, entry in scores.items():
        sys.stdout.write(
            f"== {engine_id}: {entry['hits_total']} hits · p50 {entry['p50_ms']} ms · p95 {entry['p95_ms']} ms · rss {entry['max_rss_mb']} MB\n"
        )
        for item_id, item in entry["per_item"].items():
            cer = f"  cer {item['cer']:.3f}" if "cer" in item else ""
            timing = f"  {item['median_ms']} ms" if item["median_ms"] is not None else ""
            missing = f"  missing: {','.join(item['missing'])}" if item["missing"] else ""
            sys.stdout.write(f"   {item_id:<7} {item['hits']:>2}/{item['of']}{timing}{cer}{missing}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure OCR quality, latency and footprint per engine.")
    parser.add_argument("--update", action="store_true", help="rewrite the baseline")
    parser.add_argument("--engine", action="append", help="engine id (repeatable); default: all")
    parser.add_argument("--repeats", type=int, default=3, help="timed runs per image item")
    arguments = parser.parse_args()

    engines = {name: spec for name, spec in ENGINES.items() if not arguments.engine or name in arguments.engine}
    scores = measure(engines, arguments.repeats)
    baseline = json.loads(BASELINE.read_text()) if BASELINE.exists() else {}
    regressions = compare(scores, baseline)
    report(scores)
    if arguments.update:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        baseline.update(scores)
        BASELINE.write_text(json.dumps(baseline, indent=2) + "\n")
        sys.stdout.write(f"updated {BASELINE.relative_to(ROOT)}\n")
    sys.stdout.write("".join(f"REGRESSION  {line}\n" for line in regressions))
    return 1 if regressions else 0


if __name__ == "__main__":
    raise SystemExit(main())
