"""Fetch the PP-OCR model files for `crates/oar` from pinned release URLs, verified by SHA-256.

The six embedded files (v6 tiny det, latin and cyrillic v5 rec + dicts, orientation) are needed to build the `anytomd-oar` wheel;
`--embedded` fetches only those. The small/medium sets are for a runtime `model_dir` — the engine never
downloads anything itself.

    python3 tools/get_oar_models.py --embedded
    python3 tools/get_oar_models.py
"""

import argparse
import hashlib
import sys
import urllib.request
from dataclasses import dataclass
from enum import StrEnum, auto
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "crates" / "oar" / "models"

V6 = "https://github.com/GreatV/oar-ocr/releases/download/v0.7.0"
V3 = "https://github.com/GreatV/oar-ocr/releases/download/v0.3.0"


class Kind(StrEnum):
    WHEEL = auto()
    DISK = auto()


@dataclass(frozen=True, slots=True)
class Model:
    url: str
    sha256: str
    kind: Kind


MODELS: dict[str, Model] = {
    "pp-ocrv6_tiny_det.onnx": Model(
        f"{V6}/pp-ocrv6_tiny_det.onnx", "193bab7a04fca699a6c82e6abb5b81bdb28177f0abd4062552b04908dafb19f8", Kind.WHEEL
    ),
    "latin_pp-ocrv5_mobile_rec.onnx": Model(
        f"{V3}/latin_pp-ocrv5_mobile_rec.onnx",
        "e3a6bfeea1c8a01d6fccfd480a0bd363fd907f8c65931e228bb2736f5c3e142f",
        Kind.WHEEL,
    ),
    "ppocrv5_latin_dict.txt": Model(
        f"{V3}/ppocrv5_latin_dict.txt", "ccbcc45730b3fbbd9050c5bc74db6a99067141ef1035e3d14889a84a6b9b1aff", Kind.WHEEL
    ),
    "cyrillic_pp-ocrv5_mobile_rec.onnx": Model(
        f"{V3}/cyrillic_pp-ocrv5_mobile_rec.onnx",
        "a18d96d7c8d73d90f2ed056549caa1de3a8e6cb744cccba16cd593ea8cd2d569",
        Kind.WHEEL,
    ),
    "ppocrv5_cyrillic_dict.txt": Model(
        f"{V3}/ppocrv5_cyrillic_dict.txt",
        "db40aa52ceb112055be80c694afdf655d5d2c4f7873704524cc16a447ca913ba",
        Kind.WHEEL,
    ),
    "pp-lcnet_x1_0_doc_ori.onnx": Model(
        f"{V3}/pp-lcnet_x1_0_doc_ori.onnx",
        "bbcd6c2b43ab15d2e605455aef2cd280ab87b570824522d24a69e9298875a1ac",
        Kind.WHEEL,
    ),
    "pp-ocrv6_small_det.onnx": Model(
        f"{V6}/pp-ocrv6_small_det.onnx", "d73e0058b7a8086bbd57f3d10b8bcd4ff95363f67e06e2762b5e814fe9c9410e", Kind.DISK
    ),
    "pp-ocrv6_small_rec.onnx": Model(
        f"{V6}/pp-ocrv6_small_rec.onnx", "5435fd747c9e0efe15a96d0b378d5bd157e9492ed8fd80edf08f30d02fa24634", Kind.DISK
    ),
    "pp-ocrv6_medium_det.onnx": Model(
        f"{V6}/pp-ocrv6_medium_det.onnx", "eb13b44b25bb36f89528b68720af8a61d9cf381176107f465db1757b65d086e1", Kind.DISK
    ),
    "pp-ocrv6_medium_rec.onnx": Model(
        f"{V6}/pp-ocrv6_medium_rec.onnx", "9c09abf0957f7968c7586464b7397b84ad2387a0497a351af40e9acc71b673ba", Kind.DISK
    ),
    "ppocrv6_dict.txt": Model(
        f"{V6}/ppocrv6_dict.txt", "b5f2bfe2bdd9448429e3e82b51c789775d9b42f2403d082b00662eb77e401c5d", Kind.DISK
    ),
}


def _main_fetch(name: str, model: Model) -> str:
    """One file: cached when it matches the pin, fetched and verified otherwise."""
    target = ROOT / name
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == model.sha256:
        return f"cached  {name}"
    request = urllib.request.Request(model.url, headers={"User-Agent": "anytomd-oar-models/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
    except Exception as error:  # noqa: BLE001
        return f"FAILED  {name}: {error}"
    digest = hashlib.sha256(payload).hexdigest()
    if digest != model.sha256:
        return f"FAILED  {name}: sha256 {digest[:12]} does not match the pin"
    target.write_bytes(payload)
    return f"fetched {name} ({len(payload)} bytes)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedded", action="store_true", help="only the six files the wheel embeds")
    arguments = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    wanted = {name: model for name, model in MODELS.items() if not arguments.embedded or model.kind == Kind.WHEEL}
    failures = 0
    for name, model in wanted.items():
        line = _main_fetch(name, model)
        failures += line.startswith("FAILED")
        sys.stdout.write(f"{line}\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
