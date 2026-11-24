"""Fetch the PP-OCR model files for `crates/oar` from pinned release URLs, verified by SHA-256.

The four embedded files (tiny det/rec/dict + orientation) are needed to build the `anytomd-oar` wheel;
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
    "pp-ocrv6_tiny_rec.onnx": Model(
        f"{V6}/pp-ocrv6_tiny_rec.onnx", "9ef676d6ed3c88256a2d92c640c44f25b0c40947e111b14b8be8f594091563e6", Kind.WHEEL
    ),
    "ppocrv6_tiny_dict.txt": Model(
        f"{V6}/ppocrv6_tiny_dict.txt", "c5cbe34ef40c29c4df07ed012bf96569cb69a2d2a01a07027e9f13cb832bd9cd", Kind.WHEEL
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
    parser.add_argument("--embedded", action="store_true", help="only the four files the wheel embeds")
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
