"""What local OCR still needs, asked without touching the network: the probe behind `anytomd.ocr_ready`."""

import importlib.util
import sys
from pathlib import Path
from types import MappingProxyType
from typing import Final

from anytomd.backends.oar import OarBackend
from anytomd.backends.registry import BackendRegistry
from anytomd.config.ocr import OcrSettings
from anytomd.infra.specs import OcrSpec

_TESSDATA_DIRS: Final = (".tesseract-rs/tessdata", ".local/share/liteparse/tessdata", ".cache/liteparse/tessdata")

### The files each non-embedded oar size needs in ANYTOMD_OCR_MODEL_DIR; `tiny` ships inside the wheel.
_OAR_MODELS: Final = MappingProxyType(
    {
        "small": ("pp-ocrv6_small_det.onnx", "pp-ocrv6_small_rec.onnx", "ppocrv6_dict.txt"),
        "medium": ("pp-ocrv6_medium_det.onnx", "pp-ocrv6_medium_rec.onnx", "ppocrv6_dict.txt"),
    }
)


class OcrReadiness:
    """Gaps only: the empty list means every resource the spec needs is already on disk."""

    __slots__ = ()

    ##### PRIVATE #####

    @staticmethod
    def _run_installed(library: str) -> bool:
        ### find_spec resolves without importing: readiness must not load the engine it is probing.
        ### A None entry is an uninstall (tests simulate it): not installed, and find_spec would raise on it.
        if library in sys.modules:
            return sys.modules[library] is not None
        return importlib.util.find_spec(library) is not None

    @staticmethod
    def _run_candidate_dirs(spec: OcrSpec) -> list[Path]:
        given = [Path(spec.tessdata_path)] if spec.tessdata_path else []
        env = [Path(value) for value in OcrSettings.tessdata_dirs()]
        home = [Path.home() / relative for relative in _TESSDATA_DIRS]
        return given + env + home

    @staticmethod
    def _run_extra(library: str, extra: str) -> list[str]:
        return (
            []
            if OcrReadiness._run_installed(library)
            else [f"the '{extra}' extra is not installed: pip install 'anytomd[{extra}]'"]
        )

    def _run_tesseract(self, spec: OcrSpec) -> list[str]:
        directories = self._run_candidate_dirs(spec)
        missing = [
            language
            for language in spec.language.split("+")
            if not any((directory / f"{language}.traineddata").exists() for directory in directories)
        ]
        return (
            []
            if not missing
            else [
                f"tesseract languages {missing}: convert once with network, or pass tessdata_path / set TESSDATA_PREFIX"
            ]
        )

    def _run_ocria(self) -> list[str]:
        if self._run_extra("liteparse", "ocr"):
            return []  # the caller already says the extra is missing
        models = BackendRegistry._common_module("liteparse.models")
        if models is None:
            return []
        pulled = any(model.name == "ocria-nano" and model.complete for model in models.list())
        return (
            []
            if pulled
            else ["ocria-nano model not pulled: python -c \"from liteparse import models; models.pull('ocria-nano')\""]
        )

    def _run_oar(self, spec: OcrSpec) -> list[str]:
        ### The first reason wins: a wheel without the engine makes the language question moot.
        gaps = [
            reason
            for failed, reason in (
                (not OarBackend.available, "the Rust OCR engine is not built for this platform (Intel macOS)"),
                (
                    not spec.oar_reads,
                    f"no embedded oar recognizer reads {spec.language!r}: use size 'small' or 'medium'",
                ),
            )
            if failed
        ][:1]
        needed = _OAR_MODELS.get(spec.size, ())
        if gaps or not needed:
            return gaps
        configured = OcrSettings.model_dir()
        directory = Path(configured) if configured else None
        present = directory is not None and directory.is_dir() and all((directory / name).exists() for name in needed)
        return (
            []
            if present
            else [
                f"oar {spec.size} models missing from {configured or 'ANYTOMD_OCR_MODEL_DIR unset'}: place the PP-OCRv6 files there, or use size 'tiny'"
            ]
        )

    def _run_auto(self, spec: OcrSpec) -> list[str]:
        ### auto prefers oar, then liteparse with its own default backend: the probe mirrors that order.
        if OarBackend.available and spec.oar_reads:
            return self._run_oar(spec)
        if not self._run_installed("liteparse"):
            return self._run_extra("liteparse", "ocr")
        return self._run_ocria() or self._run_tesseract(spec)

    ############################################################

    ##### PUBLIC #####

    def run(self, spec: OcrSpec) -> list[str]:
        """What `convert(..., ocr=spec)` would still need on this machine, offline."""
        match spec.backend:
            case "oar":
                return self._run_oar(spec)
            case "tesseract":
                return self._run_extra("liteparse", "ocr") + self._run_tesseract(spec)
            case "ocria-nano":
                return self._run_extra("liteparse", "ocr") + self._run_ocria()
            case _:
                return self._run_auto(spec)
