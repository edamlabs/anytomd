"""Optional backends, loaded by name only when their extra is installed."""

import functools
import importlib
import logging
from types import MappingProxyType, ModuleType
from typing import Final, NoReturn, Protocol

from anytomd.backends.base import Describer, Reader, Recognizer
from anytomd.backends.oar import OarBackend
from anytomd.infra.errors import DescriptionError, MissingExtraError
from anytomd.infra.specs import OcrSpec, VisionSpec

logger = logging.getLogger(__name__)

### Each third-party library the extras install, and the extra that installs it.
EXTRAS: Final = MappingProxyType({"liteparse": "ocr", "openai": "vlm", "pydantic_settings": "vlm"})


class _LiteparseEngine(Recognizer, Reader, Protocol):
    """Both ports at once: one liteparse instance recognizes images and reads PDFs."""


class BackendRegistry:
    """Resolves the optional backends: None when their extra is absent, never an import error at `import anytomd`."""

    __slots__ = ()

    ##### PRIVATE #####

    @staticmethod
    def _common_module(path: str) -> ModuleType | None:
        ### The one deferred import, by design: optional backends import their third-party library at module level, so
        ### they are loaded here, by name, only once their extra is installed — `import anytomd` must never need them.
        try:
            return importlib.import_module(path)
        except ModuleNotFoundError as error:
            match error.name:
                case str() as name if name in EXTRAS:
                    logger.debug("optional backend %s unavailable: install anytomd[%s]", path, EXTRAS[name])
                    return None
            raise

    @staticmethod
    @functools.lru_cache(maxsize=8)
    def _common_liteparse(ocr: OcrSpec, *, pdf_ocr: bool) -> _LiteparseEngine | None:
        ### One engine per spec, cached across instances and conversions: engines are heavy and hashable specs key them.
        module = BackendRegistry._common_module("anytomd.backends.liteparse")
        return module.LiteparseBackend(ocr, pdf_ocr=pdf_ocr) if module else None

    def _describer_spec(self, vision: VisionSpec | None) -> VisionSpec | None:
        settings = self._common_module("anytomd.config.vision")
        return vision or (settings.VisionSettings().spec() if settings else None)

    def _recognizer_liteparse(self, ocr: OcrSpec) -> Recognizer | None:
        return self._common_liteparse(ocr, pdf_ocr=True)

    def _recognizer_oar(self, ocr: OcrSpec) -> Recognizer | None:
        return self._common_oar(ocr, self._oar_model_dir())

    @staticmethod
    @functools.lru_cache(maxsize=4)
    def _common_oar(ocr: OcrSpec, model_dir: str | None) -> Recognizer | None:
        return OarBackend(ocr, model_dir=model_dir) if OarBackend.available else None

    @staticmethod
    def _oar_model_dir() -> str | None:
        settings = BackendRegistry._common_module("anytomd.config.ocr")
        return settings.OcrSettings.model_dir() if settings else None

    ############################################################

    ##### PUBLIC #####

    def recognizer(self, ocr: OcrSpec | None) -> Recognizer | None:
        """Local OCR for the spec's backend: `auto` prefers the Rust oar engine when installed and able to read the language."""
        if ocr is None:
            return None
        match ocr.backend:
            case "oar":
                return self._recognizer_oar(ocr) or self.unbuilt()
            case "auto":
                ### oar first when it reads the language; a script no embedded model reads goes to liteparse.
                return (self._recognizer_oar(ocr) if ocr.oar_reads else None) or self._recognizer_liteparse(ocr)
            case _:
                return self._recognizer_liteparse(ocr)

    def reader(self, ocr: OcrSpec | None) -> Reader:
        """The PDF reader; raises when the `pdf` extra is missing."""
        backend = self._common_liteparse(ocr or OcrSpec(), pdf_ocr=ocr is not None)
        return backend or self.missing("PDF", "pdf")

    def describer(self, vision: VisionSpec | None) -> Describer | None:
        """The vision model from `vision` or `ANYTOMD_VISION_*`, when the `vlm` extra is installed and a model is set."""
        module = self._common_module("anytomd.backends.openai")
        spec = self._describer_spec(vision) if module else None
        logger.debug("vision fallback %s", f"on with {spec.model!r}" if spec else "off: no vision model configured")
        return module.OpenaiBackend(spec) if module and spec else None

    def require(self, vision: VisionSpec | None) -> Describer:
        """The vision model, or the exact reason there is none."""
        module = self._common_module("anytomd.backends.openai")
        spec = self._describer_spec(vision) if module else self.missing("describe", "vlm")
        return module.OpenaiBackend(spec) if module and spec else self.unconfigured()

    @staticmethod
    def missing(what: str, extra: str) -> NoReturn:
        """Raise the error that names the extra to install."""
        msg = f"{what} needs the '{extra}' extra: pip install 'anytomd[{extra}]'"
        raise MissingExtraError(msg)

    @staticmethod
    def unbuilt() -> NoReturn:
        """Raise the error that says this wheel carries no Rust OCR engine (Intel macOS) and what to use instead."""
        msg = "the Rust OCR engine is not built for this platform (Intel macOS): use backend 'auto' or 'tesseract'"
        raise ValueError(msg)

    @staticmethod
    def unconfigured() -> NoReturn:
        """Raise the error that says how to configure a vision model."""
        msg = "no vision model: set ANYTOMD_VISION_MODEL or pass vision=VisionSpec(model=...)"
        raise DescriptionError(msg)
