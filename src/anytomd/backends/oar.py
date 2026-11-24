"""oar backend: local Rust OCR, PP-OCRv6 through the optional `anytomd-oar` extension. Imported only when its extra is installed."""

import asyncio
import logging

from anytomd_oar import OarEngine

from anytomd.backends.base import Recognition
from anytomd.infra.specs import OcrSpec
from anytomd.infra.warnings import DocumentWarning

logger = logging.getLogger(__name__)


class OarBackend:
    """The Rust recognizer: models on disk or embedded, the GIL released during inference."""

    __slots__ = ("_engine",)

    def __init__(self, spec: OcrSpec, model_dir: str | None = None) -> None:
        ### One ONNX session serializes its pipeline: the cheap tiny model is pooled four ways, the heavy ones stay single.
        self._engine = OarEngine(
            size=spec.size,
            columns=spec.columns,
            threads=spec.threads,
            model_dir=model_dir,
            pool=4 if spec.size == "tiny" else 1,
        )

    def recognize(self, image: bytes) -> Recognition:
        """The image's text; a failed read warns, it never passes for a letterless image."""
        try:
            text, confidence = self._engine.recognize(image)
        except (RuntimeError, ValueError) as error:
            logger.debug("oar could not read the image: %s", error)
            return Recognition("", DocumentWarning("ocr_failed", str(error), "image"))
        logger.debug("oar read %d characters at confidence %.3f", len(text), confidence or 0.0)
        return Recognition(text)

    async def arecognize(self, image: bytes) -> Recognition:
        """`recognize`, off the event loop: the engine runs native, threads handle it."""
        return await asyncio.to_thread(self.recognize, image)
