"""oar backend: local Rust OCR, PP-OCR through the optional `anytomd-oar` extension. Imported only when its extra is installed."""

import asyncio
import logging
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from anytomd_oar import OarEngine

from anytomd.backends.base import Recognition
from anytomd.infra.specs import OCR_SCRIPTS, OcrModelSize, OcrScript, OcrSpec
from anytomd.infra.warnings import DocumentWarning

logger = logging.getLogger(__name__)

### One ONNX session serializes its pipeline: the light tiny set is pooled four ways, the heavy ones stay single.
POOLS: Final[Mapping[OcrModelSize, int]] = MappingProxyType({"tiny": 4, "small": 1, "medium": 1})


class OarBackend:
    """The Rust recognizer: models on disk or embedded, the GIL released during inference."""

    __slots__ = ("_engine",)

    ##### PRIVATE #####

    @staticmethod
    def _init_script(spec: OcrSpec) -> OcrScript:
        match (spec.script, spec.oar_reads):
            case (OcrScript() as script, _):
                return script
            case (None, True):
                ### small and medium read every script with one model: the argument only picks tiny's recognizer.
                return OcrScript.LATIN
            case _:
                msg = (
                    f"no embedded oar recognizer reads {spec.language!r}: one of {sorted(OCR_SCRIPTS)}, "
                    "or size 'small'/'medium' with ANYTOMD_OCR_MODEL_DIR"
                )
                raise ValueError(msg)

    ############################################################

    ##### PUBLIC #####

    def __init__(self, spec: OcrSpec, model_dir: str | None = None) -> None:
        self._engine = OarEngine(
            size=spec.size,
            script=self._init_script(spec),
            columns=spec.columns,
            threads=spec.threads,
            model_dir=model_dir,
            pool=POOLS[spec.size],
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
