"""The image rule: OCR reads the letters; only an image without letters reaches the vision model."""

import asyncio
import logging
import threading
from typing import Final

from anytomd.backends.base import Describer, Recognition, Recognizer
from anytomd.infra.errors import ConversionError, DescriptionError
from anytomd.infra.formats import DESCRIPTION_LABEL
from anytomd.infra.warnings import DocumentWarning

logger = logging.getLogger(__name__)

KEEP_ALT: Final = "anytomd: no text and no description, original alt kept"


class KeepAltError(Exception):
    """Raised into the engine so it keeps the author's alt text; its warning is filtered out afterwards."""


class ImageFallback:
    """OCR first, vision second, nothing invented: one rule for embedded and standalone images.

    An OCR failure is never silence: its warning is collected by `drain` for the conversion to attach,
    and raised as `ConversionError` under `strict`.
    """

    __slots__ = ("_describer", "_gate", "_lock", "_recognizer", "_strict", "_warnings")

    ##### PRIVATE #####

    def _hook_nothing(self, location: str) -> str:
        logger.debug("%s: no text and no description; keeping the original alt text", location)
        ### Under strict mode the engine would turn the keep-alt warning into an error, so the alt is emptied instead.
        return "" if self._strict else self._hook_raise()

    @staticmethod
    def _hook_raise() -> str:
        raise KeepAltError(KEEP_ALT)

    @staticmethod
    def _common_label(description: str) -> str:
        return f"{DESCRIPTION_LABEL}{description}" if description else ""

    def _common_recognize(self, recognition: Recognition, where: str) -> tuple[str, tuple[DocumentWarning, ...]]:
        """The image's text with its whitespace normalized; a failure raises or becomes a warning, never silence."""
        if recognition.warning is None:
            return recognition.text, ()
        if self._strict:
            msg = f"{where}: {recognition.warning.message}"
            raise ConversionError(msg)
        logger.debug("%s: ocr failed: %s", where, recognition.warning.message)
        return "", (recognition.warning,)

    def _hook_collect(self, recognition: Recognition) -> str:
        text, warnings = self._common_recognize(recognition, "embedded image")
        with self._lock:
            self._warnings.extend(warnings)
        return " ".join(text.split())

    def _resolve_combined(
        self, image: bytes, mime_type: str, warnings: tuple[DocumentWarning, ...]
    ) -> tuple[str, tuple[DocumentWarning, ...]]:
        label, describe_warnings = self._resolve_describe(image, mime_type)
        return label, warnings + describe_warnings

    async def _aresolve_combined(
        self, image: bytes, mime_type: str, warnings: tuple[DocumentWarning, ...]
    ) -> tuple[str, tuple[DocumentWarning, ...]]:
        label, describe_warnings = await self._aresolve_describe(image, mime_type)
        return label, warnings + describe_warnings

    def _resolve_describe(self, image: bytes, mime_type: str) -> tuple[str, tuple[DocumentWarning, ...]]:
        describer = self._describer
        try:
            return self._common_label(describer.describe(image, mime_type) if describer else ""), ()
        except DescriptionError as error:
            logger.debug("vision fallback failed: %s", error)
            return "", (DocumentWarning("skipped_element", str(error), "image"),)

    async def _aresolve_describe(self, image: bytes, mime_type: str) -> tuple[str, tuple[DocumentWarning, ...]]:
        describer = self._describer
        try:
            async with self._gate:
                return self._common_label(await describer.adescribe(image, mime_type) if describer else ""), ()
        except DescriptionError as error:
            logger.debug("vision fallback failed: %s", error)
            return "", (DocumentWarning("skipped_element", str(error), "image"),)

    ############################################################

    ##### PUBLIC #####

    def __init__(
        self, recognizer: Recognizer | None, describer: Describer | None, *, strict: bool, concurrency: int
    ) -> None:
        self._recognizer = recognizer
        self._describer = describer
        self._strict = strict
        self._gate = asyncio.Semaphore(concurrency)
        self._warnings: list[DocumentWarning] = []
        ### The engine may call the sync hook from its worker threads: the list is shared, the lock is cheap.
        self._lock = threading.Lock()

    @property
    def active(self) -> bool:
        """Whether any capability is present; without one the engine's own image handling stands."""
        return bool(self._recognizer or self._describer)

    def drain(self) -> tuple[DocumentWarning, ...]:
        """The warnings collected since the last drain, taken out of the accumulator."""
        with self._lock:
            drained, self._warnings = tuple(self._warnings), []
        return drained

    def hook(self, image: bytes, mime_type: str, _prompt: str) -> str:
        """Alt text for one embedded image, on one line."""
        recognizer, describer = self._recognizer, self._describer
        text = self._hook_collect(recognizer.recognize(image)) if recognizer else ""
        logger.debug("embedded image: ocr %s", "found text" if text else "found nothing")
        described = self._common_label(describer.describe(image, mime_type)) if describer and not text else ""
        return text or described or self._hook_nothing("embedded image")

    async def ahook(self, image: bytes, mime_type: str, _prompt: str) -> str:
        """`hook`, concurrent across the document's images under one gate."""
        recognizer, describer = self._recognizer, self._describer
        async with self._gate:
            text = self._hook_collect(await recognizer.arecognize(image)) if recognizer else ""
            logger.debug("embedded image: ocr %s", "found text" if text else "found nothing")
            described = (
                self._common_label(await describer.adescribe(image, mime_type)) if describer and not text else ""
            )
        return text or described or self._hook_nothing("embedded image")

    def resolve(self, image: bytes, mime_type: str) -> tuple[str, tuple[DocumentWarning, ...]]:
        """Markdown for a standalone image: its text, else its labeled description, else `""` — with any warnings."""
        recognizer = self._recognizer
        text, warnings = (
            self._common_recognize(recognizer.recognize(image), "standalone image") if recognizer else ("", ())
        )
        logger.debug("standalone image: ocr %s", "found text" if text else "found nothing")
        return (text, warnings) if text or not self._describer else self._resolve_combined(image, mime_type, warnings)

    async def aresolve(self, image: bytes, mime_type: str) -> tuple[str, tuple[DocumentWarning, ...]]:
        """`resolve`, off the event loop."""
        recognizer = self._recognizer
        async with self._gate:
            text, warnings = (
                self._common_recognize(await recognizer.arecognize(image), "standalone image")
                if recognizer
                else ("", ())
            )
        logger.debug("standalone image: ocr %s", "found text" if text else "found nothing")
        return (
            (text, warnings)
            if text or not self._describer
            else await self._aresolve_combined(image, mime_type, warnings)
        )
