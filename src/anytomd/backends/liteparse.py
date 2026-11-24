"""liteparse: local OCR for images and the PDF reader. Imported only when the `ocr` or `pdf` extra is installed."""

import logging

import liteparse

from anytomd.backends.base import Recognition
from anytomd.infra.documents import Document
from anytomd.infra.errors import MalformedDocumentError
from anytomd.infra.specs import OcrSpec
from anytomd.infra.warnings import DocumentWarning

logger = logging.getLogger(__name__)


class LiteparseBackend:
    """One OCR engine and one PDF engine, both in-process, built from one `OcrSpec`."""

    __slots__ = ("_ocr", "_pdf")

    ##### PRIVATE #####

    @staticmethod
    def _common_kwargs(spec: OcrSpec) -> dict:
        kwargs: dict = {
            "ocr_language": spec.language,
            "dpi": float(spec.dpi),
            "skip_diagonal_text": spec.skip_diagonal_text,
            "quiet": True,
        }
        if spec.tessdata_path is not None:
            kwargs["tessdata_path"] = spec.tessdata_path
        if spec.backend in ("tesseract", "ocria-nano"):
            kwargs["ocr_backend"] = spec.backend
        return kwargs

    @staticmethod
    def _common_text(result: liteparse.ParseResult) -> str:
        return result.pages[0].text.strip() if result.pages else ""

    @staticmethod
    def _common_document(result: liteparse.ParseResult) -> Document:
        return Document(
            markdown="\n\n".join(page.markdown for page in result.pages),
            text="\n\n".join(page.text for page in result.pages),
            pages=result.num_pages,
            warnings=tuple(
                DocumentWarning("skipped_element", f"page {error.page_num}: {error.message}", f"page {error.page_num}")
                for error in result.page_errors
            ),
        )

    @staticmethod
    def _common_refusal(error: liteparse.ParseError) -> MalformedDocumentError:
        return MalformedDocumentError(f"PDF could not be read: {error}")

    ############################################################

    ##### PUBLIC #####

    def __init__(self, spec: OcrSpec, *, pdf_ocr: bool = True) -> None:
        common = self._common_kwargs(spec)
        self._ocr = liteparse.LiteParse(ocr_enabled=True, output_format="json", **common)
        self._pdf = liteparse.LiteParse(ocr_enabled=pdf_ocr, output_format="markdown", **common)

    def recognize(self, image: bytes) -> Recognition:
        """The image's text, or `""` when it holds none; a warning when reading itself failed."""
        try:
            return Recognition(self._common_text(self._ocr.parse(image)))
        except liteparse.ParseError as error:
            logger.debug("ocr could not read the image: %s", error)
            return Recognition("", DocumentWarning("ocr_failed", str(error), "image"))

    async def arecognize(self, image: bytes) -> Recognition:
        """`recognize`, off the event loop."""
        try:
            return Recognition(self._common_text(await self._ocr.aparse(image)))
        except liteparse.ParseError as error:
            logger.debug("ocr could not read the image: %s", error)
            return Recognition("", DocumentWarning("ocr_failed", str(error), "image"))

    def read(self, source: str | bytes) -> Document:
        """A PDF, every page in reading order."""
        try:
            return self._common_document(self._pdf.parse(source))
        except liteparse.ParseError as error:
            raise self._common_refusal(error) from error

    async def aread(self, source: str | bytes) -> Document:
        """`read`, off the event loop."""
        try:
            return self._common_document(await self._pdf.aparse(source))
        except liteparse.ParseError as error:
            raise self._common_refusal(error) from error
