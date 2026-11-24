"""ImageFallback: OCR first, vision only on blanks, the author's alt kept, strict mode respected, a failure never silent."""

import pytest

from anytomd.backends.base import Recognition
from anytomd.infra.errors import ConversionError, DescriptionError
from anytomd.infra.warnings import DocumentWarning
from anytomd.logic.fallback import KEEP_ALT, ImageFallback, KeepAltError
from tests.conftest import FakeVision

OCR_FAILED: DocumentWarning = DocumentWarning("ocr_failed", "traineddata download refused", "image")


class FakeOcr:
    def __init__(self, text: str, warning: DocumentWarning | None = None) -> None:
        self.text = text
        self.warning = warning

    def recognize(self, image: bytes) -> Recognition:
        return Recognition(self.text, self.warning)

    async def arecognize(self, image: bytes) -> Recognition:
        return Recognition(self.text, self.warning)


def fallback(
    ocr: str | None, vision: FakeVision | None, *, strict: bool = False, warning: DocumentWarning | None = None
) -> ImageFallback:
    return ImageFallback(FakeOcr(ocr, warning) if ocr is not None else None, vision, strict=strict, concurrency=2)


##### OCR FAILURES (P-09) #####


async def test_fallback_hook_ocr_failure_is_collected_for_drain() -> None:
    fb = fallback("", None, warning=OCR_FAILED)
    with pytest.raises(KeepAltError, match=KEEP_ALT):
        fb.hook(b"x", "image/png", "p")
    assert fb.drain() == (OCR_FAILED,)


async def test_fallback_resolve_ocr_failure_returns_its_warning() -> None:
    markdown, warnings = fallback("", None, warning=OCR_FAILED).resolve(b"x", "image/png")
    assert (markdown, warnings) == ("", (OCR_FAILED,))


async def test_fallback_resolve_ocr_failure_still_describes() -> None:
    markdown, warnings = fallback("", FakeVision(), warning=OCR_FAILED).resolve(b"x", "image/png")
    assert markdown.startswith("Image description:")
    assert warnings[0] == OCR_FAILED


async def test_fallback_strict_ocr_failure_raises() -> None:
    with pytest.raises(ConversionError, match=r"standalone image.*download refused"):
        fallback("", None, strict=True, warning=OCR_FAILED).resolve(b"x", "image/png")


async def test_fallback_aresolve_ocr_failure_warns() -> None:
    markdown, warnings = await fallback("", None, warning=OCR_FAILED).aresolve(b"x", "image/png")
    assert (markdown, warnings) == ("", (OCR_FAILED,))


async def test_fallback_hook_ocr_text_on_one_line() -> None:
    assert fallback("Title\n\nBody  text", FakeVision()).hook(b"x", "image/png", "p") == "Title Body text"


async def test_fallback_hook_vision_on_blank_ocr() -> None:
    assert fallback("", FakeVision()).hook(b"x", "image/png", "p") == "Image description: A red square"


async def test_fallback_hook_keeps_alt_when_silent() -> None:
    with pytest.raises(KeepAltError, match=KEEP_ALT):
        fallback("", None).hook(b"x", "image/png", "p")


async def test_fallback_hook_strict_empties_instead() -> None:
    assert fallback("", None, strict=True).hook(b"x", "image/png", "p") == ""


async def test_fallback_ahook_vision_on_blank_ocr() -> None:
    assert await fallback("", FakeVision()).ahook(b"x", "image/png", "p") == "Image description: A red square"


async def test_fallback_ahook_ocr_skips_vision() -> None:
    vision = FakeVision()
    assert await fallback("Text", vision).ahook(b"x", "image/png", "p") == "Text"
    assert vision.calls == []


async def test_fallback_resolve_keeps_newlines() -> None:
    assert fallback("Line one\nLine two", None).resolve(b"x", "image/png") == ("Line one\nLine two", ())


async def test_fallback_resolve_empty_without_vision() -> None:
    assert fallback("", None).resolve(b"x", "image/png") == ("", ())


async def test_fallback_resolve_empty_vision_answer_stays_empty() -> None:
    assert fallback("", FakeVision(answer="")).resolve(b"x", "image/png") == ("", ())


async def test_fallback_aresolve_vision_failure_is_a_warning() -> None:
    markdown, warnings = await fallback("", FakeVision(error=DescriptionError("down"))).aresolve(b"x", "image/png")
    assert (markdown, warnings[0].code, warnings[0].message) == ("", "skipped_element", "down")


async def test_fallback_active() -> None:
    assert fallback("", None).active
    assert fallback(None, FakeVision()).active
    assert not fallback(None, None).active
