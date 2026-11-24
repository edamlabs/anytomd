"""LiteparseBackend: the spec reaches LiteParse verbatim, real OCR reads, misses stay silent, failures warn."""

import liteparse
import pytest

from anytomd.backends.liteparse import LiteparseBackend
from anytomd.infra.errors import MalformedDocumentError
from anytomd.infra.specs import OcrSpec
from tests.conftest import text_pdf


@pytest.fixture(scope="module")
def backend() -> LiteparseBackend:
    ### Tesseract, not the ambient default: the exact string below is what the `eng` data gives, whatever liteparse bundles.
    return LiteparseBackend(OcrSpec(dpi=150, backend="tesseract"), pdf_ocr=False)


@pytest.fixture
def recorded_kwargs(monkeypatch) -> list[dict]:
    """The exact kwargs every LiteParse construction received, with the real engine kept out."""
    seen: list[dict] = []

    class FakeParse:
        def __init__(self, **kwargs) -> None:
            seen.append(kwargs)

    monkeypatch.setattr(liteparse, "LiteParse", FakeParse)
    return seen


async def test_liteparse_recognize_text(backend, raw) -> None:
    assert backend.recognize(raw("real_photo.png")).text == "This is just\nan example."


@pytest.mark.parametrize(
    ("fixture", "failed"),
    [("shot.png", False), ("vector.svg", False), ("scene.heic", True)],
    ids=["letterless", "svg", "heic"],
)
async def test_liteparse_recognize_miss(backend, raw, fixture, failed) -> None:
    recognition = backend.recognize(raw(fixture))
    assert recognition.text == ""
    assert (recognition.warning is not None) == failed


async def test_liteparse_arecognize(backend, raw) -> None:
    assert (await backend.arecognize(raw("real_photo.png"))).text == "This is just\nan example."


async def test_liteparse_arecognize_miss(backend, raw) -> None:
    recognition = await backend.arecognize(raw("scene.heic"))
    assert recognition.text == ""
    assert recognition.warning is not None


async def test_liteparse_read(backend) -> None:
    document = backend.read(text_pdf("Hello"))
    assert ("Hello" in document.markdown, document.pages, document.warnings) == (True, 1, ())


async def test_liteparse_read_path(backend, tmp_path) -> None:
    target = tmp_path / "doc.pdf"
    target.write_bytes(text_pdf("Path read"))
    assert "Path read" in backend.read(str(target)).markdown


async def test_liteparse_aread(backend) -> None:
    assert "Async read" in (await backend.aread(text_pdf("Async read"))).markdown


async def test_liteparse_read_refusal(backend) -> None:
    with pytest.raises(MalformedDocumentError, match="invalid PDF"):
        backend.read(b"%PDF-1.4 garbage")


async def test_liteparse_aread_refusal(backend) -> None:
    with pytest.raises(MalformedDocumentError):
        await backend.aread(b"%PDF-1.4 garbage")


async def test_liteparse_page_errors_become_warnings() -> None:
    page_error = type("PageError", (), {"page_num": 3, "message": "corrupt xref"})()
    result = type("Result", (), {"pages": [], "num_pages": 3, "page_errors": [page_error]})()
    warning = LiteparseBackend._common_document(result).warnings[0]
    assert (warning.code, warning.message, warning.location) == ("skipped_element", "page 3: corrupt xref", "page 3")


##### P-03 · THE SPEC REACHES THE ENGINE #####


async def test_liteparse_spec_kwargs_pass_through(recorded_kwargs) -> None:
    LiteparseBackend(OcrSpec(language="spa+eng", dpi=200, tessdata_path="/td", backend="tesseract"))
    assert recorded_kwargs[0] == {
        "ocr_enabled": True,
        "output_format": "json",
        "ocr_language": "spa+eng",
        "dpi": 200.0,
        "skip_diagonal_text": False,
        "quiet": True,
        "tessdata_path": "/td",
        "ocr_backend": "tesseract",
    }


async def test_liteparse_spec_defaults_send_no_optional_kwargs(recorded_kwargs) -> None:
    LiteparseBackend(OcrSpec(), pdf_ocr=False)
    ocr_kwargs, pdf_kwargs = recorded_kwargs
    assert ("tessdata_path" not in ocr_kwargs, "ocr_backend" not in ocr_kwargs) == (True, True)
    assert (ocr_kwargs["ocr_enabled"], pdf_kwargs["ocr_enabled"]) == (True, False)


async def test_liteparse_spec_ocria_names_its_backend(recorded_kwargs) -> None:
    LiteparseBackend(OcrSpec(backend="ocria-nano"))
    assert recorded_kwargs[0]["ocr_backend"] == "ocria-nano"


##### P-06 · A FAILED OCR IS NOT A LETTERLESS IMAGE #####


async def test_liteparse_recognize_parse_error_warns(monkeypatch) -> None:
    class ExplodingParse:
        def __init__(self, **_) -> None:
            pass

        def parse(self, image: bytes) -> None:
            msg = "OCR failed: downloading spa.traineddata HTTP 403"
            raise liteparse.ParseError(msg)

    monkeypatch.setattr(liteparse, "LiteParse", ExplodingParse)
    recognition = LiteparseBackend(OcrSpec()).recognize(b"scan")
    warning = recognition.warning
    assert warning is not None
    assert (recognition.text, warning.code, "403" in warning.message) == (
        "",
        "ocr_failed",
        True,
    )
