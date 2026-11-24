"""convert with the extras: PDF through [pdf], image text through [ocr], descriptions through [vlm]."""

import pytest

import anytomd as ad
from anytomd.infra.errors import DescriptionError
from tests.conftest import docx_with_images, text_pdf

##### PDF #####


async def test_convert_pdf_with_the_extra() -> None:
    document = ad.convert(text_pdf("Hello Anytomd"), format="pdf", ocr=False)
    assert "Hello Anytomd" in document.markdown
    assert "Hello Anytomd" in document.text
    assert document.pages == 1
    assert document.warnings == ()


async def test_convert_pdf_path(tmp_path) -> None:
    target = tmp_path / "memo.pdf"
    target.write_bytes(text_pdf("From disk"))
    assert "From disk" in ad.convert(target).markdown


async def test_convert_pdf_with_ocr_keeps_the_text_layer() -> None:
    assert "Scanned or not" in ad.convert(text_pdf("Scanned or not"), format="pdf").markdown


async def test_convert_pdf_without_the_extra(missing_extra) -> None:
    missing_extra("liteparse")
    with pytest.raises(ad.MissingExtraError, match=r"pip install 'anytomd\[pdf\]'"):
        ad.convert(text_pdf("x"), format="pdf")


async def test_convert_pdf_garbage() -> None:
    with pytest.raises(ad.MalformedDocumentError, match="PDF could not be read"):
        ad.convert(b"%PDF-1.4 garbage", format="pdf")


async def test_convert_pdf_input_too_large() -> None:
    with pytest.raises(ad.InputTooLargeError):
        ad.convert(text_pdf("x"), format="pdf", max_input_bytes=10)


##### STANDALONE IMAGES #####


async def test_convert_image_ocr_returns_its_text(path_of) -> None:
    ### The backend is pinned to tesseract: the exact line below is what its `eng` data produces, whatever liteparse bundles by default.
    assert (
        ad.convert(path_of("real_photo.png"), ocr=ad.OcrSpec(dpi=150, backend="tesseract")).markdown
        == "This is just\nan example."
    )


async def test_convert_image_ocr_default_spec_reads_the_photo(path_of) -> None:
    assert "This is just" in ad.convert(path_of("real_photo.png")).markdown


@pytest.mark.parametrize(
    ("fixture", "code"),
    [("shot.png", None), ("vector.svg", None), ("scene.heic", "ocr_failed")],
    ids=["letterless", "svg", "heic"],
)
async def test_convert_image_without_letters_is_empty(path_of, fixture, code) -> None:
    document = ad.convert(path_of(fixture))
    assert (document.markdown, document.text) == ("", "")
    assert [w.code for w in document.warnings] == ([] if code is None else [code])


async def test_convert_image_without_letters_falls_back_to_vision(path_of, fake_vision) -> None:
    document = ad.convert(path_of("shot.png"))
    assert document.markdown == "Image description: A red square"
    assert fake_vision.calls[0][1] == "image/png"


async def test_convert_image_with_letters_never_asks_vision(path_of, fake_vision) -> None:
    assert (
        ad.convert(path_of("real_photo.png"), ocr=ad.OcrSpec(dpi=150, backend="tesseract")).markdown
        == "This is just\nan example."
    )
    assert fake_vision.calls == []


async def test_convert_image_vision_only_when_ocr_is_off(path_of, fake_vision) -> None:
    assert ad.convert(path_of("real_photo.png"), ocr=False).markdown == "Image description: A red square"


async def test_convert_image_vision_failure_is_a_warning(path_of, fake_vision) -> None:
    fake_vision.error = DescriptionError("vision model 'v' failed: 429")
    document = ad.convert(path_of("shot.png"))
    assert document.markdown == ""
    assert [(w.code, w.location) for w in document.warnings] == [("skipped_element", "image")]


async def test_convert_image_without_ocr_extra(path_of, missing_extra) -> None:
    missing_extra("liteparse")
    assert ad.convert(path_of("real_photo.png")).markdown.startswith("![](image")


async def test_convert_image_input_too_large(path_of) -> None:
    with pytest.raises(ad.InputTooLargeError):
        ad.convert(path_of("real_photo.png"), max_input_bytes=10)


##### EMBEDDED IMAGES #####


async def test_convert_embedded_image_keeps_the_author_alt(raw) -> None:
    document = ad.convert(docx_with_images(1, descr="Sales chart", image=raw("shot.png")), format="docx")
    assert "![Sales chart](image1.png)" in document.markdown
    assert document.warnings == ()


async def test_convert_embedded_image_strict_empties_the_alt(raw) -> None:
    document = ad.convert(docx_with_images(1, descr="Sales chart", image=raw("shot.png")), format="docx", strict=True)
    assert "![](image1.png)" in document.markdown


async def test_convert_embedded_images_fall_back_to_vision(gallery_docx, fake_vision) -> None:
    document = ad.convert(gallery_docx, format="docx")
    assert document.markdown.count("![Image description: A red square](image") == 3
    assert document.images == ()


async def test_convert_embedded_image_ocr_text_is_one_line(raw) -> None:
    document = ad.convert(
        docx_with_images(1, image=raw("real_photo.png")), format="docx", ocr=ad.OcrSpec(dpi=150, backend="tesseract")
    )
    assert "![This is just an example.](image1.png)" in document.markdown


async def test_convert_embedded_vision_failure_keeps_alt_and_warns(single_image_docx, fake_vision) -> None:
    fake_vision.error = DescriptionError("vision model 'v' failed: 429")
    document = ad.convert(single_image_docx, format="docx")
    assert "![](image1.png)" in document.markdown
    assert [w.code for w in document.warnings] == ["skipped_element"]
    assert "429" in document.warnings[0].message


async def test_convert_vision_backend_is_closed(path_of, fake_vision) -> None:
    ad.convert(path_of("shot.png"))
    assert fake_vision.closed == ["sync"]
