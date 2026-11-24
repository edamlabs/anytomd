"""Specs: the defaults convert promises, and the OcrSpec gates (P-01)."""

import pytest

from anytomd.infra.specs import (
    OCR_BACKENDS,
    OCR_SIZES,
    ConversionSpec,
    OcrSpec,
    VisionSpec,
    as_backend_name,
    as_model_size,
)


async def test_specs_conversion_defaults() -> None:
    spec = ConversionSpec()
    assert (spec.images, spec.comments, spec.strict, spec.ocr, spec.vision) == (False, False, False, OcrSpec(), None)
    assert (spec.max_input_bytes, spec.max_image_bytes, spec.max_zip_bytes) == (8 * 1024**3, 4 * 1024**3, 16 * 1024**3)


async def test_specs_vision_defaults() -> None:
    spec = VisionSpec(model="m")
    assert (spec.base_url, spec.api_key, spec.max_tokens, spec.concurrency) == (None, None, 256, 4)


##### OCR SPEC (P-01) #####


async def test_specs_ocr_defaults() -> None:
    spec = OcrSpec()
    assert (spec.language, spec.dpi, spec.backend, spec.tessdata_path, spec.skip_diagonal_text) == (
        "eng",
        300,
        "auto",
        None,
        False,
    )
    assert (spec.size, spec.columns, spec.threads) == ("tiny", False, None)


@pytest.mark.parametrize("language", ["spa eng", "spa+EN", "spa+", "+eng", "es-ES", ""])
async def test_specs_ocr_rejects_bad_language(language) -> None:
    with pytest.raises(ValueError, match="invalid OCR language"):
        OcrSpec(language=language)


async def test_specs_ocr_accepts_combined_language() -> None:
    assert OcrSpec(language="spa+eng").language == "spa+eng"


@pytest.mark.parametrize("dpi", [71, 601, 0, -300])
async def test_specs_ocr_rejects_dpi_outside_range(dpi) -> None:
    with pytest.raises(ValueError, match="dpi out of range"):
        OcrSpec(dpi=dpi)


@pytest.mark.parametrize("backend", ["rapid", "", "OAR"])
async def test_specs_ocr_rejects_unknown_backend(backend) -> None:
    with pytest.raises(ValueError, match="invalid OCR backend"):
        OcrSpec(backend=backend)


@pytest.mark.parametrize("size", ["large", "", "TINY"])
async def test_specs_ocr_rejects_unknown_size(size) -> None:
    with pytest.raises(ValueError, match="invalid OCR model size"):
        OcrSpec(size=size)


async def test_specs_ocr_is_a_frozen_hashable_key() -> None:
    spec = OcrSpec(language="spa")
    assert hash(spec) == hash(OcrSpec(language="spa"))  # the engine cache keys on it
    frozen_field = "dpi"
    with pytest.raises(AttributeError):
        setattr(spec, frozen_field, 150)  # frozen dataclass: its __setattr__ raises FrozenInstanceError


##### NARROWING (the environment's door) #####


@pytest.mark.parametrize("value", sorted(OCR_BACKENDS))
async def test_specs_as_backend_name_accepts_the_bank(value) -> None:
    assert as_backend_name(value) == value


@pytest.mark.parametrize("value", sorted(OCR_SIZES))
async def test_specs_as_model_size_accepts_the_bank(value) -> None:
    assert as_model_size(value) == value


async def test_specs_as_narrowing_rejects_outside_the_bank() -> None:
    for bad in ("Rapid", "", "OAR"):
        with pytest.raises(ValueError, match="invalid OCR backend"):
            as_backend_name(bad)
    for bad in ("huge", "", "Tiny"):
        with pytest.raises(ValueError, match="invalid OCR model size"):
            as_model_size(bad)
