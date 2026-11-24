"""OcrSettings: the environment refines the spec, nothing more."""

import os

import pytest

from anytomd.config.ocr import OcrSettings
from anytomd.infra.specs import OcrSpec


async def test_ocr_settings_defaults_match_the_spec() -> None:
    assert OcrSettings.from_env({}) == OcrSpec()


async def test_ocr_settings_reads_every_variable() -> None:
    spec = OcrSettings.from_env(
        {
            "ANYTOMD_OCR_LANGUAGE": "spa+eng",
            "ANYTOMD_OCR_DPI": "200",
            "ANYTOMD_OCR_BACKEND": "tesseract",
            "ANYTOMD_OCR_TESSDATA": "/td",
            "ANYTOMD_OCR_SKIP_DIAGONAL": "yes",
            "ANYTOMD_OCR_SIZE": "small",
            "ANYTOMD_OCR_COLUMNS": "on",
            "ANYTOMD_OCR_THREADS": "2",
        }
    )
    assert spec == OcrSpec(
        language="spa+eng",
        dpi=200,
        backend="tesseract",
        tessdata_path="/td",
        skip_diagonal_text=True,
        size="small",
        columns=True,
        threads=2,
    )


@pytest.mark.parametrize("empty", ["", None])
async def test_ocr_settings_empty_values_keep_the_default(empty) -> None:
    environ = {} if empty is None else {"ANYTOMD_OCR_LANGUAGE": ""}
    assert OcrSettings.from_env(environ).language == "eng"


async def test_ocr_settings_invalid_values_raise_as_the_spec_does() -> None:
    with pytest.raises(ValueError, match="invalid OCR language"):
        OcrSettings.from_env({"ANYTOMD_OCR_LANGUAGE": "spa eng"})
    with pytest.raises(ValueError, match="dpi out of range"):
        OcrSettings.from_env({"ANYTOMD_OCR_DPI": "7200"})
    with pytest.raises(ValueError, match="invalid OCR model size"):
        OcrSettings.from_env({"ANYTOMD_OCR_SIZE": "huge"})
    with pytest.raises(ValueError, match="threads out of range"):
        OcrSettings.from_env({"ANYTOMD_OCR_THREADS": "0"})


async def test_ocr_settings_model_dir(monkeypatch) -> None:
    monkeypatch.delenv("ANYTOMD_OCR_MODEL_DIR", raising=False)
    assert OcrSettings.model_dir() is None
    monkeypatch.setenv("ANYTOMD_OCR_MODEL_DIR", "")
    assert OcrSettings.model_dir() is None
    assert OcrSettings.model_dir({"ANYTOMD_OCR_MODEL_DIR": "/models"}) == "/models"


@pytest.mark.parametrize("variable", ["ANYTOMD_OCR_DPI", "ANYTOMD_OCR_THREADS"])
async def test_ocr_settings_bad_integer_names_its_variable(variable) -> None:
    with pytest.raises(ValueError, match=variable):
        OcrSettings.from_env({variable: "many"})


async def test_ocr_settings_tessdata_dirs_split_the_prefix() -> None:
    assert OcrSettings.tessdata_dirs({"TESSDATA_PREFIX": f"/a{os.pathsep}{os.pathsep}/b"}) == ["/a", "/b"]


@pytest.mark.parametrize(("value", "columns"), [(None, True), ("", True), ("0", False), ("false", False), ("on", True)])
async def test_ocr_settings_columns_flag_turns_the_default_off(value, columns) -> None:
    environ = {} if value is None else {"ANYTOMD_OCR_COLUMNS": value}
    assert OcrSettings.from_env(environ).columns is columns
