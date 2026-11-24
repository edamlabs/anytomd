"""OcrReadiness: gaps reported offline, never a network call, never an engine load; and `ocr_ready` on top."""

import sys

import pytest

import anytomd as ad
from anytomd.backends.oar import OarBackend
from anytomd.backends.registry import BackendRegistry
from anytomd.infra.specs import OcrSpec
from anytomd.logic.ocrreadiness import OcrReadiness


@pytest.fixture
def readiness() -> OcrReadiness:
    return OcrReadiness()


@pytest.fixture
def oar_present(monkeypatch) -> None:
    monkeypatch.setattr(OarBackend, "available", True)


##### OAR #####


async def test_readiness_oar_unbuilt_names_the_platform(readiness) -> None:
    gaps = readiness.run(OcrSpec(backend="oar", language="chi_sim"))
    assert gaps == ["the Rust OCR engine is not built for this platform (Intel macOS)"]


async def test_readiness_oar_tiny_is_always_ready(readiness, oar_present) -> None:
    assert readiness.run(OcrSpec(backend="oar")) == []


async def test_readiness_oar_tiny_without_script_is_a_gap(readiness, oar_present) -> None:
    assert readiness.run(OcrSpec(backend="oar", language="chi_sim")) == [
        "no embedded oar recognizer reads 'chi_sim': use size 'small' or 'medium'"
    ]


async def test_readiness_auto_skips_oar_for_a_script_it_cannot_read(readiness, oar_present, missing_extra) -> None:
    missing_extra("liteparse")
    assert readiness.run(OcrSpec(language="chi_sim")) == [
        "the 'ocr' extra is not installed: pip install 'anytomd[ocr]'"
    ]


async def test_readiness_oar_small_needs_its_models_on_disk(readiness, oar_present, monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("ANYTOMD_OCR_MODEL_DIR", raising=False)
    gaps = readiness.run(OcrSpec(backend="oar", size="small"))
    assert len(gaps) == 1
    assert "ANYTOMD_OCR_MODEL_DIR unset" in gaps[0]
    monkeypatch.setenv("ANYTOMD_OCR_MODEL_DIR", str(tmp_path / "nothere"))
    assert "nothere" in readiness.run(OcrSpec(backend="oar", size="small"))[0]


async def test_readiness_oar_small_ready_with_the_files(readiness, oar_present, monkeypatch, tmp_path) -> None:
    for name in ("pp-ocrv6_small_det.onnx", "pp-ocrv6_small_rec.onnx", "ppocrv6_dict.txt"):
        (tmp_path / name).touch()
    monkeypatch.setenv("ANYTOMD_OCR_MODEL_DIR", str(tmp_path))
    assert readiness.run(OcrSpec(backend="oar", size="small")) == []


##### TESSERACT AND OCRIA #####


async def test_readiness_tesseract_missing_language(readiness, monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("TESSDATA_PREFIX", raising=False)
    gaps = readiness.run(OcrSpec(backend="tesseract", language="xal+zxx", tessdata_path=str(tmp_path)))
    assert len(gaps) == 1
    assert "xal" in gaps[0]
    assert "zxx" in gaps[0]


async def test_readiness_tesseract_language_on_disk(readiness, monkeypatch, tmp_path) -> None:
    (tmp_path / "spa.traineddata").touch()
    monkeypatch.delenv("TESSDATA_PREFIX", raising=False)
    assert readiness.run(OcrSpec(backend="tesseract", language="spa", tessdata_path=str(tmp_path))) == []


async def test_readiness_ocria_without_liteparse_names_the_extra(readiness, missing_extra) -> None:
    missing_extra("liteparse")
    assert readiness.run(OcrSpec(backend="ocria-nano")) == [
        "the 'ocr' extra is not installed: pip install 'anytomd[ocr]'"
    ]


async def test_readiness_ocria_reports_the_pull(readiness) -> None:
    ### Whatever the machine holds, the probe answers with a gap naming the pull or with readiness. Never a download.
    gaps = readiness.run(OcrSpec(backend="ocria-nano"))
    assert len(gaps) <= 1
    assert not gaps or "models.pull" in gaps[0]


##### AUTO #####


async def test_readiness_auto_prefers_the_installed_oar(readiness, oar_present) -> None:
    assert readiness.run(OcrSpec()) == []


async def test_readiness_auto_falls_to_liteparse(readiness) -> None:
    gaps = readiness.run(OcrSpec())
    assert len(gaps) <= 1  # units run without the engine: auto probes liteparse's default backend


##### THE PROBE PATHS NO MACHINE SETS UP ALONE #####


async def test_readiness_liteparse_via_find_spec_when_never_imported(readiness, monkeypatch) -> None:
    ### The path for a library that has never entered sys.modules: find_spec answers, a None entry would raise there.
    monkeypatch.delitem(sys.modules, "liteparse", raising=False)
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    assert readiness.run(OcrSpec(backend="ocria-nano")) == [
        "the 'ocr' extra is not installed: pip install 'anytomd[ocr]'"
    ]


async def test_readiness_ocria_without_the_models_module(readiness, monkeypatch) -> None:
    ### liteparse installed but its `models` submodule absent (old wheel): the probe says nothing it cannot know.
    monkeypatch.setattr(BackendRegistry, "_common_module", staticmethod(lambda path: None))
    assert readiness.run(OcrSpec(backend="ocria-nano")) == []


async def test_readiness_auto_without_liteparse_names_the_extra(readiness, missing_extra) -> None:
    missing_extra("liteparse")
    assert readiness.run(OcrSpec()) == ["the 'ocr' extra is not installed: pip install 'anytomd[ocr]'"]


##### THE API DOOR #####


async def test_api_ocr_ready_without_a_spec_reads_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("ANYTOMD_OCR_BACKEND", "tesseract")
    monkeypatch.setenv("ANYTOMD_OCR_LANGUAGE", "xal")
    monkeypatch.setenv("TESSDATA_PREFIX", "/nowhere")
    assert ad.ocr_ready() == [
        "tesseract languages ['xal']: convert once with network, or pass tessdata_path / set TESSDATA_PREFIX"
    ]


async def test_api_ocr_ready_with_a_spec_ignores_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("ANYTOMD_OCR_BACKEND", "tesseract")
    assert ad.ocr_ready(OcrSpec(backend="oar")) == ["the Rust OCR engine is not built for this platform (Intel macOS)"]
