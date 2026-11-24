"""OarBackend: the spec reaches the native engine, a read returns text, a failure warns instead of passing for blank."""

import importlib
import types

import pytest

from anytomd import _native
from anytomd.infra.specs import OcrSpec


@pytest.fixture
def oar(monkeypatch) -> types.ModuleType:
    """The real backend module over a scripted native engine: no ONNX session in a unit."""

    class OarEngine:
        def __init__(
            self, *, size: str, script: str, columns: bool, model_dir: str | None, threads: int | None, pool: int
        ) -> None:
            self.args = (size, script, columns, model_dir, threads, pool)
            self.text, self.confidence, self.error = "Rust lee", 0.91, None

        def recognize(self, image: bytes) -> tuple[str, float | None]:
            if self.error is not None:
                raise self.error
            return self.text, self.confidence

    monkeypatch.setattr(_native, "OarEngine", OarEngine, raising=False)
    return importlib.import_module("anytomd.backends.oar")


@pytest.fixture
def backend(oar):
    return oar.OarBackend(OcrSpec(backend="oar", size="small", columns=True), model_dir="/models")


async def test_oar_spec_reaches_the_engine(oar) -> None:
    engine = oar.OarBackend(OcrSpec(backend="oar", size="small", columns=True, threads=2), model_dir="/models")._engine
    assert engine.args == ("small", "latin", True, "/models", 2, 1)


async def test_oar_tiny_engines_are_pooled(oar) -> None:
    engine = oar.OarBackend(OcrSpec(backend="oar"))._engine
    assert engine.args == ("tiny", "latin", True, None, None, 4)


async def test_oar_recognize_text(backend) -> None:
    recognition = backend.recognize(b"scan")
    assert (recognition.text, recognition.warning) == ("Rust lee", None)


async def test_oar_recognize_silence_is_a_read(backend) -> None:
    backend._engine.text, backend._engine.confidence = "", None
    recognition = backend.recognize(b"blank")
    assert (recognition.text, recognition.warning) == ("", None)


async def test_oar_recognize_failure_warns(backend) -> None:
    backend._engine.error = RuntimeError("model file corrupt")
    recognition = backend.recognize(b"scan")
    assert (recognition.text, recognition.warning.code, recognition.warning.location) == ("", "ocr_failed", "image")
    assert "corrupt" in recognition.warning.message


async def test_oar_arecognize(backend) -> None:
    assert (await backend.arecognize(b"scan")).text == "Rust lee"


async def test_oar_arecognize_failure_warns(backend) -> None:
    backend._engine.error = ValueError("image could not be decoded")
    assert (await backend.arecognize(b"junk")).warning.code == "ocr_failed"


async def test_oar_script_follows_the_first_language(oar) -> None:
    engine = oar.OarBackend(OcrSpec(backend="oar", language="rus+eng"))._engine
    assert engine.args[1] == "cyrillic"


async def test_oar_small_reads_a_language_without_embedded_script(oar) -> None:
    engine = oar.OarBackend(OcrSpec(backend="oar", language="chi_sim", size="small"), model_dir="/models")._engine
    assert engine.args[:2] == ("small", "latin")


async def test_oar_tiny_without_script_names_the_way_out(oar) -> None:
    with pytest.raises(ValueError, match=r"no embedded oar recognizer reads 'chi_sim'.*size 'small'"):
        oar.OarBackend(OcrSpec(backend="oar", language="chi_sim"))
