"""BackendRegistry: optional backends resolve when installed, vanish cleanly when not, one engine per spec."""

import sys
import types

import pytest

from anytomd.backends.registry import BackendRegistry
from anytomd.infra.errors import DescriptionError, MissingExtraError
from anytomd.infra.specs import OcrSpec, VisionSpec


@pytest.fixture
def registry() -> BackendRegistry:
    return BackendRegistry()


async def test_registry_recognizer_installed(registry) -> None:
    assert registry.recognizer(OcrSpec()) is not None


async def test_registry_recognizer_off(registry) -> None:
    assert registry.recognizer(None) is None


async def test_registry_recognizer_missing(registry, missing_extra) -> None:
    missing_extra("liteparse")
    assert registry.recognizer(OcrSpec()) is None


async def test_registry_reader_missing(registry, missing_extra) -> None:
    missing_extra("liteparse")
    with pytest.raises(MissingExtraError, match=r"PDF needs the 'pdf' extra"):
        registry.reader(OcrSpec())


async def test_registry_describer_from_spec(registry) -> None:
    assert registry.describer(VisionSpec(model="m", api_key="k")).concurrency == 4


async def test_registry_describer_from_environment(registry, monkeypatch) -> None:
    monkeypatch.setenv("ANYTOMD_VISION_MODEL", "env-model")
    monkeypatch.setenv("ANYTOMD_VISION_API_KEY", "k")
    monkeypatch.setenv("ANYTOMD_VISION_CONCURRENCY", "2")
    assert registry.describer(None).concurrency == 2


async def test_registry_describer_unconfigured(registry, monkeypatch) -> None:
    monkeypatch.delenv("ANYTOMD_VISION_MODEL", raising=False)
    assert registry.describer(None) is None


async def test_registry_describer_missing(registry, missing_extra) -> None:
    missing_extra("openai")
    assert registry.describer(VisionSpec(model="m")) is None


async def test_registry_require_unconfigured(registry, monkeypatch) -> None:
    monkeypatch.delenv("ANYTOMD_VISION_MODEL", raising=False)
    with pytest.raises(DescriptionError, match="ANYTOMD_VISION_MODEL"):
        registry.require(None)


async def test_registry_require_missing(registry, missing_extra) -> None:
    missing_extra("openai")
    with pytest.raises(MissingExtraError, match="vlm"):
        registry.require(None)


async def test_registry_settings_missing_falls_back_to_the_spec(registry, missing_extra) -> None:
    missing_extra("pydantic_settings")
    assert registry.describer(None) is None


async def test_registry_unrelated_import_errors_propagate(registry, monkeypatch) -> None:
    def broken(path: str):
        msg = "No module named 'typo'"
        raise ModuleNotFoundError(msg, name="typo")

    monkeypatch.setattr("importlib.import_module", broken)
    with pytest.raises(ModuleNotFoundError, match="typo"):
        registry.recognizer(OcrSpec())


##### P-04 · BACKEND CHOICE #####


@pytest.fixture
def oar_installed(monkeypatch) -> object:
    """A stand-in oar backend module, as if the extra were installed."""
    module = types.ModuleType("anytomd.backends.oar")

    class OarBackend:
        def __init__(self, spec: OcrSpec, model_dir: str | None = None) -> None:
            self.spec, self.model_dir = spec, model_dir

    module.__dict__["OarBackend"] = OarBackend
    monkeypatch.setitem(sys.modules, "anytomd.backends.oar", module)
    return OarBackend


async def test_registry_auto_prefers_oar_when_installed(registry, oar_installed) -> None:
    assert isinstance(registry.recognizer(OcrSpec()), oar_installed)


async def test_registry_auto_skips_oar_for_a_script_it_cannot_read(registry, oar_installed) -> None:
    assert type(registry.recognizer(OcrSpec(language="chi_sim"))).__name__ == "LiteparseBackend"


async def test_registry_oar_takes_the_model_dir_from_env(registry, oar_installed, monkeypatch) -> None:
    monkeypatch.setenv("ANYTOMD_OCR_MODEL_DIR", "/models")
    assert registry.recognizer(OcrSpec(backend="oar")).model_dir == "/models"


async def test_registry_auto_falls_back_to_liteparse(registry, missing_extra) -> None:
    missing_extra("anytomd_oar")
    assert type(registry.recognizer(OcrSpec())).__name__ == "LiteparseBackend"


async def test_registry_oar_without_the_extra_names_it(registry, missing_extra) -> None:
    missing_extra("anytomd_oar")
    with pytest.raises(MissingExtraError, match=r"anytomd\[oar\]"):
        registry.recognizer(OcrSpec(backend="oar"))


async def test_registry_named_backends_take_liteparse(registry) -> None:
    for backend in ("tesseract", "ocria-nano"):
        assert type(registry.recognizer(OcrSpec(backend=backend))).__name__ == "LiteparseBackend"


##### P-05 · ONE ENGINE PER SPEC #####


async def test_registry_engine_cached_per_spec(registry) -> None:
    spec = OcrSpec(language="spa")
    assert registry.recognizer(spec) is registry.recognizer(OcrSpec(language="spa"))
    assert registry.recognizer(spec) is not registry.recognizer(OcrSpec(language="fra"))
    assert registry.reader(spec) is registry.recognizer(spec)  # one instance, both ports


async def test_registry_engine_cache_capped(registry) -> None:
    for language in ("aa", "ab", "ac", "ad", "ae", "af", "ag", "ah", "ai"):
        registry.recognizer(OcrSpec(language=language))
    assert BackendRegistry._common_liteparse.cache_info().currsize == 8


async def test_registry_module_simulating_uninstall(registry, monkeypatch) -> None:
    ### A None entry makes the import raise with the path as name: the wheel is simply not here, not an import bug.
    monkeypatch.setitem(sys.modules, "anytomd_oar", types.ModuleType("anytomd_oar"))
    monkeypatch.setitem(sys.modules, "anytomd_oar._oar", None)
    assert registry._common_module("anytomd_oar._oar") is None
