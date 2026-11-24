"""OCR battery: real engines (P-10…P-19) and packaging proofs (P-21, P-22). Run with `make ocr`.

Every test here loads real engines; none touches the network on purpose: oar models are embedded or on disk,
and the one offline proof (P-15) runs inside a network-isolated `unshare` namespace when the machine allows it.
"""

import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from random import Random

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from PIL import Image

import anytomd as ad
from anytomd.backends.base import Recognizer
from anytomd.backends.registry import BackendRegistry
from anytomd.infra.specs import OcrSpec

pytestmark = [pytest.mark.ocr, pytest.mark.timeout(300)]

_MODELS = Path("crates/oar/models")
_PANIC = "PanicException"


def _oar_installed() -> bool:
    return importlib.util.find_spec("anytomd_oar") is not None


oar_required = pytest.mark.skipif(not _oar_installed(), reason="the anytomd-oar wheel is not installed")


def _engine(spec: OcrSpec) -> Recognizer:
    """The recognizer a spec asks for: an absent engine is a broken environment, so it fails instead of narrowing."""
    recognizer = BackendRegistry().recognizer(spec)
    assert recognizer is not None
    return recognizer


@pytest.fixture(scope="module")
def tesseract() -> Recognizer:
    return _engine(OcrSpec(backend="tesseract", language="eng"))


@pytest.fixture(scope="module")
def oar_engine() -> Recognizer:
    return _engine(OcrSpec(backend="oar", threads=1))


@pytest.fixture(scope="module")
def c2(path_of) -> bytes:
    return path_of("c2_sheet.jpg").read_bytes()


@pytest.fixture(scope="module")
def c3(path_of) -> bytes:
    return path_of("c3_screenshot.png").read_bytes()


@pytest.fixture(scope="module")
def bomb() -> bytes:
    """20000x20000 grayscale JPEG: a 400 Mpx decompression bomb, small on disk."""
    buffer = io.BytesIO()
    Image.new("L", (20000, 20000), 235).save(buffer, format="JPEG", quality=40)
    return buffer.getvalue()


@pytest.fixture(scope="module")
def malformed(path_of, tmp_path_factory) -> list[bytes]:
    scratch = tmp_path_factory.mktemp("malformed")
    png_1x1, cmyk, gif, sixteen = scratch / "one.png", scratch / "cmyk.jpg", scratch / "anim.gif", scratch / "i16.png"
    Image.new("RGB", (1, 1)).save(png_1x1)
    Image.new("CMYK", (600, 400), (0, 60, 100, 10)).save(cmyk)
    frames = [Image.new("P", (200, 200), i) for i in range(2)]
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=10, loop=0)
    Image.frombytes("I;16", (600, 400), bytes(480_000)).save(sixteen)
    return [
        b"",
        path_of("c2_sheet.jpg").read_bytes()[:2048],
        png_1x1.read_bytes(),
        cmyk.read_bytes(),
        sixteen.read_bytes(),
        gif.read_bytes(),
    ]


def _survive(recognizer: Recognizer, payload: bytes) -> str:
    """The text or the name of the typed error; a Rust panic or a foreign BaseException escapes and fails the run."""
    try:
        outcome = recognizer.recognize(payload)
    except BaseException as error:
        if type(error).__name__ == _PANIC or not isinstance(error, Exception):
            msg = f"{type(error).__name__} escaped the engine"
            raise AssertionError(msg) from error
        return f"raised {type(error).__name__}"
    return outcome.text


##### P-10 · MODELS LOADED FROM DISK, NEVER DOWNLOADED #####


@oar_required
def test_ocr_p10_oar_loads_small_models_from_disk(c2, monkeypatch) -> None:
    if not (_MODELS / "pp-ocrv6_small_det.onnx").is_file():
        pytest.skip("the PP-OCRv6 small models are not on disk")
    monkeypatch.setenv("ANYTOMD_OCR_MODEL_DIR", str(_MODELS))
    BackendRegistry._common_oar.cache_clear()
    recognition = _engine(OcrSpec(backend="oar", size="small")).recognize(c2)
    BackendRegistry._common_oar.cache_clear()
    assert recognition.warning is None
    assert recognition.text.strip()


##### P-11 · READING ORDER: LABELS AND VALUES SHARE THE ROW, TILT INCLUDED #####


@oar_required
@pytest.mark.parametrize("fixture", ["rows_fixture.png", "rows_fixture_tilted.png"])
def test_ocr_p11_reading_order(path_of, fixture) -> None:
    recognizer = _engine(OcrSpec(backend="oar", columns=True))
    text = recognizer.recognize(path_of(fixture).read_bytes()).text
    assert text.splitlines() == ["Nombre Ana", "Peso 70", "Ano 2024"]


##### P-12 · ONE CACHED ENGINE, MANY THREADS #####


@pytest.mark.parametrize("engine", ["tesseract", "oar_engine"])
def test_ocr_p12_engine_is_thread_safe(engine, request, c2) -> None:
    recognizer = request.getfixturevalue(engine)
    serial = recognizer.recognize(c2)
    with ThreadPoolExecutor(max_workers=32) as pool:
        outcomes = list(pool.map(recognizer.recognize, [c2] * 32))
    assert all(outcome.text == serial.text and outcome.warning == serial.warning for outcome in outcomes)
    assert serial.text.strip()


##### P-13 · THE GIL IS RELEASED #####


@oar_required
def test_ocr_p13_four_threads_beat_serial(oar_engine, c2) -> None:
    rounds = [c2] * 8
    t0 = time.perf_counter()
    for image in rounds:
        oar_engine.recognize(image)
    serial = time.perf_counter() - t0
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(oar_engine.recognize, rounds))
    parallel = time.perf_counter() - t0
    ### Released GIL: ~1x if held. The latin recognizer saturates memory bandwidth at 2.2-2.4x (2026-11-24).
    assert serial / parallel >= 2.0, f"serial {serial:.2f}s vs four threads {parallel:.2f}s"


##### P-15 · NO NETWORK, NO SILENCE #####


def _unshare() -> str | None:
    return shutil.which("unshare")


def _isolated() -> list[str]:
    if (tool := _unshare()) is None:
        return []
    probe = subprocess.run([tool, "-rn", "true"], capture_output=True, check=False)  # noqa: S603
    return [tool, "-rn"] if probe.returncode == 0 else []


def test_ocr_p15_offline(path_of, tmp_path) -> None:
    ### netns where the machine grants it, a refused proxy everywhere else: either way the HTTP stack of the child cannot reach GitHub.
    script = f"""
import anytomd as ad
from anytomd.infra.specs import OcrSpec

blank = ad.convert({str(path_of("c2_sheet.jpg"))!r}, ocr=OcrSpec(backend="tesseract", language="ces", tessdata_path={str(tmp_path)!r}))
assert any(w.code == "ocr_failed" for w in blank.warnings), blank.warnings
rust = ad.convert({str(path_of("c2_sheet.jpg"))!r}, ocr=OcrSpec(backend="oar"))
assert rust.text.strip() and not rust.warnings, (len(rust.text), rust.warnings)
print("P-15 OK")
"""
    env = {
        **os.environ,
        "HTTP_PROXY": "http://127.0.0.1:1",
        "HTTPS_PROXY": "http://127.0.0.1:1",
        "ALL_PROXY": "http://127.0.0.1:1",
    }
    result = subprocess.run(  # noqa: S603
        [*_isolated(), sys.executable, "-c", script], capture_output=True, text=True, check=False, env=env
    )
    assert "P-15 OK" in result.stdout, result.stderr[-2000:]


##### P-14 · FIFTY SCANS THROUGH THE FALLBACK #####


def test_ocr_p14_album50(path_of) -> None:
    truth = json.loads(path_of("c8_album50.truth.json").read_text())
    t0 = time.perf_counter()
    document = ad.convert(path_of("c8_album50.docx"), format="docx", images=True)
    elapsed = time.perf_counter() - t0
    assert document.markdown.count("![") == 50
    upper = document.markdown.upper().replace(" ", "")
    hits = sum(1 for value in truth["fields"].values() if value.upper().replace(" ", "") in upper)
    assert elapsed < 120
    assert hits >= len(truth["fields"]) - 4, f"only {hits}/{len(truth['fields'])} fields on 50 scans"


##### P-17 · LETTERLESS IMAGES STAY EMPTY #####


@pytest.mark.parametrize("engine", ["tesseract", "oar_engine"])
@pytest.mark.parametrize("fixture", ["shot.png", "vector.svg", "scene.heic"])
def test_ocr_p17_letterless(engine, request, fixture, path_of) -> None:
    assert request.getfixturevalue(engine).recognize(path_of(fixture).read_bytes()).text == ""


##### P-18 · MALFORMED INPUTS NEVER PANIC #####


@pytest.mark.parametrize("engine", ["tesseract", "oar_engine"])
def test_ocr_p18_malformed_inputs(engine, request, malformed) -> None:
    recognizer = request.getfixturevalue(engine)
    for payload in malformed:
        _survive(recognizer, payload)


def _peak_rss_kb(backend: str, payloads: list[Path]) -> int:
    """The peak RSS of one clean process fed the whole nasty set: what the 1.5 GB gate really means.

    VmHWM from /proc/self/status, not ru_maxrss: Linux does not reset the latter on execve, so a
    child forked from a fat parent would report the parent's high-water mark instead of its own.
    """
    paths = repr([str(path) for path in payloads])
    script = f"""
from pathlib import Path
from anytomd.backends.registry import BackendRegistry
from anytomd.infra.specs import OcrSpec

recognizer = BackendRegistry().recognizer(OcrSpec(backend={backend!r}, language="eng"))
assert recognizer is not None, "backend did not build"
outcomes = []
for path in {paths}:
    try:
        recognition = recognizer.recognize(Path(path).read_bytes())
        outcome = "warn" if recognition.warning else ("text" if recognition.text else "blank")
    except BaseException as error:
        assert type(error).__name__ != "PanicException", error
        assert isinstance(error, Exception), error
        outcome = "typed"
    outcomes.append((Path(path).name, outcome))
for name, outcome in outcomes:
    print(name, outcome)
peak = next(line.split()[1] for line in open("/proc/self/status") if line.startswith("VmHWM"))
print("FINAL", peak)
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)  # noqa: S603
    assert result.returncode == 0, result.stderr[-3000:]
    return int(result.stdout.rsplit("FINAL", 1)[1])


@pytest.mark.parametrize("backend", ["tesseract", "oar"])
def test_ocr_p18_clean_process_rss_under_1_5_gb(backend, malformed, tmp_path, bomb) -> None:
    payloads = [tmp_path / f"payload{index}.bin" for index, payload in enumerate(malformed)]
    for path, payload in zip(payloads, malformed, strict=True):
        path.write_bytes(payload)
    bomb_file = tmp_path / "bomb.jpg"
    bomb_file.write_bytes(bomb)
    payloads.append(bomb_file)
    peak = _peak_rss_kb(backend, payloads)
    assert peak < 1_500_000, f"{backend} peaked at {peak // 1024} MB"


##### P-19 · RANDOM BYTES NEVER PANIC #####


def _mutate(seed: int, c3: bytes) -> bytes:
    rng = Random(seed)  # noqa: S311 - seeded fuzzing, not crypto
    data = bytearray(c3)
    for _ in range(rng.randint(1, 8)):
        data[rng.randrange(len(data))] = rng.randrange(256)
    return bytes(data)


@oar_required
def test_ocr_p19_fuzz(oar_engine, c3) -> None:
    @settings(max_examples=40, deadline=None, suppress_health_check=list(HealthCheck))
    @given(st.one_of(st.binary(max_size=8192), st.integers(0, 2**32 - 1).map(lambda seed: _mutate(seed, c3))))
    def fuzz(payload: bytes) -> None:
        _survive(oar_engine, payload)

    fuzz()


##### P-21 · THE WHEEL LINKS ONNX RUNTIME IN #####


@oar_required
@pytest.mark.skipif(sys.platform != "linux", reason="ldd is Linux-only; the release matrix proves the rest")
def test_ocr_p21_static_onnxruntime_and_size() -> None:
    module_spec = importlib.util.find_spec("anytomd_oar")
    assert module_spec is not None
    assert module_spec.origin is not None
    so = Path(module_spec.origin)
    ldd = shutil.which("ldd")
    assert ldd is not None, "ldd is part of the Linux toolchain"
    listed = subprocess.run([ldd, str(so)], capture_output=True, text=True, check=False)  # noqa: S603
    assert "onnxruntime" not in listed.stdout, listed.stdout
    assert so.stat().st_size <= 40 * 1024**2, f"{so} is {so.stat().st_size / 1024**2:.0f} MB"


##### P-22 · COLD START #####


@oar_required
def test_ocr_p22_cold_start(path_of) -> None:
    plain = """
import sys, time
t0 = time.perf_counter()
import anytomd as ad
from anytomd.infra.specs import OcrSpec

ocr = OcrSpec(backend="oar", threads=1)
document = ad.convert(sys.argv[1], ocr=ocr)
cold = (time.perf_counter() - t0) * 1000
warm = []
for _ in range(3):
    t = time.perf_counter()
    ad.convert(sys.argv[1], ocr=ocr)
    warm.append((time.perf_counter() - t) * 1000)
assert document.text.strip()
print(f"COLD {cold:.0f} WARM {' '.join(f'{w:.0f}' for w in warm)}")
"""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", plain, str(path_of("c2_sheet.jpg"))], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr[-2000:]
    words = result.stdout.split()
    assert words[0] == "COLD"
    cold, warm = int(words[1]), [int(value) for value in words[3:]]
    ### One intra-op thread: the latin recognizer costs ~1.07 s per 420 px sheet (all cores: ~0.22 s), measured 2026-11-24.
    assert cold < 2500, result.stdout
    assert all(value < 1500 for value in warm), result.stdout
