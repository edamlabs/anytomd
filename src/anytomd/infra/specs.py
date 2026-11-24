"""Explicit configuration values: what the caller asks a conversion for."""

import re
from dataclasses import dataclass, field
from enum import StrEnum, auto
from types import MappingProxyType
from typing import Final, Literal

DEFAULT_MAX_INPUT_BYTES: Final = 8 * 1024**3
DEFAULT_MAX_IMAGE_BYTES: Final = 4 * 1024**3
DEFAULT_MAX_ZIP_BYTES: Final = 16 * 1024**3
DEFAULT_CONCURRENCY: Final = 4

type OcrBackendName = Literal["auto", "tesseract", "ocria-nano", "oar"]
type OcrModelSize = Literal["tiny", "small", "medium"]

OCR_BACKENDS: Final = frozenset({"auto", "tesseract", "ocria-nano", "oar"})
OCR_SIZES: Final = frozenset({"tiny", "small", "medium"})
OCR_DPI_MIN: Final = 72
OCR_DPI_MAX: Final = 600
OCR_THREADS_MIN: Final = 1
OCR_THREADS_MAX: Final = 64


class OcrScript(StrEnum):
    """The alphabets the embedded oar recognizers read: one PP-OCRv5 mobile model each."""

    LATIN = auto()
    CYRILLIC = auto()


### Tesseract codes whose alphabet an embedded recognizer covers, checked against each model's dictionary.
OCR_SCRIPTS: Final = MappingProxyType(
    dict.fromkeys(
        (
            *("afr", "cat", "ces", "cym", "dan", "deu", "eng", "est", "eus", "fin", "fra", "gle", "glg", "hrv"),
            *("hun", "ind", "isl", "ita", "lat", "lav", "lit", "msa", "nld", "nor", "pol", "por", "ron", "slk"),
            *("slv", "spa", "sqi", "swe", "tur"),
        ),
        OcrScript.LATIN,
    )
    | dict.fromkeys(("bel", "bul", "kaz", "mkd", "rus", "srp", "ukr"), OcrScript.CYRILLIC)
)


@dataclass(frozen=True, slots=True)
class VisionSpec:
    """An OpenAI-compatible vision model: OpenAI, Anthropic, Gemini, local servers — `base_url` picks the provider."""

    model: str
    base_url: str | None = None
    api_key: str | None = field(default=None, repr=False)
    prompt: str = "Describe this image concisely for use as alt text."
    max_tokens: int = 256
    concurrency: int = DEFAULT_CONCURRENCY


@dataclass(frozen=True, slots=True)
class OcrSpec:
    """Local OCR settings. `language` uses Tesseract codes; '+' combines them: "spa+eng"."""

    language: str = "eng"
    dpi: int = 300  # 150 is liteparse's Tesseract default; 300 measured +11 fields on a poor scan
    backend: OcrBackendName = "auto"
    tessdata_path: str | None = None
    skip_diagonal_text: bool = False  # drops rotated stamps and watermarks
    size: OcrModelSize = (
        "tiny"  # the oar engine's model set: tiny embedded per script, small/medium multilingual on disk
    )
    columns: bool = False  # the oar engine groups lines into text columns; the other engines ignore it
    threads: int | None = (
        None  # the oar engine's ONNX Runtime intra-op threads; None lets ort choose; the other engines ignore it
    )

    @property
    def script(self) -> OcrScript | None:
        """The alphabet of the first language, the one the embedded oar recognizer reads; None when no embedded model does."""
        return OCR_SCRIPTS.get(self.language.partition("+")[0])

    @property
    def oar_reads(self) -> bool:
        """Whether the oar engine reads this spec: tiny needs an embedded script, small and medium read every one."""
        return self.script is not None or self.size != "tiny"

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z_]+(\+[a-z_]+)*", self.language):
            msg = f"invalid OCR language: {self.language!r}"
            raise ValueError(msg)
        if not OCR_DPI_MIN <= self.dpi <= OCR_DPI_MAX:
            msg = f"dpi out of range {OCR_DPI_MIN}..{OCR_DPI_MAX}: {self.dpi}"
            raise ValueError(msg)
        if self.backend not in OCR_BACKENDS:
            msg = f"invalid OCR backend: {self.backend!r}, one of {sorted(OCR_BACKENDS)}"
            raise ValueError(msg)
        if self.size not in OCR_SIZES:
            msg = f"invalid OCR model size: {self.size!r}, one of {sorted(OCR_SIZES)}"
            raise ValueError(msg)
        if self.threads is not None and not OCR_THREADS_MIN <= self.threads <= OCR_THREADS_MAX:
            msg = f"OCR threads out of range {OCR_THREADS_MIN}..{OCR_THREADS_MAX}: {self.threads}"
            raise ValueError(msg)


##### ENVIRONMENT NARROWING #####
### Unvalidated strings enter only through these: the patterns mirror the Literal banks above, and the parity test holds them equal.


def as_backend_name(value: str) -> OcrBackendName:
    """`value` checked against the backend bank, narrowed to its type; invalid raises as `OcrSpec` does."""
    match value:
        case "auto" | "tesseract" | "ocria-nano" | "oar":
            return value
        case _:
            msg = f"invalid OCR backend: {value!r}, one of {sorted(OCR_BACKENDS)}"
            raise ValueError(msg)


def as_model_size(value: str) -> OcrModelSize:
    """`value` checked against the model-size bank, narrowed to its type; invalid raises as `OcrSpec` does."""
    match value:
        case "tiny" | "small" | "medium":
            return value
        case _:
            msg = f"invalid OCR model size: {value!r}, one of {sorted(OCR_SIZES)}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ConversionSpec:
    """Every option of one conversion, as given to `convert` / `aconvert`."""

    images: bool = False
    comments: bool = False
    strict: bool = False
    ocr: OcrSpec | None = OcrSpec()
    vision: VisionSpec | None = None
    max_input_bytes: int = DEFAULT_MAX_INPUT_BYTES
    max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES
    max_zip_bytes: int = DEFAULT_MAX_ZIP_BYTES
