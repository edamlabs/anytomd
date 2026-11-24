"""Local OCR settings from the environment (`ANYTOMD_OCR_*`); stdlib only — the `ocr` extra must not need pydantic."""

import os
from collections.abc import Mapping
from typing import Final

from anytomd.infra.specs import OcrSpec, as_backend_name, as_model_size

ENV_LANGUAGE: Final = "ANYTOMD_OCR_LANGUAGE"
ENV_DPI: Final = "ANYTOMD_OCR_DPI"
ENV_BACKEND: Final = "ANYTOMD_OCR_BACKEND"
ENV_TESSDATA: Final = "ANYTOMD_OCR_TESSDATA"
ENV_SKIP_DIAGONAL: Final = "ANYTOMD_OCR_SKIP_DIAGONAL"
ENV_SIZE: Final = "ANYTOMD_OCR_SIZE"
ENV_COLUMNS: Final = "ANYTOMD_OCR_COLUMNS"
ENV_THREADS: Final = "ANYTOMD_OCR_THREADS"
ENV_MODEL_DIR: Final = "ANYTOMD_OCR_MODEL_DIR"

_TRUTHY: Final = frozenset({"1", "true", "yes", "on"})


class OcrSettings:
    """`ANYTOMD_OCR_*` refines each default of the spec; an unset variable keeps the spec's own value."""

    __slots__ = ()

    ##### PUBLIC #####

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> OcrSpec:
        """The spec the environment asks for; invalid values raise `ValueError`, as `OcrSpec` does."""
        environ = os.environ if environ is None else environ
        return OcrSpec(
            language=environ.get(ENV_LANGUAGE) or "eng",
            dpi=int(environ.get(ENV_DPI) or 300),
            backend=as_backend_name(environ.get(ENV_BACKEND) or "auto"),
            tessdata_path=environ.get(ENV_TESSDATA) or None,
            skip_diagonal_text=environ.get(ENV_SKIP_DIAGONAL, "").lower() in _TRUTHY,
            size=as_model_size(environ.get(ENV_SIZE) or "tiny"),
            columns=environ.get(ENV_COLUMNS, "").lower() in _TRUTHY,
            threads=int(threads) if (threads := environ.get(ENV_THREADS)) else None,
        )

    @classmethod
    def model_dir(cls, environ: Mapping[str, str] | None = None) -> str | None:
        """`ANYTOMD_OCR_MODEL_DIR`: the directory holding the oar models; None means the embedded ones."""
        environ = os.environ if environ is None else environ
        return environ.get(ENV_MODEL_DIR) or None
