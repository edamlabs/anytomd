"""Local OCR settings from the environment (`ANYTOMD_OCR_*`); stdlib only — the `ocr` extra must not need pydantic."""

import dataclasses
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
ENV_TESSDATA_PREFIX: Final = "TESSDATA_PREFIX"

_TRUTHY: Final = frozenset({"1", "true", "yes", "on"})


class OcrSettings:
    """`ANYTOMD_OCR_*` refines each default of the spec; an unset variable keeps the spec's own value."""

    __slots__ = ()

    ##### PRIVATE #####

    @staticmethod
    def _from_env_int(environ: Mapping[str, str], name: str) -> int | None:
        ### A bad number names its variable: int()'s own message does not say where the text came from.
        value = environ.get(name)
        try:
            return int(value) if value else None
        except ValueError as error:
            msg = f"{name} must be an integer, got {value!r}"
            raise ValueError(msg) from error

    @staticmethod
    def _from_env_flag(environ: Mapping[str, str], name: str, *, default: bool) -> bool:
        ### Unset or empty keeps the spec's default, so a default of True can still be turned off with "0" or "false".
        return value.lower() in _TRUTHY if (value := environ.get(name)) else default

    ############################################################

    ##### PUBLIC #####

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> OcrSpec:
        """The spec the environment asks for; invalid values raise `ValueError`, as `OcrSpec` does."""
        environ = os.environ if environ is None else environ
        default = OcrSpec()
        return dataclasses.replace(
            default,
            language=environ.get(ENV_LANGUAGE) or default.language,
            dpi=default.dpi if (dpi := cls._from_env_int(environ, ENV_DPI)) is None else dpi,
            backend=as_backend_name(environ.get(ENV_BACKEND) or default.backend),
            tessdata_path=environ.get(ENV_TESSDATA) or default.tessdata_path,
            skip_diagonal_text=cls._from_env_flag(environ, ENV_SKIP_DIAGONAL, default=default.skip_diagonal_text),
            size=as_model_size(environ.get(ENV_SIZE) or default.size),
            columns=cls._from_env_flag(environ, ENV_COLUMNS, default=default.columns),
            threads=cls._from_env_int(environ, ENV_THREADS),
        )

    @classmethod
    def model_dir(cls, environ: Mapping[str, str] | None = None) -> str | None:
        """`ANYTOMD_OCR_MODEL_DIR`: the directory holding the oar models; None means the embedded ones."""
        environ = os.environ if environ is None else environ
        return environ.get(ENV_MODEL_DIR) or None

    @classmethod
    def tessdata_dirs(cls, environ: Mapping[str, str] | None = None) -> list[str]:
        """`TESSDATA_PREFIX`, split on the path separator: where Tesseract looks for its language files."""
        environ = os.environ if environ is None else environ
        return [value for value in environ.get(ENV_TESSDATA_PREFIX, "").split(os.pathsep) if value]
