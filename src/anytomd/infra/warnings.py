"""Recoverable problems met while converting."""

from dataclasses import dataclass
from typing import Literal

type WarningCode = Literal[
    "ocr_failed",
    "skipped_element",
    "unsupported_feature",
    "resource_limit_reached",
    "malformed_segment",
]


@dataclass(frozen=True, slots=True)
class DocumentWarning:
    """One recoverable problem: what kind, what happened, and where when known."""

    code: WarningCode
    message: str
    location: str | None = None
