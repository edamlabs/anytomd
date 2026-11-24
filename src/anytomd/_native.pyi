"""Type stubs for the native anytomd binding."""

import os
from collections.abc import Awaitable, Callable
from typing import Literal

type WarningCode = Literal[
    "ocr_failed",
    "skipped_element",
    "unsupported_feature",
    "resource_limit_reached",
    "malformed_segment",
]
type DescribeCallable = Callable[[bytes, str, str], str]
type AsyncDescribeCallable = Callable[[bytes, str, str], Awaitable[str]]

class ConversionWarning:
    code: WarningCode
    message: str
    location: str | None

class ConversionResult:
    markdown: str
    plain_text: str
    title: str | None
    images: list[tuple[str, bytes]]
    warnings: list[ConversionWarning]

def convert_file(
    path: str | os.PathLike[str],
    *,
    extract_images: bool = False,
    extract_comments: bool = False,
    strict: bool = False,
    max_input_bytes: int = 8589934592,
    max_total_image_bytes: int = 4294967296,
    max_uncompressed_zip_bytes: int = 17179869184,
    image_describer: DescribeCallable | None = None,
) -> ConversionResult: ...
def convert_bytes(
    data: bytes | bytearray,
    format: str,
    *,
    extract_images: bool = False,
    extract_comments: bool = False,
    strict: bool = False,
    max_input_bytes: int = 8589934592,
    max_total_image_bytes: int = 4294967296,
    max_uncompressed_zip_bytes: int = 17179869184,
    image_describer: DescribeCallable | None = None,
) -> ConversionResult: ...
async def convert_file_async(
    path: str | os.PathLike[str],
    *,
    extract_images: bool = False,
    extract_comments: bool = False,
    strict: bool = False,
    max_input_bytes: int = 8589934592,
    max_total_image_bytes: int = 4294967296,
    max_uncompressed_zip_bytes: int = 17179869184,
    async_image_describer: AsyncDescribeCallable | None = None,
) -> ConversionResult: ...
async def convert_bytes_async(
    data: bytes | bytearray,
    format: str,
    *,
    extract_images: bool = False,
    extract_comments: bool = False,
    strict: bool = False,
    max_input_bytes: int = 8589934592,
    max_total_image_bytes: int = 4294967296,
    max_uncompressed_zip_bytes: int = 17179869184,
    async_image_describer: AsyncDescribeCallable | None = None,
) -> ConversionResult: ...
def detect_format(path: str | os.PathLike[str], data: bytes | bytearray) -> str | None: ...
def detect_zip_format(data: bytes | bytearray) -> str | None: ...

ENGINE_VERSION: str
OCR: bool  # the Rust OCR engine is linked in: every platform but Intel macOS, which has no ONNX Runtime build

class OarEngine:
    """A PP-OCR recognizer: tiny embedded per script, small/medium multilingual from `model_dir`. Absent where `OCR` is False."""

    def __init__(
        self,
        *,
        size: Literal["tiny", "small", "medium"] = "tiny",
        script: str = "latin",
        columns: bool = True,
        orientation: bool = True,
        model_dir: str | None = None,
        threads: int | None = None,
        pool: int = 1,
    ) -> None: ...
    @property
    def size(self) -> str: ...
    @property
    def source(self) -> str: ...
    @property
    def pool(self) -> int: ...
    def recognize(self, image: bytes) -> tuple[str, float | None]: ...
