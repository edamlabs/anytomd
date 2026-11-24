"""Ports of the optional backends: each act has a sync and an async (`a`-prefixed) form."""

from types import TracebackType
from typing import NamedTuple, Protocol, Self

from anytomd.infra.documents import Document
from anytomd.infra.warnings import DocumentWarning


class Recognition(NamedTuple):
    """The text an image yielded, plus the recoverable problem met reading it, if any."""

    text: str
    warning: DocumentWarning | None = None


class Recognizer(Protocol):
    """Reads the letters an image holds; empty text when it holds none, a warning when reading failed."""

    def recognize(self, image: bytes) -> Recognition: ...

    async def arecognize(self, image: bytes) -> Recognition: ...


class Describer(Protocol):
    """Describes an image in words with a vision model; scoped by `with` / `async with`, which own its clients."""

    @property
    def concurrency(self) -> int: ...

    def __enter__(self) -> Self: ...

    def __exit__(self, *_: type[BaseException] | BaseException | TracebackType | None) -> None: ...

    async def __aenter__(self) -> Self: ...

    async def __aexit__(self, *_: type[BaseException] | BaseException | TracebackType | None) -> None: ...

    def describe(self, image: bytes, mime_type: str) -> str: ...

    async def adescribe(self, image: bytes, mime_type: str) -> str: ...


class Reader(Protocol):
    """Reads a whole document the engine does not convert (PDF)."""

    def read(self, source: str | bytes) -> Document: ...

    async def aread(self, source: str | bytes) -> Document: ...
