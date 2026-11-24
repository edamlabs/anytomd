"""Type stubs of the native `_oar` module: the compiled surface anytomd[oar] provides."""

from typing import Literal

class OarEngine:
    """A PP-OCR recognizer: tiny embedded per script, small/medium multilingual from `model_dir`."""

    def __init__(
        self,
        *,
        size: Literal["tiny", "small", "medium"] = "tiny",
        script: str = "latin",
        columns: bool = False,
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
