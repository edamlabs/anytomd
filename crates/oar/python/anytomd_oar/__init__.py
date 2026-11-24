"""The optional Rust OCR engine for anytomd: PP-OCR through oar-ocr, ONNX Runtime."""

from importlib.metadata import version

from anytomd_oar._oar import OarEngine

__all__ = ["OarEngine"]
__version__ = version("anytomd-oar")
