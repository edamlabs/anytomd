"""The public surface: three verbs, each with its `a`-prefixed async twin."""

from anytomd.config.ocr import OcrSettings
from anytomd.infra.documents import Document
from anytomd.infra.specs import (
    DEFAULT_MAX_IMAGE_BYTES,
    DEFAULT_MAX_INPUT_BYTES,
    DEFAULT_MAX_ZIP_BYTES,
    ConversionSpec,
    OcrSpec,
    VisionSpec,
)
from anytomd.logic.conversion import Conversion
from anytomd.logic.description import Description
from anytomd.logic.detection import Detection
from anytomd.logic.ocrreadiness import OcrReadiness
from anytomd.logic.sources import InputSource, Source


def _ocr_spec(*, ocr: bool | OcrSpec) -> OcrSpec | None:
    """One normalization of the `ocr` option: a spec passes through, `True` reads the environment, `False` is off."""
    match ocr:
        case OcrSpec() as spec:
            return spec
        case True:
            return OcrSettings.from_env()
        case _:
            return None


def convert(
    source: Source,
    *,
    format: str | None = None,
    images: bool = False,
    comments: bool = False,
    strict: bool = False,
    ocr: bool | OcrSpec = True,
    vision: VisionSpec | None = None,
    max_input_bytes: int = DEFAULT_MAX_INPUT_BYTES,
    max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
    max_zip_bytes: int = DEFAULT_MAX_ZIP_BYTES,
) -> Document:
    """Convert a document — a path, or bytes with their `format` — to Markdown."""
    spec = ConversionSpec(
        images=images,
        comments=comments,
        strict=strict,
        ocr=_ocr_spec(ocr=ocr),
        vision=vision,
        max_input_bytes=max_input_bytes,
        max_image_bytes=max_image_bytes,
        max_zip_bytes=max_zip_bytes,
    )
    return Conversion(spec).run(InputSource.build(source, format))


async def aconvert(
    source: Source,
    *,
    format: str | None = None,
    images: bool = False,
    comments: bool = False,
    strict: bool = False,
    ocr: bool | OcrSpec = True,
    vision: VisionSpec | None = None,
    max_input_bytes: int = DEFAULT_MAX_INPUT_BYTES,
    max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
    max_zip_bytes: int = DEFAULT_MAX_ZIP_BYTES,
) -> Document:
    """`convert`, off the event loop; a document's images are resolved concurrently."""
    spec = ConversionSpec(
        images=images,
        comments=comments,
        strict=strict,
        ocr=_ocr_spec(ocr=ocr),
        vision=vision,
        max_input_bytes=max_input_bytes,
        max_image_bytes=max_image_bytes,
        max_zip_bytes=max_zip_bytes,
    )
    return await Conversion(spec).arun(await InputSource.abuild(source, format))


def ocr_ready(spec: OcrSpec | None = None) -> list[str]:
    """What local OCR still needs — languages without traineddata, models not pulled — without touching the network."""
    return OcrReadiness().run(OcrSettings.from_env() if spec is None else spec)


def detect(source: Source, *, name: str | None = None) -> str | None:
    """The format of a path or of bytes (`name` helps for bytes), or None."""
    return Detection().run(source, name)


async def adetect(source: Source, *, name: str | None = None) -> str | None:
    """`detect`, off the event loop."""
    return await Detection().arun(source, name)


def describe(source: Source, *, format: str | None = None, vision: VisionSpec | None = None) -> Document:
    """Describe an image with the vision model; needs the `vlm` extra and a model."""
    return Description(vision).run(InputSource.build(source, format))


async def adescribe(source: Source, *, format: str | None = None, vision: VisionSpec | None = None) -> Document:
    """`describe`, off the event loop."""
    return await Description(vision).arun(await InputSource.abuild(source, format))
