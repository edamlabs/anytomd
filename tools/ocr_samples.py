"""The OCR corpus: synthetic fixtures C2-C10 with truth files, generated into `tests/resources/ocr/`.

C1 is a real Citroen sheet (417x600): it carries a real VIN, so it stays local at
`tests/resources/media/car_sheet.png` with its truth beside it, never committed. Everything the corpus
shares with C1 is the field list — the 28 keys whose literal presence scores the engines.
"""

import io
import json
import sys
import textwrap
import zipfile
from pathlib import Path
from typing import Final

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT: Final = Path(__file__).resolve().parent.parent / "tests" / "resources" / "ocr"

_FONT_CANDIDATES: Final = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
)
_BOLD_CANDIDATES: Final = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
)

GREEN: Final = (211, 227, 207)
BAND: Final = (176, 208, 186)
INK: Final = (30, 30, 34)
STAMP_RED: Final = (165, 40, 40)
BLOCK: Final = (40, 52, 44)

### The 28 fields of the Citroen sheet: the same list scores C1 (real) and C2 (synthetic twin).
SHEET_FIELDS: Final = {
    "numero_serie": "36454879B",
    "identificacion": "VFT7YC8BHC1103048B",
    "certificado": "1103048B",
    "clasificacion": "CIT32709",
    "carroceria": "FURGON",
    "marca": "CITROEN",
    "tipo": "Y8B***",
    "variante": "VCYBMFC/BX",
    "denominacion": "FG35L3H210",
    "ano_tres": "1975",
    "mtma": "3500",
    "mtma_pe_1": "1850",
    "mtma_pe_2": "2000",
    "mmh_sif": "750/2500",
    "neumaticos": "215/70R15C",
    "carga_1": "109/107E",
    "carga_2": "112/110E",
    "altura": "2522",
    "anchura": "2050",
    "via": "1810/1790",
    "longitud": "5998",
    "voladizo": "1015",
    "entreejes": "4035",
    "motor": "D-4HU",
    "cilindrada": "4/2199",
    "potencia": "14.09/88",
    "contrasena": "C-2241",
    "sello": "2007",
}

### label, value key, left-half rows first then right-half rows, as the two columns of the form read.
SHEET_ROWS: Final = (
    ("Número de identificación:", "identificacion", True),
    ("Clasificación del vehículo:", "clasificacion", True),
    ("Carrocería: 2400", "carroceria", True),
    ("Marca:", "marca", True),
    ("Tipo:", "tipo", True),
    ("Variante:", "variante", True),
    ("Tres años:", "ano_tres", True),
    ("MTMA-N1 (Kg):", "mtma", True),
    ("MTMA-N1 P-e (kg):", "mtma_pe_1", True),
    ("MTMA-N2 P-e (kg):", "mtma_pe_2", True),
    ("Altura total (mm):", "altura", False),
    ("Anchura total (mm):", "anchura", False),
    ("Vía ant/post (mm):", "via", False),
    ("Longitud total (mm):", "longitud", False),
    ("Voladizo post.:", "voladizo", False),
    ("Dist. entre ejes:", "entreejes", False),
    ("Motor tipo:", "motor", False),
    ("Nº Cil./Cilindrada:", "cilindrada", False),
    ("Potencia fiscal:", "potencia", False),
    ("Contraseña:", "contrasena", False),
)

SHEET_LINES: Final = (
    "Nº SERIE 36454879B — FICHA TÉCNICA DE AUTOMÓVILES CITROEN ESPAÑA, S.A.",
    "Número de identificación: VFT7YC8BHC1103048B   N° CERTIFICADO 1103048B",
    "Clasificación del vehículo: CIT32709   Carrocería: 2400 FURGON",
    "Marca: CITROEN   Altura total (mm): 2522",
    "Tipo: Y8B***   Anchura total (mm): 2050",
    "Variante: VCYBMFC/BX   Vía anterior/posterior (mm): 1810/1790",
    "Tres años: 1975   Longitud total (mm): 5998",
    "MTMA-N1 (Kg): 3500   Voladizo posterior: 1015",
    "MTMA-N1 P-e (kg): 1850   Dist. entre ejes: 4035",
    "MTMA-N2 P-e (kg): 2000   Motor tipo: D-4HU",
    "Nº Cil./Cilindrada: 4/2199   Potencia fiscal: 14.09/88",
    "Contraseña: C-2241",
    "MMH SIF cf. kg: 750/2500",
    "Neumáticos: 215/70R15C",
    "NEUM. 215/70R15C(109/107E)  NEUM. 225/70R15C(112/110E)",
    "VEHICULO PROCEDENTE DE LA UE",
    "El vehículo de la contraseña C-2241 es completamente conforme con el tipo homologado.",
    "Madrid, a 14 de MARZO de 2007",
)

_LANGUAGES: Final = ("spa", "fra", "deu", "por", "rus")


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = _BOLD_CANDIDATES if bold else _FONT_CANDIDATES
    return ImageFont.truetype(next((path for path in map(Path, candidates) if path.exists()), candidates[-1]), size)


def _sheet(width: int = 420, height: int = 600) -> Image.Image:
    """C2: the synthetic twin of the Citroen sheet — form grid, green bands, rotated stamp."""
    image = Image.new("RGB", (width, height), GREEN)
    draw = ImageDraw.Draw(image)
    label, value = _font(8), _font(8, bold=True)
    draw.rectangle([8, 8, width - 8, 40], outline=INK)
    draw.text((14, 10), "Nº SERIE", font=label, fill=INK)
    draw.text((110, 10), "36454879B", font=value, fill=STAMP_RED)
    draw.text((190, 10), "FICHA TÉCNICA DE AUTOMÓVILES CITROEN", font=label, fill=INK)
    draw.line([8, 42, width - 8, 42], fill=INK)
    left = [row for row in SHEET_ROWS if row[2]]
    right = [row for row in SHEET_ROWS if not row[2]]
    for index, (label_row, value_row) in enumerate(zip(left, right, strict=True)):
        y = 48 + index * 19
        for x, (text, key) in ((12, label_row[:2]), (232, value_row[:2])):
            draw.text((x, y), text, font=label, fill=INK)
            draw.text((x + int(draw.textlength(text, font=label)) + 4, y), SHEET_FIELDS[key], font=value, fill=INK)
    draw.text((12, 248), "MMH SIF cf. kg: 750/2500   Neumáticos: 215/70R15C", font=label, fill=INK)
    draw.rectangle([12, 268, width - 12, 300], outline=INK)
    draw.text((16, 274), "Opciones incluidas en la homologación de tipo:", font=label, fill=INK)
    draw.text((16, 288), "NEUM. 215/70R15C(109/107E)", font=label, fill=INK)
    draw.text((216, 288), "NEUM. 225/70R15C(112/110E)", font=label, fill=INK)
    draw.rectangle([10, 316, width - 10, 344], fill=BAND)
    draw.text((16, 324), "VEHICULO PROCEDENTE DE LA UE", font=value, fill=INK)
    draw.rectangle([10, 356, 150, 420], outline=INK)
    draw.text((14, 360), "Observaciones:", font=label, fill=INK)
    draw.rectangle([14, 380, 120, 400], fill=BLOCK)
    draw.text((12, 432), "El vehículo de la contraseña C-2241 es completamente conforme", font=label, fill=INK)
    draw.text((12, 448), "con el tipo homologado. AUTOMÓVILES CITROEN ESPAÑA, S.A.", font=label, fill=INK)
    draw.text((216, 480), "Madrid, a 14 de MARZO", font=label, fill=INK)
    draw.text((360, 480), "de 2007", font=label, fill=INK)
    stamp = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    stamp_draw = ImageDraw.Draw(stamp)
    stamp_draw.ellipse([width * 0.55, height * 0.62, width * 0.94, height * 0.80], outline=STAMP_RED, width=2)
    stamp_draw.text((width * 0.6, height * 0.68), "ITV 2007", font=_font(13, bold=True), fill=STAMP_RED)
    return Image.alpha_composite(image.convert("RGBA"), stamp.rotate(-20, expand=False)).convert("RGB")


##### PERSPECTIVE #####


def _solve(matrix: list[list[float]], constants: list[float]) -> list[float]:
    """Gauss-Jordan over an 8x8 system: the eight perspective coefficients, exact for the four corners."""
    rows = [[*row, value] for row, value in zip(matrix, constants, strict=True)]
    for column in range(8):
        pivot = max(range(column, 8), key=lambda index: abs(rows[index][column]))
        rows[column], rows[pivot] = rows[pivot], rows[column]
        divisor = rows[column][column]
        rows[column] = [term / divisor for term in rows[column]]
        rows = [
            row if index == column else [term - factor * held for term, held in zip(row, rows[column], strict=True)]
            for index, row in enumerate(rows)
            for factor in [row[column]]
        ]
    return [row[8] for row in rows]


def _perspective(image: Image.Image, corners: tuple[float, ...], size: tuple[int, int]) -> Image.Image:
    """Map the output rectangle onto `corners` of the source: one true four-point projective warp."""
    points = list(zip(corners[::2], corners[1::2], strict=True))
    matrix, constants = [], []
    for (output_x, output_y), (source_x, source_y) in zip(
        ((0, 0), (size[0], 0), (size[0], size[1]), (0, size[1])), points, strict=True
    ):
        matrix.append([output_x, output_y, 1, 0, 0, 0, -output_x * source_x, -output_y * source_x])
        constants.append(source_x)
        matrix.append([0, 0, 0, output_x, output_y, 1, -output_x * source_y, -output_y * source_y])
        constants.append(source_y)
    return image.transform(size, Image.Transform.PERSPECTIVE, _solve(matrix, constants), Image.Resampling.BICUBIC)


def _screenshot() -> Image.Image:
    """C3: the clean English capture — the easy case, for catching regressions, not for scoring quality."""
    image = Image.new("RGB", (640, 360), (252, 252, 250))
    draw = ImageDraw.Draw(image)
    draw.text((32, 28), "Quarterly Report", font=_font(22, bold=True), fill=INK)
    for index, line in enumerate(
        (
            "Revenue grew by 18 percent in EMEA.",
            "Costs fell because the old cluster went quiet.",
            "Headcount stays at 42 engineers.",
        )
    ):
        draw.text((32, 90 + index * 38), line, font=_font(18), fill=INK)
    draw.rectangle([28, 214, 600, 252], outline=INK)
    draw.text((32, 222), "Next review: March 14, 2027.", font=_font(18), fill=INK)
    return image


def _paragraph(language: str) -> str:
    return {
        "spa": "El vehículo cargado con exceso de señalización quedó en el arcén; año 1975, contraseña C-2241.",
        "fra": "Le véhicule chargé au-delà du seuil est resté sur le bas-côté ; année 1975, code C-2241.",
        "deu": "Das überladene Fahrzeug blieb am Straßenrand stehen; Jahr 1975, Kennwort C-2241.",
        "por": "O veículo carregado além do limite ficou no acostamento; ano 1975, senha C-2241.",
        "rus": "Перегруженный автомобиль остался на обочине; год 1975, пароль C-2241.",
    }[language]


def _c4(language: str) -> Image.Image:
    """C4: one clean paragraph per language — ñ, é, ß, ö and Cyrillic against the model."""
    image = Image.new("RGB", (780, 140), (250, 250, 248))
    draw = ImageDraw.Draw(image)
    lines = textwrap.wrap(f"{language.upper()}: {_paragraph(language)}", width=68)
    for index, line in enumerate(lines):
        draw.text((24, 34 + index * 40), line, font=_font(19), fill=INK)
    return image


def _receipt_photo() -> Image.Image:
    """C5: a receipt seen through a phone camera — perspective, tilt and uneven light."""
    card = Image.new("RGB", (400, 600), (252, 252, 252))
    draw = ImageDraw.Draw(card)
    draw.text((26, 24), "TICKET FACTURA", font=_font(20, bold=True), fill=INK)
    draw.line([20, 58, 380, 58], fill=INK)
    for index, (item, cost) in enumerate(
        (("AdBlue 10L", "12,50"), ("Filtro aceite", "9,95"), ("Mano de obra", "45,00"), ("TOTAL IVA", "76,40"))
    ):
        draw.text((26, 74 + index * 40), item, font=_font(17), fill=INK)
        draw.text((250, 74 + index * 40), cost, font=_font(17, bold=True), fill=INK)
    draw.text((26, 260), "Ref. pago T-8891", font=_font(17), fill=INK)
    draw.text((26, 300), "BASE 63,14", font=_font(17), fill=INK)
    warped = _perspective(card, (14, 18, 388, 2, 400, 596, 0, 582), (480, 700))
    shade = Image.new("L", (480, 700))
    ImageDraw.Draw(shade).ellipse([40, 30, 460, 660], fill=255)
    shade = shade.filter(ImageFilter.GaussianBlur(80))
    return Image.composite(warped, Image.eval(warped, lambda p: p * 2 // 3), shade).rotate(
        5, expand=True, fillcolor=(70, 68, 64)
    )


def _c10() -> Image.Image:
    """C10: 8 px x-height — the resolution limit: half of this is expected to be lost."""
    image = Image.new("RGB", (720, 140), (250, 250, 248))
    draw = ImageDraw.Draw(image)
    for index, line in enumerate(("Código de verificación: K-4471", "Referencia: 78 04 921 337 / B")):
        draw.text((24, 30 + index * 60), line, font=_font(11), fill=INK)
    return image


def _jpeg(image: Image.Image, quality: int = 60) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=quality)
    return buffer.getvalue()


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def _pdf(pages: list[Image.Image]) -> bytes:
    buffer = io.BytesIO()
    pages[0].save(buffer, "PDF", save_all=True, append_images=pages[1:], resolution=72.0)
    return buffer.getvalue()


def _gallery(scan: bytes) -> bytes:
    """C8: a DOCX with 50 embedded scans — the concurrency and embedded-image route."""
    declarations = "".join(
        f'<w:p><w:r><w:drawing><wp:inline><wp:docPr descr="escaneado"/><a:graphic><a:graphicData><pic:pic>'
        f'<pic:blipFill><a:blip r:embed="rId{index + 2}"/></pic:blipFill></pic:pic></a:graphicData></a:graphic>'
        f"</wp:inline></w:drawing></w:r></w:p>"
        for index in range(50)
    )
    body = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
        ' xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
        ' xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"'
        ' xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'
        ' xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><w:body>'
        f"<w:p><w:r><w:t>Album</w:t></w:r></w:p>{declarations}</w:body></w:document>"
    )
    rels = "".join(
        f'<Relationship Id="rId{index + 2}" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
        f'relationships/image" Target="media/image{index + 1}.png"/>'
        for index in range(50)
    )
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Default Extension="png" ContentType="image/png"/>'
            '<Override PartName="/word/document.xml" ContentType='
            '"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>',
        )
        bundle.writestr(
            "_rels/.rels",
            '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/'
            'relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/'
            '2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>',
        )
        bundle.writestr(
            "word/_rels/document.xml.rels",
            '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/'
            f'relationships">{rels}</Relationships>',
        )
        bundle.writestr("word/document.xml", body)
        for index in range(50):
            bundle.writestr(f"word/media/image{index + 1}.png", scan)
    return archive.getvalue()


def _truth(fields: dict[str, str], text: str) -> bytes:
    return (json.dumps({"fields": fields, "text": text}, indent=2, ensure_ascii=False) + "\n").encode()


##### CORPUS #####

C3_FIELDS: Final = {
    "titulo": "Quarterly Report",
    "crecimiento": "18 percent",
    "region": "EMEA",
    "headcount": "42 engineers",
    "revision": "March 14, 2027",
}
C5_FIELDS: Final = {
    "titulo": "TICKET FACTURA",
    "adblue": "AdBlue",
    "obra": "45,00",
    "total": "76,40",
    "pago": "T-8891",
    "base": "63,14",
}
C10_FIELDS: Final = {"verificacion": "K-4471", "referencia": "78 04 921 337"}


C3_LINES: Final = (
    "Quarterly Report",
    "Revenue grew by 18 percent in EMEA.",
    "Costs fell because the old cluster went quiet.",
    "Headcount stays at 42 engineers.",
    "Next review: March 14, 2027.",
)


def build() -> dict[str, bytes]:
    """Every corpus file, images and truth JSONs alike, keyed by name."""
    sheet, scan, receipt = _sheet(), _screenshot(), _receipt_photo()
    c2 = _jpeg(sheet)
    files: dict[str, bytes] = {
        "c2_sheet.jpg": c2,
        "c2_sheet.truth.json": _truth(SHEET_FIELDS, "\n".join(SHEET_LINES)),
        "c3_screenshot.png": _png(scan),
        "c3_screenshot.truth.json": _truth(C3_FIELDS, "\n".join(C3_LINES)),
        "c5_receipt.jpg": _jpeg(receipt, quality=75),
        "c5_receipt.truth.json": _truth(
            C5_FIELDS,
            "TICKET FACTURA AdBlue 10L 12,50 Filtro aceite 9,95 Mano de obra 45,00 "
            "TOTAL IVA 76,40 Ref. pago T-8891 BASE 63,14",
        ),
        "c6_rot90.png": _png(scan.rotate(-90, expand=True)),
        "c6_rot180.png": _png(scan.rotate(180, expand=True)),
        "c6_orientation.truth.json": _truth(C3_FIELDS, "\n".join(C3_LINES)),
        "c8_album50.docx": _gallery(c2),
        "c8_album50.truth.json": _truth(SHEET_FIELDS, "\n".join(SHEET_LINES)),
        "c9_scan10.pdf": _pdf([sheet] * 10),
        "c9_scan10.truth.json": _truth(SHEET_FIELDS, "\n".join(SHEET_LINES)),
        "c10_small.png": _png(_c10()),
        "c10_small.truth.json": _truth(C10_FIELDS, "Código de verificación: K-4471 Referencia: 78 04 921 337 / B"),
        "c1_sheet.truth.json": _truth(SHEET_FIELDS, "\n".join(SHEET_LINES)),
    }
    files |= {f"c4_{language}.png": _png(_c4(language)) for language in _LANGUAGES}
    files["c4_languages.truth.json"] = _truth(
        {"ano": "1975", "clave": "C-2241"},
        " ".join(_paragraph(language) for language in _LANGUAGES),
    )
    return files


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    files = build()
    for name, payload in files.items():
        (ROOT / name).write_bytes(payload)
    sys.stdout.write(f"generated {len(files)} ocr corpus files in {ROOT}\n")


if __name__ == "__main__":
    main()
