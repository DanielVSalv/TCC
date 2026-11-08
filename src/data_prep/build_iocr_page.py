"""
Construção do "iocr_page" exigido pelo TFPredictor (TableFormer) — ver o TODO
original em src/models/tableformer_runner.py.

Confirmado lendo diretamente docling_ibm_models/tableformer/data_management/
tf_cell_matcher.py (third_party/docling-ibm-models/, clonado para inspeção): apesar
do esquema completo de página usado internamente pelo Docling ser bem mais elaborado
(blocks, text_lines, fonts, etc. — ver tests/test_data/samples/*.json no repositório
oficial), o TFPredictor de fato só lê os seguintes campos do "iocr_page":

  - iocr_page["image"]   -> imagem da página (array OpenCV/BGR)
  - iocr_page["height"], iocr_page["width"] -> dimensões da página
  - iocr_page["tokens"]  -> lista de palavras, cada uma como dict com EXATAMENTE:
        {"id": <int>, "text": <str>, "bbox": [l, t, r, b]}
    (bbox em pixels, origem no canto superior-esquerdo; também aceita bbox como
    dict {"l":, "t":, "r":, "b":} — convertido automaticamente pelo próprio
    tf_cell_matcher.py)

Ou seja, não é necessário replicar o schema interno completo do Docling — só esses
três campos por palavra, mais imagem/altura/largura no nível da página.

O Tesseract é usado para imagens sem texto nativo. Para PDFs digitais, o módulo
`src/data_prep/text_recovery.py` extrai palavras com PyMuPDF, transforma as caixas
da página em coordenadas do recorte padronizado e fornece esses mesmos tokens ao
TableFormer e aos demais modelos.
"""

from pathlib import Path
import os
from typing import Any

import cv2


def build_iocr_page_from_image(image_path: Path, min_confidence: float = 0.0) -> dict[str, Any]:
    """
    Executa OCR (Tesseract) sobre a imagem de uma tabela já recortada e padronizada
    (ver src/data_prep/extract_table_regions.py e standardize_input.py) e monta o
    "iocr_page" no formato mínimo exigido pelo TFPredictor.

    `min_confidence` filtra palavras com confiança de OCR abaixo do limiar (escala
    0-100 do próprio Tesseract) — útil para reduzir ruído em digitalizações de baixa
    qualidade.
    """
    import pytesseract
    tesseract_cmd = os.environ.get("TESSERACT_CMD")
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Não foi possível ler a imagem em {image_path}.")
    height, width = image.shape[:2]

    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    tokens: list[dict[str, Any]] = []
    next_id = 0
    for i in range(len(ocr_data["text"])):
        text = ocr_data["text"][i].strip()
        if not text:
            continue
        confidence = float(ocr_data["conf"][i])
        if confidence < min_confidence:
            continue

        left = ocr_data["left"][i]
        top = ocr_data["top"][i]
        cell_width = ocr_data["width"][i]
        cell_height = ocr_data["height"][i]

        tokens.append(
            {
                "id": next_id,
                "text": text,
                "bbox": [left, top, left + cell_width, top + cell_height],
            }
        )
        next_id += 1

    return {
        "image": image,
        "height": height,
        "width": width,
        "tokens": tokens,
    }


def build_iocr_page_from_words(image_path: Path, words: list[dict]) -> dict[str, Any]:
    """Build a TableFormer page from words with pixel-space `bbox` values."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Não foi possível ler a imagem em {image_path}.")
    height, width = image.shape[:2]
    tokens = [
        {"id": i, "text": word["text"], "bbox": word["bbox"]}
        for i, word in enumerate(words)
        if str(word.get("text", "")).strip()
    ]
    return {"image": image, "height": height, "width": width, "tokens": tokens}


def build_iocr_page_from_pdf_text(image_path: Path, pdf_words: list[dict]) -> dict[str, Any]:
    """
    Alternativa ao OCR para tabelas extraídas de PDFs com texto nativo (não
    escaneados): monta o "iocr_page" a partir de palavras já extraídas com
    coordenadas nativas do PDF (ex.: via pdfplumber `page.extract_words()`), evitando
    o ruído do OCR.

    `pdf_words` deve conter `text`, `x0`, `top`, `x1`, `bottom` já em pixels do
    recorte. A conversão da página PDF para as coordenadas do recorte acontece em
    `src/data_prep/text_recovery.py`.
    """
    words = [
        {
            "text": word["text"],
            "bbox": [word["x0"], word["top"], word["x1"], word["bottom"]],
        }
        for word in pdf_words
    ]
    return build_iocr_page_from_words(image_path, words)


if __name__ == "__main__":
    pass
