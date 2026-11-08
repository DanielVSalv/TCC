"""Text extraction and cell assignment for table recognition evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any


@dataclass
class RecoveredText:
    words: list[dict[str, Any]]
    source: str
    engine: str
    config: dict[str, Any]


def _parse_size(value: str) -> tuple[int, int]:
    width, height = value.lower().split("x", maxsplit=1)
    return int(width), int(height)


def extract_pdf_words_for_crop(
    pdf_path: Path,
    page_index: int,
    dpi: int,
    crop_bbox_pixels: tuple[float, float, float, float],
    crop_size: tuple[int, int],
    standardized_size: tuple[int, int],
) -> list[dict[str, Any]]:
    """Map PyMuPDF word boxes from page points to standardized crop pixels."""
    try:
        import pymupdf
    except ImportError:
        import fitz as pymupdf

    document = pymupdf.open(pdf_path)
    try:
        page = document[page_index]
        crop_left, crop_top, _, _ = crop_bbox_pixels
        scale_to_pixels = dpi / 72
        scale_x = standardized_size[0] / crop_size[0]
        scale_y = standardized_size[1] / crop_size[1]
        words = []

        for word_id, item in enumerate(page.get_text("words")):
            x0, y0, x1, y1, text = item[:5]
            text = str(text).strip()
            if not text:
                continue

            page_bbox = pymupdf.Rect(x0, y0, x1, y1)
            if page.rotation:
                page_bbox *= page.rotation_matrix

            left = (page_bbox.x0 * scale_to_pixels - crop_left) * scale_x
            top = (page_bbox.y0 * scale_to_pixels - crop_top) * scale_y
            right = (page_bbox.x1 * scale_to_pixels - crop_left) * scale_x
            bottom = (page_bbox.y1 * scale_to_pixels - crop_top) * scale_y
            left = max(0.0, min(float(standardized_size[0]), left))
            top = max(0.0, min(float(standardized_size[1]), top))
            right = max(0.0, min(float(standardized_size[0]), right))
            bottom = max(0.0, min(float(standardized_size[1]), bottom))
            if right <= left or bottom <= top:
                continue

            words.append(
                {
                    "id": word_id,
                    "text": text,
                    "bbox": [left, top, right, bottom],
                    "confidence": None,
                }
            )

        return words
    finally:
        document.close()


def extract_ocr_words(
    image_path: Path,
    language: str = "eng",
    page_segmentation_mode: int = 6,
    min_confidence: float = 0.0,
    tesseract_cmd: str | None = None,
) -> list[dict[str, Any]]:
    """Run Tesseract OCR and return word boxes in image pixel coordinates."""
    import pytesseract
    from pytesseract import Output

    executable = tesseract_cmd or os.environ.get("TESSERACT_CMD")
    if executable:
        pytesseract.pytesseract.tesseract_cmd = executable

    data = pytesseract.image_to_data(
        str(image_path),
        lang=language,
        config=f"--psm {page_segmentation_mode}",
        output_type=Output.DICT,
    )
    words = []
    for index, raw_text in enumerate(data["text"]):
        text = str(raw_text).strip()
        if not text:
            continue
        confidence = float(data["conf"][index])
        if confidence < min_confidence:
            continue
        left = int(data["left"][index])
        top = int(data["top"][index])
        words.append(
            {
                "id": len(words),
                "text": text,
                "bbox": [
                    left,
                    top,
                    left + int(data["width"][index]),
                    top + int(data["height"][index]),
                ],
                "confidence": confidence,
            }
        )
    return words


def recover_sample_text(
    image_path: Path,
    dataset_name: str,
    dataset_cfg: dict,
    text_cfg: dict,
    manifest_entry: dict[str, str] | None = None,
) -> RecoveredText:
    """Use OCR for raster inputs and force it for pseudonymized corporate samples."""
    mode = text_cfg.get("mode", "auto")
    if mode not in {"auto", "pdf_text", "ocr"}:
        raise ValueError(f"Modo de recuperação textual desconhecido: {mode}")
    privacy_redacted = (
        dataset_name == "corporate"
        and manifest_entry is not None
        and manifest_entry.get("privacy_redacted", "").lower() == "true"
    )
    if privacy_redacted:
        mode = "ocr"

    language = text_cfg.get("ocr_language", "eng")
    page_segmentation_mode = int(text_cfg.get("ocr_page_segmentation_mode", 6))
    min_confidence = float(text_cfg.get("ocr_min_confidence", 0.0))
    min_native_words = int(text_cfg.get("min_native_words", 3))
    tesseract_cmd = text_cfg.get("tesseract_cmd") or os.environ.get("TESSERACT_CMD")
    base_config = {
        "mode": mode,
        "ocr_language": language,
        "ocr_page_segmentation_mode": page_segmentation_mode,
        "ocr_min_confidence": min_confidence,
        "min_native_words": min_native_words,
        "tesseract_cmd": tesseract_cmd or "PATH",
    }

    if mode == "pdf_text" and (
        dataset_name != "corporate" or manifest_entry is None
    ):
        raise ValueError(
            f"Modo pdf_text requer uma amostra corporativa com manifesto: {image_path.name}."
        )

    if mode != "ocr" and dataset_name == "corporate" and manifest_entry is not None:
        raw_dir = Path(dataset_cfg["raw_dir"])
        pdf_path = raw_dir / manifest_entry["source_pdf"]
        crop_bbox = tuple(
            float(value) for value in json.loads(manifest_entry["crop_bbox_pixels"])
        )
        crop_size = _parse_size(manifest_entry["original_size"])
        standardized_size = _parse_size(manifest_entry["standardized_size"])
        words = extract_pdf_words_for_crop(
            pdf_path,
            page_index=int(manifest_entry["page_index"]),
            dpi=int(manifest_entry["dpi"]),
            crop_bbox_pixels=crop_bbox,
            crop_size=crop_size,
            standardized_size=standardized_size,
        )
        if len(words) >= min_native_words:
            return RecoveredText(
                words=words,
                source="pdf_text",
                engine="pymupdf",
                config={**base_config, "native_word_count": len(words)},
            )
        if mode == "pdf_text":
            raise ValueError(
                f"PDF sem texto nativo suficiente para {image_path.name}: {len(words)} palavras."
            )

    words = extract_ocr_words(
        image_path,
        language=language,
        page_segmentation_mode=page_segmentation_mode,
        min_confidence=min_confidence,
        tesseract_cmd=tesseract_cmd,
    )
    return RecoveredText(
        words=words,
        source="ocr",
        engine="tesseract",
        config={**base_config, "recognized_word_count": len(words)},
    )


def assign_words_to_cells(
    table,
    words: list[dict[str, Any]],
    min_word_overlap: float = 0.2,
) -> int:
    """Assign each recognized word to the cell with greatest word-box overlap.

    Returns the number of words that did not overlap any cell sufficiently.
    Cells without geometry remain empty and are not eligible for assignment.
    """
    cell_words: list[list[tuple[float, float, str]]] = [
        [] for _ in table.cells
    ]
    unassigned = 0

    for word in words:
        word_box = word["bbox"]
        word_width = max(0.0, word_box[2] - word_box[0])
        word_height = max(0.0, word_box[3] - word_box[1])
        word_area = word_width * word_height
        if word_area <= 0:
            unassigned += 1
            continue

        best_cell_index = None
        best_overlap = 0.0
        for cell_index, cell in enumerate(table.cells):
            if cell.bbox is None:
                continue
            cell_box = cell.bbox
            overlap_width = max(
                0.0, min(word_box[2], cell_box[2]) - max(word_box[0], cell_box[0])
            )
            overlap_height = max(
                0.0, min(word_box[3], cell_box[3]) - max(word_box[1], cell_box[1])
            )
            overlap_ratio = overlap_width * overlap_height / word_area
            if overlap_ratio > best_overlap:
                best_overlap = overlap_ratio
                best_cell_index = cell_index

        if best_cell_index is None or best_overlap < min_word_overlap:
            unassigned += 1
            continue

        center_y = (word_box[1] + word_box[3]) / 2
        cell_words[best_cell_index].append((center_y, word_box[0], word["text"]))

    for cell, assigned_words in zip(table.cells, cell_words):
        if cell.bbox is None:
            continue
        assigned_words.sort(key=lambda item: (item[0], item[1]))
        cell.text = " ".join(item[2] for item in assigned_words)

    return unassigned
