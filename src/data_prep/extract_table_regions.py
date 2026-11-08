"""
Extração da região da tabela dentro do documento (Etapa TD — Table Detection, ver
Seção 2.2 do TCC).

Para os datasets públicos (PubTabNet, FinTabNet, SciTSR, IBTR), as regiões de tabela
já costumam vir recortadas ou com bounding boxes anotadas — nesse caso, esta etapa se
resume a validar/recortar conforme a anotação original (`extract_from_annotated_dataset`).

Para os documentos corporativos reais (PDF sem anotação prévia), a detecção é feita
pelo Table Transformer na variante de DETECÇÃO (checkpoint
`microsoft/table-transformer-detection`, ver src/data_prep/table_detector.py) — a
mesma família de modelo já usada como um dos 4 algoritmos de TSR do estudo
(src/models/tabletransformer_runner.py), mas com checkpoint e tarefa diferentes:
aqui a saída é a localização da tabela na página, não sua estrutura interna.

Fluxo para PDFs não anotados (`extract_from_pdf`):
  1. Rasterizar cada página do PDF em uma imagem (via PyMuPDF/`pymupdf`, sem exigir
     um binário externo como o Poppler).
  2. Rodar o `TableDetector` sobre cada página para localizar as tabelas.
  3. Recortar cada região detectada, devolvendo uma `TableRegion` por tabela
     encontrada (um PDF/página pode conter mais de uma tabela).
"""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image


@dataclass
class TableRegion:
    """Representa uma região de tabela extraída de um documento."""

    source_path: Path
    page_index: int
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1)
    image: Optional[Image.Image] = None
    detection_score: Optional[float] = None
    detection_index: int = 0
    rotated: bool = False
    crop_bbox: Optional[tuple[float, float, float, float]] = None


def extract_from_annotated_dataset(
    image_path: Path, bbox: tuple[float, float, float, float]
) -> TableRegion:
    """
    Recorta a região da tabela a partir de uma imagem de página e de um bounding box
    já anotado (caso comum para PubTabNet/FinTabNet/SciTSR/IBTR).
    """
    img = Image.open(image_path).convert("RGB")
    cropped = img.crop(bbox)
    return TableRegion(
        source_path=image_path,
        page_index=0,
        bbox=bbox,
        image=cropped,
        crop_bbox=bbox,
    )


def rasterize_pdf_page(pdf_path: Path, page_index: int, dpi: int = 200) -> Image.Image:
    """
    Rasteriza uma página do PDF em uma imagem PIL (RGB), no DPI informado. Usa
    PyMuPDF (`pymupdf`), que não depende de nenhum binário externo (ao contrário de
    rotas baseadas em Poppler/`pdf2image`).
    """
    import pymupdf

    doc = pymupdf.open(str(pdf_path))
    try:
        page = doc[page_index]
        zoom = dpi / 72  # PDF usa 72 pontos por polegada como unidade nativa
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
        mode = "RGB" if pixmap.n < 4 else "RGBA"
        image = Image.frombytes(mode, (pixmap.width, pixmap.height), pixmap.samples)
        return image.convert("RGB")
    finally:
        doc.close()


def count_pdf_pages(pdf_path: Path) -> int:
    import pymupdf

    doc = pymupdf.open(str(pdf_path))
    try:
        return doc.page_count
    finally:
        doc.close()


def extract_from_pdf(
    pdf_path: Path,
    detector=None,
    dpi: int = 200,
    pages: Optional[list[int]] = None,
    include_rotated: bool = False,
    page_records: Optional[list[dict]] = None,
    crop_padding_px: float = 8,
) -> list[TableRegion]:
    """
    Localiza e recorta candidatas a região de tabela em um PDF sem anotação prévia
    (fluxo pensado para o subconjunto de documentos corporativos reais).

    `detector` deve ser uma instância de `TableDetector` (src/data_prep/
    table_detector.py) já carregada via `load_model()`. Se omitido, uma nova
    instância é criada e carregada aqui — útil para uso avulso, mas ineficiente ao
    processar muitos PDFs (nesse caso, carregue o detector uma vez e passe-o
    explicitamente, reaproveitando entre chamadas).

    `pages`: lista opcional de índices de página a processar (0-based). Por padrão,
    processa todas as páginas do PDF. Detecções rotacionadas são excluídas por
    padrão; com `include_rotated=True`, retornam como registros sem imagem para que
    o chamador possa registrar explicitamente a exclusão no manifesto.

    `page_records`, quando informado, recebe uma entrada por página processada com
    o número de detecções, inclusive páginas em que nenhuma tabela foi localizada.
    """
    from src.data_prep.table_detector import TableDetector

    if detector is None:
        detector = TableDetector()
        detector.load_model()

    page_indices = pages if pages is not None else range(count_pdf_pages(pdf_path))

    regions: list[TableRegion] = []
    for page_index in page_indices:
        page_image = rasterize_pdf_page(pdf_path, page_index, dpi=dpi)
        detections = detector.detect(page_image, page_index=page_index)
        if page_records is not None:
            page_records.append(
                {"page_index": page_index, "detection_count": len(detections)}
            )

        for detection_index, detection in enumerate(detections):
            if detection.rotated:
                if include_rotated:
                    regions.append(
                        TableRegion(
                            source_path=pdf_path,
                            page_index=page_index,
                            bbox=detection.bbox,
                            detection_score=detection.score,
                            detection_index=detection_index,
                            rotated=True,
                        )
                    )
                continue

            x0, y0, x1, y1 = detection.bbox
            crop_bbox = (
                max(0.0, x0 - crop_padding_px),
                max(0.0, y0 - crop_padding_px),
                min(float(page_image.width), x1 + crop_padding_px),
                min(float(page_image.height), y1 + crop_padding_px),
            )
            cropped = page_image.crop(crop_bbox)
            regions.append(
                TableRegion(
                    source_path=pdf_path,
                    page_index=page_index,
                    bbox=detection.bbox,
                    image=cropped,
                    detection_score=detection.score,
                    detection_index=detection_index,
                    crop_bbox=crop_bbox,
                )
            )

    return regions


def prepare_corporate_dataset(
    raw_dir: Path,
    processed_dir: Path,
    detector,
    dpi: int = 200,
    max_dimension: int = 1024,
    crop_padding_px: float = 12,
    pages_by_pdf: Optional[dict[str, list[int]]] = None,
    manifest_path: Optional[Path] = None,
    authorization_confirmed: bool = False,
) -> list[dict]:
    """Create standardized table crops and a CSV manifest from corporate PDFs.

    `pages_by_pdf` maps PDF filenames to one-based physical page numbers. When a
    mapping is provided, only listed PDFs/pages are processed; use `None` to process
    every page in every PDF. Rotated detections are written to the manifest as
    excluded; they are not supplied to recognition runners.
    """
    if authorization_confirmed is not True:
        raise PermissionError(
            "Confirme a autorização formal antes de processar documentos corporativos."
        )

    from src.data_prep.standardize_input import (
        StandardizationConfig,
        standardize_image,
    )

    pdf_paths = (
        sorted(raw_dir.glob("*.pdf"))
        if pages_by_pdf is None
        else [raw_dir / filename for filename in sorted(pages_by_pdf)]
    )
    if not pdf_paths:
        raise FileNotFoundError(f"Nenhum PDF encontrado em {raw_dir}.")
    missing_pdfs = [path for path in pdf_paths if not path.is_file()]
    if missing_pdfs:
        missing_names = ", ".join(path.name for path in missing_pdfs)
        raise FileNotFoundError(f"PDFs configurados não encontrados em {raw_dir}: {missing_names}")

    processed_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_path or processed_dir / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    standardization = StandardizationConfig(max_dimension=max_dimension)
    manifest_rows: list[dict] = []
    saved_by_page: dict[tuple[str, int], int] = {}

    for pdf_path in pdf_paths:
        page_numbers = pages_by_pdf.get(pdf_path.name) if pages_by_pdf is not None else None
        pages = [number - 1 for number in page_numbers] if page_numbers is not None else None
        page_records: list[dict] = []
        regions = extract_from_pdf(
            pdf_path,
            detector=detector,
            dpi=dpi,
            pages=pages,
            include_rotated=True,
            page_records=page_records,
        )

        detections_by_page: dict[int, list[TableRegion]] = {}
        for region in regions:
            detections_by_page.setdefault(region.page_index, []).append(region)

        for page_record in page_records:
            page_index = page_record["page_index"]
            page_regions = detections_by_page.get(page_index, [])
            if not page_regions:
                manifest_rows.append(
                    {
                        "source_pdf": pdf_path.name,
                        "page_index": page_index,
                        "page_number": page_index + 1,
                        "dpi": dpi,
                        "detection_index": "",
                        "status": "no_detection",
                        "bbox_pixels": "",
                        "crop_bbox_pixels": "",
                        "confidence": "",
                        "rotated": False,
                        "image_file": "",
                        "original_size": "",
                        "standardized_size": "",
                    }
                )
                continue

            for region in page_regions:
                bbox_json = json.dumps(
                    [round(value, 2) for value in region.bbox], separators=(",", ":")
                )
                row = {
                    "source_pdf": pdf_path.name,
                    "page_index": region.page_index,
                    "page_number": region.page_index + 1,
                    "dpi": dpi,
                    "detection_index": region.detection_index,
                    "status": "excluded_rotated" if region.rotated else "saved",
                    "bbox_pixels": bbox_json,
                    "crop_bbox_pixels": "",
                    "confidence": region.detection_score,
                    "rotated": region.rotated,
                    "image_file": "",
                    "original_size": "",
                    "standardized_size": "",
                }
                if not region.rotated and region.image is not None:
                    row["crop_bbox_pixels"] = json.dumps(
                        [round(value, 2) for value in region.crop_bbox or region.bbox],
                        separators=(",", ":"),
                    )
                    page_key = (pdf_path.name, region.page_index)
                    table_index = saved_by_page.get(page_key, 0)
                    saved_by_page[page_key] = table_index + 1
                    image_name = (
                        f"{pdf_path.stem}_pg{region.page_index}_tab{table_index}.png"
                    )
                    image_path = processed_dir / image_name
                    standardized = standardize_image(region.image, standardization)
                    standardized.save(image_path, format=standardization.output_format)
                    row["image_file"] = image_name
                    row["original_size"] = (
                        f"{region.image.width}x{region.image.height}"
                    )
                    row["standardized_size"] = (
                        f"{standardized.width}x{standardized.height}"
                    )
                manifest_rows.append(row)

    fieldnames = [
        "source_pdf",
        "page_index",
        "page_number",
        "dpi",
        "detection_index",
        "status",
        "bbox_pixels",
        "crop_bbox_pixels",
        "confidence",
        "rotated",
        "image_file",
        "original_size",
        "standardized_size",
    ]
    with open(manifest_path, "w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    saved_count = sum(row["status"] == "saved" for row in manifest_rows)
    rotated_count = sum(row["status"] == "excluded_rotated" for row in manifest_rows)
    no_detection_count = sum(row["status"] == "no_detection" for row in manifest_rows)
    print(
        f"Corporate extraction: saved={saved_count} excluded_rotated={rotated_count} "
        f"pages_without_detection={no_detection_count} manifest={manifest_path}"
    )
    return manifest_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera recortes corporativos e manifesto a partir dos PDFs."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment_config.yaml"),
        help="Arquivo YAML com caminhos, páginas e parâmetros de detecção.",
    )
    args = parser.parse_args()

    import yaml

    with open(args.config, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    dataset_cfg = config["datasets"]["corporate"]
    detection_cfg = dataset_cfg["table_detection"]
    privacy_cfg = dataset_cfg.get("privacy", {})
    if privacy_cfg.get("authorization_confirmed") is not True:
        raise PermissionError(
            "Confirme a autorização formal antes de processar documentos corporativos."
        )
    pages_path = Path(privacy_cfg["source_pages_path"])
    with pages_path.open("r", encoding="utf-8") as pages_file:
        pages_by_pdf = yaml.safe_load(pages_file)["pages_by_pdf"]

    from src.data_prep.table_detector import TableDetector

    detector = TableDetector(
        score_threshold=detection_cfg.get("score_threshold", 0.8)
    )
    detector.load_model(detection_cfg.get("checkpoint", "microsoft/table-transformer-detection"))
    manifest_path = Path(
        privacy_cfg.get(
            "source_manifest_path",
            str(
                Path(
                    privacy_cfg.get(
                        "source_processed_dir", dataset_cfg["processed_dir"]
                    )
                )
                / "manifest.csv"
            ),
        )
    )
    prepare_corporate_dataset(
        raw_dir=Path(dataset_cfg["raw_dir"]),
        processed_dir=Path(
            privacy_cfg.get("source_processed_dir", dataset_cfg["processed_dir"])
        ),
        detector=detector,
        dpi=detection_cfg.get("dpi", 200),
        max_dimension=detection_cfg.get("max_dimension", 1024),
        crop_padding_px=detection_cfg.get("crop_padding_px", 12),
        pages_by_pdf=pages_by_pdf,
        manifest_path=manifest_path,
        authorization_confirmed=privacy_cfg["authorization_confirmed"],
    )


if __name__ == "__main__":
    main()
