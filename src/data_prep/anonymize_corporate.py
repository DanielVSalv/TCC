"""Create pseudonymous corporate crops and ground truth from manual redactions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from typing import Any

from PIL import Image, ImageDraw


def _validate_sample_rules(
    image_path: Path,
    ground_truth_path: Path,
    rules: dict[str, Any],
) -> tuple[str, list[tuple[int, int, int, int]], list[tuple[int, int]]]:
    if not isinstance(rules, dict):
        raise ValueError(f"Regras inválidas para a amostra {image_path.name}.")
    alias = rules.get("sample_id")
    if not isinstance(alias, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", alias):
        raise ValueError(
            f"Identificador pseudônimo inválido para a amostra {image_path.name}."
        )

    raw_regions = rules.get("image_regions")
    raw_cells = rules.get("cells")
    if not isinstance(raw_regions, list) or not isinstance(raw_cells, list):
        raise ValueError(
            f"Informe image_regions e cells como listas em {image_path.name}."
        )
    if bool(raw_regions) != bool(raw_cells):
        raise ValueError(
            f"Regiões e células devem ser informadas em conjunto em {image_path.name}."
        )
    if not raw_regions and rules.get("reviewed_no_sensitive_data") is not True:
        raise ValueError(
            f"Confirme a revisão manual sem dados sensíveis para {image_path.name}."
        )

    with Image.open(image_path) as source_image:
        width, height = source_image.size

    regions = []
    for region in raw_regions:
        if (
            not isinstance(region, list)
            or len(region) != 4
            or any(type(value) is not int for value in region)
        ):
            raise ValueError(
                f"Região inválida em {image_path.name}; use [x0, y0, x1, y1] inteiros."
            )
        x0, y0, x1, y1 = region
        if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0 or x1 > width or y1 > height:
            raise ValueError(
                f"Região fora dos limites da imagem {image_path.name} ({width}x{height})."
            )
        regions.append((x0, y0, x1, y1))

    ground_truth = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    cells_by_position = {
        (cell["row"], cell["col"]): cell for cell in ground_truth["cells"]
    }
    cells = []
    for position in raw_cells:
        if (
            not isinstance(position, list)
            or len(position) != 2
            or any(type(value) is not int for value in position)
        ):
            raise ValueError(
                f"Célula inválida em {image_path.name}; use [linha, coluna] inteiros."
            )
        cell_position = (position[0], position[1])
        cell = cells_by_position.get(cell_position)
        if cell is None or not str(cell.get("text", "")).strip():
            raise ValueError(
                f"Célula {cell_position} ausente ou sem texto em {ground_truth_path.name}."
            )
        cells.append(cell_position)
    if len(set(cells)) != len(cells):
        raise ValueError(f"Células duplicadas nas regras de {image_path.name}.")

    return alias, regions, cells


def anonymize_corporate_dataset(
    source_images_dir: Path,
    source_ground_truth_dir: Path,
    output_images_dir: Path,
    output_ground_truth_dir: Path,
    output_manifest_path: Path,
    sample_rules: dict[str, Any],
    authorization_confirmed: bool,
) -> list[dict[str, Any]]:
    """Mask manually identified image regions and clear corresponding cell text.

    Outputs use pseudonymous IDs and omit original PDF names, page positions,
    source paths, and image metadata. Every source crop must have explicit rules.
    """
    if authorization_confirmed is not True:
        raise PermissionError(
            "Confirme a autorização formal de uso antes de processar documentos "
            "corporativos."
        )

    source_dirs = (source_images_dir.resolve(), source_ground_truth_dir.resolve())
    output_dirs = (output_images_dir.resolve(), output_ground_truth_dir.resolve())
    if (
        output_dirs[0] == output_dirs[1]
        or output_dirs[0] in output_dirs[1].parents
        or output_dirs[1] in output_dirs[0].parents
        or any(
            source == output or source in output.parents or output in source.parents
            for source in source_dirs
            for output in output_dirs
        )
    ):
        raise ValueError(
            "As pastas de saída da anonimização devem ser distintas e separadas "
            "das pastas-fonte."
        )

    image_paths = sorted(source_images_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"Nenhuma imagem corporativa encontrada em {source_images_dir}.")

    expected_names = {path.name for path in image_paths}
    configured_names = set(sample_rules)
    if expected_names != configured_names:
        missing_rules = sorted(expected_names - configured_names)
        unknown_rules = sorted(configured_names - expected_names)
        raise ValueError(
            "As regras de anonimização devem corresponder exatamente às imagens "
            f"corporativas. Sem regra: {missing_rules}; sem imagem: {unknown_rules}."
        )

    validated = []
    aliases = set()
    for image_path in image_paths:
        ground_truth_path = source_ground_truth_dir / f"{image_path.stem}.json"
        if not ground_truth_path.is_file():
            raise FileNotFoundError(
                f"Gabarito corporativo não encontrado para {image_path.name}."
            )
        alias, regions, cells = _validate_sample_rules(
            image_path, ground_truth_path, sample_rules[image_path.name]
        )
        if alias in aliases:
            raise ValueError(f"Identificador pseudônimo duplicado: {alias}.")
        aliases.add(alias)
        validated.append((image_path, ground_truth_path, alias, regions, cells))

    expected_image_names = {f"{alias}.png" for alias in aliases}
    expected_ground_truth_names = {f"{alias}.json" for alias in aliases}
    existing_image_names = (
        {path.name for path in output_images_dir.glob("*.png")}
        if output_images_dir.exists()
        else set()
    )
    existing_ground_truth_names = (
        {path.name for path in output_ground_truth_dir.glob("*.json")}
        if output_ground_truth_dir.exists()
        else set()
    )
    if (
        existing_image_names - expected_image_names
        or existing_ground_truth_names - expected_ground_truth_names
    ):
        raise FileExistsError(
            "As pastas de saída contêm amostras que não pertencem às regras atuais; "
            "revise-as manualmente antes de continuar."
        )

    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_ground_truth_dir.mkdir(parents=True, exist_ok=True)
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for image_path, ground_truth_path, alias, regions, cell_positions in validated:
        with Image.open(image_path) as source_image:
            redacted_image = Image.new("RGB", source_image.size, "white")
            rgb_image = source_image.convert("RGB")
            redacted_image.paste(rgb_image)
            rgb_image.close()
        draw = ImageDraw.Draw(redacted_image)
        for x0, y0, x1, y1 in regions:
            draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill="black")
        output_image_path = output_images_dir / f"{alias}.png"
        redacted_image.save(output_image_path, format="PNG")

        ground_truth = json.loads(ground_truth_path.read_text(encoding="utf-8"))
        sanitized_cells = []
        for cell in ground_truth["cells"]:
            sanitized_cell = {
                "row": cell["row"],
                "col": cell["col"],
                "row_span": cell.get("row_span", 1),
                "col_span": cell.get("col_span", 1),
                "is_header": cell.get("is_header", False),
                "text": cell.get("text", ""),
            }
            if (cell["row"], cell["col"]) in cell_positions:
                sanitized_cell["text"] = ""
            sanitized_cells.append(sanitized_cell)
        sanitized_ground_truth = {
            "n_rows": ground_truth["n_rows"],
            "n_cols": ground_truth["n_cols"],
            "cells": sanitized_cells,
        }
        (output_ground_truth_dir / f"{alias}.json").write_text(
            json.dumps(sanitized_ground_truth, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        manifest_rows.append(
            {
                "sample_id": alias,
                "image_file": output_image_path.name,
                "privacy_redacted": "true",
                "redacted_cell_count": len(cell_positions),
                "width": redacted_image.width,
                "height": redacted_image.height,
            }
        )
        redacted_image.close()

    with output_manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        fieldnames = [
            "sample_id",
            "image_file",
            "privacy_redacted",
            "redacted_cell_count",
            "width",
            "height",
        ]
        writer = csv.DictWriter(manifest_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    return manifest_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera recortes e gabaritos corporativos anonimizados."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment_config.yaml"),
        help="Configuração do experimento e dos caminhos corporativos.",
    )
    args = parser.parse_args()

    import yaml

    with args.config.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    corporate_cfg = config["datasets"]["corporate"]
    privacy_cfg = corporate_cfg["privacy"]
    rules_path = Path(privacy_cfg["rules_path"])
    with rules_path.open("r", encoding="utf-8") as rules_file:
        rules = yaml.safe_load(rules_file)

    anonymize_corporate_dataset(
        source_images_dir=Path(privacy_cfg["source_processed_dir"]),
        source_ground_truth_dir=Path(privacy_cfg["source_ground_truth_dir"]),
        output_images_dir=Path(corporate_cfg["processed_dir"]),
        output_ground_truth_dir=Path(corporate_cfg["ground_truth_dir"]),
        output_manifest_path=Path(
            corporate_cfg["table_detection"]["manifest_path"]
        ),
        sample_rules=rules["samples"],
        authorization_confirmed=privacy_cfg.get("authorization_confirmed", False),
    )


if __name__ == "__main__":
    main()
