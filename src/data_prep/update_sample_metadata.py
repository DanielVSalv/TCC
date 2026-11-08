"""Add configured samples to the metadata table without guessing visual attributes."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml

from src.run_experiment import load_dataset_samples, load_ground_truth

FIELDNAMES = [
    "dataset",
    "sample",
    "n_rows",
    "n_cols",
    "cell_count",
    "merged_cell_count",
    "has_merged_cells",
    "hierarchical_header",
    "header_rows",
    "visible_rules",
    "row_banding",
    "scanned_or_digital",
    "complexity_score",
    "complexity",
    "structure_basis",
    "merged_basis",
    "header_basis",
    "visual_basis",
    "origin_basis",
]


def _metadata_for_sample(
    dataset_name: str, sample_path: Path, ground_truth_ref: object, dataset_cfg: dict
) -> dict[str, str]:
    table = load_ground_truth(dataset_name, ground_truth_ref, dataset_cfg)
    merged_count = sum(
        cell.row_span > 1 or cell.col_span > 1 for cell in table.cells
    )
    return {
        "dataset": dataset_name,
        "sample": sample_path.name,
        "n_rows": str(table.n_rows),
        "n_cols": str(table.n_cols),
        "cell_count": str(len(table.cells)),
        "merged_cell_count": str(merged_count),
        "has_merged_cells": str(merged_count > 0).lower(),
        "hierarchical_header": "not_annotated",
        "header_rows": "not_annotated",
        "visible_rules": "not_reviewed",
        "row_banding": "not_reviewed",
        "scanned_or_digital": "not_reviewed",
        "complexity_score": "",
        "complexity": "not_reviewed",
        "structure_basis": "dataset_annotation",
        "merged_basis": "dataset_annotation",
        "header_basis": "not_annotated",
        "visual_basis": "not_reviewed",
        "origin_basis": "not_reviewed",
    }


def update_sample_metadata(
    config_path: Path, metadata_path: Path, root: Path
) -> tuple[int, int]:
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    existing: dict[tuple[str, str], dict[str, str]] = {}
    if metadata_path.is_file():
        with metadata_path.open("r", newline="", encoding="utf-8") as metadata_file:
            reader = csv.DictReader(metadata_file)
            if reader.fieldnames != FIELDNAMES:
                raise ValueError(
                    f"Colunas inesperadas em {metadata_path}: {reader.fieldnames}"
                )
            for row in reader:
                key = (row["dataset"], row["sample"])
                if key in existing:
                    raise ValueError(f"Metadados duplicados para {key[0]}/{key[1]}")
                existing[key] = row

    added = 0
    for dataset_name, dataset_cfg in config["datasets"].items():
        if not dataset_cfg.get("enabled"):
            continue
        dataset_cfg = dict(dataset_cfg)
        for path_key in ("raw_dir", "ground_truth_dir", "processed_dir"):
            configured_path = dataset_cfg.get(path_key)
            if configured_path and not Path(configured_path).is_absolute():
                dataset_cfg[path_key] = str(root / configured_path)
        for sample_path, ground_truth_ref in load_dataset_samples(
            dataset_name, dataset_cfg
        ):
            key = (dataset_name, sample_path.name)
            if key in existing:
                continue
            existing[key] = _metadata_for_sample(
                dataset_name, sample_path, ground_truth_ref, dataset_cfg
            )
            added += 1

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", newline="", encoding="utf-8") as metadata_file:
        writer = csv.DictWriter(metadata_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(
            existing[key] for key in sorted(existing)
        )
    return added, len(existing)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inclui amostras configuradas nos metadados sem presumir atributos visuais."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment_config.yaml"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/ground_truth/sample_metadata.csv"),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    added, total = update_sample_metadata(args.config, args.metadata, args.root)
    print(f"metadata_samples={total} added={added} output={args.metadata}")


if __name__ == "__main__":
    main()
