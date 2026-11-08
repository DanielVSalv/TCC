"""Valida os datasets habilitados antes da inferência dos modelos."""

import argparse
from collections import Counter
import json
from pathlib import Path

import yaml
from PIL import Image

from src.run_experiment import load_dataset_samples, load_ground_truth


def _unpaired_corporate_files(dataset_cfg: dict) -> tuple[int, int]:
    processed_dir = Path(dataset_cfg.get("processed_dir", "data/processed/corporate"))
    ground_truth_dir = Path(dataset_cfg["ground_truth_dir"])
    image_stems = {path.stem for path in processed_dir.glob("*.png")}
    ground_truth_stems = {path.stem for path in ground_truth_dir.glob("*.json")}
    return len(image_stems - ground_truth_stems), len(ground_truth_stems - image_stems)


def _unpaired_scitsr_files(dataset_cfg: dict) -> tuple[int, int]:
    raw_dir = Path(dataset_cfg["raw_dir"])
    split = dataset_cfg.get("split", "test")
    structure_dir = raw_dir / "structure" / split
    image_dir = raw_dir / "img" / split
    annotation_stems = {path.stem for path in structure_dir.glob("*.json")}
    image_stems = {
        path.stem
        for extension in ("*.png", "*.jpg", "*.jpeg")
        for path in image_dir.glob(extension)
    }
    return len(annotation_stems - image_stems), len(image_stems - annotation_stems)


def _unpaired_public_files(
    dataset_name: str, dataset_cfg: dict, candidates: list[tuple[Path, object]]
) -> tuple[int, int]:
    raw_dir = Path(dataset_cfg["raw_dir"])
    split = dataset_cfg.get("split")
    annotation_names = {
        Path(gt_ref["filename"]).name for _, gt_ref in candidates if "filename" in gt_ref
    }
    image_paths = [path for path, _ in candidates]
    image_subdir = raw_dir / str(split) if split else raw_dir
    if not image_subdir.is_dir():
        image_subdir = raw_dir
    image_names = {
        path.name
        for pattern in ("*.png", "*.jpg", "*.jpeg")
        for path in image_subdir.glob(pattern)
    }
    missing_images = sum(not path.is_file() for path in image_paths)
    missing_ground_truth = len(image_names - annotation_names)
    return missing_images, missing_ground_truth


def _validate_table(table) -> str | None:
    if table.n_rows < 1 or table.n_cols < 1 or not table.cells:
        return "empty_ground_truth"

    for cell in table.cells:
        if cell.row < 0 or cell.col < 0:
            return "negative_cell_position"
        if cell.row_span < 1 or cell.col_span < 1:
            return "invalid_span"
        if cell.row + cell.row_span > table.n_rows:
            return "row_span_out_of_bounds"
        if cell.col + cell.col_span > table.n_cols:
            return "col_span_out_of_bounds"

    return None


def validate_dataset(dataset_name: str, dataset_cfg: dict) -> dict:
    full_cfg = dict(dataset_cfg)
    limit = dataset_cfg.get("sample_size")
    full_cfg["sample_size"] = None
    candidates = load_dataset_samples(dataset_name, full_cfg)
    selected = candidates
    reasons: Counter[str] = Counter()
    loaded = 0
    sample_reports = []

    for image_path, gt_ref in selected:
        sample_report = {
            "sample": image_path.name,
            "image": str(image_path),
            "ground_truth": (
                str(gt_ref)
                if isinstance(gt_ref, Path)
                else str(gt_ref.get("filename", "inline_annotation"))
            ),
            "status": "ignored",
        }
        if not image_path.is_file():
            reasons["missing_image"] += 1
            sample_report["reason"] = "missing_image"
            sample_reports.append(sample_report)
            continue

        try:
            with Image.open(image_path) as image:
                image.verify()
        except Exception as exc:
            reasons["unreadable_image"] += 1
            sample_report["reason"] = "unreadable_image"
            sample_report["error"] = f"{type(exc).__name__}: {exc}"
            sample_reports.append(sample_report)
            continue

        if isinstance(gt_ref, Path) and not gt_ref.is_file():
            reasons["missing_ground_truth"] += 1
            sample_report["reason"] = "missing_ground_truth"
            sample_reports.append(sample_report)
            continue

        try:
            table = load_ground_truth(dataset_name, gt_ref, dataset_cfg)
        except Exception as exc:
            reasons["unreadable_ground_truth"] += 1
            sample_report["reason"] = "unreadable_ground_truth"
            sample_report["error"] = f"{type(exc).__name__}: {exc}"
            sample_reports.append(sample_report)
            continue

        invalid_reason = _validate_table(table)
        if invalid_reason is not None:
            reasons[invalid_reason] += 1
            sample_report["reason"] = invalid_reason
            sample_reports.append(sample_report)
            continue

        loaded += 1
        row_spans = [cell.row_span for cell in table.cells]
        col_spans = [cell.col_span for cell in table.cells]
        sample_report.update(
            {
                "status": "loaded",
                "rows": table.n_rows,
                "columns": table.n_cols,
                "cells": len(table.cells),
                "merged_cells": sum(
                    cell.row_span > 1 or cell.col_span > 1 for cell in table.cells
                ),
                "row_span_range": [min(row_spans), max(row_spans)],
                "column_span_range": [min(col_spans), max(col_spans)],
                "row_span_values": sorted(set(row_spans)),
                "column_span_values": sorted(set(col_spans)),
            }
        )
        sample_reports.append(sample_report)

    unmatched_images = 0
    unmatched_ground_truth = 0
    if dataset_name == "corporate":
        unmatched_images, unmatched_ground_truth = _unpaired_corporate_files(dataset_cfg)
    elif dataset_name == "scitsr":
        unmatched_images, unmatched_ground_truth = _unpaired_scitsr_files(dataset_cfg)
    elif dataset_name in ("pubtabnet", "fintabnet"):
        unmatched_images, unmatched_ground_truth = _unpaired_public_files(
            dataset_name, dataset_cfg, candidates
        )

    extra_unmatched_images = max(
        0, unmatched_images - reasons.get("missing_image", 0)
    )
    ignored = (
        sum(reasons.values())
        + extra_unmatched_images
        + unmatched_ground_truth
    )
    result = {
        "dataset": dataset_name,
        "available": len(candidates),
        "configured_sample_size": limit,
        "selected_for_run": (
            len(candidates) if limit is None else min(limit, len(candidates))
        ),
        "validated": len(selected),
        "loaded": loaded,
        "ignored": ignored,
        "reasons": dict(reasons),
        "unmatched_images": unmatched_images,
        "unmatched_ground_truth": unmatched_ground_truth,
        "samples": sample_reports,
    }
    print(
        f"{dataset_name}: available={result['available']} "
        f"selected_for_run={result['selected_for_run']} validated={result['validated']} "
        f"loaded={loaded} ignored={ignored}"
    )
    if reasons:
        print(f"  ignored samples: {dict(reasons)}")
    if unmatched_images or unmatched_ground_truth:
        print(
            "  unmatched files: "
            f"images_without_ground_truth={unmatched_images}, "
            f"ground_truth_without_image={unmatched_ground_truth}"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Valida imagens e gabaritos dos datasets habilitados."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment_config.yaml"),
        help="Arquivo YAML de configuração do experimento.",
    )
    parser.add_argument(
        "--dataset",
        help="Validar somente um dataset (por padrão, todos os habilitados).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Salvar o relatório JSON detalhado, incluindo imagens, gabaritos e spans.",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    datasets = config["datasets"]
    if args.dataset:
        if args.dataset not in datasets:
            raise ValueError(f"Dataset desconhecido: {args.dataset}")
        selected_datasets = {args.dataset: datasets[args.dataset]}
    else:
        selected_datasets = {
            name: dataset_cfg
            for name, dataset_cfg in datasets.items()
            if dataset_cfg.get("enabled")
        }

    failed = False
    reports = []
    for dataset_name, dataset_cfg in selected_datasets.items():
        try:
            result = validate_dataset(dataset_name, dataset_cfg)
            failed = failed or result["loaded"] == 0
            reports.append(result)
        except Exception as exc:
            failed = True
            error_report = {
                "dataset": dataset_name,
                "available": 0,
                "selected_for_run": 0,
                "validated": 0,
                "loaded": 0,
                "ignored": 0,
                "error": f"{type(exc).__name__}: {exc}",
                "samples": [],
            }
            reports.append(error_report)
            print(
                f"{dataset_name}: available=0 selected_for_run=0 "
                f"validated=0 loaded=0 ignored=0 error={error_report['error']}"
            )

    if args.report:
        report_path = args.report
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(
                {
                    "config": str(args.config),
                    "datasets": reports,
                    "totals": {
                        "available": sum(row.get("available", 0) for row in reports),
                        "validated": sum(row.get("validated", 0) for row in reports),
                        "loaded": sum(row.get("loaded", 0) for row in reports),
                        "ignored": sum(row.get("ignored", 0) for row in reports),
                    },
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"Relatório detalhado salvo em: {report_path}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
