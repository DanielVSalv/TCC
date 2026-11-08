"""Aggregate model metrics by manually reviewed table attributes."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import yaml

from src.evaluation.teds import METRIC_PROTOCOL_VERSION

ATTRIBUTES = (
    "has_merged_cells",
    "hierarchical_header",
    "visible_rules",
    "row_banding",
    "scanned_or_digital",
    "complexity",
)
METRICS = ("teds", "s_teds", "cell_f1")


def _read_metadata(path: Path) -> list[dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as metadata_file:
        rows = list(csv.DictReader(metadata_file))
    keys = [(row["dataset"], row["sample"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Metadata contém amostras duplicadas: {path}")
    return rows


def _read_metric_files(
    metrics_dir: Path,
    datasets: set[str],
    configured_models: set[str] | None = None,
) -> tuple[dict, list[str], list[str]]:
    results: dict[tuple[str, str, str], dict[str, float | None]] = {}
    models = set(configured_models or ())
    legacy_files: list[str] = []
    for path in sorted(metrics_dir.glob("*.csv")):
        dataset, separator, model = path.stem.rpartition("_")
        if not separator or dataset not in datasets:
            continue
        with open(path, "r", newline="", encoding="utf-8") as metrics_file:
            reader = csv.DictReader(metrics_file)
            if "metric_protocol" not in (reader.fieldnames or []):
                legacy_files.append(path.name)
                continue
            file_rows = list(reader)
            protocols = {row["metric_protocol"] for row in file_rows}
            if protocols and protocols != {METRIC_PROTOCOL_VERSION}:
                legacy_files.append(path.name)
                continue
            models.add(model)
            for row in file_rows:
                key = (dataset, Path(row["sample"]).name, model)
                values = {
                    metric: float(row[metric])
                    if row.get(metric) not in (None, "")
                    else None
                    for metric in METRICS
                }
                results[key] = values
    return results, sorted(models), legacy_files


def _aggregate(
    metadata_rows: list[dict[str, str]],
    results: dict,
    models: list[str],
    output_path: Path,
) -> list[dict[str, object]]:
    scopes = ["all", *sorted({row["dataset"] for row in metadata_rows})]
    summary_rows: list[dict[str, object]] = []

    for scope in scopes:
        scope_rows = [
            row for row in metadata_rows if scope == "all" or row["dataset"] == scope
        ]
        active_models = models or ["no_current_metrics"]
        for attribute in ATTRIBUTES:
            groups: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in scope_rows:
                groups[row[attribute]].append(row)

            for group_value, group_rows in sorted(groups.items()):
                for model in active_models:
                    metric_rows = [
                        results[(row["dataset"], row["sample"], model)]
                        for row in group_rows
                        if (row["dataset"], row["sample"], model) in results
                    ]
                    record: dict[str, object] = {
                        "scope": scope,
                        "attribute": attribute,
                        "value": group_value,
                        "model": model,
                        "metric_protocol": (
                            METRIC_PROTOCOL_VERSION if model != "no_current_metrics" else ""
                        ),
                        "metrics_status": (
                            "current" if model != "no_current_metrics" else "no_current_metrics"
                        ),
                        "sample_count": len(group_rows),
                        "evaluated_count": len(metric_rows),
                        "merged_sample_count": sum(
                            row["has_merged_cells"] == "true" for row in group_rows
                        ),
                        "mean_rows": mean(int(row["n_rows"]) for row in group_rows),
                        "mean_cols": mean(int(row["n_cols"]) for row in group_rows),
                        "mean_cells": mean(int(row["cell_count"]) for row in group_rows),
                    }
                    for metric in METRICS:
                        available = [
                            row[metric] for row in metric_rows if row[metric] is not None
                        ]
                        record[f"{metric}_count"] = len(available)
                        record[f"mean_{metric}"] = mean(available) if available else ""
                    summary_rows.append(record)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scope",
        "attribute",
        "value",
        "model",
        "metric_protocol",
        "metrics_status",
        "sample_count",
        "evaluated_count",
        "merged_sample_count",
        "mean_rows",
        "mean_cols",
        "mean_cells",
        "teds_count",
        "mean_teds",
        "s_teds_count",
        "mean_s_teds",
        "cell_f1_count",
        "mean_cell_f1",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume métricas por atributos de tabela e modelo."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/ground_truth/sample_metadata.csv"),
    )
    parser.add_argument("--metrics-dir", type=Path, default=Path("results/metrics"))
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiment_config.yaml")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/metrics/by_attribute_summary.csv"),
    )
    args = parser.parse_args()

    metadata_rows = _read_metadata(args.metadata)
    with open(args.config, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    configured_models = {
        name
        for name, model_cfg in config["models"].items()
        if model_cfg.get("enabled")
    }
    results, models, legacy_files = _read_metric_files(
        args.metrics_dir,
        {row["dataset"] for row in metadata_rows},
        configured_models,
    )
    summary_rows = _aggregate(metadata_rows, results, models, args.output)
    print(
        f"attribute summaries={len(summary_rows)} models={','.join(models) or 'none'} "
        f"metadata_samples={len(metadata_rows)} output={args.output}"
    )
    if legacy_files:
        print(f"excluded legacy metric files: {', '.join(legacy_files)}")
    for dataset in sorted({row["dataset"] for row in metadata_rows}):
        print(
            f"{dataset}: metadata_samples="
            f"{sum(row['dataset'] == dataset for row in metadata_rows)}"
        )


if __name__ == "__main__":
    main()
