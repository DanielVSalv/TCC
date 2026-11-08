"""Run fixed, paired model benchmarks and write aggregate/audit reports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from openpyxl import load_workbook

from src.run_experiment import load_config, load_dataset_samples
from src.evaluation.summarize_by_attributes import (
    _aggregate as _aggregate_by_attributes,
    _read_metadata,
    _read_metric_files,
)

METRICS = ("teds", "s_teds", "cell_f1")
MODELS = ("tableformer", "tabstructnet", "tabletransformer", "slanet")
EXACT_SIGN_FLIP_MAX_PAIRS = 15
MONTE_CARLO_SIGN_FLIP_ITERATIONS = 10_000


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _code_fingerprint(root: Path, config_path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted((root / "src").rglob("*.py"))
    files.extend(sorted((root / "environment").glob("requirements*.txt")))
    files.append(config_path)
    for path in files:
        try:
            label = path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            label = str(path.resolve())
        digest.update(label.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_sample_manifest(
    config: dict, root: Path, manifest_path: Path
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for dataset_name, dataset_cfg in config["datasets"].items():
        if not dataset_cfg.get("enabled"):
            continue
        samples = load_dataset_samples(dataset_name, dataset_cfg)
        if not samples:
            raise ValueError(f"Dataset habilitado sem amostras disponíveis: {dataset_name}")
        for index, (image_path, ground_truth_ref) in enumerate(samples):
            if not image_path.is_file():
                raise FileNotFoundError(f"Imagem ausente: {image_path}")
            if isinstance(ground_truth_ref, Path) and not ground_truth_ref.is_file():
                raise FileNotFoundError(f"Gabarito ausente: {ground_truth_ref}")
            rows.append(
                {
                    "dataset": dataset_name,
                    "split": str(dataset_cfg.get("split") or "curated"),
                    "sample_index": str(index),
                    "sample": image_path.name,
                    "image": str(image_path.resolve()),
                    "ground_truth": (
                        str(ground_truth_ref.resolve())
                        if isinstance(ground_truth_ref, Path)
                        else str(ground_truth_ref.get("filename", "inline_annotation"))
                    ),
                }
            )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _filter_metadata_to_manifest(
    metadata_rows: list[dict[str, str]], manifest_rows: list[dict[str, str]]
) -> list[dict[str, str]]:
    expected = {
        (row["dataset"], Path(row["sample"]).name) for row in manifest_rows
    }
    available = {
        (row["dataset"], row["sample"]) for row in metadata_rows
    }
    missing = sorted(expected - available)
    if missing:
        raise ValueError(
            "Metadados ausentes para amostras do manifesto: "
            + ", ".join(f"{dataset}/{sample}" for dataset, sample in missing)
        )
    return [
        row
        for row in metadata_rows
        if (row["dataset"], row["sample"]) in expected
    ]


def _checkpoint_metadata(
    model_name: str, model_cfg: dict[str, Any], root: Path
) -> dict[str, Any]:
    configured = model_cfg.get("checkpoint_path")
    checkpoint: dict[str, Any] = {"configured": configured}
    if model_name == "slanet":
        checkpoint.update(
            {
                "repository": "PaddleOCR PP-StructureV2",
                "model_name": (
                    "en_ppstructure_mobile_v2.0_SLANet"
                    if model_cfg.get("language", "en") == "en"
                    else "ch_ppstructure_mobile_v2.0_SLANet"
                ),
                "model_url": (
                    "https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/"
                    "paddle3.0b2/en_ppstructure_mobile_v2.0_SLANet_infer.tar"
                    if model_cfg.get("language", "en") == "en"
                    else "https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/"
                    "paddle3.0b2/ch_ppstructure_mobile_v2.0_SLANet_infer.tar"
                ),
                "model_source": "PaddleOCR 2.10 default inference model",
                "table_algorithm": model_cfg.get("table_algorithm", "SLANet"),
                "language": model_cfg.get("language", "en"),
            }
        )
    if configured is None:
        if model_name == "tableformer":
            checkpoint.update(
                {
                    "repository": "ds4sd/docling-models",
                    "revision": model_cfg.get("revision", "v2.1.0"),
                    "variant": model_cfg.get("variant", "fast"),
                }
            )
        return checkpoint

    if model_name == "tabletransformer":
        checkpoint["repository"] = configured
        checkpoint["revision"] = model_cfg.get("revision")
        return checkpoint

    path = Path(configured)
    if not path.is_absolute():
        path = root / path
    checkpoint["resolved_path"] = str(path.resolve())
    if path.is_file():
        checkpoint["size_bytes"] = path.stat().st_size
        checkpoint["sha256"] = _sha256_file(path)
    elif path.is_dir():
        checkpoint["directory_exists"] = True
    else:
        checkpoint["exists"] = False
    return checkpoint


def _environment_metadata(python_executable: Path) -> dict[str, Any]:
    freeze = subprocess.run(
        [str(python_executable), "-m", "pip", "freeze"],
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "python_executable": str(python_executable.resolve()),
        "python_version": subprocess.run(
            [
                str(python_executable),
                "-c",
                "import platform; print(platform.python_version())",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "platform": platform.platform(),
        "pip_freeze_exit_code": freeze.returncode,
        "pip_freeze": freeze.stdout.splitlines(),
        "pip_freeze_stderr": freeze.stderr.strip() or None,
    }


def _read_metric_rows(metrics_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(metrics_dir.glob("*.csv")):
        dataset, _, model = metrics_path.stem.rpartition("_")
        with open(metrics_path, "r", newline="", encoding="utf-8") as metrics_file:
            for row in csv.DictReader(metrics_file):
                parsed: dict[str, Any] = {
                    "dataset": dataset,
                    "model": model,
                    "sample": row["sample"],
                    "status": row.get("status", "completed"),
                    "error_type": row.get("error_type"),
                    "error": row.get("error"),
                    "predicted_rows": row.get("predicted_rows"),
                    "predicted_columns": row.get("predicted_columns"),
                    "predicted_cells": row.get("predicted_cells"),
                    "ground_truth_rows": row.get("ground_truth_rows"),
                    "ground_truth_columns": row.get("ground_truth_columns"),
                    "ground_truth_cells": row.get("ground_truth_cells"),
                    "cells_with_text": row.get("cells_with_text"),
                    "text_coverage": row.get("text_coverage"),
                    "recognized_words": row.get("recognized_words"),
                    "assigned_words": row.get("assigned_words"),
                    "unassigned_words": row.get("unassigned_words"),
                }
                for metric in METRICS:
                    value = row.get(metric, "")
                    parsed[metric] = float(value) if value not in ("", None) else None
                rows.append(parsed)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sample_std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    average = sum(values) / len(values)
    return math.sqrt(sum((value - average) ** 2 for value in values) / (len(values) - 1))


def _percentile(sorted_values: list[float], probability: float) -> float:
    position = probability * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def _bootstrap_ci(
    values: list[float], rng: random.Random, iterations: int = 10_000
) -> tuple[float | None, float | None]:
    if len(values) < 2:
        return None, None
    means = sorted(
        sum(rng.choice(values) for _ in values) / len(values)
        for _ in range(iterations)
    )
    return _percentile(means, 0.025), _percentile(means, 0.975)


def _sign_flip_p_value(
    differences: list[float], rng: random.Random
) -> tuple[float, str, int]:
    observed = abs(sum(differences) / len(differences))
    if len(differences) <= EXACT_SIGN_FLIP_MAX_PAIRS:
        extreme_count = 0
        total = 1 << len(differences)
        for signs in itertools.product((-1, 1), repeat=len(differences)):
            permutation_mean = sum(
                sign * difference
                for sign, difference in zip(signs, differences)
            ) / len(differences)
            if abs(permutation_mean) >= observed - 1e-12:
                extreme_count += 1
        return extreme_count / total, "exact", total

    extreme_count = 0
    for _ in range(MONTE_CARLO_SIGN_FLIP_ITERATIONS):
        permutation_mean = sum(
            rng.choice((-1, 1)) * difference for difference in differences
        ) / len(differences)
        if abs(permutation_mean) >= observed - 1e-12:
            extreme_count += 1
    return (
        (extreme_count + 1) / (MONTE_CARLO_SIGN_FLIP_ITERATIONS + 1),
        "monte_carlo",
        MONTE_CARLO_SIGN_FLIP_ITERATIONS,
    )


def _summarize_metrics(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        for metric in METRICS:
            value = row[metric]
            if value is not None:
                groups.setdefault((row["dataset"], row["model"], metric), []).append(value)

    output = []
    for key, values in sorted(groups.items()):
        rng = random.Random(f"{seed}:{key}")
        ci_low, ci_high = _bootstrap_ci(values, rng)
        output.append(
            {
                "dataset": key[0],
                "model": key[1],
                "metric": key[2],
                "n": len(values),
                "mean": sum(values) / len(values),
                "sample_std": _sample_std(values),
                "median": _percentile(sorted(values), 0.5),
                "min": min(values),
                "max": max(values),
                "bootstrap_95_ci_low": ci_low,
                "bootstrap_95_ci_high": ci_high,
                "ci_note": (
                    "exploratory: small paired sample"
                    if len(values) < 10
                    else "percentile bootstrap over tables"
                ),
            }
        )
    return output


def _paired_comparisons(
    rows: list[dict[str, Any]], seed: int
) -> list[dict[str, Any]]:
    samples: dict[tuple[str, str, str], dict[str, float]] = {}
    for row in rows:
        for metric in METRICS:
            value = row[metric]
            if value is not None:
                samples.setdefault((row["dataset"], metric, row["sample"]), {})[
                    row["model"]
                ] = value

    grouped: dict[tuple[str, str, str, str], list[tuple[str, float]]] = {}
    dataset_metrics = sorted({(dataset, metric) for dataset, metric, _ in samples})
    for dataset, metric in dataset_metrics:
        models = sorted(
            {
                model
                for (row_dataset, row_metric, _), values in samples.items()
                if row_dataset == dataset and row_metric == metric
                for model in values
            }
        )
        for model_a, model_b in itertools.combinations(models, 2):
            pairs = []
            for (row_dataset, row_metric, sample), values in samples.items():
                if row_dataset == dataset and row_metric == metric:
                    if model_a in values and model_b in values:
                        pairs.append((sample, values[model_a] - values[model_b]))
            if pairs:
                grouped[(dataset, metric, model_a, model_b)] = pairs

    output = []
    for key, pairs in sorted(grouped.items()):
        differences = [difference for _, difference in pairs]
        mean_difference = sum(differences) / len(differences)
        rng = random.Random(f"{seed}:paired:{key}")
        ci_low, ci_high = _bootstrap_ci(differences, rng)
        sign_flip_p, sign_flip_method, sign_flip_permutations = _sign_flip_p_value(
            differences, rng
        )
        output.append(
            {
                "dataset": key[0],
                "metric": key[1],
                "model_a": key[2],
                "model_b": key[3],
                "difference_definition": "model_a - model_b",
                "n_paired": len(differences),
                "mean_paired_difference": mean_difference,
                "paired_difference_std": _sample_std(differences),
                "bootstrap_95_ci_low": ci_low,
                "bootstrap_95_ci_high": ci_high,
                "sign_flip_p_two_sided": sign_flip_p,
                "sign_flip_method": sign_flip_method,
                "sign_flip_permutations": sign_flip_permutations,
                "paired_samples": json.dumps([sample for sample, _ in pairs]),
                "inference_note": (
                    "exploratory only; very small paired n"
                    if len(differences) < 10
                    else f"{sign_flip_method} paired sign-flip test and table-level bootstrap CI"
                ),
            }
        )
    return output


def _audit_spreadsheets(spreadsheets_dir: Path) -> list[dict[str, Any]]:
    audit: list[dict[str, Any]] = []
    for path in sorted(spreadsheets_dir.rglob("*.xlsx")):
        workbook = load_workbook(path, data_only=True, read_only=False)
        try:
            sheet = workbook.active
            area = sheet.max_row * sheet.max_column
            nonempty = [
                cell
                for row in sheet.iter_rows()
                for cell in row
                if cell.value is not None and not isinstance(cell.value, str) or
                (isinstance(cell.value, str) and bool(cell.value.strip()))
            ]
            merged_slots = sum(
                (merged.max_row - merged.min_row + 1)
                * (merged.max_col - merged.min_col + 1)
                - 1
                for merged in sheet.merged_cells.ranges
            )
            first_row = [
                sheet.cell(1, column).value
                for column in range(1, sheet.max_column + 1)
            ]
            audit.append(
                {
                    "dataset": path.parent.parent.name,
                    "model": path.parent.name,
                    "sample": path.stem,
                    "workbook": str(path),
                    "rows": sheet.max_row,
                    "columns": sheet.max_column,
                    "nonempty_cells": len(nonempty),
                    "nonempty_text_cells": sum(
                        isinstance(cell.value, str) and bool(cell.value.strip())
                        for cell in nonempty
                    ),
                    "empty_unmerged_grid_slots": max(area - len(nonempty) - merged_slots, 0),
                    "merged_ranges": len(sheet.merged_cells.ranges),
                    "merged_range_addresses": ";".join(
                        str(merged) for merged in sheet.merged_cells.ranges
                    ),
                    "bold_cells": sum(cell.font.bold for cell in nonempty),
                    "first_row_values": json.dumps(first_row, ensure_ascii=False),
                    "text_preview": json.dumps(
                        [
                            str(cell.value)[:100]
                            for cell in nonempty
                            if isinstance(cell.value, str) and cell.value.strip()
                        ][:10],
                        ensure_ascii=False,
                    ),
                }
            )
        finally:
            workbook.close()
    return audit


def run_benchmark(
    root: Path,
    config_path: Path,
    run_id: str,
    selected_models: list[str],
) -> int:
    config = load_config(config_path)
    run_dir = root / config["experiment"]["output_dir"] / "benchmarks" / run_id
    output_dir = run_dir / "output"
    logs_dir = run_dir / "logs"
    metadata_dir = run_dir / "metadata"
    for path in (output_dir, logs_dir, metadata_dir):
        path.mkdir(parents=True, exist_ok=True)

    snapshot_path = run_dir / "config_snapshot.yaml"
    snapshot_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    manifest_path = run_dir / "sample_manifest.csv"
    sample_rows = _write_sample_manifest(config, root, manifest_path)
    manifest_hash = _sha256_file(manifest_path)
    code_hash = _code_fingerprint(root, config_path)
    seed = int(config.get("experiment", {}).get("seed", 0))
    overall_status = 0
    model_statuses: dict[str, dict[str, Any]] = {}

    for model_name in selected_models:
        model_cfg = config["models"].get(model_name)
        if not model_cfg or not model_cfg.get("enabled"):
            model_statuses[model_name] = {"status": "disabled_or_missing"}
            overall_status = 1
            continue

        environment_path = Path(model_cfg.get("venv", "")) / (
            "Scripts/python.exe" if platform.system() == "Windows" else "bin/python"
        )
        if not environment_path.is_absolute():
            environment_path = root / environment_path
        started_at = datetime.now(timezone.utc).isoformat()
        started = time.monotonic()
        metadata = {
            "run_id": run_id,
            "model": model_name,
            "status": "running",
            "started_at_utc": started_at,
            "config_snapshot": str(snapshot_path),
            "config_sha256": _sha256_file(snapshot_path),
            "code_version": f"sha256:{code_hash}",
            "sample_manifest": str(manifest_path),
            "sample_manifest_sha256": manifest_hash,
            "datasets_and_sample_counts": {
                dataset: sum(row["dataset"] == dataset for row in sample_rows)
                for dataset in config["datasets"]
                if any(row["dataset"] == dataset for row in sample_rows)
            },
            "model_parameters": model_cfg,
            "checkpoint": _checkpoint_metadata(model_name, model_cfg, root),
            "environment": (
                _environment_metadata(environment_path)
                if environment_path.is_file()
                else {"python_executable_missing": str(environment_path)}
            ),
        }
        model_statuses[model_name] = metadata

        if not environment_path.is_file():
            metadata["status"] = "environment_missing"
            metadata["return_code"] = None
            overall_status = 1
            (metadata_dir / f"{model_name}.json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            continue

        command = [
            str(environment_path),
            "-m",
            "src.run_experiment",
            "--config",
            str(config_path.resolve()),
            "--model",
            model_name,
            "--sample-manifest",
            str(manifest_path.resolve()),
            "--output-dir",
            str(output_dir.resolve()),
        ]
        metadata["command"] = command
        log_path = logs_dir / f"{model_name}.log"
        with open(log_path, "w", encoding="utf-8") as log_file:
            completed = subprocess.run(
                command,
                cwd=root,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
        metadata["return_code"] = completed.returncode
        metadata["elapsed_seconds"] = round(time.monotonic() - started, 3)
        metadata["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        metadata["log"] = str(log_path)
        metadata["status"] = "completed" if completed.returncode == 0 else "failed"
        model_rows = [
            row
            for row in _read_metric_rows(output_dir / "metrics")
            if row["model"] == model_name
        ]
        metadata["sample_outcomes"] = {
            status: sum(row["status"] == status for row in model_rows)
            for status in sorted({row["status"] for row in model_rows})
        }
        if completed.returncode:
            overall_status = 1
        (metadata_dir / f"{model_name}.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    metric_rows = _read_metric_rows(output_dir / "metrics")
    summary_rows = _summarize_metrics(metric_rows, seed)
    paired_rows = _paired_comparisons(metric_rows, seed)
    xlsx_rows = _audit_spreadsheets(output_dir / "spreadsheets")
    attribute_metadata_path = root / "data" / "ground_truth" / "sample_metadata.csv"
    attribute_summary_path = run_dir / "attribute_summary.csv"
    if attribute_metadata_path.is_file():
        attribute_metadata = _filter_metadata_to_manifest(
            _read_metadata(attribute_metadata_path), sample_rows
        )
        attribute_results, attribute_models, _ = _read_metric_files(
            output_dir / "metrics",
            {row["dataset"] for row in attribute_metadata},
            set(selected_models),
        )
        _aggregate_by_attributes(
            attribute_metadata,
            attribute_results,
            attribute_models,
            attribute_summary_path,
        )
    _write_csv(
        run_dir / "metrics_by_sample.csv",
        metric_rows,
        [
            "dataset",
            "model",
            "sample",
            "status",
            "error_type",
            "error",
            *METRICS,
            "predicted_rows",
            "predicted_columns",
            "predicted_cells",
            "ground_truth_rows",
            "ground_truth_columns",
            "ground_truth_cells",
            "cells_with_text",
            "text_coverage",
            "recognized_words",
            "assigned_words",
            "unassigned_words",
        ],
    )
    _write_csv(
        run_dir / "summary_by_dataset_model.csv",
        summary_rows,
        [
            "dataset",
            "model",
            "metric",
            "n",
            "mean",
            "sample_std",
            "median",
            "min",
            "max",
            "bootstrap_95_ci_low",
            "bootstrap_95_ci_high",
            "ci_note",
        ],
    )
    _write_csv(
        run_dir / "paired_comparisons.csv",
        paired_rows,
        [
            "dataset",
            "metric",
            "model_a",
            "model_b",
            "difference_definition",
            "n_paired",
            "mean_paired_difference",
            "paired_difference_std",
            "bootstrap_95_ci_low",
            "bootstrap_95_ci_high",
            "sign_flip_p_two_sided",
            "sign_flip_method",
            "sign_flip_permutations",
            "paired_samples",
            "inference_note",
        ],
    )
    _write_csv(
        run_dir / "xlsx_audit.csv",
        xlsx_rows,
        [
            "dataset",
            "model",
            "sample",
            "workbook",
            "rows",
            "columns",
            "nonempty_cells",
            "nonempty_text_cells",
            "empty_unmerged_grid_slots",
            "merged_ranges",
            "merged_range_addresses",
            "bold_cells",
            "first_row_values",
            "text_preview",
        ],
    )
    report = {
        "run_id": run_id,
        "status": "completed" if overall_status == 0 else "partial_or_failed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_version": f"sha256:{code_hash}",
        "seed": seed,
        "sample_manifest_sha256": manifest_hash,
        "sample_count": len(sample_rows),
        "sample_counts_by_dataset": {
            dataset: sum(row["dataset"] == dataset for row in sample_rows)
            for dataset in config["datasets"]
            if any(row["dataset"] == dataset for row in sample_rows)
        },
        "model_statuses": {
            name: metadata.get("status") for name, metadata in model_statuses.items()
        },
        "sample_outcomes_by_model": {
            model: {
                status: sum(
                    row["model"] == model and row["status"] == status
                    for row in metric_rows
                )
                for status in sorted(
                    {row["status"] for row in metric_rows if row["model"] == model}
                )
            }
            for model in selected_models
        },
        "paired_inference_note": (
            "Paired sign-flip p-values are exact up to 15 pairs and use 10,000 "
            "seeded Monte Carlo sign flips above that; intervals and tests remain "
            "exploratory and do not establish model superiority."
        ),
        "outputs": {
            "metrics_by_sample": "metrics_by_sample.csv",
            "summary_by_dataset_model": "summary_by_dataset_model.csv",
            "paired_comparisons": "paired_comparisons.csv",
            "xlsx_audit": "xlsx_audit.csv",
            "attribute_summary": (
                "attribute_summary.csv"
                if attribute_summary_path.is_file()
                else None
            ),
        },
    }
    (run_dir / "run_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return overall_status


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Executa um benchmark pareado em venvs isolados e gera relatórios."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/experiment_config.yaml"))
    parser.add_argument("--run-id", required=True, help="Identificador novo e único da execução.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODELS,
        default=list(MODELS),
        help="Modelos a incluir; por padrão tenta todos os configurados.",
    )
    args = parser.parse_args()
    if any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in args.run_id):
        raise ValueError("--run-id aceita apenas letras, números, hífen e underscore.")
    raise SystemExit(
        run_benchmark(Path.cwd(), args.config.resolve(), args.run_id, args.models)
    )


if __name__ == "__main__":
    main()
