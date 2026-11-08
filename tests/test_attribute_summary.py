from __future__ import annotations

import csv

from src.evaluation.summarize_by_attributes import (
    _aggregate,
    _read_metric_files,
)
from src.evaluation.teds import METRIC_PROTOCOL_VERSION


def _metadata(sample: str, merged: str) -> dict[str, str]:
    return {
        "dataset": "corporate",
        "sample": sample,
        "n_rows": "2",
        "n_cols": "2",
        "cell_count": "4",
        "merged_cell_count": "1" if merged == "true" else "0",
        "has_merged_cells": merged,
        "hierarchical_header": "false",
        "visible_rules": "full_grid",
        "row_banding": "false",
        "scanned_or_digital": "digital",
        "complexity": "low",
    }


def test_legacy_metric_csv_is_excluded(tmp_path) -> None:
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    legacy_path = metrics_dir / "corporate_tableformer.csv"
    legacy_path.write_text(
        "sample,teds,s_teds,cell_f1\na.png,0.9,0.9,0.9\n",
        encoding="utf-8",
    )

    results, models, legacy_files = _read_metric_files(metrics_dir, {"corporate"})

    assert results == {}
    assert models == []
    assert legacy_files == [legacy_path.name]


def test_versioned_results_are_aggregated_with_group_counts(tmp_path) -> None:
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    metrics_path = metrics_dir / "corporate_tableformer.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as metrics_file:
        writer = csv.DictWriter(
            metrics_file,
            fieldnames=["sample", "metric_protocol", "teds", "s_teds", "cell_f1"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "sample": "a.png",
                    "metric_protocol": METRIC_PROTOCOL_VERSION,
                    "teds": 0.8,
                    "s_teds": 0.9,
                    "cell_f1": 0.75,
                },
                {
                    "sample": "b.png",
                    "metric_protocol": METRIC_PROTOCOL_VERSION,
                    "teds": "",
                    "s_teds": 0.7,
                    "cell_f1": 0.5,
                },
            ]
        )

    results, models, legacy_files = _read_metric_files(metrics_dir, {"corporate"})
    output_path = tmp_path / "summary.csv"
    summary = _aggregate(
        [_metadata("a.png", "true"), _metadata("b.png", "false")],
        results,
        models,
        output_path,
    )
    merged_group = next(
        row
        for row in summary
        if row["scope"] == "all"
        and row["attribute"] == "has_merged_cells"
        and row["value"] == "true"
    )

    assert models == ["tableformer"]
    assert legacy_files == []
    assert merged_group["sample_count"] == 1
    assert merged_group["evaluated_count"] == 1
    assert merged_group["mean_teds"] == 0.8
    assert merged_group["mean_cell_f1"] == 0.75
    assert output_path.is_file()


def test_models_without_current_scores_remain_visible(tmp_path) -> None:
    output_path = tmp_path / "summary.csv"
    summary = _aggregate(
        [_metadata("a.png", "true")],
        {},
        ["tableformer", "tabletransformer"],
        output_path,
    )
    merged_rows = [
        row
        for row in summary
        if row["scope"] == "all"
        and row["attribute"] == "has_merged_cells"
        and row["value"] == "true"
    ]

    assert {row["model"] for row in merged_rows} == {
        "tableformer",
        "tabletransformer",
    }
    assert all(row["sample_count"] == 1 for row in merged_rows)
    assert all(row["evaluated_count"] == 0 for row in merged_rows)
    assert all(row["metrics_status"] == "current" for row in merged_rows)
