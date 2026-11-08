from __future__ import annotations

from copy import copy
from pathlib import Path

from openpyxl import Workbook

from src.evaluation.benchmark import (
    _audit_spreadsheets,
    _checkpoint_metadata,
    _filter_metadata_to_manifest,
    _paired_comparisons,
    _summarize_metrics,
)


def test_summary_and_paired_comparison_use_same_samples() -> None:
    rows = [
        {
            "dataset": "corporate",
            "model": "tableformer",
            "sample": "a.png",
            "teds": 0.8,
            "s_teds": 0.9,
            "cell_f1": 0.7,
        },
        {
            "dataset": "corporate",
            "model": "tableformer",
            "sample": "b.png",
            "teds": 1.0,
            "s_teds": 0.8,
            "cell_f1": 0.9,
        },
        {
            "dataset": "corporate",
            "model": "tabstructnet",
            "sample": "a.png",
            "teds": 0.6,
            "s_teds": 0.8,
            "cell_f1": 0.5,
        },
        {
            "dataset": "corporate",
            "model": "tabstructnet",
            "sample": "b.png",
            "teds": 0.8,
            "s_teds": 0.7,
            "cell_f1": 0.7,
        },
    ]

    summary = _summarize_metrics(rows, seed=42)
    teds_summary = next(
        row
        for row in summary
        if row["model"] == "tableformer" and row["metric"] == "teds"
    )
    paired = next(
        row
        for row in _paired_comparisons(rows, seed=42)
        if row["metric"] == "teds"
    )

    assert teds_summary["n"] == 2
    assert teds_summary["mean"] == 0.9
    assert teds_summary["sample_std"] > 0
    assert teds_summary["bootstrap_95_ci_low"] <= 0.9
    assert teds_summary["bootstrap_95_ci_high"] >= 0.9
    assert paired["n_paired"] == 2
    assert paired["mean_paired_difference"] == 0.2
    assert paired["sign_flip_p_two_sided"] == 0.5
    assert paired["sign_flip_method"] == "exact"
    assert paired["paired_samples"] == '["a.png", "b.png"]'


def test_large_paired_comparison_uses_bounded_monte_carlo() -> None:
    rows = [
        {
            "dataset": "scitsr",
            "model": model,
            "sample": f"sample-{index}.png",
            "teds": 0.8 if model == "tableformer" else 0.6,
            "s_teds": 0.9 if model == "tableformer" else 0.7,
            "cell_f1": 0.7 if model == "tableformer" else 0.5,
        }
        for index in range(50)
        for model in ("tableformer", "slanet")
    ]

    paired = _paired_comparisons(rows, seed=42)

    assert paired
    assert all(row["n_paired"] == 50 for row in paired)
    assert all(row["sign_flip_method"] == "monte_carlo" for row in paired)
    assert all(row["sign_flip_permutations"] == 10_000 for row in paired)
    assert all(0 < row["sign_flip_p_two_sided"] <= 1 for row in paired)


def test_attribute_metadata_is_limited_to_frozen_manifest() -> None:
    metadata_rows = [
        {"dataset": "scitsr", "sample": "selected.png"},
        {"dataset": "scitsr", "sample": "stale.png"},
    ]
    manifest_rows = [
        {"dataset": "scitsr", "sample": "selected.png"},
    ]

    assert _filter_metadata_to_manifest(metadata_rows, manifest_rows) == [
        {"dataset": "scitsr", "sample": "selected.png"}
    ]


def test_xlsx_audit_counts_text_merges_and_empty_slots(tmp_path) -> None:
    workbook_path = tmp_path / "corporate" / "tableformer" / "sample.xlsx"
    workbook_path.parent.mkdir(parents=True)
    workbook = Workbook()
    worksheet = workbook.active
    worksheet["A1"] = "Header"
    font = copy(worksheet["A1"].font)
    font.bold = True
    worksheet["A1"].font = font
    worksheet.merge_cells("A1:B1")
    worksheet["A2"] = "Value"
    worksheet["B2"] = ""
    workbook.save(workbook_path)
    workbook.close()

    audit = _audit_spreadsheets(tmp_path)[0]

    assert audit["rows"] == 2
    assert audit["columns"] == 2
    assert audit["nonempty_text_cells"] == 2
    assert audit["merged_ranges"] == 1
    assert audit["empty_unmerged_grid_slots"] == 1
    assert audit["bold_cells"] == 1
    assert audit["text_preview"] == '["Header", "Value"]'


def test_slanet_checkpoint_metadata_identifies_default_model() -> None:
    metadata = _checkpoint_metadata(
        "slanet",
        {
            "checkpoint_path": None,
            "table_algorithm": "SLANet",
            "language": "en",
        },
        Path("."),
    )

    assert metadata["model_name"] == "en_ppstructure_mobile_v2.0_SLANet"
    assert metadata["table_algorithm"] == "SLANet"
    assert metadata["language"] == "en"
    assert metadata["model_url"].endswith("en_ppstructure_mobile_v2.0_SLANet_infer.tar")
