from __future__ import annotations

import json

import pytest

from src import web_app


def test_safe_run_dir_rejects_path_traversal(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(web_app, "BENCHMARKS_DIR", tmp_path / "benchmarks")

    assert web_app._safe_run_dir("sample_01") == (tmp_path / "benchmarks" / "sample_01").resolve()
    with pytest.raises(ValueError, match="inválido"):
        web_app._safe_run_dir("../outside")


def test_audit_summary_counts_text_and_structure() -> None:
    summary = web_app._summarize_audit(
        [
            {
                "model": "tableformer",
                "nonempty_text_cells": "12",
                "empty_unmerged_grid_slots": "3",
                "merged_ranges": "1",
                "bold_cells": "4",
            },
            {
                "model": "tableformer",
                "nonempty_text_cells": "0",
                "empty_unmerged_grid_slots": "8",
                "merged_ranges": "2",
                "bold_cells": "0",
            },
        ]
    )

    assert summary == [
        {
            "model": "tableformer",
            "workbooks": 2,
            "with_text": 1,
            "text_cells": 12,
            "empty_slots": 11,
            "merged_ranges": 3,
            "bold_cells": 4,
        }
    ]


def test_dashboard_payload_reads_persisted_run_without_running_models(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path
    benchmarks = root / "results" / "benchmarks"
    run_dir = benchmarks / "existing-run"
    run_dir.mkdir(parents=True)
    scitsr_images = root / "data" / "raw" / "scitsr" / "img" / "test"
    scitsr_structures = root / "data" / "raw" / "scitsr" / "structure" / "test"
    scitsr_images.mkdir(parents=True)
    scitsr_structures.mkdir(parents=True)
    config_path = root / "config.yaml"
    config_path.write_text(
        """
experiment:
  name: demo
datasets:
  scitsr:
    enabled: true
    raw_dir: data/raw/scitsr
    split: test
    sample_size: 5
models:
  tableformer:
    enabled: true
""",
        encoding="utf-8",
    )
    (run_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "status": "partial_or_failed",
                "sample_count": 1,
                "sample_counts_by_dataset": {"scitsr": 1},
                "model_statuses": {"tableformer": "completed"},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "sample_manifest.csv").write_text(
        "dataset,split,sample\nscitsr,test,one.png\n", encoding="utf-8"
    )
    (run_dir / "summary_by_dataset_model.csv").write_text(
        "dataset,model,metric,n,mean,sample_std\nscitsr,tableformer,cell_f1,1,1.0,0.0\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(web_app, "ROOT", root)
    monkeypatch.setattr(web_app, "CONFIG_PATH", config_path)
    monkeypatch.setattr(web_app, "BENCHMARKS_DIR", benchmarks)

    payload = web_app.dashboard_payload()

    assert payload["report"]["run_id"] == "existing-run"
    assert payload["summary"][0]["mean"] == "1.0"
    scitsr = next(item for item in payload["dataset_readiness"] if item["name"] == "scitsr")
    assert scitsr["manifest_samples"] == 1
    assert scitsr["required_paths_present"]


def test_dashboard_payload_reads_latest_validation_report(tmp_path, monkeypatch) -> None:
    root = tmp_path
    validation_dir = root / "results" / "validation"
    validation_dir.mkdir(parents=True)
    report_path = validation_dir / "datasets.json"
    report_path.write_text(
        json.dumps(
            {
                "totals": {"available": 2, "loaded": 2, "ignored": 0},
                "datasets": [
                    {
                        "dataset": "scitsr",
                        "available": 2,
                        "validated": 2,
                        "loaded": 2,
                        "ignored": 0,
                        "unmatched_images": 0,
                        "unmatched_ground_truth": 0,
                        "samples": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config_path = root / "config.yaml"
    config_path.write_text(
        "experiment:\n  name: demo\ndatasets: {}\nmodels: {}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(web_app, "ROOT", root)
    monkeypatch.setattr(web_app, "CONFIG_PATH", config_path)
    monkeypatch.setattr(web_app, "BENCHMARKS_DIR", root / "results" / "benchmarks")

    payload = web_app.dashboard_payload()

    assert payload["validation"]["totals"]["loaded"] == 2
    assert payload["validation"]["report_path"] == "results\\validation\\datasets.json"
