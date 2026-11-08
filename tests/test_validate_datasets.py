from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from src.data_prep import validate_datasets
from src.normalization.normalize_output import NormalizedCell, NormalizedTable


def test_validator_records_image_ground_truth_and_span_details(
    tmp_path: Path, monkeypatch
) -> None:
    image_path = tmp_path / "table.png"
    Image.new("RGB", (20, 10), "white").save(image_path)
    ground_truth_path = tmp_path / "table.json"
    ground_truth_path.write_text("{}", encoding="utf-8")
    table = NormalizedTable(
        cells=[NormalizedCell(row=0, col=0, row_span=2), NormalizedCell(row=0, col=1)],
        n_rows=2,
        n_cols=2,
    )
    monkeypatch.setattr(
        validate_datasets,
        "load_dataset_samples",
        lambda *_args: [(image_path, ground_truth_path)],
    )
    monkeypatch.setattr(validate_datasets, "load_ground_truth", lambda *_args: table)

    result = validate_datasets.validate_dataset(
        "scitsr",
        {"raw_dir": str(tmp_path), "split": "test", "sample_size": 1},
    )
    sample = result["samples"][0]

    assert result["available"] == 1
    assert result["selected_for_run"] == 1
    assert result["validated"] == result["loaded"] == 1
    assert sample["image"] == str(image_path)
    assert sample["ground_truth"] == str(ground_truth_path)
    assert sample["row_span_range"] == [1, 2]
    assert sample["column_span_range"] == [1, 1]
    assert sample["merged_cells"] == 1


def test_validator_writes_detailed_json_report(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "datasets:\n  scitsr:\n    enabled: true\n    raw_dir: data/scitsr\n"
        "    split: test\n    sample_size: 1\n",
        encoding="utf-8",
    )
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (10, 10), "white").save(image_path)
    gt_path = tmp_path / "sample.json"
    gt_path.write_text("{}", encoding="utf-8")
    table = NormalizedTable(cells=[NormalizedCell(row=0, col=0)], n_rows=1, n_cols=1)
    monkeypatch.setattr(validate_datasets, "load_dataset_samples", lambda *_args: [(image_path, gt_path)])
    monkeypatch.setattr(validate_datasets, "load_ground_truth", lambda *_args: table)
    monkeypatch.setattr(
        "sys.argv",
        [
            "validate_datasets",
            "--config",
            str(config_path),
            "--report",
            str(tmp_path / "validation.json"),
        ],
    )

    validate_datasets.main()
    report = json.loads((tmp_path / "validation.json").read_text(encoding="utf-8"))

    assert report["totals"]["loaded"] == 1
    assert report["datasets"][0]["samples"][0]["status"] == "loaded"
