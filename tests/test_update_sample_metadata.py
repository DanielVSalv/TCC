from __future__ import annotations

import csv

from src.data_prep.update_sample_metadata import FIELDNAMES, update_sample_metadata


def test_metadata_update_adds_derived_structure_and_marks_visual_fields_unknown(
    tmp_path,
) -> None:
    raw = tmp_path / "data" / "raw" / "scitsr"
    images = raw / "img" / "test"
    structures = raw / "structure" / "test"
    images.mkdir(parents=True)
    structures.mkdir(parents=True)
    (images / "merged.png").touch()
    (structures / "merged.json").write_text(
        """
        {"cells": [
          {"start_row": 0, "end_row": 1, "start_col": 0, "end_col": 0, "content": []},
          {"start_row": 0, "end_row": 0, "start_col": 1, "end_col": 1, "content": ["A"]}
        ]}
        """,
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        """
datasets:
  scitsr:
    enabled: true
    raw_dir: data/raw/scitsr
    split: test
    sample_size: 50
    sample_seed: 42
""",
        encoding="utf-8",
    )
    metadata_path = tmp_path / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as metadata:
        writer = csv.DictWriter(metadata, fieldnames=FIELDNAMES)
        writer.writeheader()
        original = {field: "existing" for field in FIELDNAMES}
        original.update(dataset="corporate", sample="manual.png")
        writer.writerow(original)

    added, total = update_sample_metadata(config, metadata_path, tmp_path)

    with metadata_path.open("r", newline="", encoding="utf-8") as metadata:
        rows = list(csv.DictReader(metadata))
    scitsr_row = next(row for row in rows if row["dataset"] == "scitsr")
    preserved = next(row for row in rows if row["dataset"] == "corporate")

    assert (added, total) == (1, 2)
    assert scitsr_row["n_rows"] == "2"
    assert scitsr_row["n_cols"] == "2"
    assert scitsr_row["cell_count"] == "2"
    assert scitsr_row["merged_cell_count"] == "1"
    assert scitsr_row["has_merged_cells"] == "true"
    assert scitsr_row["visible_rules"] == "not_reviewed"
    assert scitsr_row["hierarchical_header"] == "not_annotated"
    assert preserved["visual_basis"] == "existing"
