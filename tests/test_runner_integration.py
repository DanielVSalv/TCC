from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from src.models.base_runner import RawModelOutput
from src.normalization.normalize_output import (
    NormalizedCell,
    NormalizedTable,
    normalize_output,
)
from src.run_experiment import get_runner, validate_normalized_table


@pytest.mark.parametrize(
    ("model_name", "raw_content", "expected_rows", "expected_cols"),
    [
        (
            "tableformer",
            "<thead><tr><td></td><td colspan='2'></td></tr>"
            "<tr><td></td><td></td><td></td></tr></thead>"
            "<tr><td></td><td></td><td></td></tr>",
            3,
            3,
        ),
        (
            "tabletransformer",
            [
                {"label": "table row", "score": 0.99, "box": [0, 0, 20, 10]},
                {"label": "table row", "score": 0.99, "box": [0, 10, 20, 20]},
                {"label": "table column", "score": 0.99, "box": [0, 0, 10, 20]},
                {"label": "table column", "score": 0.99, "box": [10, 0, 20, 20]},
                {
                    "label": "table column header",
                    "score": 0.99,
                    "box": [0, 0, 20, 10],
                },
                {
                    "label": "table spanning cell",
                    "score": 0.99,
                    "box": [0, 10, 20, 20],
                },
            ],
            2,
            2,
        ),
        (
            "tabstructnet",
            "<annotations><object><cells>"
            "<tablecell><start_row>0</start_row><end_row>1</end_row>"
            "<start_col>0</start_col><end_col>0</end_col>"
            "<x0>0</x0><y0>0</y0><x1>10</x1><y1>20</y1></tablecell>"
            "<tablecell><start_row>0</start_row><end_row>0</end_row>"
            "<start_col>1</start_col><end_col>1</end_col></tablecell>"
            "</cells></object></annotations>",
            2,
            2,
        ),
        (
            "slanet",
            {
                "html": "<table><tr><td colspan='2'>ignored OCR</td></tr>"
                "<tr><td></td><td></td></tr></table>",
                "cell_bbox": [
                    [0, 0, 10, 10],
                    [10, 10, 20, 20],
                    [0, 10, 10, 20],
                ],
            },
            2,
            2,
        ),
    ],
)
def test_runner_output_normalizes_to_valid_table(
    model_name: str, raw_content, expected_rows: int, expected_cols: int
) -> None:
    raw_output = RawModelOutput(
        model_name=model_name,
        source_image_path=Path("sample.png"),
        content=raw_content,
    )

    normalized = normalize_output(raw_output.model_name, raw_output.content)
    validate_normalized_table(normalized, model_name, raw_output.source_image_path)

    assert isinstance(normalized, NormalizedTable)
    assert normalized.n_rows == expected_rows
    assert normalized.n_cols == expected_cols
    assert normalized.cells
    if model_name == "slanet":
        assert all(not cell.text for cell in normalized.cells)
    assert all(cell.row_span > 0 and cell.col_span > 0 for cell in normalized.cells)
    assert any(
        cell.row_span > 1 or cell.col_span > 1 for cell in normalized.cells
    )


def test_tableformer_normalization_keeps_matched_text_and_geometry() -> None:
    raw_content = {
        "html": "<table><thead><tr><td></td><td></td></tr></thead>"
        "<tr><td></td><td></td></tr></table>",
        "matched_cells": [
            {
                "start_row_offset_idx": 0,
                "start_col_offset_idx": 1,
                "column_header": True,
                "bbox": {"l": 20, "t": 0, "r": 40, "b": 10},
                "text_cell_bboxes": [
                    {"token": "B", "l": 30, "t": 2},
                    {"token": "Header", "l": 20, "t": 2},
                ],
            }
        ],
    }

    normalized = normalize_output("tableformer", raw_content)
    cell = next(cell for cell in normalized.cells if (cell.row, cell.col) == (0, 1))

    assert normalized.n_rows == 2
    assert normalized.n_cols == 2
    assert cell.text == "Header B"
    assert cell.is_header
    assert cell.bbox == (20.0, 0.0, 40.0, 10.0)


@pytest.mark.parametrize(
    ("model_name", "module_name", "class_name", "model_cfg", "expected_kwargs"),
    [
        (
            "tableformer",
            "src.models.tableformer_runner",
            "TableFormerRunner",
            {
                "variant": "accurate",
                "device": "cpu",
                "revision": "revision-sha",
            },
            {
                "variant": "accurate",
                "device": "cpu",
                "revision": "revision-sha",
            },
        ),
        (
            "tabstructnet",
            "src.models.tabstructnet_runner",
            "TabStructNetRunner",
            {"repo_dir": "models/tabstructnet"},
            {"repo_dir": Path("models/tabstructnet")},
        ),
        (
            "tabletransformer",
            "src.models.tabletransformer_runner",
            "TableTransformerRunner",
            {
                "score_threshold": 0.42,
                "device": "cpu",
                "revision": "revision-sha",
            },
            {
                "score_threshold": 0.42,
                "device": "cpu",
                "revision": "revision-sha",
            },
        ),
        (
            "slanet",
            "src.models.slanet_runner",
            "SLANetRunner",
            {
                "language": "en",
                "device": "cpu",
                "table_algorithm": "SLANet",
            },
            {
                "language": "en",
                "device": "cpu",
                "table_algorithm": "SLANet",
            },
        ),
    ],
)
def test_get_runner_passes_config_and_checkpoint(
    monkeypatch,
    tmp_path: Path,
    model_name: str,
    module_name: str,
    class_name: str,
    model_cfg: dict,
    expected_kwargs: dict,
) -> None:
    model_module = importlib.import_module(module_name)
    checkpoint_path = (
        "microsoft/table-transformer-structure-recognition"
        if model_name == "tabletransformer"
        else tmp_path / "checkpoint"
    )
    captured = {}

    class FakeRunner:
        def __init__(self, **kwargs) -> None:
            captured["kwargs"] = kwargs

        def load_model(self, checkpoint) -> None:
            captured["checkpoint"] = checkpoint

    monkeypatch.setattr(model_module, class_name, FakeRunner)

    get_runner(
        model_name,
        {**model_cfg, "checkpoint_path": str(checkpoint_path)},
    )

    assert captured["kwargs"] == expected_kwargs
    assert captured["checkpoint"] == checkpoint_path


def test_validation_rejects_empty_or_out_of_bounds_tables() -> None:
    with pytest.raises(ValueError, match="não produziu uma tabela válida"):
        validate_normalized_table(NormalizedTable(), "tableformer", Path("sample.png"))

    invalid = NormalizedTable(
        cells=[NormalizedCell(row=1, col=0)],
        n_rows=1,
        n_cols=1,
    )
    with pytest.raises(ValueError, match="coordenadas inválidas"):
        validate_normalized_table(invalid, "tableformer", Path("sample.png"))
