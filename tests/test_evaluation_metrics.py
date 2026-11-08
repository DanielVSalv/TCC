from __future__ import annotations

import pytest

from src.evaluation import teds
from src.evaluation.cell_f1 import compute_cell_f1
from src.evaluation.s_teds import compute_s_teds
from src.evaluation.teds import compute_teds
from src.normalization.normalize_output import NormalizedCell, NormalizedTable


def _table(
    cells: list[NormalizedCell], n_rows: int = 1, n_cols: int = 3
) -> NormalizedTable:
    return NormalizedTable(cells=cells, n_rows=n_rows, n_cols=n_cols)


def test_identical_table_scores_perfectly() -> None:
    ground_truth = _table(
        [NormalizedCell(0, 0, text="Revenue"), NormalizedCell(0, 1, text="100")],
        n_cols=2,
    )

    assert compute_teds(ground_truth, ground_truth) == 1.0
    assert compute_s_teds(ground_truth, ground_truth) == 1.0
    assert compute_cell_f1(ground_truth, ground_truth).f1 == 1.0


def test_teds_is_clamped_at_zero_when_edit_distance_exceeds_tree_size(
    monkeypatch,
) -> None:
    class OversizedDistance:
        def __init__(self, *_args) -> None:
            pass

        def compute_edit_distance(self) -> int:
            return 100

    monkeypatch.setattr(teds, "APTED", OversizedDistance)
    table = _table([NormalizedCell(0, 0, text="A")], n_cols=1)

    assert teds.compute_teds(table, table) == 0.0


def test_wrong_text_only_changes_text_sensitive_teds() -> None:
    ground_truth = _table(
        [NormalizedCell(0, 0, text="A"), NormalizedCell(0, 1, text="B")],
        n_cols=2,
    )
    predicted = _table(
        [NormalizedCell(0, 0, text="X"), NormalizedCell(0, 1, text="B")],
        n_cols=2,
    )

    # Matches PubTabNet reference TEDS for <td>A</td><td>B</td> vs X/B.
    assert compute_teds(predicted, ground_truth) == pytest.approx(2 / 3)
    assert compute_s_teds(predicted, ground_truth) == 1.0
    assert compute_cell_f1(predicted, ground_truth).f1 == 1.0


def test_coordinate_shift_is_penalized_by_teds_and_cell_f1() -> None:
    ground_truth = _table(
        [NormalizedCell(0, 0, text="A"), NormalizedCell(0, 1, text="B")],
        n_cols=2,
    )
    predicted = _table(
        [NormalizedCell(0, 1, text="A"), NormalizedCell(0, 2, text="B")],
        n_cols=3,
    )

    assert compute_teds(predicted, ground_truth) == 0.75
    assert compute_s_teds(predicted, ground_truth) == 0.75
    assert compute_cell_f1(predicted, ground_truth).f1 == 0.5


def test_internal_empty_column_matches_pubtabnet_teds_reference() -> None:
    ground_truth = _table(
        [NormalizedCell(0, 0, text="A"), NormalizedCell(0, 1, text="B")],
        n_cols=2,
    )
    predicted = _table(
        [NormalizedCell(0, 0, text="A"), NormalizedCell(0, 2, text="B")],
        n_cols=3,
    )

    # The missing logical slot is serialized as an empty <td> node in the TEDS tree.
    assert compute_teds(predicted, ground_truth) == 0.75
    assert compute_cell_f1(predicted, ground_truth).f1 == 0.5


def test_missing_cell_is_counted_as_deletion() -> None:
    ground_truth = _table(
        [NormalizedCell(0, 0, text="A"), NormalizedCell(0, 1, text="B")],
        n_cols=2,
    )
    predicted = _table([NormalizedCell(0, 0, text="A")], n_cols=2)

    assert compute_teds(predicted, ground_truth) == pytest.approx(2 / 3)
    assert compute_s_teds(predicted, ground_truth) == 1.0
    assert compute_cell_f1(predicted, ground_truth).f1 == pytest.approx(2 / 3)


def test_correct_merge_scores_perfectly_and_wrong_merge_is_penalized() -> None:
    merged = _table([NormalizedCell(0, 0, col_span=2, text="AB")], n_cols=2)
    separate = _table(
        [NormalizedCell(0, 0, text="A"), NormalizedCell(0, 1, text="B")],
        n_cols=2,
    )

    assert compute_teds(merged, merged) == 1.0
    assert compute_s_teds(merged, merged) == 1.0
    assert compute_cell_f1(merged, merged).f1 == 1.0
    assert compute_teds(separate, merged) < 1.0
    assert compute_s_teds(separate, merged) < 1.0
    assert compute_cell_f1(separate, merged).f1 == 0.0


def test_s_teds_ignores_unannotated_header_tag_but_teds_preserves_it() -> None:
    ground_truth = _table(
        [
            NormalizedCell(0, 0, text="H"),
            NormalizedCell(0, 1, text="V"),
        ],
        n_cols=2,
    )
    predicted = _table(
        [
            NormalizedCell(0, 0, is_header=True, text="H"),
            NormalizedCell(0, 1, text="V"),
        ],
        n_cols=2,
    )

    assert compute_teds(predicted, ground_truth) == pytest.approx(2 / 3)
    assert compute_s_teds(predicted, ground_truth) == 1.0
    assert compute_cell_f1(predicted, ground_truth).f1 == 1.0
