from __future__ import annotations

from pathlib import Path

from src.data_prep.dataset_loaders.scitsr_loader import list_samples
from src.run_experiment import load_dataset_samples


def test_seeded_sample_is_reproducible_and_preserves_unseeded_order(tmp_path: Path) -> None:
    images = tmp_path / "img" / "test"
    structures = tmp_path / "structure" / "test"
    images.mkdir(parents=True)
    structures.mkdir(parents=True)
    for index in range(80):
        sample_id = f"table_{index:03d}"
        (images / f"{sample_id}.png").touch()
        (structures / f"{sample_id}.json").touch()

    first = list_samples(tmp_path, sample_size=50, seed=42)
    repeated = list_samples(tmp_path, sample_size=50, seed=42)
    unseeded = list_samples(tmp_path, sample_size=5)
    configured = load_dataset_samples(
        "scitsr",
        {
            "raw_dir": str(tmp_path),
            "split": "test",
            "sample_size": 50,
            "sample_seed": 42,
        },
    )

    assert len(first) == 50
    assert first == repeated == configured
    assert [image.name for image, _ in unseeded] == [
        f"table_{index:03d}.png" for index in range(5)
    ]
    assert first != unseeded
