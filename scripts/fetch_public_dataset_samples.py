"""Download deterministic PubTabNet/FinTabNet samples from pinned HF revisions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import shutil
import tempfile
import time
from typing import TypedDict
import urllib.error
import urllib.request


class DatasetConfig(TypedDict):
    repo: str
    split: str
    revision: str
    annotations: str
    parquet_files: list[str]


DATASETS: dict[str, DatasetConfig] = {
    "pubtabnet": {
        "repo": "docling-project/PubTabNet_OTSL",
        "split": "val",
        "revision": "c6b519724c09c81dc4797b742826681ebf5c7dde",
        "annotations": "PubTabNet_OTSL_sample_seed{seed}.jsonl",
        "parquet_files": [
            "data/val-00000-of-00001-2d685be412ba1e61.parquet",
        ],
    },
    "fintabnet": {
        "repo": "docling-project/FinTabNet_OTSL",
        "split": "test",
        "revision": "1fe37f0f9478b4c571c51c5c4d11e3ef357e75c4",
        "annotations": "FinTabNet_OTSL_sample_seed{seed}.jsonl",
        "parquet_files": [
            "data/test-00000-of-00002-7d6f82c6161b0377.parquet",
            "data/test-00001-of-00002-7104c7ad37a00357.parquet",
        ],
    },
}


def open_url(request: urllib.request.Request):
    for attempt in range(5):
        try:
            return urllib.request.urlopen(request, timeout=90)
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 4:
                raise
            time.sleep(min(2**attempt, 10))
    raise RuntimeError("The retry loop exited unexpectedly.")


def download_parquet(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url, headers={"User-Agent": "tcc-table-recognition/1.0"}
    )
    with open_url(request) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def image_extension(content: bytes) -> str:
    if content.startswith(b"\xff\xd8"):
        return ".jpg"
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    raise ValueError("The Parquet image has an unsupported or invalid format.")


def select_rows(
    parquet_paths: list[Path], seed: int, sample_size: int, dataset: str
) -> list[tuple[int, dict]]:
    import pyarrow.parquet as pq

    parquet_files = [pq.ParquetFile(path) for path in parquet_paths]
    try:
        row_counts = [parquet.metadata.num_rows for parquet in parquet_files]
        available = sum(row_counts)
        if sample_size > available:
            raise ValueError(
                f"Requested {sample_size} rows from {dataset}, but {available} are available."
            )

        selected_indexes = sorted(
            random.Random(seed).sample(range(available), sample_size)
        )
        selected_rows = []
        global_offset = 0
        for parquet, row_count in zip(parquet_files, row_counts):
            local_indexes = [
                (index - global_offset, index)
                for index in selected_indexes
                if global_offset <= index < global_offset + row_count
            ]
            row_group_offset = 0
            for group_index in range(parquet.metadata.num_row_groups):
                group_count = parquet.metadata.row_group(group_index).num_rows
                group_rows = [
                    (local_index - row_group_offset, global_index)
                    for local_index, global_index in local_indexes
                    if row_group_offset <= local_index < row_group_offset + group_count
                ]
                if group_rows:
                    table = parquet.read_row_group(group_index)
                    for local_index, global_index in group_rows:
                        selected_rows.append(
                            (global_index, table.slice(local_index, 1).to_pylist()[0])
                        )
                row_group_offset += group_count
            global_offset += row_count
        return selected_rows
    finally:
        for parquet in parquet_files:
            parquet.close()


def write_dataset(
    root: Path,
    dataset: str,
    config: DatasetConfig,
    seed: int,
    sample_size: int,
) -> list[dict[str, str]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "Install pyarrow in the active environment to read the pinned dataset files."
        ) from error

    with tempfile.TemporaryDirectory(prefix=f"{dataset}-parquet-") as temp_dir:
        parquet_paths = []
        for relative_path in config["parquet_files"]:
            url = (
                f"https://huggingface.co/datasets/{config['repo']}/resolve/"
                f"{config['revision']}/{relative_path}"
            )
            parquet_path = Path(temp_dir) / Path(relative_path).name
            download_parquet(url, parquet_path)
            parquet_paths.append(parquet_path)

        selected_rows = select_rows(
            parquet_paths, seed=seed, sample_size=sample_size, dataset=dataset
        )

    image_dir = root / "data" / "raw" / dataset / config["split"]
    image_dir.mkdir(parents=True, exist_ok=True)
    records = []
    manifest_rows = []
    for row_index, source_row in selected_rows:
        image_field = source_row["image"]
        image = image_field.get("bytes") if isinstance(image_field, dict) else image_field
        if not image:
            raise ValueError(f"Missing image bytes in {dataset} source row {row_index}.")
        filename = Path(source_row["filename"]).with_suffix(image_extension(image)).name
        record = {
            "filename": filename,
            "split": config["split"],
            "imgid": source_row["imgid"],
            "html": {
                "structure": {"tokens": source_row["html"]},
                "cells": source_row["cells"],
            },
        }
        image_path = image_dir / filename
        image_hash = hashlib.sha256(image).digest()
        if image_path.exists() and hashlib.sha256(image_path.read_bytes()).digest() != image_hash:
            image_path = image_dir / (
                f"{Path(filename).stem}_{row_index}{image_extension(image)}"
            )
            record["filename"] = image_path.name
        if image_path.exists() and hashlib.sha256(image_path.read_bytes()).digest() != image_hash:
            raise FileExistsError(f"Refusing to overwrite a different image: {image_path}")
        image_path.write_bytes(image)
        records.append(record)
        manifest_rows.append(
            {
                "dataset": dataset,
                "image": image_path.relative_to(root).as_posix(),
                "ground_truth": (
                    root / "data" / "raw" / dataset / config["annotations"].format(
                        seed=seed
                    )
                ).relative_to(root).as_posix(),
                "split": config["split"],
                "seed": str(seed),
                "source_dataset": config["repo"],
                "source_revision": config["revision"],
                "source_row_idx": str(row_index),
                "source_sample_id": Path(record["filename"]).stem,
            }
        )

    annotation_path = root / "data" / "raw" / dataset / config["annotations"].format(
        seed=seed
    )
    annotation_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return manifest_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive.")

    root = Path.cwd()
    manifest_rows = []
    for dataset, config in DATASETS.items():
        manifest_rows.extend(
            write_dataset(root, dataset, config, args.seed, args.sample_size)
        )

    from src.data_prep.dataset_loaders.scitsr_loader import list_samples

    scitsr_samples = list_samples(
        root / "data" / "raw" / "scitsr",
        split="test",
        sample_size=args.sample_size,
        seed=args.seed,
    )
    if len(scitsr_samples) != args.sample_size:
        raise ValueError(
            f"Selected {len(scitsr_samples)} SciTSR samples, expected {args.sample_size}."
        )
    for image_path, ground_truth_path in scitsr_samples:
        manifest_rows.append(
            {
                "dataset": "scitsr",
                "image": image_path.relative_to(root).as_posix(),
                "ground_truth": ground_truth_path.relative_to(root).as_posix(),
                "split": "test",
                "seed": str(args.seed),
                "source_dataset": "SciTSR (local dataset copy)",
                "source_revision": "not recorded in the local copy",
                "source_row_idx": "",
                "source_sample_id": image_path.stem,
            }
        )

    manifest_path = args.manifest or Path(
        f"data/manifests/public-samples-seed{args.seed}.csv"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(
        f"Downloaded {len(manifest_rows)} images and wrote the sample manifest "
        f"to {manifest_path}."
    )


if __name__ == "__main__":
    main()
