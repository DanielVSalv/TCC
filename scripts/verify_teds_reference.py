from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
import types
import urllib.request
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_URL = (
    "https://raw.githubusercontent.com/ibm-aur-nlp/PubTabNet/"
    "8ffde9024bd331f5a61c3c549fdf30dd3c3e43a5/src/metric.py"
)
REFERENCE_SHA256 = (
    "a9aedd313678b42e55b06c4b0a365d658ef495aaf3abb7fd6012a6e9f334f9f9"
)
SAMPLE_FILENAME = "PMC5755158_010_01.jpg"

sys.path.insert(0, str(PROJECT_ROOT))


def parallel_execution_not_supported(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("This verification uses the serial PubTabNet reference only.")


def load_reference(source_path: Path | None) -> Any:
    if source_path is None:
        request = urllib.request.Request(
            REFERENCE_URL, headers={"User-Agent": "TEDS reference verification"}
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            source = response.read()
    else:
        source = source_path.read_bytes()

    digest = hashlib.sha256(source).hexdigest()
    if digest != REFERENCE_SHA256:
        raise ValueError(
            "The reference implementation hash does not match the pinned "
            f"PubTabNet source: {digest}"
        )

    with tempfile.TemporaryDirectory(prefix="pubtabnet-teds-") as temp_dir:
        reference_path = Path(temp_dir) / "metric.py"
        reference_path.write_bytes(source)
        spec = importlib.util.spec_from_file_location(
            "pubtabnet_reference_metric", reference_path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not load the pinned PubTabNet TEDS source.")

        module = importlib.util.module_from_spec(spec)
        parallel_stub = types.ModuleType("parallel")
        parallel_stub.parallel_process = parallel_execution_not_supported
        previous_parallel = sys.modules.get("parallel")
        sys.modules["parallel"] = parallel_stub
        try:
            spec.loader.exec_module(module)
        finally:
            if previous_parallel is None:
                sys.modules.pop("parallel", None)
            else:
                sys.modules["parallel"] = previous_parallel
    return module


def annotation_to_html(record: dict[str, Any]) -> str:
    structure_tokens = record["html"]["structure"]["tokens"]
    cells = record["html"]["cells"]
    parts = ["<html><body><table>"]
    cell_index = 0
    token_index = 0

    while token_index < len(structure_tokens):
        if structure_tokens[token_index : token_index + 2] == ["[", "]"]:
            if cell_index >= len(cells):
                raise ValueError("The annotation has more cell markers than cells.")
            # Preserve inline markup tokens for the upstream HTML parser.
            parts.append("".join(cells[cell_index]["tokens"]))
            cell_index += 1
            token_index += 2
        else:
            parts.append(structure_tokens[token_index])
            token_index += 1

    if cell_index != len(cells):
        raise ValueError(
            f"The annotation has {len(cells)} cells but {cell_index} cell markers."
        )
    parts.append("</table></body></html>")
    return "".join(parts)


def load_sample(annotation_path: Path) -> dict[str, Any]:
    with annotation_path.open(encoding="utf-8") as annotation_file:
        for line in annotation_file:
            record = json.loads(line)
            if record.get("filename") == SAMPLE_FILENAME:
                if record.get("split") != "val":
                    raise ValueError(f"{SAMPLE_FILENAME} is not in the val split.")
                return record
    raise FileNotFoundError(
        f"{SAMPLE_FILENAME} was not found in {annotation_path}."
    )


def perturb_first_text_token(record: dict[str, Any]) -> tuple[int, str, str]:
    for cell_index, cell in enumerate(record["html"]["cells"]):
        for token_index, token in enumerate(cell["tokens"]):
            if token and not (token.startswith("<") and token.endswith(">")):
                replacement = "Y" if token == "X" else "X"
                cell["tokens"][token_index] = replacement
                return cell_index, token, replacement
    raise ValueError("The sample contains no text token to perturb.")


def score_case(
    label: str,
    predicted_html: str,
    ground_truth_html: str,
    reference: Any,
    compute_teds: Any,
    normalize_from_html: Any,
    expected: tuple[float, float] | None = None,
) -> tuple[float, float]:
    reference_score = reference.evaluate(predicted_html, ground_truth_html)
    local_score = compute_teds(
        normalize_from_html(predicted_html),
        normalize_from_html(ground_truth_html),
    )
    print(f"{label}: reference={reference_score:.9f}; local={local_score:.9f}")
    if expected is not None and (
        abs(reference_score - expected[0]) > 1e-12
        or abs(local_score - expected[1]) > 1e-12
    ):
        raise AssertionError(
            f"{label} diverged from expected scores "
            f"{expected[0]:.9f}/{expected[1]:.9f}."
        )
    return reference_score, local_score


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare local TEDS with the pinned PubTabNet reference."
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "pubtabnet" / "PubTabNet_2.0.0.jsonl",
        help="PubTabNet JSONL containing the sample annotation.",
    )
    parser.add_argument(
        "--reference-source",
        type=Path,
        help="Optional local copy of metric.py at the pinned PubTabNet revision.",
    )
    args = parser.parse_args()

    reference_module = load_reference(args.reference_source)
    reference = reference_module.TEDS(n_jobs=1)

    from src.evaluation.teds import compute_teds
    from src.normalization.normalize_output import normalize_from_html

    reference_revision = REFERENCE_URL.split("/")[-3]
    print(f"Reference: PubTabNet {reference_revision}")
    print(f"SHA-256: {REFERENCE_SHA256}")

    cases = [
        (
            "Control: identical table",
            "<html><body><table><tr><td>A</td></tr></table></body></html>",
            "<html><body><table><tr><td>A</td></tr></table></body></html>",
            (1.0, 1.0),
        ),
        (
            "Control: text replacement",
            "<html><body><table><tr><td>X</td><td>B</td></tr></table></body></html>",
            "<html><body><table><tr><td>A</td><td>B</td></tr></table></body></html>",
            (2 / 3, 2 / 3),
        ),
        (
            "Control: explicit empty cell",
            "<html><body><table><tr><td>A</td><td></td><td>B</td></tr></table></body></html>",
            "<html><body><table><tr><td>A</td><td>B</td></tr></table></body></html>",
            (0.75, 0.75),
        ),
        (
            "Control: identical merged cell",
            "<html><body><table><tr><td colspan='2'>AB</td></tr></table></body></html>",
            "<html><body><table><tr><td colspan='2'>AB</td></tr></table></body></html>",
            (1.0, 1.0),
        ),
    ]
    for label, predicted_html, ground_truth_html, expected in cases:
        score_case(
            label,
            predicted_html,
            ground_truth_html,
            reference,
            compute_teds,
            normalize_from_html,
            expected,
        )

    sample = load_sample(args.annotation)
    ground_truth_html = annotation_to_html(sample)
    score_case(
        "Real PubTabNet sample, intact",
        ground_truth_html,
        ground_truth_html,
        reference,
        compute_teds,
        normalize_from_html,
        (1.0, 1.0),
    )

    changed_sample = copy.deepcopy(sample)
    cell_index, original_token, replacement = perturb_first_text_token(changed_sample)
    score_case(
        f"Real PubTabNet sample: cell {cell_index}, "
        f"token {original_token!r}->{replacement!r}",
        annotation_to_html(changed_sample),
        ground_truth_html,
        reference,
        compute_teds,
        normalize_from_html,
        (0.9955555555555555, 0.9928571428571429),
    )


if __name__ == "__main__":
    main()
