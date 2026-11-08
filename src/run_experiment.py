"""
Orquestrador principal do pipeline experimental (Seção 4.2 da metodologia).

Fluxo, para cada combinação (dataset, modelo) habilitada em configs/experiment_config.yaml:
  1. Carrega as amostras do dataset (imagem + referência de gabarito) via o loader
     correspondente em src/data_prep/dataset_loaders/ — PubTabNet/FinTabNet/SciTSR lêem
     direto de `raw_dir` (já vêm com anotação própria); o dataset corporativo lê de
     `processed_dir` (recortes gerados por extract_table_regions.extract_from_pdf) +
     `ground_truth_dir` (gabarito manual).
  2. Executa o modelo (via seu runner) sobre cada imagem.
  3. Normaliza a saída para NormalizedTable.
  4. Calcula TEDS, S-TEDS e F1 por célula contra o gabarito.
  5. Exporta a estrutura reconstruída para .xlsx em results/spreadsheets/.
  6. Grava as métricas agregadas em results/metrics/<dataset>_<modelo>.csv.

IMPORTANTE — isolamento de ambientes:
Os 4 modelos têm dependências MUTUAMENTE INCOMPATÍVEIS no mesmo processo Python
(TableFormer/Table Transformer: PyTorch 2.x atual; TabStructNet: TensorFlow 1.15,
Python ≤3.9; SLANet: PaddlePaddle/PaddleOCR 2.x) — ver README.md. Por isso, `main()`
abaixo exige `--model` (um único modelo por execução)
e SEMPRE deve ser invocado já de dentro do `.venv-<modelo>` correspondente (nunca do
`.venv-base`, que só tem as dependências comuns de orquestração/avaliação). Rodar os
4 modelos é, portanto, 4 invocações separadas, cada uma no seu venv:

    .venv-tableformer/bin/python -m src.run_experiment --model tableformer
    .venv-tabstructnet/bin/python -m src.run_experiment --model tabstructnet
    .venv-tabletransformer/bin/python -m src.run_experiment --model tabletransformer
    .venv-slanet/bin/python -m src.run_experiment --model slanet

Dentro de uma mesma invocação (um único modelo), todos os datasets habilitados em
`configs/experiment_config.yaml` são processados em sequência, já que a avaliação
(TEDS/S-TEDS/F1) e a exportação não têm esse problema de dependências conflitantes.

Datasets suportados (ver src/data_prep/dataset_loaders/): PubTabNet, FinTabNet,
SciTSR e o conjunto reduzido de documentos corporativos. PubTabNet e FinTabNet
compartilham o mesmo loader, pois usam exatamente o mesmo schema de anotação.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import yaml

from src.evaluation.cell_f1 import compute_cell_f1
from src.evaluation.s_teds import compute_s_teds
from src.evaluation.teds import METRIC_PROTOCOL_VERSION, compute_teds
from src.data_prep.text_recovery import assign_words_to_cells, recover_sample_text
from src.export.spreadsheet_exporter import export_to_xlsx
from src.normalization.normalize_output import NormalizedTable, normalize_output


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataset_samples(dataset_name: str, dataset_cfg: dict) -> list[tuple[Path, Any]]:
    """
    Retorna uma lista de (caminho_da_imagem, referência_de_gabarito) para o dataset
    informado. O tipo da "referência de gabarito" varia por dataset (um dict já
    carregado do .jsonl para PubTabNet/FinTabNet; um Path para o .json de estrutura
    no SciTSR e no corporativo) — por isso ela é repassada, sem modificação, direto
    para `load_ground_truth` abaixo.
    """
    raw_dir = Path(dataset_cfg["raw_dir"])
    sample_size = dataset_cfg.get("sample_size")

    if dataset_name in ("pubtabnet", "fintabnet"):
        from src.data_prep.dataset_loaders.pubtabnet_fintabnet_loader import list_samples

        # Nome do arquivo de anotações varia entre as duas distribuições oficiais;
        # ajustar aqui (ou via uma chave própria no YAML) se o nome baixado diferir.
        annotations_filename = dataset_cfg.get(
            "annotations_filename",
            "PubTabNet_2.0.0.jsonl" if dataset_name == "pubtabnet" else "FinTabNet_1.0.0.jsonl",
        )
        return list_samples(
            raw_dir,
            annotations_filename=annotations_filename,
            split=dataset_cfg.get("split"),
            sample_size=sample_size,
        )

    if dataset_name == "scitsr":
        from src.data_prep.dataset_loaders.scitsr_loader import list_samples

        return list_samples(
            raw_dir,
            split=dataset_cfg.get("split", "test"),
            sample_size=sample_size,
            seed=dataset_cfg.get("sample_seed"),
        )

    if dataset_name == "corporate":
        if dataset_cfg.get("privacy", {}).get("authorization_confirmed") is not True:
            raise PermissionError(
                "Confirme a autorização formal antes de executar o conjunto corporativo."
            )
        from src.data_prep.dataset_loaders.corporate_loader import list_samples

        processed_dir = Path(dataset_cfg.get("processed_dir", "data/processed/corporate"))
        ground_truth_dir = Path(dataset_cfg["ground_truth_dir"])
        return list_samples(processed_dir, ground_truth_dir, sample_size=sample_size)

    raise ValueError(f"Dataset desconhecido para carregamento de amostras: '{dataset_name}'.")


def load_ground_truth(dataset_name: str, gt_ref: Any, dataset_cfg: dict) -> NormalizedTable:
    """Carrega e normaliza a estrutura de referência a partir da gt_ref retornada por
    `load_dataset_samples` (ver nota acima sobre o tipo variar por dataset)."""
    if dataset_name in ("pubtabnet", "fintabnet"):
        from src.data_prep.dataset_loaders.pubtabnet_fintabnet_loader import (
            load_ground_truth as _load,
        )
    elif dataset_name == "scitsr":
        from src.data_prep.dataset_loaders.scitsr_loader import load_ground_truth as _load
    elif dataset_name == "corporate":
        from src.data_prep.dataset_loaders.corporate_loader import load_ground_truth as _load
    else:
        raise ValueError(f"Dataset desconhecido para carregamento de gabarito: '{dataset_name}'.")

    return _load(gt_ref)


def get_runner(model_name: str, model_cfg: dict):
    """Instancia e carrega o runner do modelo indicado."""
    if model_name == "tableformer":
        from src.models.tableformer_runner import TableFormerRunner

        runner = TableFormerRunner(
            variant=model_cfg.get("variant", "fast"),
            device=model_cfg.get("device", "cpu"),
            revision=model_cfg.get("revision", "v2.1.0"),
        )
    elif model_name == "tabstructnet":
        from src.models.tabstructnet_runner import TabStructNetRunner

        runner = TabStructNetRunner(
            repo_dir=Path(model_cfg.get("repo_dir", "third_party/TabStructNet"))
        )
    elif model_name == "tabletransformer":
        from src.models.tabletransformer_runner import TableTransformerRunner

        runner = TableTransformerRunner(
            score_threshold=model_cfg.get("score_threshold", 0.7),
            device=model_cfg.get("device", "cpu"),
            revision=model_cfg.get("revision"),
        )
    elif model_name == "slanet":
        from src.models.slanet_runner import SLANetRunner

        runner = SLANetRunner(
            language=model_cfg.get("language", "en"),
            device=model_cfg.get("device", "cpu"),
            table_algorithm=model_cfg.get("table_algorithm", "SLANet"),
        )
    else:
        raise ValueError(f"Modelo desconhecido: {model_name}")

    checkpoint_path = model_cfg.get("checkpoint_path")
    if model_name == "tabletransformer":
        # Hugging Face repo IDs require forward slashes; Path() rewrites them on Windows.
        runner.load_model(checkpoint_path)
    else:
        runner.load_model(Path(checkpoint_path) if checkpoint_path else None)
    return runner


def validate_normalized_table(
    table: NormalizedTable, model_name: str, image_path: Path
) -> None:
    """Raise an actionable error instead of accepting an empty or malformed result."""
    if not table.cells or table.n_rows <= 0 or table.n_cols <= 0:
        raise ValueError(
            f"{model_name} não produziu uma tabela válida para {image_path}: "
            f"{table.n_rows}x{table.n_cols}, {len(table.cells)} células."
        )

    for cell in table.cells:
        if (
            cell.row < 0
            or cell.col < 0
            or cell.row_span <= 0
            or cell.col_span <= 0
            or cell.row + cell.row_span > table.n_rows
            or cell.col + cell.col_span > table.n_cols
        ):
            raise ValueError(
                f"{model_name} produziu coordenadas inválidas para {image_path}: "
                f"célula ({cell.row}, {cell.col}) com extensão "
                f"{cell.row_span}x{cell.col_span} em tabela "
                f"{table.n_rows}x{table.n_cols}."
            )


def _corporate_manifest_entries(dataset_cfg: dict) -> dict[str, dict[str, str]]:
    manifest_path = Path(
        dataset_cfg.get("table_detection", {}).get(
            "manifest_path",
            str(Path(dataset_cfg["processed_dir"]) / "manifest.csv"),
        )
    )
    if not manifest_path.is_file():
        return {}

    with open(manifest_path, "r", newline="", encoding="utf-8") as manifest_file:
        return {
            row["image_file"]: row
            for row in csv.DictReader(manifest_file)
            if row.get("image_file")
        }


def select_manifest_samples(
    samples: list[tuple[Path, Any]], dataset_name: str, manifest_path: Path
) -> list[tuple[Path, Any]]:
    """Restrict a model run to the exact sample paths frozen in a benchmark manifest."""
    with open(manifest_path, "r", newline="", encoding="utf-8") as manifest_file:
        manifest_rows = [
            row
            for row in csv.DictReader(manifest_file)
            if row.get("dataset") == dataset_name
        ]

    expected_paths = {str(Path(row["image"]).resolve()) for row in manifest_rows}
    selected = [
        (image_path, gt_ref)
        for image_path, gt_ref in samples
        if str(image_path.resolve()) in expected_paths
    ]
    selected_paths = {str(image_path.resolve()) for image_path, _ in selected}
    if selected_paths != expected_paths:
        missing = sorted(expected_paths - selected_paths)
        raise ValueError(
            f"O manifesto do benchmark contém amostras ausentes em "
            f"{dataset_name}: {missing}"
        )
    if len(selected) != len(manifest_rows):
        raise ValueError(
            f"O manifesto do benchmark contém caminhos duplicados para {dataset_name}."
        )
    return selected


def run_smoke_test(
    dataset_name: str,
    dataset_cfg: dict,
    model_name: str,
    model_cfg: dict,
    text_cfg: dict,
) -> None:
    """Run one sample and print bounded raw/normalized output for preflight checks."""
    samples = load_dataset_samples(dataset_name, dataset_cfg)
    if not samples:
        raise ValueError(f"Nenhuma amostra encontrada para o dataset '{dataset_name}'.")

    sample_path, _ = samples[0]
    manifest_entries = (
        _corporate_manifest_entries(dataset_cfg) if dataset_name == "corporate" else {}
    )
    text_result = recover_sample_text(
        image_path=sample_path,
        dataset_name=dataset_name,
        dataset_cfg=dataset_cfg,
        text_cfg=text_cfg,
        manifest_entry=manifest_entries.get(sample_path.name),
    )
    runner = get_runner(model_name, model_cfg)
    if model_name == "tableformer":
        raw_output = runner.predict_with_words(sample_path, text_result.words)
    else:
        raw_output = runner.predict(sample_path)
    if raw_output.model_name != model_name:
        raise ValueError(
            f"Runner configurado como '{model_name}' retornou "
            f"model_name='{raw_output.model_name}'."
        )

    normalized = normalize_output(model_name, raw_output.content)
    validate_normalized_table(normalized, model_name, sample_path)
    raw_preview = repr(raw_output.content)
    if len(raw_preview) > 500:
        raw_preview = raw_preview[:497] + "..."

    print(
        json.dumps(
            {
                "model": model_name,
                "sample": str(sample_path),
                "raw_type": type(raw_output.content).__name__,
                "raw_preview": raw_preview,
                "normalized": {
                    "rows": normalized.n_rows,
                    "columns": normalized.n_cols,
                    "cells": len(normalized.cells),
                    "first_cells": [
                        {
                            "row": cell.row,
                            "column": cell.col,
                            "row_span": cell.row_span,
                            "column_span": cell.col_span,
                            "text": cell.text[:60],
                        }
                        for cell in normalized.cells[:5]
                    ],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def run_single_combination(
    dataset_name: str,
    dataset_cfg: dict,
    model_name: str,
    model_cfg: dict,
    output_dir: Path,
    text_cfg: dict,
    sample_manifest: Path | None = None,
) -> None:
    samples = load_dataset_samples(dataset_name, dataset_cfg)
    if sample_manifest is not None:
        samples = select_manifest_samples(samples, dataset_name, sample_manifest)
    runner = get_runner(model_name, model_cfg)
    manifest_entries = (
        _corporate_manifest_entries(dataset_cfg) if dataset_name == "corporate" else {}
    )

    metrics_rows = []
    spreadsheets_dir = output_dir / "spreadsheets" / dataset_name / model_name
    metrics_path = output_dir / "metrics" / f"{dataset_name}_{model_name}.csv"

    for sample_path, gt_ref in samples:
        try:
            text_result = recover_sample_text(
                image_path=sample_path,
                dataset_name=dataset_name,
                dataset_cfg=dataset_cfg,
                text_cfg=text_cfg,
                manifest_entry=manifest_entries.get(sample_path.name),
            )
            if model_name == "tableformer":
                raw_output = runner.predict_with_words(sample_path, text_result.words)
            else:
                raw_output = runner.predict(sample_path)
            predicted_table = normalize_output(model_name, raw_output.content)
            validate_normalized_table(predicted_table, model_name, sample_path)
            ground_truth_table = load_ground_truth(dataset_name, gt_ref, dataset_cfg)

            recognized_words = len(text_result.words)
            has_cell_geometry = any(cell.bbox is not None for cell in predicted_table.cells)
            unassigned_words = None
            assigned_words = None
            if model_name != "tableformer":
                unassigned_words = assign_words_to_cells(
                    predicted_table,
                    text_result.words,
                    min_word_overlap=float(text_cfg.get("min_word_overlap", 0.2)),
                )
                assigned_words = recognized_words - unassigned_words

            text_score_status = "scored"
            if recognized_words == 0:
                text_score_status = "no_text_detected"
            elif model_name != "tableformer" and not has_cell_geometry:
                text_score_status = "no_cell_geometry"
            teds_score = (
                compute_teds(predicted_table, ground_truth_table)
                if text_score_status == "scored"
                else None
            )
            s_teds_score = compute_s_teds(predicted_table, ground_truth_table)
            cell_f1_result = compute_cell_f1(predicted_table, ground_truth_table)
            cells_with_text = sum(bool(cell.text.strip()) for cell in predicted_table.cells)
            text_coverage = (
                cells_with_text / len(predicted_table.cells)
                if predicted_table.cells
                else 0.0
            )

            export_to_xlsx(
                predicted_table, spreadsheets_dir / f"{sample_path.stem}.xlsx"
            )

            metrics_rows.append(
                {
                    "sample": sample_path.name,
                    "status": "completed",
                    "error_type": None,
                    "error": None,
                    "metric_protocol": METRIC_PROTOCOL_VERSION,
                    "teds": teds_score,
                    "s_teds": s_teds_score,
                    "cell_f1": cell_f1_result.f1,
                    "cell_precision": cell_f1_result.precision,
                    "cell_recall": cell_f1_result.recall,
                    "text_source": text_result.source,
                    "text_engine": text_result.engine,
                    "text_config": json.dumps(
                        text_result.config, ensure_ascii=False, sort_keys=True
                    ),
                    "recognized_words": recognized_words,
                    "assigned_words": assigned_words,
                    "unassigned_words": unassigned_words,
                    "predicted_rows": predicted_table.n_rows,
                    "predicted_columns": predicted_table.n_cols,
                    "predicted_cells": len(predicted_table.cells),
                    "ground_truth_rows": ground_truth_table.n_rows,
                    "ground_truth_columns": ground_truth_table.n_cols,
                    "ground_truth_cells": len(ground_truth_table.cells),
                    "cells_with_text": cells_with_text,
                    "text_coverage": text_coverage,
                    "text_score_status": text_score_status,
                }
            )
        except Exception as exc:
            print(
                f"[run_experiment] Falha explícita em {model_name}/{dataset_name}/"
                f"{sample_path.name}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            metrics_rows.append(
                {
                    "sample": sample_path.name,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "metric_protocol": METRIC_PROTOCOL_VERSION,
                    "teds": None,
                    "s_teds": None,
                    "cell_f1": None,
                    "cell_precision": None,
                    "cell_recall": None,
                    "text_source": None,
                    "text_engine": None,
                    "text_config": None,
                    "recognized_words": None,
                    "assigned_words": None,
                    "unassigned_words": None,
                    "predicted_rows": None,
                    "predicted_columns": None,
                    "predicted_cells": None,
                    "ground_truth_rows": None,
                    "ground_truth_columns": None,
                    "ground_truth_cells": None,
                    "cells_with_text": None,
                    "text_coverage": None,
                    "text_score_status": "sample_error",
                }
            )

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(metrics_rows[0].keys()) if metrics_rows else []
        )
        writer.writeheader()
        writer.writerows(metrics_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Executa o pipeline experimental de TSR/TR para UM modelo por vez "
            "(ver nota de isolamento de ambientes no topo deste arquivo) — "
            "invocar a partir do .venv-<modelo> correspondente."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment_config.yaml"),
        help="Caminho para o arquivo de configuração do experimento.",
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["tableformer", "tabstructnet", "tabletransformer", "slanet"],
        help=(
            "Modelo a executar nesta invocação. Obrigatório: rodar mais de um "
            "modelo no mesmo processo não é suportado, por conflito de "
            "dependências entre os ambientes (ver docstring do módulo)."
        ),
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Processar apenas este dataset (por padrão, processa todos os habilitados no config).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Executa uma amostra e exibe uma prévia limitada da saída bruta e normalizada.",
    )
    parser.add_argument(
        "--sample-manifest",
        type=Path,
        default=None,
        help="CSV com a lista congelada de amostras a usar em todos os modelos.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Sobrescreve experiment.output_dir para isolar os artefatos desta execução.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    model_cfg = config["models"].get(args.model)
    if model_cfg is None or not model_cfg.get("enabled"):
        raise ValueError(f"Modelo '{args.model}' não encontrado ou desabilitado em {args.config}.")

    if args.smoke_test:
        if not args.dataset:
            raise ValueError("--smoke-test requer --dataset para escolher a amostra.")
        if args.dataset not in config["datasets"]:
            raise ValueError(f"Dataset '{args.dataset}' não encontrado em {args.config}.")
        run_smoke_test(
            args.dataset,
            config["datasets"][args.dataset],
            args.model,
            model_cfg,
            config.get("text_recovery", {}),
        )
        return

    output_dir = args.output_dir or Path(config["experiment"]["output_dir"])
    datasets_to_run = (
        {args.dataset: config["datasets"][args.dataset]}
        if args.dataset
        else {name: cfg for name, cfg in config["datasets"].items() if cfg.get("enabled")}
    )
    if args.dataset and args.dataset not in config["datasets"]:
        raise ValueError(f"Dataset '{args.dataset}' não encontrado em {args.config}.")

    for dataset_name, dataset_cfg in datasets_to_run.items():
        print(f"[run_experiment] Executando {args.model} em {dataset_name}...")
        run_single_combination(
            dataset_name,
            dataset_cfg,
            args.model,
            model_cfg,
            output_dir,
            config.get("text_recovery", {}),
            args.sample_manifest,
        )


if __name__ == "__main__":
    main()
