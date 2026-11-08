"""
Loader para derivados pseudonimizados do subconjunto corporativo. Os recortes e
gabaritos carregados pelo experimento são produzidos por
`src.data_prep.anonymize_corporate`; os arquivos-fonte permanecem em diretórios
privados e não devem ser usados diretamente pelo benchmark.

Diferente dos datasets públicos, aqui NÃO existe um formato de anotação oficial para
seguir — o gabarito precisa ser produzido manualmente (dado que a amostra é pequena,
conforme a metodologia). Por isso, ao contrário dos outros três loaders (que precisam
interpretar um formato externo específico), este módulo DEFINE o formato do gabarito,
escolhido para ser o mais simples possível de produzir manualmente: um JSON que
espelha diretamente os campos de `NormalizedCell`/`NormalizedTable`.

Formato esperado em data/ground_truth/corporate-anonymized/<sample_id>.json:

    {
      "n_rows": 3,
      "n_cols": 2,
      "cells": [
        {"row": 0, "col": 0, "row_span": 1, "col_span": 2, "is_header": true, "text": "Resumo"},
        {"row": 1, "col": 0, "row_span": 1, "col_span": 1, "is_header": false, "text": "Receita"},
        {"row": 1, "col": 1, "row_span": 1, "col_span": 1, "is_header": false, "text": "1000"},
        ...
      ]
    }

Cada gabarito corresponde a um recorte pseudonimizado em
data/processed/corporate-anonymized/<sample_id>.png. Os nomes de arquivo de origem
não são usados como identificadores pelo loader.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.normalization.normalize_output import NormalizedCell, NormalizedTable


def list_samples(
    processed_dir: Path, ground_truth_dir: Path, sample_size: int | None = None
) -> list[tuple[Path, Path]]:
    """
    Retorna (caminho_da_imagem_recortada, caminho_do_json_de_gabarito) para cada
    tabela já extraída e com gabarito disponível. Só inclui pares onde AMBOS os
    arquivos existem — tabelas extraídas mas ainda sem gabarito anotado são
    silenciosamente ignoradas (útil durante a anotação incremental do conjunto).
    """
    if not processed_dir.exists():
        raise FileNotFoundError(
            f"Pasta de recortes processados não encontrada em {processed_dir}. "
            "Rode a extração (src/data_prep/extract_table_regions.extract_from_pdf) "
            "sobre os PDFs em data/corporate/ antes de carregar este dataset."
        )

    samples = []
    for image_path in sorted(processed_dir.glob("*.png")):
        gt_path = ground_truth_dir / f"{image_path.stem}.json"
        if not gt_path.exists():
            continue
        samples.append((image_path, gt_path))
        if sample_size is not None and len(samples) >= sample_size:
            break

    return samples


def load_ground_truth(gt_json_path: Path) -> NormalizedTable:
    with open(gt_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cells = [
        NormalizedCell(
            row=c["row"],
            col=c["col"],
            row_span=c.get("row_span", 1),
            col_span=c.get("col_span", 1),
            is_header=c.get("is_header", False),
            text=c.get("text", ""),
        )
        for c in data["cells"]
    ]

    return NormalizedTable(cells=cells, n_rows=data["n_rows"], n_cols=data["n_cols"])


if __name__ == "__main__":
    pass
