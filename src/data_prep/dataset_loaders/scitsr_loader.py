"""
Loader para o SciTSR (Chi et al., "Complicated Table Structure Recognition", 2019).

Formato confirmado via repositório oficial (github.com/Academic-Hammer/SciTSR) e via
amostras reais publicadas em huggingface.co/datasets/bevaya/SciTSR-cc-by-nc-sa:

    SciTSR/
    ├── img/{split}/[ID].png            (ou .jpg, conforme a distribuição)
    ├── structure/{split}/[ID].json     -> lista de células:
    │     [{"id": 40, "tex": "China", "content": ["China"],
    │       "start_row": 10, "end_row": 10, "start_col": 2, "end_col": 2}, ...]
    └── chunk/{split}/[ID].chunk        (texto com coordenadas em PDF — não usado
                                          aqui, pois já temos `content` no structure/)

Diferente do PubTabNet/FinTabNet, o SciTSR já fornece start_row/end_row/start_col/
end_col diretamente por célula — o mesmo conceito estrutural normalizado da saída do
TabStructNet (ver src/normalization/normalize_output.py), então a
conversão para NormalizedTable é direta, sem precisar passar por HTML.

O SciTSR não distingue células de cabeçalho (sem campo equivalente a `is_header`) —
`NormalizedCell.is_header` fica sempre `False` para este dataset, mesma observação
já registrada para PubTabNet/FinTabNet (mas lá o motivo é o uso de `<td>` dentro de
`<thead>`; aqui é porque o dataset simplesmente não anota essa distinção).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from src.normalization.normalize_output import NormalizedCell, NormalizedTable


def list_samples(
    raw_dir: Path,
    split: str = "test",
    sample_size: int | None = None,
    seed: int | None = None,
) -> list[tuple[Path, Path]]:
    """
    Retorna uma lista de (caminho_da_imagem, caminho_do_json_de_estrutura) para o
    split pedido. `raw_dir` deve ser a pasta raiz do SciTSR já extraída (contendo
    img/, structure/ e chunk/).
    """
    img_dir = raw_dir / "img" / split
    structure_dir = raw_dir / "structure" / split

    if not img_dir.exists() or not structure_dir.exists():
        raise FileNotFoundError(
            f"Estrutura de pastas do SciTSR não encontrada em {raw_dir} "
            f"(esperado img/{split}/ e structure/{split}/). Baixe o dataset oficial "
            "(github.com/Academic-Hammer/SciTSR) e aponte raw_dir para a pasta extraída."
        )

    samples = []
    for json_path in sorted(structure_dir.glob("*.json")):
        sample_id = json_path.stem
        # A extensão da imagem varia entre distribuições do SciTSR (.png mais comum);
        # tenta as alternativas mais usuais antes de desistir da amostra.
        image_path = None
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = img_dir / f"{sample_id}{ext}"
            if candidate.exists():
                image_path = candidate
                break
        if image_path is None:
            continue  # imagem ausente para este id; pula a amostra

        samples.append((image_path, json_path))

    if sample_size is None or len(samples) <= sample_size:
        return samples
    if seed is None:
        return samples[:sample_size]
    return sorted(
        random.Random(seed).sample(samples, sample_size),
        key=lambda sample: sample[0].name,
    )


def load_ground_truth(structure_json_path: Path) -> NormalizedTable:
    with open(structure_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_cells = data["cells"] if isinstance(data, dict) else data

    cells: list[NormalizedCell] = []
    max_row, max_col = 0, 0

    for raw_cell in raw_cells:
        start_row = raw_cell["start_row"]
        end_row = raw_cell["end_row"]
        start_col = raw_cell["start_col"]
        end_col = raw_cell["end_col"]
        text = " ".join(raw_cell.get("content", []))

        cells.append(
            NormalizedCell(
                row=start_row,
                col=start_col,
                row_span=max(end_row - start_row + 1, 1),
                col_span=max(end_col - start_col + 1, 1),
                is_header=False,  # SciTSR não anota cabeçalho — ver docstring do módulo
                text=text,
            )
        )
        max_row, max_col = max(max_row, end_row), max(max_col, end_col)

    return NormalizedTable(cells=cells, n_rows=max_row + 1, n_cols=max_col + 1)


if __name__ == "__main__":
    pass
