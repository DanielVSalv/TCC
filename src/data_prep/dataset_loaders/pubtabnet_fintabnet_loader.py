"""
Loader para PubTabNet e FinTabNet — compartilhado porque os dois datasets usam
EXATAMENTE o mesmo schema de anotação (mesma equipe/IBM, mesmo formato jsonl),
diferindo apenas na origem dos documentos (PubTabNet: artigos científicos do PubMed
Central; FinTabNet: relatórios financeiros anuais).

Formato confirmado via documentação oficial (github.com/ibm-aur-nlp/PubTabNet e
descrição do dataset em deepwiki.com/poloclub/unitable, que documenta o mesmo schema
usado por ambos): um arquivo .jsonl, uma linha por amostra, cada linha:

    {
      "filename": "nome_da_imagem.png",
      "split": "train" | "val" | "test",
      "imgid": <int>,
      "html": {
        "cells": [
          {"tokens": ["T", "e", "x", "t", "o"], "bbox": [x0, y0, x1, y1]},  # bbox só em células não-vazias
          {"tokens": []},  # célula vazia: sem "bbox"
          ...
        ],
        "structure": {
          "tokens": ["<thead>", "<tr>", "<td", ' colspan="2"', ">", "[", "]", "</td>", ...]
        }
      }
    }

NOTA METODOLÓGICA (confirmada testando a reconstrução com um exemplo sintético fiel
ao schema): o PubTabNet/FinTabNet usa `<td>` mesmo para células dentro de `<thead>`
— não usa `<th>`. Ou seja, `NormalizedCell.is_header` será sempre `False` no gabarito
desses dois datasets, independentemente de a célula estar no cabeçalho ou não. Isso
é relevante para a métrica TEDS (a tag td/th entra no custo de comparação em
src/evaluation/teds.py): se um algoritmo prever `<th>` para células de cabeçalho
(como o Table Transformer, via `normalize_from_object_detection`), essa previsão
será penalizada como divergência de tag, mesmo semanticamente correta. Vale discutir
essa limitação na Seção 6 (Resultados) do TCC caso apareça nos números.

RECONSTRUÇÃO DO HTML: os tokens de "structure" já formam um HTML válido quando
concatenados diretamente (sem separador) — EXCETO que, para cada célula NÃO VAZIA,
o conteúdo real fica em outro lugar (`cells[i]["tokens"]`) e sua posição no esqueleto
é marcada por um par de tokens placeholder "[" e "]" (células vazias não têm esse par,
só "<td></td>" direto). A função `_reconstruct_html` abaixo substitui cada par "["/"]"
pelo texto da célula correspondente, produzindo um HTML completo e válido, que então é
passado para `normalize_from_html()` (já testada em src/normalization/normalize_output.py)
em vez de duplicar a lógica de reconstrução de grade.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from src.normalization.normalize_output import NormalizedTable, normalize_from_html


def _reconstruct_html(structure_tokens: list[str], cells: list[dict]) -> str:
    cell_idx = 0
    out: list[str] = []

    for tok in structure_tokens:
        if tok == "[":
            cell_text = "".join(cells[cell_idx].get("tokens", []))
            out.append(cell_text)
        elif tok == "]":
            pass  # marcador de fechamento do placeholder; não entra no HTML
        else:
            out.append(tok)

        if tok in ("</td>", "</th>"):
            cell_idx += 1

    return "".join(out)


def iter_annotations(jsonl_path: Path, split: str | None = None) -> Iterator[dict]:
    """Itera as anotações do arquivo .jsonl, opcionalmente filtrando por split."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if split is not None and record.get("split") != split:
                continue
            yield record


def list_samples(
    raw_dir: Path,
    annotations_filename: str,
    split: str | None = None,
    sample_size: int | None = None,
) -> list[tuple[Path, dict]]:
    """
    Retorna uma lista de (caminho_da_imagem, registro_de_anotacao) para o split
    pedido. `raw_dir` deve conter o .jsonl de anotações e as imagens (geralmente em
    subpastas train/val/test, mas o caminho exato depende de como o dataset foi
    extraído — ajustar `image_subdir` abaixo se necessário).
    """
    jsonl_path = raw_dir / annotations_filename
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"Arquivo de anotações não encontrado em {jsonl_path}. Baixe o dataset "
            "(ver link oficial) e aponte raw_dir para a pasta extraída."
        )

    samples = []
    for record in iter_annotations(jsonl_path, split=split):
        # A maioria das distribuições de PubTabNet/FinTabNet organiza as imagens em
        # subpastas por split (train/val/test); ajuste aqui se a sua extração for
        # diferente (ex.: tudo em uma única pasta "images/").
        image_subdir = record.get("split", split or "")
        image_path = raw_dir / image_subdir / record["filename"]
        if not image_path.exists():
            image_path = raw_dir / record["filename"]  # fallback: pasta única
        samples.append((image_path, record))

        if sample_size is not None and len(samples) >= sample_size:
            break

    return samples


def load_ground_truth(record: dict) -> NormalizedTable:
    """Reconstrói a NormalizedTable a partir de um registro de anotação já carregado."""
    structure_tokens = record["html"]["structure"]["tokens"]
    cells = record["html"]["cells"]
    html = _reconstruct_html(structure_tokens, cells)
    # Os tokens de "structure" codificam apenas o CONTEÚDO interno da tabela
    # (thead/tbody/tr/td), sem a tag externa <table> — confirmado testando contra o
    # schema documentado. normalize_from_html() exige essa tag externa para
    # localizar a tabela via BeautifulSoup, daí o wrap abaixo.
    return normalize_from_html(f"<table>{html}</table>")


if __name__ == "__main__":
    pass
