"""
S-TEDS — variante estrutural do TEDS (Structure-only TEDS), que desconsidera o
conteúdo textual das células e avalia apenas a correção da grade lógica (linhas,
colunas, mesclagens), conforme Yu et al. (2024) e Kasem et al. (2024).

Reaproveita a mesma lógica de `teds.py`, mas com o texto das células removido antes
da conversão para árvore, de forma que apenas a estrutura seja comparada.
"""

from copy import deepcopy

from src.evaluation.teds import compute_teds
from src.normalization.normalize_output import NormalizedTable


def _strip_text(table: NormalizedTable) -> NormalizedTable:
    stripped = deepcopy(table)
    for cell in stripped.cells:
        cell.text = ""
        cell.is_header = False
    return stripped


def compute_s_teds(predicted: NormalizedTable, ground_truth: NormalizedTable) -> float:
    """
    Calcula S-TEDS ignorando texto e distinção `td`/`th`. Cabeçalhos não são
    anotados de modo compatível nos gabaritos públicos; TEDS textual preserva a
    semântica da referência PubTabNet.
    """
    return compute_teds(_strip_text(predicted), _strip_text(ground_truth))


if __name__ == "__main__":
    pass
