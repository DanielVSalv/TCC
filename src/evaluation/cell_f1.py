"""
F1 por célula — avaliação granular da correspondência entre as células identificadas
pelo algoritmo e as células da estrutura de referência (Seção 2.3 do TCC).

Diferente do TEDS/S-TEDS (que avaliam a árvore da tabela como um todo), esta métrica
compara diretamente os conjuntos de células previstas e de referência, considerando
uma célula "correta" quando sua posição (linha/coluna inicial) e sua extensão
(rowspan/colspan) coincidem com o gabarito. Isso permite observar, célula a célula,
onde o algoritmo erra (ex.: mesclagens não detectadas, deslocamento de coluna).

Esta é a única das três métricas que não depende de bibliotecas externas de árvore
(apted) nem de uma implementação de referência externa — por isso já está totalmente
implementada aqui, ao contrário de teds.py/s_teds.py.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.normalization.normalize_output import NormalizedTable


@dataclass
class CellF1Result:
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int


def _cell_signature(cell) -> tuple[int, int, int, int]:
    """Identifica uma célula pela sua posição e extensão (ignora o texto)."""
    return (cell.row, cell.col, cell.row_span, cell.col_span)


def compute_cell_f1(predicted: NormalizedTable, ground_truth: NormalizedTable) -> CellF1Result:
    """
    Calcula precisão, revocação e F1 por célula entre a tabela predita e o gabarito.

    Uma célula prevista é considerada verdadeiro-positivo se existir uma célula de
    mesma posição (linha/coluna inicial) e mesma extensão (rowspan/colspan) no
    gabarito.
    """
    predicted_signatures = {_cell_signature(c) for c in predicted.cells}
    gt_signatures = {_cell_signature(c) for c in ground_truth.cells}

    true_positives = len(predicted_signatures & gt_signatures)
    false_positives = len(predicted_signatures - gt_signatures)
    false_negatives = len(gt_signatures - predicted_signatures)

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return CellF1Result(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )


if __name__ == "__main__":
    # Exemplo de uso manual (com tabelas normalizadas fictícias):
    # from src.normalization.normalize_output import normalize_from_html
    # pred = normalize_from_html("<table><tr><td>A</td><td>B</td></tr></table>")
    # gt = normalize_from_html("<table><tr><td>A</td><td>B</td></tr></table>")
    # result = compute_cell_f1(pred, gt)
    # print(result)
    pass
