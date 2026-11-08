"""
TEDS — Tree-Edit-Distance-based Similarity (Zhong et al., "Image-based table
recognition: data, model, and evaluation", ECCV 2020 — métrica introduzida junto ao
dataset PubTabNet; usada como referência por Yu et al., 2024 e Kasem et al., 2024,
ambos citados no referencial teórico deste TCC).

Ideia do algoritmo (reimplementada aqui a partir da descrição do método, usando a
biblioteca `apted` para o cálculo eficiente da distância de edição entre árvores —
Pawlik; Augsten, "Efficient Computation of the Tree Edit Distance", 2015/2016, a
mesma base algorítmica usada pela implementação de referência do PubTabNet):

  1. A tabela (predita e gabarito) é representada como uma árvore: raiz "table",
     filhos "tr" (uma por linha), cada "tr" com filhos "td"/"th" (uma por célula).
  2. Calcula-se a distância de edição mínima entre as duas árvores, com um custo de
     edição específico para células de tabela:
       - trocar uma célula por outra de mesmo rowspan/colspan custa a distância de
         edição normalizada entre os textos das duas células (0 se idênticos);
       - trocar uma célula por outra de rowspan/colspan DIFERENTE, ou por um nó de
         tag diferente (ex.: "td" por "tr"), custa o máximo (1.0);
       - inserir ou remover qualquer nó custa 1.0.
  3. TEDS = 1 - (distância de edição) / max(nº de nós descendentes da raiz em cada
      árvore) — igual à referência PubTabNet, que exclui a raiz `<table>` do
      denominador. O resultado fica entre 0.0 e 1.0.

S-TEDS (ver s_teds.py) reaproveita esta mesma função após remover o texto e neutralizar
`td`/`th`, pois os gabaritos públicos não anotam cabeçalhos de forma compatível. Assim,
avalia linhas, colunas ordenadas e spans sem usar texto ou a distinção do cabeçalho.

Como a árvore não codifica coordenadas absolutas de coluna nem lacunas implícitas,
deslocamentos que preservam a ordem dos nós podem manter TEDS=1. Use F1 por célula
para avaliar posição e extensão exatas.
"""

from __future__ import annotations

from apted import APTED, Config
from apted.helpers import Tree

from src.normalization.normalize_output import NormalizedTable

METRIC_PROTOCOL_VERSION = "teds_grid_v3"


class _TableTreeNode(Tree):
    """
    Nó de árvore compatível com apted.helpers.Tree (usa `.name` para a tag e
    `.children` para os filhos — ambos lidos pela Config padrão da biblioteca),
    com campos extras (`colspan`, `rowspan`, `text`) usados apenas por `_TEDSConfig`
    para nós do tipo "td"/"th".
    """

    def __init__(self, tag: str, colspan: int = 1, rowspan: int = 1, text: str = "", *children):
        super().__init__(tag, *children)
        self.colspan = colspan
        self.rowspan = rowspan
        self.text = text


def _levenshtein(a: str, b: str) -> int:
    """
    Distância de edição (Levenshtein) padrão entre duas strings, usada para comparar
    o conteúdo textual de células correspondentes. Implementação direta do algoritmo
    clássico de programação dinâmica (sem dependência externa).
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous_row = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current_row = [i]
        for j, char_b in enumerate(b, start=1):
            insert_cost = current_row[j - 1] + 1
            delete_cost = previous_row[j] + 1
            replace_cost = previous_row[j - 1] + (char_a != char_b)
            current_row.append(min(insert_cost, delete_cost, replace_cost))
        previous_row = current_row

    return previous_row[-1]


class _TEDSConfig(Config):
    """Custo de edição específico para árvores de tabela, conforme descrito acima."""

    def rename(self, node1: _TableTreeNode, node2: _TableTreeNode) -> float:
        if node1.name != node2.name:
            return 1.0

        if node1.name not in ("td", "th"):
            # Nós estruturais sem atributos (table, tr): mesma tag já basta.
            return 0.0

        if node1.colspan != node2.colspan or node1.rowspan != node2.rowspan:
            return 1.0

        if node1.text == node2.text:
            return 0.0

        max_len = max(len(node1.text), len(node2.text), 1)
        return _levenshtein(node1.text, node2.text) / max_len

    def children(self, node: _TableTreeNode) -> list:
        return node.children

    def delete(self, node: _TableTreeNode) -> float:
        return 1.0

    def insert(self, node: _TableTreeNode) -> float:
        return 1.0


def _build_tree(table: NormalizedTable) -> _TableTreeNode:
    by_row: dict[int, list] = {}
    for cell in table.cells:
        by_row.setdefault(cell.row, []).append(cell)

    row_nodes = []
    occupied: set[tuple[int, int]] = set()
    for row_idx in range(table.n_rows):
        cells_in_row = sorted(by_row.get(row_idx, []), key=lambda c: c.col)
        cell_nodes = []
        cursor = 0
        for cell in cells_in_row:
            while cursor < min(cell.col, table.n_cols):
                if (row_idx, cursor) not in occupied:
                    cell_nodes.append(_TableTreeNode("td", text=""))
                    occupied.add((row_idx, cursor))
                cursor += 1

            if cell.col >= table.n_cols or (row_idx, cell.col) in occupied:
                continue

            cell_nodes.append(
                _TableTreeNode(
                    "th" if cell.is_header else "td",
                    colspan=cell.col_span,
                    rowspan=cell.row_span,
                    text=cell.text,
                )
            )
            for occupied_row in range(row_idx, min(row_idx + cell.row_span, table.n_rows)):
                for occupied_col in range(cell.col, min(cell.col + cell.col_span, table.n_cols)):
                    occupied.add((occupied_row, occupied_col))
            cursor = cell.col + cell.col_span

        while cursor < table.n_cols:
            if (row_idx, cursor) not in occupied:
                cell_nodes.append(_TableTreeNode("td", text=""))
                occupied.add((row_idx, cursor))
            cursor += 1

        row_nodes.append(_TableTreeNode("tr", 1, 1, "", *cell_nodes))

    return _TableTreeNode("table", 1, 1, "", *row_nodes)


def _count_nodes(node: _TableTreeNode) -> int:
    return 1 + sum(_count_nodes(child) for child in node.children)


def compute_teds(predicted: NormalizedTable, ground_truth: NormalizedTable) -> float:
    """
    Calcula o TEDS entre a tabela predita e o gabarito.

    Retorna um valor entre 0.0 (nenhuma similaridade estrutural/textual) e 1.0
    (estruturas e conteúdos idênticos).
    """
    tree_pred = _build_tree(predicted)
    tree_gt = _build_tree(ground_truth)

    n_pred = _count_nodes(tree_pred) - 1
    n_gt = _count_nodes(tree_gt) - 1

    if n_pred == 0 and n_gt == 0:
        return 1.0
    if n_pred == 0 or n_gt == 0:
        return 0.0

    edit_distance = APTED(tree_pred, tree_gt, _TEDSConfig()).compute_edit_distance()

    return max(0.0, 1.0 - (edit_distance / max(n_pred, n_gt)))


if __name__ == "__main__":
    pass
