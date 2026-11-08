"""
Exportação da estrutura de tabela normalizada para uma planilha .xlsx
(Seção 4.2, aplicação prática do preenchimento automatizado de planilhas).

Usa openpyxl diretamente (em vez de pandas.to_excel) porque precisamos controlar
mesclagem de células (rowspan/colspan), o que o DataFrame comum não representa.
"""

from pathlib import Path

from openpyxl import Workbook
from openpyxl.utils import get_column_letter

from src.normalization.normalize_output import NormalizedTable


def export_to_xlsx(table: NormalizedTable, output_path: Path, sheet_name: str = "Tabela") -> Path:
    """
    Escreve a `NormalizedTable` em um arquivo .xlsx, aplicando mesclagem de células
    conforme rowspan/colspan, e aplicando negrito às células marcadas como cabeçalho.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wb = Workbook()
    ws = wb.active
    ws.title = sheet_name

    for cell in table.cells:
        # openpyxl é 1-indexado; NormalizedTable é 0-indexado.
        row = cell.row + 1
        col = cell.col + 1
        target = ws.cell(row=row, column=col, value=cell.text)
        if cell.is_header:
            target.font = target.font.copy(bold=True)

        if cell.row_span > 1 or cell.col_span > 1:
            end_row = row + cell.row_span - 1
            end_col = col + cell.col_span - 1
            ws.merge_cells(
                start_row=row, start_column=col, end_row=end_row, end_column=end_col
            )

    # Ajuste simples de largura de coluna com base no conteúdo.
    for col_idx in range(1, table.n_cols + 1):
        col_letter = get_column_letter(col_idx)
        max_len = max(
            (len(c.text) for c in table.cells if c.col + 1 == col_idx),
            default=10,
        )
        ws.column_dimensions[col_letter].width = min(max(max_len + 2, 10), 40)

    wb.save(output_path)
    return output_path


if __name__ == "__main__":
    # Exemplo de uso manual:
    # from src.normalization.normalize_output import normalize_from_html
    # table = normalize_from_html(
    #     "<table><tr><th>Nome</th><th>Valor</th></tr>"
    #     "<tr><td>Item A</td><td>10</td></tr></table>"
    # )
    # export_to_xlsx(table, Path("results/spreadsheets/exemplo.xlsx"))
    pass
