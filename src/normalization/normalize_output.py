"""
Normalização das saídas dos diferentes modelos para uma representação tabular comum
(Seção 4.2, etapa "f").

Os algoritmos do estudo produzem saídas em formatos bem diferentes entre si
(confirmado lendo o código-fonte oficial de cada um — ver docstrings dos respectivos
runners em src/models/):

  - TableFormer        -> HTML estrutural + tokens reconhecidos nas células casadas
  - Table Transformer   -> detecções de objetos (caixas de linha/coluna/spanning cell)
  - TabStructNet        -> XML próprio (<tablecell><start_row>...</start_row>...)
  - SLANet               -> HTML estrutural + caixas das células

Este módulo converte qualquer um desses formatos para `NormalizedTable`, usada de
forma uniforme pela avaliação (TEDS, S-TEDS, F1 por célula) e pela exportação para
planilha.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from xml.etree import ElementTree as ET

from bs4 import BeautifulSoup


@dataclass
class NormalizedCell:
    row: int
    col: int
    row_span: int = 1
    col_span: int = 1
    is_header: bool = False
    text: str = ""
    bbox: tuple[float, float, float, float] | None = None


@dataclass
class NormalizedTable:
    cells: list[NormalizedCell] = field(default_factory=list)
    n_rows: int = 0
    n_cols: int = 0

    def to_html(self) -> str:
        """Reconstrói uma representação HTML simplificada, usada pelo cálculo de TEDS."""
        rows_html = []
        by_row: dict[int, list[NormalizedCell]] = {}
        for cell in self.cells:
            by_row.setdefault(cell.row, []).append(cell)

        for row_idx in sorted(by_row):
            cells_html = []
            for cell in sorted(by_row[row_idx], key=lambda c: c.col):
                tag = "th" if cell.is_header else "td"
                attrs = ""
                if cell.row_span > 1:
                    attrs += f' rowspan="{cell.row_span}"'
                if cell.col_span > 1:
                    attrs += f' colspan="{cell.col_span}"'
                cells_html.append(f"<{tag}{attrs}>{cell.text}</{tag}>")
            rows_html.append(f"<tr>{''.join(cells_html)}</tr>")

        return f"<table>{''.join(rows_html)}</table>"


def normalize_from_html(raw_html: str) -> NormalizedTable:
    """
    Converte uma saída em HTML (formato comum a TableFormer, VAST e outros) para
    NormalizedTable. Também aceita fragmentos contendo <thead>/<tr> sem <table>,
    formato emitido pelo TableFormer. Útil como implementação de referência para
    adaptar os demais formatos (matrizes, sequências) a um HTML equivalente.
    """
    soup = BeautifulSoup(raw_html, "html.parser")
    table = soup.find("table")
    if table is None:
        if soup.find("tr") is None:
            return NormalizedTable()
        # TableFormer may return an HTML table fragment with <thead>/<tr> but no
        # enclosing <table>; treat the fragment as the table root.
        table = soup

    cells: list[NormalizedCell] = []
    max_col = 0
    # Ocupação de grade para lidar corretamente com rowspan/colspan.
    occupied: dict[tuple[int, int], bool] = {}

    for row_idx, tr in enumerate(table.find_all("tr")):
        col_idx = 0
        for cell_tag in tr.find_all(["td", "th"]):
            while occupied.get((row_idx, col_idx)):
                col_idx += 1

            row_span = int(cell_tag.get("rowspan", 1))
            col_span = int(cell_tag.get("colspan", 1))
            is_header = cell_tag.name == "th"
            text = cell_tag.get_text(strip=True)

            cells.append(
                NormalizedCell(
                    row=row_idx,
                    col=col_idx,
                    row_span=row_span,
                    col_span=col_span,
                    is_header=is_header,
                    text=text,
                )
            )

            for r in range(row_idx, row_idx + row_span):
                for c in range(col_idx, col_idx + col_span):
                    occupied[(r, c)] = True

            max_col = max(max_col, col_idx + col_span - 1)
            col_idx += col_span

    n_rows = max((c.row + c.row_span for c in cells), default=0)
    n_cols = max_col + 1 if cells else 0

    return NormalizedTable(cells=cells, n_rows=n_rows, n_cols=n_cols)


def normalize_from_tableformer(raw_content) -> NormalizedTable:
    """Normalize TableFormer structure and merge text returned by cell matching."""
    if isinstance(raw_content, str):
        return normalize_from_html(raw_content)
    if not isinstance(raw_content, dict):
        raise TypeError("A saída estruturada do TableFormer deve ser texto HTML ou dict.")

    table = normalize_from_html(raw_content["html"])
    cells_by_position = {(cell.row, cell.col): cell for cell in table.cells}
    for matched_cell in raw_content.get("matched_cells", []):
        position = (
            int(matched_cell["start_row_offset_idx"]),
            int(matched_cell["start_col_offset_idx"]),
        )
        cell = cells_by_position.get(position)
        if cell is None:
            raise ValueError(
                f"TableFormer retornou texto para uma célula fora da estrutura: {position}."
            )

        text_boxes = sorted(
            matched_cell.get("text_cell_bboxes", []),
            key=lambda box: (float(box.get("t", 0)), float(box.get("l", 0))),
        )
        cell.text = " ".join(
            str(box["token"]).strip()
            for box in text_boxes
            if str(box.get("token", "")).strip()
        )
        cell.is_header = bool(matched_cell.get("column_header", False))

        bbox = matched_cell.get("bbox")
        if bbox and all(key in bbox for key in ("l", "t", "r", "b")):
            cell.bbox = (
                float(bbox["l"]),
                float(bbox["t"]),
                float(bbox["r"]),
                float(bbox["b"]),
            )

    return table


def normalize_from_slanet(raw_content) -> NormalizedTable:
    """Normalize PP-Structure's HTML and attach SLANet's cell boxes in order."""
    if not isinstance(raw_content, dict) or not isinstance(
        raw_content.get("html"), str
    ):
        raise TypeError("A saída do SLANet deve conter HTML e caixas das células.")

    table = normalize_from_html(raw_content["html"])
    for cell in table.cells:
        cell.text = ""

    cell_boxes = raw_content.get("cell_bbox", [])
    if not isinstance(cell_boxes, list) or len(cell_boxes) != len(table.cells):
        raise ValueError(
            "SLANet retornou quantidade de caixas incompatível com as células: "
            f"{len(cell_boxes) if isinstance(cell_boxes, list) else 'inválida'} "
            f"caixas para {len(table.cells)} células."
        )
    for cell, box in zip(table.cells, cell_boxes):
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError(f"Caixa de célula SLANet inválida: {box!r}")
        cell.bbox = tuple(float(value) for value in box)
    return table


def normalize_from_object_detection(
    detections: list[dict], iou_threshold: float = 0.5
) -> NormalizedTable:
    """
    Converte a saída bruta do Table Transformer (lista de detecções, cada uma com
    `label` em {"table row", "table column", "table column header",
    "table projected row header", "table spanning cell", ...}, `score` e `box`
    [xmin, ymin, xmax, ymax]) em uma NormalizedTable.

    Diferente de um modelo que já emite HTML/estrutura lógica, aqui a grade precisa
    ser RECONSTRUÍDA geometricamente:
      1. Ordena as caixas de "table row" por y (topo->base) e as de "table column"
         por x (esquerda->direita) — cada uma define uma faixa de coordenadas.
      2. Cada célula da grade é a interseção entre uma faixa de linha e uma faixa de
         coluna.
      3. Caixas de "table spanning cell" que cobrem mais de uma célula da grade
         (por sobreposição/IoU) definem mesclagens (rowspan/colspan > 1).
      4. Linhas cuja faixa se sobrepõe a uma detecção de "table column header" são
         marcadas como cabeçalho.

    Esta é a mesma estratégia (interseção geométrica linha x coluna, com mesclagem
    por sobreposição de "spanning cell") descrita informalmente na documentação do
    Table Transformer para converter detecções em uma tabela navegável — aqui
    reimplementada para produzir diretamente uma `NormalizedTable`.
    """
    rows = sorted(
        (d for d in detections if d["label"] == "table row"), key=lambda d: d["box"][1]
    )
    cols = sorted(
        (d for d in detections if d["label"] == "table column"), key=lambda d: d["box"][0]
    )
    headers = [d for d in detections if d["label"] == "table column header"]
    spanning_cells = [d for d in detections if d["label"] == "table spanning cell"]

    if not rows or not cols:
        return NormalizedTable()

    def _overlaps_vertically(box_a, box_b) -> bool:
        return box_a[1] < box_b[3] and box_b[1] < box_a[3]

    def _overlaps_horizontally(box_a, box_b) -> bool:
        return box_a[0] < box_b[2] and box_b[0] < box_a[2]

    def _row_is_header(row_box) -> bool:
        return any(_overlaps_vertically(row_box, h["box"]) for h in headers)

    cells: list[NormalizedCell] = []
    covered: set[tuple[int, int]] = set()

    for row_idx, row in enumerate(rows):
        for col_idx, col in enumerate(cols):
            if (row_idx, col_idx) in covered:
                continue

            row_span, col_span = 1, 1
            for span in spanning_cells:
                if _overlaps_vertically(span["box"], row["box"]) and _overlaps_horizontally(
                    span["box"], col["box"]
                ):
                    extra_rows = sum(
                        1 for r in rows[row_idx:] if _overlaps_vertically(span["box"], r["box"])
                    )
                    extra_cols = sum(
                        1 for c in cols[col_idx:] if _overlaps_horizontally(span["box"], c["box"])
                    )
                    row_span, col_span = max(row_span, extra_rows), max(col_span, extra_cols)

            covered_rows = rows[row_idx : row_idx + row_span]
            covered_cols = cols[col_idx : col_idx + col_span]
            cell_bbox = (
                min(c["box"][0] for c in covered_cols),
                min(r["box"][1] for r in covered_rows),
                max(c["box"][2] for c in covered_cols),
                max(r["box"][3] for r in covered_rows),
            )

            for r in range(row_idx, row_idx + row_span):
                for c in range(col_idx, col_idx + col_span):
                    covered.add((r, c))

            cells.append(
                NormalizedCell(
                    row=row_idx,
                    col=col_idx,
                    row_span=row_span,
                    col_span=col_span,
                    is_header=_row_is_header(row["box"]),
                    text="",  # o Table Transformer não extrai texto; requer OCR à parte
                    bbox=cell_bbox,
                )
            )

    return NormalizedTable(cells=cells, n_rows=len(rows), n_cols=len(cols))


def normalize_from_tabstructnet_xml(xml_content: str) -> NormalizedTable:
    """
    Converte o XML produzido pelo pós-processador oficial do TabStructNet
    (xml_generating_postprocessor/cell_postprocessor_adj.py) em NormalizedTable.

    Formato confirmado no código-fonte do repositório oficial:
        <annotations><object><cells>
          <tablecell>
            <start_row>0</start_row><end_row>0</end_row>
            <start_col>0</start_col><end_col>0</end_col>
          </tablecell>
          ...
        </cells></object></annotations>

    O TabStructNet não produz o texto da célula (apenas a estrutura física); o campo
    `text` fica vazio aqui, a ser preenchido por uma etapa separada de TCR/OCR se
    necessário.
    """
    root = ET.fromstring(xml_content)
    cells: list[NormalizedCell] = []
    max_row, max_col = 0, 0

    for tablecell in root.iterfind(".//tablecell"):
        start_row = int(tablecell.findtext("start_row", "0"))
        end_row = int(tablecell.findtext("end_row", str(start_row)))
        start_col = int(tablecell.findtext("start_col", "0"))
        end_col = int(tablecell.findtext("end_col", str(start_col)))
        coordinates = [
            tablecell.findtext(name)
            for name in ("x0", "y0", "x1", "y1")
        ]
        bbox = (
            tuple(float(value) for value in coordinates)
            if all(value is not None for value in coordinates)
            else None
        )

        cells.append(
            NormalizedCell(
                row=start_row,
                col=start_col,
                row_span=max(end_row - start_row + 1, 1),
                col_span=max(end_col - start_col + 1, 1),
                is_header=False,  # TabStructNet não distingue cabeçalho na saída
                text="",
                bbox=bbox,
            )
        )
        max_row, max_col = max(max_row, end_row), max(max_col, end_col)

    return NormalizedTable(cells=cells, n_rows=max_row + 1, n_cols=max_col + 1)


def normalize_output(model_name: str, raw_content) -> NormalizedTable:
    """
    Ponto de entrada único usado por run_experiment.py. Despacha para a função de
    normalização adequada conforme o formato de saída real de cada modelo (confirmado
    lendo o código-fonte oficial de cada um — ver docstrings dos runners).
    """
    if model_name == "tableformer":
        return normalize_from_tableformer(raw_content)

    if model_name == "tabletransformer":
        return normalize_from_object_detection(raw_content)

    if model_name == "tabstructnet":
        return normalize_from_tabstructnet_xml(raw_content)

    if model_name == "slanet":
        return normalize_from_slanet(raw_content)

    raise ValueError(f"Modelo desconhecido para normalização: {model_name}")
