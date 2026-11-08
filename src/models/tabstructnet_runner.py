"""
Runner para TabStructNet (Raja; Mondal; Jawahar, ECCV 2020).

Este arquivo deve ser executado no ambiente `.venv-tabstructnet`
(ver environment/requirements-tabstructnet.txt — TensorFlow 1.15 / Keras 2.2, exige
Python 3.7–3.9).

Implementação de referência: repositório oficial dos autores,
https://github.com/sachinraja13/TabStructNet, já clonado em
third_party/TabStructNet/. Pesos oficiais (arquivo .h5, treinado em SciTSR):
Google Drive — links exatos encontrados no próprio repositório, em
trained_model/tab/logs/tab20200821T0923/note.txt e
trained_model/tab/logs/tab20200821T0923_2/note.txt (variante fine-tuned em UNLV):
  - mask_rcnn_tab_0100.h5 (SciTSR):
    https://drive.google.com/drive/folders/1DRFbOW5FSl4A73wgw5i9O0JKkRegyO6-
  - mask_rcnn_tab_0120.h5 (SciTSR + fine-tuning UNLV):
    https://drive.google.com/drive/folders/1t5Io8q_g9MkM2KyYXG3s9b0NOVo3RIDw

DETALHES CONFIRMADOS NO CÓDIGO-FONTE (lidos diretamente de samples/tabnet/tabnet.py
e xml_generating_postprocessor/cell_postprocessor_adj.py):
  - Ao contrário do que o nome “Mask R-CNN” sugere isoladamente, o modelo usado aqui
    é uma versão MODIFICADA do Mask_RCNN do matterport (pasta mrcnn/ do próprio
    repositório): além de rois/scores/masks/class_ids padrão, `model.detect([image])`
    retorna também `row_adj` e `col_adj` — matrizes de adjacência que indicam quais
    células pertencem à mesma linha/coluna lógica. Ou seja, a rede já prevê a
    estrutura relacional, não apenas caixas de célula soltas (coerente com a proposta
    do paper de modelar a estrutura como um grafo retilíneo).
  - NÃO é necessário usar o fluxo de CLI baseado em dataset COCO/"evaluate" do
    tabnet.py para uma única imagem: `model.detect([image_rgb], verbose=0)[0]` é a
    API padrão de inferência do Mask_RCNN e funciona diretamente sobre uma imagem
    (função usada internamente por evaluate_tabnet() dentro de um loop).
  - O pós-processamento oficial (cell_postprocessor_adj.py) normalmente lê
    detecções/adjacências de arquivos em disco (pensado para rodar em lote), mas as
    funções de reconstrução da estrutura (`get_column_structure_indices`,
    `get_row_structure_indices`, `get_aligned_column_coordinates`,
    `get_aligned_row_coordinates`, `get_final_table_details_without_merge`,
    `create_root`, `create_cell_annotation`) são funções Python comuns, importáveis e
    chamáveis diretamente sobre os arrays em memória — sem precisar do loop de
    arquivos do `main()` original nem da etapa de OCR (o conteúdo textual da célula é
    opcional na saída oficial: `ADD_CONNTENT_IN_XML = False` por padrão no próprio
    repositório, ou seja, o TabStructNet oficialmente não garante o texto da célula
    na saída — friso isso porque também é a convenção já adotada em
    src/normalization/normalize_output.py, que trata `text=""` para este modelo).

LIMITAÇÃO DESTA IMPLEMENTAÇÃO: por rodar sobre TensorFlow 1.15 / Mask_RCNN (framework
sem suporte a Python >= 3.10), não foi possível testar a execução ponta a ponta no
ambiente de desenvolvimento usado para este repositório (Python 3.12). A integração
abaixo foi escrita diretamente a partir da leitura do código-fonte oficial, mas
precisa de validação assim que o `.venv-tabstructnet` (Python 3.7–3.9) estiver criado
e os pesos baixados.
"""

from __future__ import annotations

import sys
from pathlib import Path

from src.models.base_runner import BaseModelRunner, RawModelOutput

REPO_DIR_DEFAULT = Path("third_party/TabStructNet")


class TabStructNetRunner(BaseModelRunner):
    model_name = "tabstructnet"

    def __init__(self, repo_dir: Path = REPO_DIR_DEFAULT) -> None:
        self._repo_dir = repo_dir
        self._model = None
        self._postproc = None

    def _add_repo_to_syspath(self) -> None:
        for sub in [self._repo_dir, self._repo_dir / "samples" / "tabnet"]:
            sub_str = str(sub)
            if sub_str not in sys.path:
                sys.path.insert(0, sub_str)

    def load_model(self, checkpoint_path: Path | None) -> None:
        import os

        if not self._repo_dir.exists():
            raise FileNotFoundError(
                f"Repositório do TabStructNet não encontrado em {self._repo_dir}. "
                "Clonar github.com/sachinraja13/TabStructNet nesse caminho."
            )
        if checkpoint_path is None or not Path(checkpoint_path).exists():
            raise FileNotFoundError(
                f"Pesos do TabStructNet não encontrados em {checkpoint_path}. "
                "Baixar do Google Drive oficial (ver links na docstring do módulo)."
            )

        self._add_repo_to_syspath()
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf

        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

        # `tabnet` é o módulo samples/tabnet/tabnet.py do repositório oficial; seu
        # bloco de CLI está protegido por `if __name__ == "__main__":`, então importar
        # como módulo não dispara o parsing de argumentos nem nenhum treino/avaliação.
        import tabnet  # noqa: E402
        from mrcnn import model as modellib  # noqa: E402

        class InferenceConfig(tabnet.TabNetConfig):
            # Mesma configuração usada pelo próprio tabnet.py no modo "evaluate".
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
            # The official configuration pads adjacency logits to 2000x2000
            # before a dense layer; cap inference proposals to keep CPU memory bounded.
            DETECTION_MAX_INSTANCES = 300

        checkpoint_path = Path(checkpoint_path)
        model_dir = checkpoint_path.parent if checkpoint_path.suffix == ".h5" else checkpoint_path

        model = modellib.MaskRCNN(
            mode="inference", config=InferenceConfig(), model_dir=str(model_dir)
        )

        weights_path = (
            str(checkpoint_path)
            if checkpoint_path.suffix == ".h5"
            else model.find_last()
        )
        model.load_weights(weights_path, by_name=True)

        self._model = model

        # Módulo de pós-processamento (funções puras, reaproveitadas fora do loop de
        # arquivos original — ver docstring do módulo).
        postproc_dir = self._repo_dir / "xml_generating_postprocessor"
        postproc_dir_str = str(postproc_dir)
        if postproc_dir_str not in sys.path:
            sys.path.insert(0, postproc_dir_str)
        import cell_postprocessor_adj as postproc  # noqa: E402

        self._postproc = postproc

    def predict(self, image_path: Path) -> RawModelOutput:
        if self._model is None:
            raise RuntimeError("Chame load_model() antes de predict().")

        import cv2
        import numpy as np
        from xml.etree import ElementTree as ET

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise FileNotFoundError(f"Não foi possível ler a imagem em {image_path}.")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = image_bgr.shape[:2]

        result = self._model.detect([image_rgb], verbose=0)[0]
        rois = result["rois"]          # formato do Mask_RCNN: [y1, x1, y2, x2]
        row_adj = result["row_adj"]
        col_adj = result["col_adj"]

        if len(rois) == 0:
            # Nenhuma célula detectada; devolve XML de tabela vazia.
            root = self._postproc.create_root(str(image_path), image_path.name, width, height, 3)
            xml_content = ET.tostring(root, encoding="unicode")
            return RawModelOutput(model_name=self.model_name, source_image_path=image_path, content=xml_content)

        # Conversão de [y1,x1,y2,x2] (Mask_RCNN) para [x1,y1,x2,y2] (esperado pelas
        # funções do pós-processador, conforme lido em cell_postprocessor_adj.py).
        table_cells = np.array(
            [[int(r[1]), int(r[0]), int(r[3]), int(r[2])] for r in rois],
            dtype=np.int64,
        )

        start_col, end_col = self._postproc.get_column_structure_indices(col_adj, table_cells)
        start_row, end_row = self._postproc.get_row_structure_indices(row_adj, table_cells)

        # Mesma lógica de saneamento aplicada em cell_postprocessor_adj.main() antes
        # de montar `table_details` (grudar células de borda na margem da tabela,
        # garantir start <= end).
        for i in range(len(table_cells)):
            if end_row[i] < start_row[i]:
                end_row[i] = start_row[i]
            if end_col[i] < start_col[i]:
                end_col[i] = start_col[i]
            if start_row[i] == 0:
                table_cells[i][1] = 0
            if end_row[i] == max(end_row):
                table_cells[i][3] = height
            if start_col[i] == 0:
                table_cells[i][0] = 0
            if end_col[i] == max(end_col):
                table_cells[i][2] = width

        col_starts, col_ends = self._postproc.get_aligned_column_coordinates(
            table_cells, start_col, end_col
        )
        row_starts, row_ends = self._postproc.get_aligned_row_coordinates(
            table_cells, start_row, end_row
        )
        table_details = self._postproc.get_final_table_details_without_merge(
            row_starts, col_starts, row_ends, col_ends
        )

        root = self._postproc.create_root(str(image_path), image_path.name, width, height, 3)
        root = self._postproc.create_cell_annotation(
            root, table_details, [0, 0, width, height], image_bgr, add_content=False
        )

        xml_content = ET.tostring(root, encoding="unicode")

        return RawModelOutput(
            model_name=self.model_name, source_image_path=image_path, content=xml_content
        )


if __name__ == "__main__":
    pass
