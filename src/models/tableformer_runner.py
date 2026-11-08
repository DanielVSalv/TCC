"""
Runner para TableFormer (Nassar et al., CVPR 2022).

Este arquivo deve ser executado no ambiente `.venv-tableformer`
(ver environment/requirements-tableformer.txt).

Implementação de referência: pacote `docling-ibm-models` (IBM/Docling),
https://github.com/docling-project/docling-ibm-models — inspecionado diretamente
(clone raso) para confirmar a API real abaixo. Checkpoint oficial publicado no
Hugging Face em `ds4sd/docling-models` (baixado via `huggingface_hub.snapshot_download`).

DETALHE IMPORTANTE DA ARQUITETURA REAL (confirmado no código-fonte):
O TFPredictor NÃO recebe apenas uma imagem "crua". Ele espera uma "iocr_page" com a
imagem e as palavras já extraídas via OCR/parsing de PDF (ver
src/data_prep/build_iocr_page.py — schema mínimo confirmado por leitura direta de
tf_cell_matcher.py: cada palavra precisa apenas de `id`, `text` e `bbox`). O modelo
prevê a estrutura (sequência OTSL) e casa (matching) as células com esse texto.

A saída inclui a estrutura HTML e os tokens de texto casados em
`multi_table_predict()`; ambos são normalizados em
`src/normalization/normalize_output.py`.

TODO:
  - Definir a variante do checkpoint (`fast` ou `accurate`) em
    configs/experiment_config.yaml (ambas ficam em
    model_artifacts/tableformer/<variante> dentro do snapshot baixado).
  - Para tabelas extraídas de PDF com texto nativo (não escaneado), considerar usar
    `build_iocr_page_from_pdf_text` em vez de `build_iocr_page_from_image` (OCR),
    pela maior precisão do texto nativo — ver docstring de build_iocr_page.py.

NOTA: "dataset_wordmap" (necessário para o decodificador da sequência OTSL) já vem
preenchido em DEFAULT_MODEL_CONFIG — ver src/models/tableformer_wordmap.py para a
proveniência exata (extraído programaticamente do test_config oficial do
docling-ibm-models) e o aviso sobre compatibilidade caso um checkpoint diferente do
padrão (`ds4sd/docling-models`) seja usado.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.models.base_runner import BaseModelRunner, RawModelOutput
from src.models.tableformer_wordmap import DEFAULT_DATASET_WORDMAP

# Configuração-modelo baseada no `test_config` real usado nos testes do docling-ibm-models
# (tests/test_tf_predictor.py). Os campos de "dataset_wordmap" foram omitidos aqui por
# brevidade — devem ser copiados do arquivo de teste do repositório ou do wordmap
# distribuído junto ao checkpoint.
DEFAULT_MODEL_CONFIG: dict[str, Any] = {
    "dataset": {
        "type": "TF_prepared",
        "name": "TF",
        "load_cells": True,
        "bbox_format": "5plet",
        "resized_image": 448,
        "keep_AR": False,
        "up_scaling_enabled": True,
        "down_scaling_enabled": True,
        "padding_mode": "null",
        "padding_color": [0, 0, 0],
        "image_normalization": {
            "state": True,
            "mean": [0.94247851, 0.94254675, 0.94292611],
            "std": [0.17910956, 0.17940403, 0.17931663],
        },
        "color_jitter": False,
        "rand_crop": False,
        "rand_pad": False,
        "image_grayscale": False,
    },
    "model": {
        "type": "TableModel04_rs",
        "name": "14_128_256_4_true",
        # "save_dir" é preenchido em load_model() com o caminho do checkpoint baixado.
        "backbone": "resnet18",
        "enc_image_size": 28,
        "tag_embed_dim": 16,
        "hidden_dim": 512,
        "tag_decoder_dim": 512,
        "bbox_embed_dim": 256,
        "tag_attention_dim": 256,
        "bbox_attention_dim": 512,
        "enc_layers": 4,
        "dec_layers": 2,
        "nheads": 8,
        "dropout": 0.1,
        "bbox_classes": 2,
    },
     "train": {
        "save_periodicity": 1,
        "disable_cuda": False,
        "epochs": 23,
        "batch_size": 50,
        "clip_gradient": 0.1,
        "clip_max_norm": 0.1,
        "bbox": True,
        "validation": False,
    },
    "predict": {
        "max_steps": 1024,
        "beam_size": 5,
        "bbox": True,
        "pdf_cell_iou_thres": 0.05,
        "padding": False,
        "padding_size": 50,
        "disable_post_process": False,
        "profiling": False,
    },
    # Copiado do test_config oficial (ver src/models/tableformer_wordmap.py para a
    # proveniência exata e o aviso sobre compatibilidade com outros checkpoints).
    "dataset_wordmap": DEFAULT_DATASET_WORDMAP,
}


class TableFormerRunner(BaseModelRunner):
    model_name = "tableformer"

    def __init__(
        self,
        device: str = "cpu",
        variant: str = "fast",
        revision: str = "v2.1.0",
    ) -> None:
        self._predictor = None
        self._device = device
        self._variant = variant  # "fast" ou "accurate"
        self._revision = revision

    def load_model(self, checkpoint_path: Path | None) -> None:
        """
        Baixa o checkpoint oficial se `checkpoint_path` for None; caso contrário,
        usa o diretório local configurado. O diretório local deve conter os pesos
        da variante selecionada.
        """
        from docling_ibm_models.tableformer.data_management.tf_predictor import (
            TFPredictor,
        )

        if checkpoint_path is None:
            from huggingface_hub import snapshot_download

            download_path = snapshot_download(
                repo_id="ds4sd/docling-models", revision=self._revision
            )
            save_dir = (
                Path(download_path)
                / "model_artifacts"
                / "tableformer"
                / self._variant
            )
        else:
            save_dir = Path(checkpoint_path)

        config = dict(DEFAULT_MODEL_CONFIG)
        config["model"] = dict(config["model"])
        config["model"]["save_dir"] = str(save_dir)

        self._predictor = TFPredictor(config, device=self._device, num_threads=4)


    def predict(self, image_path: Path) -> RawModelOutput:
        """
        Modo standalone: constrói o "iocr_page" via Tesseract. O orquestrador usa
        `predict_with_words()` para fornecer os tokens compartilhados (OCR ou PDF
        vetorial). `image_path` deve ser o recorte de uma tabela, então
        `table_bboxes` cobre a imagem inteira.
        """
        from src.data_prep.build_iocr_page import build_iocr_page_from_image

        iocr_page = build_iocr_page_from_image(image_path)
        return self._predict_iocr_page(image_path, iocr_page)

    def predict_with_words(self, image_path: Path, words: list[dict]) -> RawModelOutput:
        """Run prediction with shared OCR/native-PDF words in pixel coordinates."""
        from src.data_prep.build_iocr_page import build_iocr_page_from_words

        iocr_page = build_iocr_page_from_words(image_path, words)
        return self._predict_iocr_page(image_path, iocr_page)

    def _predict_iocr_page(self, image_path: Path, iocr_page: dict) -> RawModelOutput:
        if self._predictor is None:
            raise RuntimeError("Chame load_model() antes de predict().")

        table_bboxes = [[0, 0, iocr_page["width"], iocr_page["height"]]]

        multi_tf_output = self._predictor.multi_table_predict(
            iocr_page,
            table_bboxes,
            do_matching=True,
            correct_overlapping_cells=False,
            sort_row_col_indexes=True,
        )
        
        prediction = multi_tf_output[0]["predict_details"]["prediction"]
        html_seq = prediction["html_seq"]
        html = "".join(html_seq) if isinstance(html_seq, list) else html_seq

        return RawModelOutput(
            model_name=self.model_name,
            source_image_path=image_path,
            content={
                "html": html,
                "matched_cells": multi_tf_output[0]["tf_responses"],
            },
        )


if __name__ == "__main__":
    pass
