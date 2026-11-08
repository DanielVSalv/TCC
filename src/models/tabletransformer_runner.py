"""
Runner para Table Transformer / TATR (Smock et al., "PubTables-1M: Towards
Comprehensive Table Extraction From Unstructured Documents", 2022).

Este arquivo deve ser executado no ambiente `.venv-tabletransformer`
(ver environment/requirements-tabletransformer.txt).

Implementação de referência: integrado oficialmente à biblioteca `transformers` da
Hugging Face como `TableTransformerForObjectDetection`. Código-fonte original em
https://github.com/microsoft/table-transformer. Checkpoint oficial no Hugging Face:
`microsoft/table-transformer-structure-recognition` (licença MIT).

DETALHE IMPORTANTE DA ARQUITETURA REAL:
Diferente do TableFormer (que gera HTML diretamente), o Table Transformer é um
detector de objetos (arquitetura DETR) que prevê caixas delimitadoras com rótulos
como "table row", "table column", "table column header" e "table spanning cell" —
ele NÃO gera uma estrutura de tabela pronta. A reconstrução da grade lógica
(quais células existem, quais estão mescladas) precisa ser feita a partir da
interseção geométrica entre as caixas de linha e de coluna previstas, com as caixas
de "spanning cell" indicando mesclagens. Essa lógica de pós-processamento está
implementada em `src/normalization/normalize_output.py`
(`normalize_from_object_detection`).

TODO:
  - Ajustar o limiar de confiança (`score_threshold`) conforme os experimentos.
  - Validar se a variante do checkpoint usada (`microsoft/table-transformer-structure-recognition`,
    treinado em PubTables-1M, ou a variante `-v1.1-all`, que combina PubTables-1M e
    FinTabNet.c) é a mais adequada para os datasets do estudo — documentar a escolha
    na Seção 4/6 do TCC.
"""

from __future__ import annotations

from pathlib import Path

from src.models.base_runner import BaseModelRunner, RawModelOutput

DEFAULT_CHECKPOINT = "microsoft/table-transformer-structure-recognition"


class TableTransformerRunner(BaseModelRunner):
    model_name = "tabletransformer"

    def __init__(
        self,
        score_threshold: float = 0.7,
        device: str = "cpu",
        revision: str | None = None,
    ) -> None:
        self._model = None
        self._image_processor = None
        self._score_threshold = score_threshold
        self._device = device
        self._revision = revision

    def load_model(self, checkpoint_path: str | Path | None) -> None:
        """
        `checkpoint_path` pode ser o identificador do Hugging Face Hub
        (DEFAULT_CHECKPOINT) ou um diretório local com um checkpoint já baixado via
        `from_pretrained(..., use_pretrained_backbone=False)` + `save_pretrained(...)`
        (necessário para uso totalmente offline — ver discussão oficial do modelo).
        """
        import torch
        from transformers import AutoImageProcessor, TableTransformerForObjectDetection

        checkpoint = str(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT

        self._model = TableTransformerForObjectDetection.from_pretrained(
            checkpoint, revision=self._revision
        )
        self._model.to(self._device)
        self._model.eval()

        # O pré-processador de imagem "detr" é compatível com a família DETR/Table
        # Transformer (redimensionamento + normalização ImageNet).
        self._image_processor = AutoImageProcessor.from_pretrained(
            checkpoint, revision=self._revision
        )
        self._torch = torch

    def predict(self, image_path: Path) -> RawModelOutput:
        if self._model is None:
            raise RuntimeError("Chame load_model() antes de predict().")

        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        inputs = self._image_processor(images=image, return_tensors="pt").to(self._device)

        with self._torch.no_grad():
            outputs = self._model(**inputs)

        target_sizes = self._torch.tensor([image.size[::-1]])
        results = self._image_processor.post_process_object_detection(
            outputs, threshold=self._score_threshold, target_sizes=target_sizes
        )[0]

        # `content` carrega as detecções brutas (boxes + labels + scores); a
        # reconstrução da grade lógica acontece em normalize_output.py, pois depende
        # de heurísticas de interseção geométrica compartilhadas com outros formatos
        # baseados em detecção de objetos.
        detections = [
            {
                "label": self._model.config.id2label[label.item()],
                "score": score.item(),
                "box": box.tolist(),  # [xmin, ymin, xmax, ymax]
            }
            for score, label, box in zip(
                results["scores"], results["labels"], results["boxes"]
            )
        ]

        return RawModelOutput(
            model_name=self.model_name,
            source_image_path=image_path,
            content=detections,
        )


if __name__ == "__main__":
    pass
