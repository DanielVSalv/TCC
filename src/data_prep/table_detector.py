"""
Detecção de tabela em página de documento (Table Detection, TD — ver Seção 2.2 do
TCC), usada para localizar a tabela DENTRO de documentos corporativos reais não
anotados, antes de recortar a região e alimentá-la nos algoritmos de TSR/TR do
estudo (TableFormer, TabStructNet, Table Transformer, SLANet).

Os datasets públicos (PubTabNet, FinTabNet, SciTSR, IBTR) já vêm com a tabela
recortada/anotada — esta etapa só é necessária para o subconjunto de documentos
corporativos (ver extract_table_regions.extract_from_pdf).

Modelo escolhido: Table Transformer (TATR), variante de DETECÇÃO —
checkpoint oficial `microsoft/table-transformer-detection` (Smock et al.,
"PubTables-1M", 2022; mesma família do modelo de Table Transformer já usado para
reconhecimento de estrutura em src/models/tabletransformer_runner.py, mas com um
checkpoint diferente e treinado para uma tarefa diferente — aqui a saída é só
"onde está a tabela na página", não a estrutura interna dela).

Confirmado via documentação oficial do modelo: usa a MESMA classe
`TableTransformerForObjectDetection` do modelo de estrutura, mudando apenas o
checkpoint. Rótulos previstos pelo checkpoint de detecção: "table" e
"table rotated" (tabelas detectadas em páginas giradas ~90°). Esta implementação
localiza ambas, mas não faz a rotação automática da imagem para "table rotated" —
ver TODO abaixo.
"""

from dataclasses import dataclass
from pathlib import Path

DEFAULT_CHECKPOINT = "microsoft/table-transformer-detection"


@dataclass
class DetectedTable:
    page_index: int
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1), em pixels da página rasterizada
    score: float
    rotated: bool = False


class TableDetector:
    """
    Carrega uma vez (`load_model`) e reutiliza para detectar tabelas em várias
    páginas (`detect`). Mantido como classe separada dos runners de
    src/models/ porque pertence à etapa de PREPARAÇÃO de dados (Table Detection),
    não à etapa de comparação dos algoritmos de TSR/TR propriamente ditos — embora
    reuse a mesma biblioteca (`transformers`) e o mesmo ambiente
    (`.venv-tabletransformer`, já que ambos dependem de `transformers`+`torch`).
    """

    def __init__(self, score_threshold: float = 0.8, device: str = "cpu") -> None:
        self._model = None
        self._image_processor = None
        self._score_threshold = score_threshold
        self._device = device

    def load_model(self, checkpoint: str = DEFAULT_CHECKPOINT) -> None:
        from transformers import AutoImageProcessor, TableTransformerForObjectDetection

        self._model = TableTransformerForObjectDetection.from_pretrained(checkpoint)
        self._model.to(self._device)
        self._model.eval()
        self._image_processor = AutoImageProcessor.from_pretrained(checkpoint)

        import torch

        self._torch = torch

    def detect(self, page_image, page_index: int = 0) -> list[DetectedTable]:
        """
        `page_image` deve ser uma imagem PIL.Image (RGB) de UMA página inteira já
        rasterizada (ver rasterize_pdf_page em extract_table_regions.py).
        """
        if self._model is None:
            raise RuntimeError("Chame load_model() antes de detect().")

        inputs = self._image_processor(images=page_image, return_tensors="pt").to(self._device)

        with self._torch.no_grad():
            outputs = self._model(**inputs)

        target_sizes = self._torch.tensor([page_image.size[::-1]])
        results = self._image_processor.post_process_object_detection(
            outputs, threshold=self._score_threshold, target_sizes=target_sizes
        )[0]

        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            label_name = self._model.config.id2label[label.item()]
            detections.append(
                DetectedTable(
                    page_index=page_index,
                    bbox=tuple(box.tolist()),
                    score=score.item(),
                    rotated=(label_name == "table rotated"),
                )
            )
        return detections


if __name__ == "__main__":
    pass
