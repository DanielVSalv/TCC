"""SLANet table-structure runner using PaddleOCR PP-StructureV2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.models.base_runner import BaseModelRunner, RawModelOutput


class SLANetRunner(BaseModelRunner):
    model_name = "slanet"

    def __init__(
        self,
        language: str = "en",
        device: str = "cpu",
        table_algorithm: str = "SLANet",
    ) -> None:
        if device not in {"cpu", "gpu"}:
            raise ValueError("SLANet device deve ser 'cpu' ou 'gpu'.")
        self._language = language
        self._device = device
        self._table_algorithm = table_algorithm
        self._engine: Any = None

    def load_model(self, checkpoint_path: Path | None) -> None:
        from paddleocr import PPStructure

        options: dict[str, Any] = {
            "table": True,
            "layout": False,
            "ocr": False,
            "image_orientation": False,
            "show_log": False,
            "lang": self._language,
            "use_gpu": self._device == "gpu",
            "table_algorithm": self._table_algorithm,
        }
        if checkpoint_path is not None:
            options["table_model_dir"] = str(checkpoint_path)
        self._engine = PPStructure(**options)

    def predict(self, image_path: Path) -> RawModelOutput:
        if self._engine is None:
            raise RuntimeError("Chame load_model() antes de predict().")

        import cv2

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Não foi possível abrir a imagem: {image_path}")

        results = self._engine(image)
        table_results = [
            result
            for result in results
            if isinstance(result, dict) and result.get("type") == "table"
        ]
        if len(table_results) != 1:
            raise ValueError(
                f"SLANet/PP-Structure deveria retornar uma tabela para o recorte "
                f"{image_path}, retornou {len(table_results)}."
            )

        table_result = table_results[0].get("res")
        if not isinstance(table_result, dict) or not table_result.get("html"):
            raise ValueError(f"SLANet não retornou HTML de tabela para {image_path}.")

        return RawModelOutput(
            model_name=self.model_name,
            source_image_path=image_path,
            content={
                "html": table_result["html"],
                "cell_bbox": table_result.get("cell_bbox", []),
            },
        )
