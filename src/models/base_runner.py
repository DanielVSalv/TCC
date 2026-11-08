"""
Interface comum que todo runner de modelo (TableFormer, TabStructNet, Table
Transformer, SLANet) deve implementar. Isso permite que src/run_experiment.py
trate qualquer modelo de forma uniforme, e que a normalização/avaliação (Seção 4.2,
etapas "f" e "g") independam de qual algoritmo gerou a saída.

Cada modelo roda no seu próprio ambiente virtual (ver README.md e configs/experiment_config.yaml).
Se optar por não rodar cada modelo como subprocesso isolado, alternativa é implementar
cada runner como um script standalone que é chamado via subprocess a partir do venv
correspondente, e que lê/escreve arquivos intermediários em disco (mais simples de
garantir isolamento de dependências).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class RawModelOutput:
    """
    Saída bruta de um modelo, antes da normalização. O formato de `content` varia por
    modelo — estrutura e tokens casados para TableFormer, lista de detecções de
    objetos para o Table Transformer, XML próprio para TabStructNet, HTML estrutural
    e caixas de célula para SLANet —
    e é resolvido pelo respectivo módulo em src/normalization/.
    """

    model_name: str
    source_image_path: Path
    content: Any
    raw_confidence: float | None = None


class BaseModelRunner(ABC):
    """Contrato que cada `<modelo>_runner.py` deve implementar."""

    model_name: str = "base"

    @abstractmethod
    def load_model(self, checkpoint_path: Path | None) -> None:
        """Carrega os pesos/checkpoint do modelo em memória."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, image_path: Path) -> RawModelOutput:
        """Executa a inferência sobre uma imagem de tabela já padronizada."""
        raise NotImplementedError

    def predict_batch(self, image_paths: list[Path]) -> list[RawModelOutput]:
        """Implementação padrão sequencial; pode ser sobrescrita para batching real."""
        return [self.predict(p) for p in image_paths]
