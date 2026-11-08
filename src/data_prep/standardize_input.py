"""
Padronização das entradas antes de serem enviadas a cada algoritmo (Seção 4.2, etapa "d").

Os recortes corporativos passam por `standardize_image()` durante a extração e são
salvos como RGB PNG com dimensão máxima configurada antes de `run_experiment.py`
carregá-los. Os datasets públicos preservam os arquivos originais para não alterar a
entrada oficial; cada runner usa o pré-processador nativo do modelo (por exemplo,
processor DETR do Table Transformer e redimensionamento interno do TableFormer).
Este módulo concentra apenas a padronização comum dos recortes corporativos.
"""

from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass
class StandardizationConfig:
    max_dimension: int = 1024       # redimensiona mantendo proporção, lado maior <= isso
    convert_to_rgb: bool = True
    output_format: str = "PNG"


def standardize_image(image: Image.Image, config: StandardizationConfig) -> Image.Image:
    """Aplica redimensionamento e normalização básica de formato à imagem da tabela."""
    if config.convert_to_rgb:
        image = image.convert("RGB")

    width, height = image.size
    scale = config.max_dimension / max(width, height)
    if scale < 1:
        new_size = (int(width * scale), int(height * scale))
        image = image.resize(new_size, Image.LANCZOS)

    return image


def save_standardized(image: Image.Image, output_path: Path, config: StandardizationConfig) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format=config.output_format)
    return output_path


if __name__ == "__main__":
    # Exemplo de uso manual durante o desenvolvimento:
    # cfg = StandardizationConfig()
    # img = Image.open("data/processed/exemplo_table.png")
    # std_img = standardize_image(img, cfg)
    # save_standardized(std_img, Path("data/processed/exemplo_table_std.png"), cfg)
    pass
