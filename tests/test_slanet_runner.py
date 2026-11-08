from __future__ import annotations

import sys
from types import ModuleType
from pathlib import Path

from src.models.slanet_runner import SLANetRunner


def test_slanet_runner_loads_cpu_structure_only_and_returns_raw_table(
    tmp_path: Path, monkeypatch
) -> None:
    captured = {}

    class FakeEngine:
        def __init__(self, **options) -> None:
            captured["options"] = options

        def __call__(self, image):
            captured["image"] = image
            return [
                {
                    "type": "table",
                    "res": {
                        "html": "<table><tr><td>A</td></tr></table>",
                        "cell_bbox": [[1, 2, 3, 4]],
                    },
                }
            ]

    paddleocr = ModuleType("paddleocr")
    paddleocr.PPStructure = FakeEngine
    monkeypatch.setitem(sys.modules, "paddleocr", paddleocr)

    cv2 = ModuleType("cv2")
    cv2.imread = lambda path: {"image_path": path}
    monkeypatch.setitem(sys.modules, "cv2", cv2)

    runner = SLANetRunner(language="en", device="cpu", table_algorithm="SLANet")
    runner.load_model(tmp_path / "table-model")
    output = runner.predict(tmp_path / "sample.png")

    assert captured["options"] == {
        "table": True,
        "layout": False,
        "ocr": False,
        "image_orientation": False,
        "show_log": False,
        "lang": "en",
        "use_gpu": False,
        "table_algorithm": "SLANet",
        "table_model_dir": str(tmp_path / "table-model"),
    }
    assert output.model_name == "slanet"
    assert output.content == {
        "html": "<table><tr><td>A</td></tr></table>",
        "cell_bbox": [[1, 2, 3, 4]],
    }


def test_slanet_runner_rejects_missing_image(tmp_path: Path, monkeypatch) -> None:
    paddleocr = ModuleType("paddleocr")
    paddleocr.PPStructure = lambda **options: lambda image: []
    monkeypatch.setitem(sys.modules, "paddleocr", paddleocr)
    cv2 = ModuleType("cv2")
    cv2.imread = lambda path: None
    monkeypatch.setitem(sys.modules, "cv2", cv2)

    runner = SLANetRunner()
    runner.load_model(None)

    try:
        runner.predict(tmp_path / "missing.png")
    except ValueError as error:
        assert "Não foi possível abrir a imagem" in str(error)
    else:
        raise AssertionError("Expected missing image to be reported")
