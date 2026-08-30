import os
import zipfile
from types import SimpleNamespace

import gradio as gr
from PIL import Image

from core import all_model_search
from core.exporters import save_plot


class FakeModelManager:
    def __init__(self, names):
        self.models = {name: SimpleNamespace(requires_multiband=False) for name in names}

    def get_model(self, name):
        model = self.models.get(name)
        return (model, None) if model is not None else (None, f"{name} unavailable")


def _single_model_generator(_manager, _query, _threshold, model_name, _filters):
    image = Image.new("RGB", (8, 8), "green")
    yield gr.update(), "Encoding...", gr.update(), gr.update(), gr.update(), gr.update()
    yield (
        [(image, "Score: 1.0000")],
        f"{model_name} complete",
        image,
        [image, image, f"{model_name} report", [], model_name],
        {"model": model_name},
        gr.update(value=image, visible=True),
    )


def test_all_text_models_report_progress_and_build_export_state(monkeypatch):
    monkeypatch.setattr(all_model_search, "search_text", _single_model_generator)
    manager = FakeModelManager(["SigLIP", "FarSLIP"])

    outputs = list(
        all_model_search.search_all_text_models(
            manager,
            "rainforest",
            7,
            ["SigLIP", "FarSLIP"],
        )
    )
    final = outputs[-1]

    assert any("[1/2] - SigLIP" in output[1] for output in outputs if isinstance(output[1], str))
    assert any("[2/2] - FarSLIP" in output[1] for output in outputs if isinstance(output[1], str))
    assert "Completed 2/2 available models." in final[1]
    assert len(final[0]) == 2
    assert final[0][0][1].startswith("SigLIP\n")
    assert final[3]["kind"] == "all_models"
    assert [item["model_name"] for item in final[3]["models"]] == ["SigLIP", "FarSLIP"]


def test_all_model_export_names_each_models_figures(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    image = Image.new("RGB", (8, 8), "green")
    bundle = {
        "kind": "all_models",
        "query_mode": "text",
        "query_label": "rainforest and river",
        "models": [
            {
                "model_name": "SigLIP",
                "distribution": image,
                "overview": image,
                "results_text": "results",
                "results_meta": [],
            },
            {
                "model_name": "FarSLIP",
                "distribution": image,
                "overview": image,
                "results_text": "results",
                "results_meta": [],
            },
        ],
    }

    archive = save_plot(bundle, {}, "thumbnail")

    assert os.path.basename(archive).startswith("earth_embedding_explorer_text_all_models_rainforest_and_river_")
    with zipfile.ZipFile(archive) as zip_file:
        names = set(zip_file.namelist())
    assert "manifest.txt" in names
    assert "01_siglip/siglip_text_distribution.png" in names
    assert "01_siglip/siglip_text_top5.png" in names
    assert "02_farslip/farslip_text_distribution.png" in names
    assert "02_farslip/farslip_text_top5.png" in names
