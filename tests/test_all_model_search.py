import os
import zipfile
from types import SimpleNamespace

import gradio as gr
from PIL import Image

from core import all_model_search
from core.exporters import save_comparison_figures, save_plot


class FakeModelManager:
    def __init__(self, names, multiband_names=()):
        self.models = {name: SimpleNamespace(requires_multiband=name in multiband_names) for name in names}

    def get_model(self, name):
        model = self.models.get(name)
        return (model, None) if model is not None else (None, f"{name} unavailable")


def _single_model_generator(_manager, _query, _threshold, model_name, _filters, **_kwargs):
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


def test_all_image_models_require_downloaded_multispectral_state():
    manager = FakeModelManager(["SigLIP", "Clay"], multiband_names={"Clay"})

    output = list(
        all_model_search.search_all_image_models(
            manager,
            Image.new("RGB", (8, 8)),
            7,
            ["SigLIP", "Clay"],
        )
    )[-1]

    assert "requires a downloaded multispectral image" in output[1]


def test_all_image_models_include_rgb_and_multispectral_models(monkeypatch):
    calls = []

    def fake_search(*args, **kwargs):
        calls.append((args[3], kwargs["multiband_data"], kwargs["image_metadata"]))
        yield from _single_model_generator(*args, **kwargs)

    monkeypatch.setattr(all_model_search, "search_image", fake_search)
    manager = FakeModelManager(["SigLIP", "Clay"], multiband_names={"Clay"})
    multiband = object()
    metadata = {"timestamp": "20221115T161819"}

    final = list(
        all_model_search.search_all_image_models(
            manager,
            Image.new("RGB", (8, 8)),
            7,
            ["SigLIP", "Clay"],
            multiband_data=multiband,
            image_metadata=metadata,
        )
    )[-1]

    assert [item[0] for item in calls] == ["SigLIP", "Clay"]
    assert all(item[1] is multiband and item[2] is metadata for item in calls)
    assert [item["model_name"] for item in final[3]["models"]] == ["SigLIP", "Clay"]


def test_figure_only_export_is_flat_and_never_downloads_images(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(
        "core.exporters.download_and_process_image",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected image download")),
    )
    image = Image.new("RGB", (8, 8), "green")
    bundle = {
        "kind": "all_models",
        "query_mode": "image",
        "query_label": "image_query",
        "models": [
            {"model_name": "SigLIP", "distribution": image, "overview": image},
            {"model_name": "Clay", "distribution": image, "overview": image},
        ],
    }

    archive = save_comparison_figures(bundle)

    with zipfile.ZipFile(archive) as zip_file:
        assert set(zip_file.namelist()) == {
            "SigLIP_distribution.png",
            "SigLIP_Top5.png",
            "Clay_distribution.png",
            "Clay_Top5.png",
        }
