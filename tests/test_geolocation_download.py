"""Geolocation queries must retain raw pixels and metadata for every model."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from clay_metadata import resolve_clay_metadata
from ui.callbacks import download_image_by_location


@pytest.fixture
def source():
    return pd.DataFrame(
        {
            "product_id": ["sample"],
            "timestamp": ["20200124T074211"],
            "centre_lat": [27.261384963989258],
            "centre_lon": [43.07999038696289],
        }
    )


@pytest.mark.parametrize(
    "model_name,requires_multiband",
    [
        ("DINOv2", False),
        ("SigLIP", False),
        ("FarSLIP", False),
        ("TIPSv2", False),
        ("Qwen3VL", False),
        ("SatCLIP", True),
        ("Clay", True),
        ("OlmoEarth-v1_2", True),
    ],
)
@pytest.mark.parametrize("pid", ["", "sample"])
def test_every_model_downloads_raw_bands_and_full_metadata(monkeypatch, source, model_name, requires_multiband, pid):
    # Only the selected model is loaded: even an RGB-only installation can
    # obtain a multispectral query without loading Clay/SatCLIP/OlmoEarth.
    models = {model_name: SimpleNamespace(df_embed=source, requires_multiband=requires_multiband)}
    preview = Image.new("RGB", (4, 4))
    bands = np.arange(4 * 4 * 12, dtype=np.uint16).reshape(4, 4, 12)
    metadata = resolve_clay_metadata(
        time_candidates=[("20200124T074211", "parquet_product_datetime")],
        latlon_candidates=[(27.261384963989258, 43.07999038696289, "tiff_bounds")],
    )
    metadata["product_datetime"] = "20200124T074211"
    calls = []

    def download(product_id, **kwargs):
        calls.append(product_id)
        assert kwargs["df_source"] is source
        assert kwargs["mode"] == "multiband"
        assert kwargs["return_metadata"] is True
        return preview, preview, bands, metadata

    monkeypatch.setattr("ui.callbacks.download_and_process_image", download)
    image, status, returned_bands, returned_metadata = download_image_by_location(27.2, 43, pid, model_name, models)

    assert calls == ["sample"]
    assert image is preview
    assert returned_bands is bands
    assert returned_bands.dtype == np.uint16
    assert returned_metadata["product_id"] == "sample"
    assert returned_metadata["timestamp"] == "20200124T074211"
    assert returned_metadata["product_datetime"] == "20200124T074211"
    assert returned_metadata["clay_time_input_source"] == "parquet_product_datetime"
    assert returned_metadata["clay_latlon_input_source"] == "tiff_bounds"
    np.testing.assert_array_equal(returned_metadata["clay_latlon_input"], metadata["clay_latlon_input"])
    assert "multispectral + metadata" in status
    assert "RGB preview only" in status


@pytest.mark.parametrize("missing_preview", [False, True])
def test_incomplete_download_clears_query_instead_of_falling_back_to_thumbnail(monkeypatch, source, missing_preview):
    preview = None if missing_preview else Image.new("RGB", (4, 4))
    bands = np.zeros((4, 4, 12), dtype=np.uint16) if missing_preview else None
    monkeypatch.setattr(
        "ui.callbacks.download_and_process_image",
        lambda *args, **kwargs: (preview, None, bands, None),
    )
    models = {"SigLIP": SimpleNamespace(df_embed=source, requires_multiband=False)}

    image, status, returned_bands, metadata = download_image_by_location(27.2, 43, "", "SigLIP", models)

    assert image is returned_bands is metadata is None
    assert "Failed to download multispectral image" in status


def test_download_button_is_independent_of_all_model_checkbox():
    # Inspect the event binding without importing app.py, which loads all
    # model weights. Toggling all-model search must not change download inputs.
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    tree = ast.parse(app_path.read_text())
    bindings = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "btn_download_img"
        and node.func.attr == "click"
    ]
    assert len(bindings) == 1
    keywords = {keyword.arg: keyword.value for keyword in bindings[0].keywords}
    assert keywords["fn"].id == "_download_image_by_location"
    assert [node.id for node in keywords["inputs"].elts] == ["img_lat", "img_lon", "img_pid", "model_selector_img"]
    assert [node.id for node in keywords["outputs"].elts] == [
        "image_input",
        "img_click_status",
        "multiband_state",
        "image_metadata_state",
    ]
