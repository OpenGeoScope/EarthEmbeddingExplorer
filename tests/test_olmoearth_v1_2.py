"""Focused tests for OlmoEarth v1.2 timestamp handling."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from data_utils import _prepare_row_dict
from generate_embeddings import _rewrite_row_meta_parquet_location
from MajorTOM.embedder.MajorTOM_Embedder import MajorTOM_Embedder
from models.olmoearth_model import OlmoEarthModel
from ui.callbacks import download_image_by_location


def test_major_tom_timestamp_formats_are_converted_to_model_components():
    assert OlmoEarthModel._parse_timestamp("20221115T161819") == (15, 10, 2022)
    assert OlmoEarthModel._parse_timestamp("2019-10-22T16:13:49Z") == (22, 9, 2019)
    assert OlmoEarthModel._parse_timestamp(pd.Timestamp("2024-02-29")) == (29, 1, 2024)


def test_missing_timestamp_uses_valid_midpoint_date():
    assert OlmoEarthModel._parse_timestamp(None) == (1, 6, 2020)


def test_effective_input_resolution_tracks_resize_scale():
    model = object.__new__(OlmoEarthModel)
    model.size = (128, 128)
    model.native_input_res = 10.0
    assert model._effective_input_res(128, 128) == 10.0
    assert model._effective_input_res(384, 384) == 30.0


def test_timestamp_batch_shape_and_month_indexing():
    result = OlmoEarthModel._prepare_timestamps(
        ["20221115T161819", "20191022T161349"],
        batch_size=2,
        device="cpu",
    )
    assert result.shape == (2, 1, 3)
    assert result.tolist() == [[[15, 10, 2022]], [[22, 9, 2019]]]


def test_major_tom_embedder_forwards_timestamps_only_to_temporal_models():
    class TemporalModel:
        supports_timestamps = True

        def __call__(self, images, timestamps=None):
            self.timestamps = timestamps
            return images

    wrapper = object.__new__(MajorTOM_Embedder)
    wrapper.embedder = TemporalModel()
    images = torch.ones(2, 1)
    result = wrapper._encode_images(images, timestamps=["2020-01-01", "2021-01-01"])
    assert result is images
    assert wrapper.embedder.timestamps == ["2020-01-01", "2021-01-01"]


def test_download_callback_returns_selected_product_timestamp(monkeypatch):
    df = pd.DataFrame(
        {
            "product_id": ["sample"],
            "timestamp": ["20221115T161819"],
            "centre_lat": [1.0],
            "centre_lon": [2.0],
        }
    )
    model = SimpleNamespace(df_embed=df, requires_multiband=True)
    multiband = np.zeros((4, 4, 12), dtype=np.uint16)

    monkeypatch.setattr(
        "ui.callbacks.download_and_process_image",
        lambda *args, **kwargs: (SimpleNamespace(), None, multiband),
    )

    _image, _status, returned_bands, metadata = download_image_by_location(
        1.0,
        2.0,
        "sample",
        "OlmoEarth-v1_2",
        {"OlmoEarth-v1_2": model},
    )
    assert returned_bands is multiband
    assert metadata == {"product_id": "sample", "timestamp": "20221115T161819"}


def test_local_subset_rewrite_updates_url_and_row_index(tmp_path):
    subset_dir = tmp_path / "Core-S2L2A-249k" / "images_249k"
    subset_dir.mkdir(parents=True)
    parquet_path = subset_dir / "part_00001.parquet"
    parquet_path.touch()
    row_meta = pd.DataFrame(
        {
            "parquet_url": ["https://example.test/Core-S2L2A/images/part_00001.parquet"],
            "parquet_row": [107],
        }
    )

    rewritten = _rewrite_row_meta_parquet_location(row_meta, str(parquet_path), row_idx=0)

    assert rewritten["parquet_row"].iloc[0] == 0
    assert rewritten["parquet_url"].iloc[0].endswith("Core-S2L2A-249k/resolve/master/images_249k/part_00001.parquet")


def test_source_url_uses_selected_modelscope_endpoint(monkeypatch):
    df = pd.DataFrame(
        {
            "product_id": ["sample"],
            "parquet_url": [
                "https://www.modelscope.cn/datasets/Major-TOM/Core-S2L2A-249k/resolve/master/images_249k/part_00001.parquet"
            ],
            "parquet_row": [0],
        }
    )
    monkeypatch.setenv("DOWNLOAD_ENDPOINT", "modelscope.ai")
    monkeypatch.delenv("MAJOR_TOM_LOCAL_IMAGES", raising=False)

    row, error = _prepare_row_dict("sample", df, verbose=False)

    assert error is None
    assert row["parquet_url"].startswith("https://www.modelscope.ai/")
