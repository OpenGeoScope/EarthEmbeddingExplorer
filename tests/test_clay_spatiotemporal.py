"""Tests for Clay v1.5 spatiotemporal metadata handling."""

from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from clay_metadata import (
    MISSING_SOURCE,
    clay_metadata_status,
    encode_clay_latlon,
    encode_clay_time,
    resolve_clay_metadata,
    timestamp_from_tiff_tags,
)
from data_utils import MULTIBAND_COLUMNS, download_and_process_image
from models.clay_model import ClayModel
from ui.callbacks import download_image_by_location


def test_clay_time_encoding_matches_official_formula():
    encoded = encode_clay_time("20221115T161819")
    week_angle = 46 * 2 * np.pi / 52
    hour_angle = 16 * 2 * np.pi / 24
    expected = [np.sin(week_angle), np.cos(week_angle), np.sin(hour_angle), np.cos(hour_angle)]
    np.testing.assert_allclose(encoded, expected, rtol=1e-6)


def test_clay_latlon_encoding_matches_official_formula():
    encoded = encode_clay_latlon(30, -120)
    expected = [np.sin(np.pi / 6), np.cos(np.pi / 6), np.sin(-2 * np.pi / 3), np.cos(-2 * np.pi / 3)]
    np.testing.assert_allclose(encoded, expected, rtol=1e-6)


def test_metadata_precedence_and_zero_fallback():
    resolved = resolve_clay_metadata(
        time_candidates=[(None, "tiff_tag"), ("20221115T161819", "parquet_product_datetime")],
        latlon_candidates=[(None, None, "tiff_bounds"), (10, 20, "embedding_center")],
    )
    assert resolved["clay_time_input_source"] == "parquet_product_datetime"
    assert resolved["clay_latlon_input_source"] == "embedding_center"
    assert resolved["clay_missing_metadata"] == []

    missing = resolve_clay_metadata()
    assert missing["clay_time_input_source"] == MISSING_SOURCE
    assert missing["clay_latlon_input_source"] == MISSING_SOURCE
    np.testing.assert_array_equal(missing["clay_time_input"], np.zeros(4, dtype=np.float32))
    assert "time" in clay_metadata_status(missing)


def test_tiff_datetime_tag_parsing():
    assert timestamp_from_tiff_tags({"TIFFTAG_DATETIME": "2022:11:15 16:18:19"}) == "2022:11:15 16:18:19"
    assert timestamp_from_tiff_tags({"unrelated": "value"}) is None


def test_clay_metadata_batch_broadcasts_and_validates_shape():
    metadata = resolve_clay_metadata(
        time_candidates=[("20221115T161819", "test")],
        latlon_candidates=[(10, 20, "test")],
    )
    latlon, time = ClayModel._metadata_inputs(metadata, batch_size=2, device="cpu")
    assert latlon.shape == time.shape == (2, 4)
    torch.testing.assert_close(latlon[0], latlon[1])

    with pytest.raises(ValueError, match="Expected 2 metadata records"):
        ClayModel._metadata_inputs([metadata, metadata, metadata], batch_size=2, device="cpu")


def _tiff_bytes(value=1000):
    array = np.full((8, 8), value, dtype=np.uint16)
    with MemoryFile() as memory_file:
        with memory_file.open(
            driver="GTiff",
            height=8,
            width=8,
            count=1,
            dtype=array.dtype,
            crs="EPSG:32631",
            transform=from_origin(500000, 1000, 10, 10),
        ) as dataset:
            dataset.write(array, 1)
        return memory_file.read()


def _thumbnail_bytes():
    output = BytesIO()
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(output, format="PNG")
    return output.getvalue()


def test_multiband_download_returns_real_tiff_location_and_parquet_time(tmp_path):
    columns = {
        "product_id": ["sample"],
        "grid_cell": ["cell"],
        "product_datetime": ["20221115T161819"],
        "thumbnail": [_thumbnail_bytes()],
    }
    columns.update({band: [_tiff_bytes()] for band in MULTIBAND_COLUMNS})
    parquet_path = tmp_path / "part_00001.parquet"
    pq.write_table(pa.table(columns), parquet_path, row_group_size=1)

    source = pd.DataFrame(
        {
            "product_id": ["sample"],
            "parquet_url": [str(parquet_path)],
            "parquet_row": [0],
            "timestamp": ["2000-01-01"],
            "centre_lat": [-50.0],
            "centre_lon": [-50.0],
        }
    )

    preview, _full, bands, metadata = download_and_process_image(
        "sample", df_source=source, verbose=False, mode="multiband", return_metadata=True
    )

    assert isinstance(preview, Image.Image)
    assert bands.shape == (8, 8, len(MULTIBAND_COLUMNS))
    assert metadata["clay_time_input_source"] == "parquet_product_datetime"
    assert metadata["clay_latlon_input_source"] == "tiff_bounds"
    assert metadata["timestamp"] == "20221115T161819"
    assert metadata["latitude"] != -50.0
    assert metadata["longitude"] != -50.0


def test_multiband_download_rejects_mismatched_source_row(tmp_path):
    columns = {
        "product_id": ["different-product"],
        "product_datetime": ["20221115T161819"],
        "thumbnail": [_thumbnail_bytes()],
    }
    columns.update({band: [_tiff_bytes()] for band in MULTIBAND_COLUMNS})
    parquet_path = tmp_path / "part_00001.parquet"
    pq.write_table(pa.table(columns), parquet_path, row_group_size=1)
    source = pd.DataFrame(
        {
            "product_id": ["requested-product"],
            "parquet_url": [str(parquet_path)],
            "parquet_row": [0],
        }
    )

    _preview, _full, bands, metadata = download_and_process_image(
        "requested-product", df_source=source, verbose=False, mode="multiband", return_metadata=True
    )

    assert bands is None
    assert metadata is None


def test_clay_build_datacube_uses_metadata_instead_of_zeros():
    model = object.__new__(ClayModel)
    model.clay_waves = torch.ones(10)
    model.clay_gsd = torch.tensor([10.0])
    image = torch.zeros(2, 10, 4, 4)
    metadata = resolve_clay_metadata(
        time_candidates=[("20221115T161819", "test")],
        latlon_candidates=[(10, 20, "test")],
    )
    latlon, time = model._metadata_inputs(metadata, batch_size=2, device="cpu")
    datacube = model._build_datacube(image, latlon=latlon, time=time)
    assert torch.count_nonzero(datacube["time"]) > 0
    assert torch.count_nonzero(datacube["latlon"]) > 0


def test_download_callback_stores_clay_metadata_and_reports_status(monkeypatch):
    source = pd.DataFrame(
        {
            "product_id": ["sample"],
            "timestamp": ["20221115T161819"],
            "centre_lat": [10.0],
            "centre_lon": [20.0],
        }
    )
    model = SimpleNamespace(
        df_embed=source,
        requires_multiband=True,
        supports_spatiotemporal_metadata=True,
    )
    bands = np.zeros((4, 4, 12), dtype=np.uint16)
    metadata = resolve_clay_metadata(
        time_candidates=[("20221115T161819", "parquet_product_datetime")],
        latlon_candidates=[(10, 20, "tiff_bounds")],
    )
    monkeypatch.setattr(
        "ui.callbacks.download_and_process_image",
        lambda *args, **kwargs: (Image.new("RGB", (4, 4)), None, bands, metadata),
    )

    _image, status, returned_bands, returned_metadata = download_image_by_location(
        10,
        20,
        "sample",
        "Clay-v1_5",
        {"Clay-v1_5": model},
    )

    assert returned_bands is bands
    assert returned_metadata["clay_time_input_source"] == "parquet_product_datetime"
    assert returned_metadata["clay_latlon_input_source"] == "tiff_bounds"
    assert "Clay metadata input: time + latitude/longitude" in status


def test_search_image_forwards_clay_metadata():
    from core.search_engine import search_image

    class Model:
        requires_multiband = True
        supports_spatiotemporal_metadata = True
        bands = MULTIBAND_COLUMNS

        def encode_image(self, image, metadata=None):
            self.received_metadata = metadata
            return torch.ones(1, 2)

        def search(self, _features, top_percent=None):
            return np.asarray([1.0]), np.asarray([0]), np.asarray([0])

    model = Model()
    model_manager = SimpleNamespace(get_model=lambda _name: (model, None))
    metadata = resolve_clay_metadata(
        time_candidates=[("20221115T161819", "parquet_product_datetime")],
        latlon_candidates=[(10, 20, "tiff_bounds")],
    )
    generator = search_image(
        model_manager,
        Image.new("RGB", (4, 4)),
        10,
        "Clay-v1_5",
        multiband_data=np.zeros((4, 4, 12), dtype=np.uint16),
        image_metadata=metadata,
    )

    next(generator)
    next(generator)
    assert model.received_metadata is metadata
