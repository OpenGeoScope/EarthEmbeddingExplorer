"""Unit tests for search-engine helper logic.

These tests load no models, need no GPU, and run in seconds. They cover:
- ``_normalize_scores`` min-max normalization and the constant-score degenerate case
- ``_align_on_grid_cell`` cross-model alignment on unique grid cells
- ``apply_filters`` warning behavior (lat swap, antimeridian split, bad dates)
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from core.filters import apply_filters, build_filter_options
from core.search_engine import (
    _align_on_grid_cell,
    _device_matmul_scores,
    _generate_status_msg,
    _normalize_scores,
    _rgb_query_tensor_from_multiband,
    search_image,
)
from data_utils import MULTIBAND_COLUMNS


class TestRGBQueryFromMultiband:
    def test_reorders_raw_bands_and_returns_nchw_tensor(self):
        multiband = np.arange(2 * 3 * len(MULTIBAND_COLUMNS), dtype=np.uint16).reshape(2, 3, len(MULTIBAND_COLUMNS))
        model = SimpleNamespace(bands=["B04", "B03", "B02"])

        query = _rgb_query_tensor_from_multiband(model, multiband)

        expected = multiband[..., [MULTIBAND_COLUMNS.index(name) for name in model.bands]]
        assert query.shape == (1, 3, 2, 3)
        np.testing.assert_array_equal(query.squeeze(0).permute(1, 2, 0).numpy(), expected)

    def test_applies_optional_index_alignment_hook(self):
        calls = []

        def prepare(image):
            calls.append(image.shape)
            return np.zeros((5, 7, 3), dtype=np.uint16)

        model = SimpleNamespace(bands=["B04", "B03", "B02"], prepare_index_aligned_image=prepare)
        query = _rgb_query_tensor_from_multiband(model, np.zeros((2, 3, len(MULTIBAND_COLUMNS)), dtype=np.uint16))

        assert calls == [(2, 3, 3)]
        assert query.shape == (1, 3, 5, 7)

    def test_rgb_search_prefers_raw_bands_over_preview(self):
        class Model:
            requires_multiband = False

            def __init__(self):
                self.bands = ["B04", "B03", "B02"]

            def encode_image(self, image):
                self.received = image
                return torch.ones(1, 2)

        model = Model()
        manager = SimpleNamespace(get_model=lambda _name: (model, None))
        preview = Image.new("RGB", (4, 4), "green")
        generator = search_image(
            manager,
            preview,
            10,
            "RGBModel",
            multiband_data=np.zeros((4, 4, len(MULTIBAND_COLUMNS)), dtype=np.uint16),
            image_metadata={"timestamp": "20200124T074211"},
        )

        next(generator)
        next(generator)

        assert isinstance(model.received, torch.Tensor)
        assert model.received.shape == (1, 3, 4, 4)

    def test_rgb_upload_without_multiband_keeps_preview_path(self):
        class Model:
            requires_multiband = False

            def encode_image(self, image):
                self.received = image
                return torch.ones(1, 2)

        model = Model()
        manager = SimpleNamespace(get_model=lambda _name: (model, None))
        preview = Image.new("RGB", (4, 4), "green")
        generator = search_image(manager, preview, 10, "RGBModel")

        next(generator)
        next(generator)

        assert model.received is preview

    def test_tipsv2_alignment_matches_embedding_generator_resize(self):
        from models.tipsv2_model import TIPSv2Model

        model = object.__new__(TIPSv2Model)
        model.size = (448, 448)
        image = np.arange(384 * 384 * 3, dtype=np.uint16).reshape(384, 384, 3)

        aligned = model.prepare_index_aligned_image(image)

        assert aligned.shape == (448, 448, 3)
        assert aligned.dtype == image.dtype


class TestNormalizeScores:
    """Min-max normalization, including the max == min degenerate case."""

    def test_normal_range(self):
        scores = np.array([0.1, 0.5, 0.9])
        normed, warning = _normalize_scores(scores)
        assert warning is None
        np.testing.assert_allclose(normed, [0.0, 0.5, 1.0])

    def test_constant_scores_return_zeros_with_warning(self):
        scores = np.full(5, 0.42)
        normed, warning = _normalize_scores(scores, modality="Text")
        np.testing.assert_array_equal(normed, np.zeros(5))
        assert warning is not None
        assert "Text" in warning
        assert "constant" in warning

    def test_constant_scores_without_modality_label(self):
        _normed, warning = _normalize_scores(np.ones(3))
        assert warning is not None
        assert "constant" in warning


def _make_embed_df(grid_cells, product_ids):
    """Build a synthetic embedding metadata DataFrame."""
    return pd.DataFrame(
        {
            "grid_cell": grid_cells,
            "product_id": product_ids,
            "centre_lat": np.zeros(len(grid_cells)),
            "centre_lon": np.zeros(len(grid_cells)),
        }
    )


class TestGridCellAlignment:
    """Cross-model alignment must use unique grid_cell, not duplicated product_id."""

    def test_alignment_matches_grid_cell_intersection_size(self):
        # product_id is duplicated across grid cells (one product covers several cells)
        df_a = _make_embed_df(["c1", "c2", "c3", "c4"], ["p1", "p1", "p2", "p2"])
        df_b = _make_embed_df(["c2", "c3", "c4", "c5"], ["p1", "p2", "p2", "p3"])

        idx_a, idx_b = _align_on_grid_cell(df_a, df_b)

        common = sorted(set(df_a["grid_cell"]) & set(df_b["grid_cell"]))
        assert len(idx_a) == len(idx_b) == len(common) == 3
        # Row-aligned pairwise, in sorted (deterministic) grid_cell order
        assert list(df_a["grid_cell"].values[idx_a]) == common
        assert list(df_b["grid_cell"].values[idx_b]) == common

    def test_alignment_is_deterministic(self):
        df_a = _make_embed_df(["c3", "c1", "c2"], ["p1", "p2", "p3"])
        df_b = _make_embed_df(["c2", "c3", "c1"], ["p3", "p1", "p2"])
        idx_a1, idx_b1 = _align_on_grid_cell(df_a, df_b)
        idx_a2, idx_b2 = _align_on_grid_cell(df_a, df_b)
        np.testing.assert_array_equal(idx_a1, idx_a2)
        np.testing.assert_array_equal(idx_b1, idx_b2)

    def test_no_common_grid_cells_raises(self):
        with pytest.raises(ValueError, match="No common grid cells"):
            _align_on_grid_cell(_make_embed_df(["a"], ["p"]), _make_embed_df(["b"], ["p"]))

    def test_missing_grid_cell_column_raises(self):
        df_no_cell = pd.DataFrame({"product_id": ["p1"]})
        with pytest.raises(ValueError, match="grid_cell"):
            _align_on_grid_cell(df_no_cell, _make_embed_df(["c1"], ["p1"]))


def _search_df():
    """Small metadata frame exercising both hemispheres and the antimeridian."""
    return pd.DataFrame(
        {
            "timestamp": ["2020-06-01"] * 4,
            "centre_lat": [10.0, 20.0, -10.0, 0.0],
            "centre_lon": [170.0, -170.0, 0.0, 179.0],
        }
    )


def _run_filters(df, filter_options):
    """Run apply_filters over the full index range of ``df``."""
    probs = np.linspace(0.1, 0.9, len(df))
    indices = np.arange(len(df))
    return apply_filters(df, probs, indices, indices.copy(), filter_options)


class TestApplyFilters:
    """Filter warnings: lat swap, antimeridian split, inverted/invalid dates."""

    def test_lat_bounds_swapped_with_warning(self):
        opts = build_filter_options(enable_geo=True, lat_min=15, lat_max=5)
        _new_f, new_t, df_geo, _probs_geo, warnings = _run_filters(_search_df(), opts)
        assert any("swapped" in w for w in warnings)
        # Swapped box is lat in [5, 15] -> only the lat=10 row survives
        assert df_geo["centre_lat"].tolist() == [10.0]
        assert len(new_t) == 1

    def test_lon_antimeridian_split_with_warning(self):
        opts = build_filter_options(enable_geo=True, lon_min=160, lon_max=-160)
        _new_f, _new_t, df_geo, _probs_geo, warnings = _run_filters(_search_df(), opts)
        assert any("antimeridian" in w for w in warnings)
        # Union [160, 180] + [-180, -160] -> lons 170, 179, -170 survive
        assert sorted(df_geo["centre_lon"].tolist()) == [-170.0, 170.0, 179.0]

    def test_start_after_end_warns_and_returns_empty(self):
        opts = build_filter_options(enable_time=True, start_date="2023-01-01", end_date="2020-01-01")
        new_f, new_t, _df_geo, _probs_geo, warnings = _run_filters(_search_df(), opts)
        assert any("after end date" in w for w in warnings)
        assert len(new_f) == 0
        assert len(new_t) == 0

    def test_invalid_date_warns_and_filter_not_applied(self):
        opts = build_filter_options(enable_time=True, start_date="not-a-date", end_date="2020-01-01")
        new_f, _new_t, _df_geo, _probs_geo, warnings = _run_filters(_search_df(), opts)
        assert any("not applied" in w for w in warnings)
        # Parse failure -> filter skipped, all rows kept
        assert len(new_f) == 4

    def test_no_filter_options_returns_empty_warnings(self):
        df = _search_df()
        probs = np.linspace(0.1, 0.9, len(df))
        indices = np.arange(len(df))
        new_f, new_t, df_geo, _probs_geo, warnings = apply_filters(df, probs, indices, indices.copy(), None)
        assert warnings == []
        assert len(new_f) == len(new_t) == 4
        assert df_geo is df


class TestDeviceMatmulScores:
    """On-device similarity computation must match the numpy reference exactly."""

    def test_matches_numpy_reference(self):
        rng = np.random.default_rng(0)
        emb = rng.normal(size=(50, 8)).astype(np.float32)
        feat = rng.normal(size=(1, 8)).astype(np.float32)
        expected = (emb @ feat.T).ravel()
        result = _device_matmul_scores(torch.from_numpy(emb), torch.from_numpy(feat))
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_accepts_numpy_inputs_and_1d_feature(self):
        emb = np.eye(4, dtype=np.float32)
        feat = np.ones(4, dtype=np.float32)  # (D,) instead of (1, D)
        result = _device_matmul_scores(emb, feat)
        np.testing.assert_allclose(result, np.ones(4), rtol=1e-6)

    def test_indices_align_rows(self):
        emb = torch.from_numpy(np.eye(5, dtype=np.float32))
        feat = torch.ones(1, 5)
        result = _device_matmul_scores(emb, feat, indices=np.array([4, 2]))
        # Rows 4 and 2 of the identity matrix, each dot all-ones = 1
        np.testing.assert_allclose(result, np.ones(2), rtol=1e-6)


def _status_results(count):
    return [
        {
            "id": f"product-{index}",
            "lat": 30.0 + index,
            "lon": 120.0 + index,
            "score": 0.9 - index / 100,
        }
        for index in range(1, count + 1)
    ]


class TestStatusTopResults:
    def test_lists_all_five_displayed_results(self):
        status = _generate_status_msg(1741, 0.07, _status_results(5))

        assert "Found 1741 matches in top 7‰." in status
        assert "Top 5 similar images:" in status
        assert status.count("Product ID:") == 5
        assert "5. Product ID: product-5" in status

    def test_heading_tracks_fewer_successful_downloads(self):
        status = _generate_status_msg(100, 0.03, _status_results(3))

        assert "Top 3 similar images:" in status
        assert status.count("Product ID:") == 3
        assert "4. Product ID:" not in status


class TestQwenEnforceDevice:
    """``Qwen3VLEmbeddingModel._enforce_device`` migrates the official embedder's
    inner model onto the caller-requested device (the official class hard-codes
    cuda-when-available). Tested here with a dummy inner model, no weights needed."""

    def _make_wrapper(self, device):
        from models.qwen3vl_embedding_model import Qwen3VLEmbeddingModel

        wrapper = object.__new__(Qwen3VLEmbeddingModel)  # bypass __init__ (no model load)
        wrapper.device = device
        return wrapper

    def test_cpu_to_cpu_is_noop(self):
        from types import SimpleNamespace

        wrapper = self._make_wrapper("cpu")
        wrapper.model = SimpleNamespace(model=torch.nn.Linear(4, 4))
        wrapper._enforce_device()
        assert next(wrapper.model.model.parameters()).device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
    def test_moves_inner_model_to_requested_cuda_device(self):
        from types import SimpleNamespace

        wrapper = self._make_wrapper("cuda:0")
        wrapper.model = SimpleNamespace(model=torch.nn.Linear(4, 4))  # starts on CPU
        wrapper._enforce_device()
        assert next(wrapper.model.model.parameters()).device == torch.device("cuda:0")
        wrapper.model.model.to("cpu")  # free GPU memory
