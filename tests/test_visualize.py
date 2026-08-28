"""Focused tests for the single-row retrieval overview."""

import pandas as pd
from PIL import Image

from visualize import (
    _OVERVIEW_COLUMN_SPACING,
    _OVERVIEW_TITLE_FONTSIZE,
    _build_top5_figure,
    _result_overview_title,
)


def _result(rank, timestamp="20221115T161819"):
    return {
        "id": f"sample-{rank}",
        "lat": 30.1234 + rank,
        "lon": 120.5678 + rank,
        "timestamp": timestamp,
        "score": 0.9 - rank / 100,
        "image_384": Image.new("RGB", (384, 384), "green"),
        "image_full": Image.new("RGB", (768, 768), "red"),
    }


def test_result_title_combines_rank_score_time_and_coordinates():
    title = _result_overview_title(1, _result(1))

    assert title == (
        "Rank 1, Score: 0.8900\n"
        "2022-11-15 16:18:19 | (31.12, 121.57)"
    )


def test_missing_timestamp_is_explicit():
    assert "\nN/A |" in _result_overview_title(1, _result(1, timestamp=pd.NaT))


def test_text_search_overview_has_only_one_axis_per_result():
    fig = _build_top5_figure(None, [_result(1), _result(2)])

    assert len(fig.axes) == 2
    assert all(axis.get_subplotspec().rowspan.start == 0 for axis in fig.axes)
    assert all(axis.title.get_fontsize() == _OVERVIEW_TITLE_FONTSIZE for axis in fig.axes)
    assert all("Original" not in axis.get_title() for axis in fig.axes)
    assert fig.subplotpars.wspace == _OVERVIEW_COLUMN_SPACING
    assert fig.subplotpars.wspace > 0


def test_image_or_location_overview_adds_query_to_same_row():
    query = Image.new("RGB", (384, 384), "blue")
    fig = _build_top5_figure(query, [_result(1), _result(2)], query_info="Image Query")

    assert len(fig.axes) == 3
    assert all(axis.get_subplotspec().rowspan.start == 0 for axis in fig.axes)
    assert fig.axes[0].get_title() == "Query\nImage Query"
    assert fig.axes[0].title.get_fontsize() == _OVERVIEW_TITLE_FONTSIZE
