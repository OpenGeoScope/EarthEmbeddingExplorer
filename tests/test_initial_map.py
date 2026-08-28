"""Regression tests for the global map used on the initial page."""

from types import SimpleNamespace

import pandas as pd

from ui import callbacks


def test_global_map_is_rendered_once_and_reused(monkeypatch):
    callbacks._GLOBAL_MAP_CACHE.clear()
    df = pd.DataFrame(
        {
            "centre_lat": [0.0],
            "centre_lon": [0.0],
            "product_id": ["sample"],
        }
    )
    rendered_image = object()
    render_calls = []

    def fake_render(source_df):
        render_calls.append(source_df)
        return rendered_image, source_df

    monkeypatch.setattr(callbacks, "plot_global_map_static", fake_render)
    models = {"model": SimpleNamespace(df_embed=df)}

    first = callbacks.get_global_map(models)
    second = callbacks.get_global_map(models)

    assert first == (rendered_image, df)
    assert second == first
    assert render_calls == [df]


def test_global_map_handles_missing_models():
    assert callbacks.get_global_map({}) == (None, None)
    assert callbacks.get_global_map({"model": SimpleNamespace(df_embed=None)}) == (None, None)
