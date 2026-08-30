"""Regression tests for the global map used on the initial page."""

from types import SimpleNamespace

import gradio as gr
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


def test_hidden_tab_examples_fix_is_limited_to_affected_gradio_versions():
    assert not callbacks.needs_hidden_tab_examples_fix("5.49.1")
    assert not callbacks.needs_hidden_tab_examples_fix("6.16.2")
    assert callbacks.needs_hidden_tab_examples_fix("6.17.0")
    assert callbacks.needs_hidden_tab_examples_fix("6.20.3")
    assert not callbacks.needs_hidden_tab_examples_fix("6.21.0")
    assert not callbacks.needs_hidden_tab_examples_fix("unknown")


def test_tab_map_visibility_bypasses_queue_and_progress_overlay():
    with gr.Blocks() as demo:
        with gr.Tabs():
            with gr.Tab("Image") as tab:
                plot_map = gr.Image()
        callbacks.bind_tab_map_visibility(tab, plot_map)

    dependency = demo.get_config_file()["dependencies"][0]
    assert dependency["queue"] is False
    assert dependency["show_progress"] == "hidden"
    assert dependency["api_visibility"] == "private"
