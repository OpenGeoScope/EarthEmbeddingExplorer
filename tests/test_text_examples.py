import json

import gradio as gr

from ui.example_controls import render_text_example_buttons, text_example_js


def test_text_example_js_returns_exact_query_as_single_output():
    query = 'coastline with "quoted" text'
    js = text_example_js(query)

    assert js.startswith("() => ")
    assert json.loads(js.removeprefix("() => ")) == [query]


def test_text_example_buttons_have_no_backend_dependency():
    queries = ["first query", "second query"]
    with gr.Blocks() as demo:
        target = gr.Textbox()
        buttons = render_text_example_buttons(queries, target)

    config = demo.get_config_file()
    button_values = [component["props"]["value"] for component in config["components"] if component["type"] == "button"]
    assert button_values == queries

    dependencies = config["dependencies"]
    assert len(dependencies) == len(queries)
    assert all(dependency["backend_fn"] is False for dependency in dependencies)
    assert all(dependency["queue"] is False for dependency in dependencies)
    assert all(dependency["show_progress"] == "hidden" for dependency in dependencies)
    assert [button.value for button in buttons] == queries
