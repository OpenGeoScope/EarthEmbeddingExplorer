import json

import gradio as gr


def text_example_js(query):
    """Build a JS-only output payload for one text example."""
    return f"() => {json.dumps([query])}"


def render_text_example_buttons(queries, target):
    """Render text examples that update ``target`` without a backend request."""
    buttons = []
    with gr.Column(elem_classes=["text-example-list"]):
        gr.Markdown("**Text Examples**", elem_classes=["text-example-label"])
        for query in queries:
            button = gr.Button(
                query,
                variant="secondary",
                size="sm",
                elem_classes=["text-example-button"],
            )
            button.click(
                fn=None,
                outputs=[target],
                js=text_example_js(query),
                queue=False,
                show_progress="hidden",
            )
            buttons.append(button)
    return buttons
