import time

import gradio as gr

from .search_engine import search_image, search_text


def _prefix_gallery(gallery, model_name):
    prefixed = []
    for item in gallery or []:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            image, caption = item
            prefixed.append((image, f"{model_name}\n{caption}"))
        else:
            prefixed.append(item)
    return prefixed


def _comparison_plots(artifacts):
    plots = []
    for artifact in artifacts:
        model_name = artifact["model_name"]
        if artifact.get("distribution") is not None:
            plots.append((artifact["distribution"], f"{model_name} - Distribution"))
        if artifact.get("overview") is not None:
            plots.append((artifact["overview"], f"{model_name} - Top 5"))
    return plots


def _all_model_search(
    model_manager,
    mode,
    query,
    threshold,
    model_names,
    filter_options=None,
    multiband_data=None,
    image_metadata=None,
):
    if mode == "text" and not query:
        yield gr.update(), "Please enter a query.", gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        return
    if mode == "image" and query is None:
        yield gr.update(), "Please upload an image.", gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        return
    if mode == "image" and (multiband_data is None or image_metadata is None):
        yield (
            gr.update(),
            (
                "Search with all models requires a downloaded multispectral image.\n"
                "Select a location and click 'Download Image by Geolocation' first."
            ),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(visible=False),
        )
        return

    available = []
    unavailable = []
    for model_name in model_names:
        model, error = model_manager.get_model(model_name)
        if error or model is None:
            unavailable.append(model_name)
        elif mode == "text" and getattr(model, "requires_multiband", False):
            unavailable.append(model_name)
        else:
            available.append(model_name)

    if not available:
        yield (
            gr.update(),
            "No text-image models are available.",
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )
        return

    started = time.time()
    artifacts = []
    combined_gallery = []
    failures = []
    last_df = gr.update()
    last_map = gr.update()

    for index, model_name in enumerate(available, start=1):
        prefix = f"All-model {mode} search [{index}/{len(available)}] - {model_name}"
        try:
            if mode == "text":
                generator = search_text(model_manager, query, threshold, model_name, filter_options)
            else:
                generator = search_image(
                    model_manager,
                    query,
                    threshold,
                    model_name,
                    filter_options,
                    multiband_data=multiband_data,
                    image_metadata=image_metadata,
                )

            final_output = None
            for output in generator:
                final_output = output
                values = list(output)
                if isinstance(values[1], str):
                    values[1] = f"{prefix}\n\n{values[1]}"
                yield (*values, gr.update(visible=False))

            package = final_output[3] if final_output is not None else None
            if not isinstance(package, (list, tuple)) or len(package) < 5:
                status = final_output[1] if final_output is not None else "No result returned."
                failures.append((model_name, str(status)))
                continue

            artifact = {
                "model_name": model_name,
                "distribution": package[0],
                "overview": package[1],
                "results_text": package[2],
                "results_meta": package[3],
                "status": final_output[1],
            }
            artifacts.append(artifact)
            combined_gallery.extend(_prefix_gallery(final_output[0], model_name))
            last_df = final_output[4]
            last_map = final_output[5]
        except Exception as exc:
            failures.append((model_name, str(exc)))

    if not artifacts:
        details = "\n".join(f"- {name}: {message}" for name, message in failures)
        yield (
            [],
            f"All-model {mode} search failed.\n{details}",
            None,
            gr.update(),
            last_df,
            last_map,
            gr.update(visible=False),
        )
        return

    elapsed = time.time() - started
    lines = [
        f"All-model {mode} search complete in {elapsed:.1f}s.",
        f"Completed {len(artifacts)}/{len(available)} available models.",
        "",
        *[f"✓ {artifact['model_name']}" for artifact in artifacts],
    ]
    if unavailable:
        lines.extend(["", f"Unavailable: {', '.join(unavailable)}"])
    if failures:
        lines.extend(["", "Failures:", *[f"- {name}: {message}" for name, message in failures]])

    state = {
        "kind": "all_models",
        "query_mode": mode,
        "query_label": query if mode == "text" else "image_query",
        "models": artifacts,
    }
    comparison = _comparison_plots(artifacts)
    yield (
        combined_gallery,
        "\n".join(lines),
        artifacts[0]["overview"],
        state,
        last_df,
        last_map,
        gr.update(value=comparison, visible=True),
    )


def search_all_text_models(model_manager, query, threshold, model_names, filter_options=None):
    yield from _all_model_search(model_manager, "text", query, threshold, model_names, filter_options)


def search_all_image_models(
    model_manager,
    image,
    threshold,
    model_names,
    filter_options=None,
    multiband_data=None,
    image_metadata=None,
):
    yield from _all_model_search(
        model_manager,
        "image",
        image,
        threshold,
        model_names,
        filter_options,
        multiband_data=multiband_data,
        image_metadata=image_metadata,
    )
