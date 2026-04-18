import os
import gradio as gr

from core.exporters import save_plot
from core.filters import build_filter_options

# Import extracted modules
from core.model_manager import ModelManager
from core.search_engine import (
    search_image as _search_image,
)
from core.search_engine import (
    search_location as _search_location,
)
from core.search_engine import (
    search_mixed as _search_mixed,
)
from core.search_engine import (
    search_text as _search_text,
)
from ui.callbacks import (
    download_image_by_location,
    get_initial_plot,
    handle_map_click,
    reset_to_global_map,
)


# Environment variable for controlling download endpoint
# Options: 'modelscope.cn', 'modelscope.ai', 'huggingface'
DOWNLOAD_ENDPOINT = os.getenv("DOWNLOAD_ENDPOINT", "modelscope.cn")
if DOWNLOAD_ENDPOINT not in ("modelscope.cn", "modelscope.ai", "huggingface"):
    print("\n====== Warning: DOWNLOAD_ENDPOINT should be modelscope.cn, modelscope.ai, or huggingface! =====\n")

# Initialize ModelManager (loads all models)
model_manager = ModelManager()
models = model_manager.models  # Keep for backward compatibility with existing code

INTRODUCTION_ZH = "EarthEmbeddingExplorer 是一款工具跨模态遥感图像检索工具，允许您使用自然语言描述、图像、地理位置或简单地在地图上点击来搜索地球的卫星图像。例如，您可以输入“热带雨林”或“有城市的海岸线”，系统就会找到地球上与您描述相符的位置。然后，它会在世界地图上可视化这些位置的卫星图像嵌入和您的输入嵌入的相似度，并显示最相似的图像。您可以下载检索结果和最相似的图像。"
INTRODUCTION_EN = "EarthEmbeddingExplorer is a tool that allows you to search for satellite images of the Earth using natural language descriptions, images, geolocations, or a simple a click on the map. For example, you can type \"tropical rainforest\" or \"coastline with a city,\" and the system will find locations on Earth that match your description. It then visualizes these locations on a world map and displays the top matching images."

if DOWNLOAD_ENDPOINT == 'modelscope.cn':
    introduction = INTRODUCTION_ZH
else:
    introduction = INTRODUCTION_EN

def get_active_model(model_name):
    """Wrapper for backward compatibility."""
    return model_manager.get_model(model_name)


# Wrapper functions for UI callbacks (pass models dict to extracted functions)
def _get_initial_plot():
    return get_initial_plot(models)


def _reset_to_global_map():
    return reset_to_global_map(models)


def _handle_map_click(evt, df_vis):
    """Handle map click event - extract lat/lon from click coordinates."""
    return handle_map_click(evt, df_vis)


def _download_image_by_location(lat, lon, pid, model_name):
    return download_image_by_location(lat, lon, pid, model_name, models)




# Gradio Blocks Interface
with gr.Blocks(
    title="EarthEmbeddingExplorer",
    css="""
.filter-checkbox {
    background: transparent !important;
    border: 1px solid #d1d5db !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
    margin: 4px 0 !important;
    box-shadow: none !important;
    outline: none !important;
}
.filter-checkbox > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}
.filter-checkbox label {
    background: transparent !important;
    font-weight: bold !important;
}
.filter-checkbox label span {
    background: transparent !important;
}
/* Remove gray border from Gradio form group wrapping the filter checkboxes */
.form:has(> .filter-checkbox),
div.form:has(.filter-checkbox) {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
}
""",
) as demo:
    gr.Markdown("# EarthEmbeddingExplorer")
    gr.HTML(f"""
    <div style="font-size: 1.2em;">
    {introduction}
    </div>

    <div style="display: flex; gap: 0.2em; align-items: center; justify-content: center;">
    <a href="https://modelscope.cn/studios/Major-TOM/EarthEmbeddingExplorer/"><img src="https://img.shields.io/badge/Open in ModelScope.cn-xGPU-624aff"></a>
    <a href="https://modelscope.ai/studios/Major-TOM/EarthEmbeddingExplorer/"><img src="https://img.shields.io/badge/Open in ModelScope.ai-xGPU-624aff"></a>
    <a href="https://modelscope.cn/collections/Major-TOM/Core-S2L2A-249k"><img src="https://img.shields.io/badge/👾 MS-Dataset-624aff"></a>
    <a href="https://huggingface.co/datasets/ML4RS-Anonymous/EarthEmbeddings"><img src="https://img.shields.io/badge/🤗 HF-Dataset-FFD21E"></a>
    <a href="https://arxiv.org/abs/2603.29441"> <img src="https://img.shields.io/badge/arXiv-@ICLR26📖-B31B1B"> </a>
    <a href="https://openreview.net/forum?id=LSsEenJVqD"> <img src="https://img.shields.io/badge/OpenReview-@ICLR26📖-007bff"> </a>
    </div>

    """)

    with gr.Row():
        with gr.Column(scale=4):
            with gr.Tabs():
                with gr.TabItem("Text Search") as tab_text:
                    model_selector_text = gr.Dropdown(choices=["SigLIP", "FarSLIP"], value="FarSLIP", label="Model")
                    query_input = gr.Textbox(label="Query", placeholder="e.g., rainforest, glacier")

                    gr.Examples(
                        examples=[
                            ["a satellite image of a river around a city"],
                            ["a satellite image of a rainforest"],
                            ["a satellite image of a slum"],
                            ["a satellite image of a glacier"],
                            ["a satellite image of snow covered mountains"],
                        ],
                        inputs=[query_input],
                        label="Text Examples",
                    )

                    search_btn = gr.Button("Search by Text", variant="primary")

                with gr.TabItem("Image Search") as tab_image:
                    model_selector_img = gr.Dropdown(
                        choices=["SigLIP", "FarSLIP", "SatCLIP", "DINOv2"], value="FarSLIP", label="Model"
                    )

                    gr.Markdown("### Option 1: Upload or Select Image")
                    image_input = gr.Image(type="pil", label="Upload Image")

                    gr.Examples(
                        examples=[
                            ["./examples/example1.png"],
                            ["./examples/example2.png"],
                            ["./examples/example3.png"],
                        ],
                        inputs=[image_input],
                        label="Image Examples",
                    )

                    gr.Markdown("### Option 2: Click Map or Enter Coordinates")
                    btn_reset_map_img = gr.Button("🔄 Reset Map to Global View", variant="secondary", size="sm")

                    with gr.Row():
                        img_lat = gr.Number(label="Latitude", interactive=True)
                        img_lon = gr.Number(label="Longitude", interactive=True)

                    img_pid = gr.Textbox(label="Product ID (auto-filled)", visible=False)
                    img_click_status = gr.Markdown("")

                    btn_download_img = gr.Button("Download Image by Geolocation", variant="secondary")

                    search_img_btn = gr.Button("Search by Image", variant="primary")

                with gr.TabItem("Location Search") as tab_location:
                    gr.Markdown("Search using **SatCLIP** location encoder.")

                    gr.Markdown("### Click Map or Enter Coordinates")
                    btn_reset_map_loc = gr.Button("🔄 Reset Map to Global View", variant="secondary", size="sm")

                    with gr.Row():
                        lat_input = gr.Number(label="Latitude", value=30.0, interactive=True)
                        lon_input = gr.Number(label="Longitude", value=120.0, interactive=True)

                    loc_pid = gr.Textbox(label="Product ID (auto-filled)", visible=False)
                    loc_click_status = gr.Markdown("")

                    gr.Examples(
                        examples=[
                            [30.32, 120.15],
                            [40.7128, -74.0060],
                            [24.65, 46.71],
                            [-3.4653, -62.2159],
                            [64.4, 16.8],
                        ],
                        inputs=[lat_input, lon_input],
                        label="Location Examples",
                    )

                    search_loc_btn = gr.Button("Search by Location", variant="primary")

                with gr.TabItem("Mixed Search") as tab_mixed:
                    gr.Markdown("""
                    ### Multi-Modal Fusion Search
                    Combine **Text**, **Image**, and **Location** queries with adjustable weights.
                    Text/Image use FarSLIP or SigLIP; Location uses SatCLIP. Scores are normalized and fused.
                    """)

                    model_selector_mixed = gr.Dropdown(
                        choices=["FarSLIP", "SigLIP"], value="FarSLIP", label="Model for Text/Image"
                    )

                    gr.Markdown("#### 📝 Text Query")
                    mixed_text_input = gr.Textbox(
                        label="Text Query (optional)", placeholder="e.g., tropical rainforest, glacier, urban area"
                    )

                    gr.Examples(
                        examples=[
                            ["a satellite image of a river around a city"],
                            ["a satellite image of a rainforest"],
                            ["a satellite image of a glacier"],
                            ["a satellite image of snow covered mountains"],
                        ],
                        inputs=[mixed_text_input],
                        label="Text Examples",
                    )

                    gr.Markdown("#### 🖼️ Image Query")
                    mixed_image_input = gr.Image(type="pil", label="Upload Image (optional)")

                    gr.Examples(
                        examples=[
                            ["./examples/example1.png"],
                            ["./examples/example2.png"],
                            ["./examples/example3.png"],
                        ],
                        inputs=[mixed_image_input],
                        label="Image Examples",
                    )

                    gr.Markdown("#### 📍 Location Query")
                    btn_reset_map_mixed = gr.Button("🔄 Reset Map to Global View", variant="secondary", size="sm")
                    with gr.Row():
                        mixed_lat = gr.Number(label="Latitude", interactive=True)
                        mixed_lon = gr.Number(label="Longitude", interactive=True)
                    mixed_pid = gr.Textbox(label="Product ID (auto-filled)", visible=False)
                    mixed_click_status = gr.Markdown("")

                    gr.Markdown("#### ⚖️ Fusion Weights")
                    gr.Markdown("_Weights are auto-normalized. Set weight to 0 to disable a modality._")
                    with gr.Row():
                        weight_text = gr.Slider(minimum=0, maximum=1, value=0.33, step=0.01, label="Text Weight")
                        weight_image = gr.Slider(minimum=0, maximum=1, value=0.33, step=0.01, label="Image Weight")
                        weight_location = gr.Slider(
                            minimum=0, maximum=1, value=0.33, step=0.01, label="Location Weight"
                        )

                    search_mixed_btn = gr.Button("🔍 Mixed Search", variant="primary")

            threshold_slider = gr.Slider(minimum=1, maximum=30, value=7, step=1, label="Top Percentage (‰)")

            # Filter controls
            enable_time_filter = gr.Checkbox(
                label="📅 Enable Time Filter", value=False, elem_classes=["filter-checkbox"]
            )
            with gr.Row():
                time_start = gr.Textbox(label="Start Date", placeholder="YYYY-MM-DD", value="2016-01-01", visible=False)
                time_end = gr.Textbox(label="End Date", placeholder="YYYY-MM-DD", value="2024-12-31", visible=False)
            enable_geo_filter = gr.Checkbox(
                label="🌍 Enable Geo Filter (Bounding Box)", value=False, elem_classes=["filter-checkbox"]
            )
            with gr.Row():
                geo_lat_min = gr.Number(label="Lat Min", value=-90, visible=False)
                geo_lat_max = gr.Number(label="Lat Max", value=90, visible=False)
            with gr.Row():
                geo_lon_min = gr.Number(label="Lon Min", value=-180, visible=False)
                geo_lon_max = gr.Number(label="Lon Max", value=180, visible=False)

            status_output = gr.Textbox(label="Status", lines=10)
            download_mode = gr.Dropdown(
                choices=["thumbnail", "rgb", "multiband"],
                value="thumbnail",
                label="Image Download Mode",
                info="thumbnail: fast preview | rgb: B04/B03/B02 composite | multiband: all 12 S2 bands",
            )
            save_btn = gr.Button("Download Result")
            download_file = gr.File(label="Zipped Results", height=40)

        with gr.Column(scale=6):
            plot_map = gr.Image(
                label="Geographical Distribution", type="pil", interactive=False, height=400, width=800, visible=True
            )
            plot_map_interactive = gr.Plot(label="Geographical Distribution (Interactive)", visible=False)
            results_plot = gr.Image(label="Top 5 Matched Images", type="pil")
            gallery_images = gr.Gallery(label="Top Retrieved Images (Zoom)", columns=3, height="auto")

    current_fig = gr.State()
    map_data_state = gr.State()
    multiband_state = gr.State(value=None)  # Stores 12-band numpy array for SatCLIP encoding
    image_source = gr.State(value="upload")  # Tracks whether image came from "upload" or "download"

    # Clear multiband state only when user uploads a new image manually,
    # NOT when the image was programmatically set by the download button.
    def _clear_multiband_on_upload(img, source):
        if source == "download":
            # Image was set by the download button — keep multiband, reset source flag
            return gr.update(), "download"
        # User manually uploaded/changed image — discard stale multiband data
        return None, "upload"

    image_input.change(
        fn=_clear_multiband_on_upload, inputs=[image_input, image_source], outputs=[multiband_state, image_source]
    )

    # Initial Load
    demo.load(fn=_get_initial_plot, outputs=[plot_map, current_fig, map_data_state, plot_map_interactive])

    # Reset Map Buttons
    btn_reset_map_img.click(fn=_reset_to_global_map, outputs=[plot_map, current_fig, map_data_state])

    btn_reset_map_loc.click(fn=_reset_to_global_map, outputs=[plot_map, current_fig, map_data_state])

    btn_reset_map_mixed.click(fn=_reset_to_global_map, outputs=[plot_map, current_fig, map_data_state])

    # Map Click Event - updates Image Search coordinates
    def _map_click_handler(evt: gr.SelectData, state_data):
        """Wrapper for map click that passes map_data_state."""
        return _handle_map_click(evt, state_data)

    plot_map.select(fn=_map_click_handler, inputs=[map_data_state], outputs=[img_lat, img_lon, img_pid, img_click_status])
    plot_map.select(fn=_map_click_handler, inputs=[map_data_state], outputs=[lat_input, lon_input, loc_pid, loc_click_status])
    plot_map.select(fn=_map_click_handler, inputs=[map_data_state], outputs=[mixed_lat, mixed_lon, mixed_pid, mixed_click_status])

    # Download Image by Geolocation
    def _download_and_mark_source(lat, lon, pid, model_name):
        img, status, multiband = download_image_by_location(lat, lon, pid, model_name, models)
        return img, status, multiband, "download"

    btn_download_img.click(
        fn=_download_and_mark_source,
        inputs=[img_lat, img_lon, img_pid, model_selector_img],
        outputs=[image_input, img_click_status, multiband_state, image_source],
    )

    # Filter toggle events
    def toggle_time_filter(enabled):
        return gr.update(visible=enabled), gr.update(visible=enabled)

    def toggle_geo_filter(enabled):
        return (
            gr.update(visible=enabled),
            gr.update(visible=enabled),
            gr.update(visible=enabled),
            gr.update(visible=enabled),
        )

    enable_time_filter.change(fn=toggle_time_filter, inputs=[enable_time_filter], outputs=[time_start, time_end])

    enable_geo_filter.change(
        fn=toggle_geo_filter, inputs=[enable_geo_filter], outputs=[geo_lat_min, geo_lat_max, geo_lon_min, geo_lon_max]
    )

    # Wrapper functions: pack UI filter controls into filter_options dict
    _filter_inputs = [
        enable_time_filter,
        time_start,
        time_end,
        enable_geo_filter,
        geo_lat_min,
        geo_lat_max,
        geo_lon_min,
        geo_lon_max,
    ]

    def _wrap_search_text(
        query, threshold, model_name, enable_time, start_date, end_date, enable_geo, lat_min, lat_max, lon_min, lon_max
    ):
        fo = build_filter_options(enable_time, start_date, end_date, enable_geo, lat_min, lat_max, lon_min, lon_max)
        yield from _search_text(model_manager, query, threshold, model_name, fo)

    def _wrap_search_image(
        image_input,
        threshold,
        model_name,
        enable_time,
        start_date,
        end_date,
        enable_geo,
        lat_min,
        lat_max,
        lon_min,
        lon_max,
        multiband_data=None,
    ):
        fo = build_filter_options(enable_time, start_date, end_date, enable_geo, lat_min, lat_max, lon_min, lon_max)
        yield from _search_image(model_manager, image_input, threshold, model_name, fo, multiband_data=multiband_data)

    def _wrap_search_location(
        lat, lon, threshold, enable_time, start_date, end_date, enable_geo, f_lat_min, f_lat_max, f_lon_min, f_lon_max
    ):
        fo = build_filter_options(
            enable_time, start_date, end_date, enable_geo, f_lat_min, f_lat_max, f_lon_min, f_lon_max
        )
        yield from _search_location(model_manager, lat, lon, threshold, fo)

    def _wrap_search_mixed(
        query_text,
        query_image,
        lat,
        lon,
        w_text,
        w_image,
        w_location,
        threshold,
        model_name,
        enable_time,
        start_date,
        end_date,
        enable_geo,
        f_lat_min,
        f_lat_max,
        f_lon_min,
        f_lon_max,
        multiband_data=None,
    ):
        fo = build_filter_options(
            enable_time, start_date, end_date, enable_geo, f_lat_min, f_lat_max, f_lon_min, f_lon_max
        )
        yield from _search_mixed(
            model_manager,
            query_text, query_image, lat, lon, w_text, w_image, w_location, threshold, model_name, fo, multiband_data
        )

    # Search Event (Text)
    search_btn.click(
        fn=_wrap_search_text,
        inputs=[
            query_input,
            threshold_slider,
            model_selector_text,
            *_filter_inputs,
        ],
        outputs=[
            plot_map_interactive,
            gallery_images,
            status_output,
            results_plot,
            current_fig,
            map_data_state,
            plot_map,
        ],
    )

    # Search Event (Image)
    search_img_btn.click(
        fn=_wrap_search_image,
        inputs=[image_input, threshold_slider, model_selector_img, *_filter_inputs, multiband_state],
        outputs=[
            plot_map_interactive,
            gallery_images,
            status_output,
            results_plot,
            current_fig,
            map_data_state,
            plot_map,
        ],
    )

    # Search Event (Location)
    search_loc_btn.click(
        fn=_wrap_search_location,
        inputs=[lat_input, lon_input, threshold_slider, *_filter_inputs],
        outputs=[
            plot_map_interactive,
            gallery_images,
            status_output,
            results_plot,
            current_fig,
            map_data_state,
            plot_map,
        ],
    )

    # Search Event (Mixed)
    search_mixed_btn.click(
        fn=_wrap_search_mixed,
        inputs=[
            mixed_text_input,
            mixed_image_input,
            mixed_lat,
            mixed_lon,
            weight_text,
            weight_image,
            weight_location,
            threshold_slider,
            model_selector_mixed,
            *_filter_inputs,
            multiband_state,
        ],
        outputs=[
            plot_map_interactive,
            gallery_images,
            status_output,
            results_plot,
            current_fig,
            map_data_state,
            plot_map,
        ],
    )

    # Save Event — download_mode controls the image format in the exported zip
    # Save/Download Results
    def _save_results(current_fig, download_mode):
        return save_plot(current_fig, models, download_mode)

    save_btn.click(fn=_save_results, inputs=[current_fig, download_mode], outputs=[download_file])

    # Tab Selection Events
    def show_static_map():
        return gr.update(visible=True), gr.update(visible=False)

    tab_text.select(fn=show_static_map, outputs=[plot_map, plot_map_interactive])
    tab_image.select(fn=show_static_map, outputs=[plot_map, plot_map_interactive])
    tab_location.select(fn=show_static_map, outputs=[plot_map, plot_map_interactive])
    tab_mixed.select(fn=show_static_map, outputs=[plot_map, plot_map_interactive])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7859, share=False)
