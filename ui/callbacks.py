import gradio as gr
import pandas as pd

from data_utils import download_and_process_image
from visualize import plot_global_map_static


def get_initial_plot(models):
    """Find the first available embedding to plot the global map."""
    if models is None:
        print("Warning: models is None in get_initial_plot")
        return gr.update(visible=True), [], None, gr.update(visible=False)

    df_vis = None
    img = None
    first_model_name = next(iter(models), None)
    if first_model_name is not None and models[first_model_name].df_embed is not None:
        img, df_vis = plot_global_map_static(models[first_model_name].df_embed)
    else:
        print("No embedding data available for initial plot.")
        img, df_vis = None, None

    return gr.update(value=img, visible=True), [img] if img else [], df_vis, gr.update(visible=False)


def handle_map_click(evt: gr.SelectData, df_vis):
    if evt is None:
        return None, None, None, "No point selected."

    try:
        x, y = evt.index[0], evt.index[1]

        # Image dimensions
        img_width = 3500
        img_height = 1750

        # Scaled Margins (Proportional to 4000x2000)
        left_margin = 110 * 0.875
        right_margin = 110 * 0.875
        top_margin = 100 * 0.875
        bottom_margin = 67 * 0.875

        plot_width = img_width - left_margin - right_margin
        plot_height = img_height - top_margin - bottom_margin

        # Adjust for aspect ratio preservation
        map_aspect = 360.0 / 180.0  # 2.0
        plot_aspect = plot_width / plot_height

        if plot_aspect > map_aspect:
            actual_map_width = plot_height * map_aspect
            actual_map_height = plot_height
            h_offset = (plot_width - actual_map_width) / 2
            v_offset = 0
        else:
            actual_map_width = plot_width
            actual_map_height = plot_width / map_aspect
            h_offset = 0
            v_offset = (plot_height - actual_map_height) / 2

        # Calculate relative position within the plot area
        x_in_plot = x - left_margin
        y_in_plot = y - top_margin

        # Check if click is within the actual map bounds
        if (
            x_in_plot < h_offset
            or x_in_plot > h_offset + actual_map_width
            or y_in_plot < v_offset
            or y_in_plot > v_offset + actual_map_height
        ):
            return None, None, None, "Click outside map area. Please click on the map."

        # Calculate relative position within the map (0 to 1)
        x_rel = (x_in_plot - h_offset) / actual_map_width
        y_rel = (y_in_plot - v_offset) / actual_map_height

        # Clamp to [0, 1]
        x_rel = max(0, min(1, x_rel))
        y_rel = max(0, min(1, y_rel))

        # Convert to geographic coordinates
        lon = x_rel * 360 - 180
        lat = 90 - y_rel * 180

        # Find nearest point in df_vis if available
        pid = ""
        if df_vis is not None:
            dists = (df_vis["centre_lat"] - lat) ** 2 + (df_vis["centre_lon"] - lon) ** 2
            min_idx = dists.idxmin()
            nearest_row = df_vis.loc[min_idx]

            if dists[min_idx] < 25:
                lat = nearest_row["centre_lat"]
                lon = nearest_row["centre_lon"]
                pid = nearest_row["product_id"]

    except Exception as e:
        print(f"Error handling click: {e}")
        import traceback

        traceback.print_exc()
        return None, None, None, f"Error: {e}"

    return lat, lon, pid, f"Selected Point: ({lat:.4f}, {lon:.4f})"


def download_image_by_location(lat, lon, pid, model_name, models):
    """Download and return the image at the specified location.

    For SatCLIP, automatically downloads multiband data and stores it in the
    returned tuple for subsequent encoding.

    Returns:
        (thumbnail_img, status_msg, multiband_array_or_None)
    """
    print(f"DEBUG download_image_by_location: lat={lat}, lon={lon}, model_name={model_name}")
    if lat is None or lon is None:
        return None, "Please specify coordinates first.", None

    model = models.get(model_name)
    if model is None:
        return None, f"Model {model_name} not loaded.", None

    try:
        # Convert to float to ensure proper formatting
        lat = float(lat)
        lon = float(lon)

        # Find Product ID if not provided
        if not pid:
            df = model.df_embed
            lats = pd.to_numeric(df["centre_lat"], errors="coerce")
            lons = pd.to_numeric(df["centre_lon"], errors="coerce")
            dists = (lats - lat) ** 2 + (lons - lon) ** 2
            nearest_idx = dists.idxmin()
            pid = df.loc[nearest_idx, "product_id"]

        # For SatCLIP: download multiband for encoding; thumbnail for display
        multiband_array = None
        if model_name == "SatCLIP":
            result = download_and_process_image(pid, df_source=model.df_embed, verbose=True, mode="multiband")
            img_384, _, multiband_array = result
            if img_384 is None:
                return None, f"Failed to download image for location ({lat:.4f}, {lon:.4f})", None
            return img_384, f"Downloaded image at ({lat:.4f}, {lon:.4f}) [multiband for SatCLIP]", multiband_array
        else:
            img_384, _ = download_and_process_image(pid, df_source=model.df_embed, verbose=True, mode="thumbnail")
            if img_384 is None:
                return None, f"Failed to download image for location ({lat:.4f}, {lon:.4f})", None
            return img_384, f"Downloaded image at ({lat:.4f}, {lon:.4f})", None

    except Exception as e:
        import traceback

        traceback.print_exc()
        return None, f"Error: {e!s}", None


def reset_to_global_map(models):
    """Reset the map to the initial global distribution view."""
    if models is None:
        print("Warning: models is None in reset_to_global_map")
        return gr.update(visible=True), [], None

    img = None
    df_vis = None
    first_model_name = next(iter(models), None)
    if first_model_name is not None and models[first_model_name].df_embed is not None:
        img, df_vis = plot_global_map_static(models[first_model_name].df_embed)
    else:
        print("No embedding data available for initial plot.")
        img, df_vis = None, None

    return gr.update(value=img, visible=True), [img] if img else [], df_vis
