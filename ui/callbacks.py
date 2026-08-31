import re
from functools import lru_cache

import cartopy.crs as ccrs
import gradio as gr
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from clay_metadata import clay_metadata_status
from data_utils import download_and_process_image
from visualize import plot_global_map_static

# The global map is identical for every session, so render it once and reuse
# it. Re-rendering the dpi=350 scatter map on every page load used to make
# the initial load slow (and pile up under concurrent sessions).
_GLOBAL_MAP_CACHE = {}


def needs_hidden_tab_examples_fix(version):
    """Return whether this Gradio version needs the hidden-tab Examples workaround."""
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", str(version))
    if match is None:
        return False
    parsed = tuple(int(part) for part in match.groups())
    return (6, 17, 0) <= parsed < (6, 21, 0)


def bind_tab_map_visibility(tab, plot_map):
    """Keep the shared map visible without entering Gradio's prediction queue."""

    def show_static_map():
        return gr.update(visible=True)

    return tab.select(
        fn=show_static_map,
        outputs=[plot_map],
        queue=False,
        show_progress="hidden",
    )


def get_global_map(models):
    """Return cached (img, df_vis) for the global sample map."""
    first_model_name = next(iter(models), None)
    if first_model_name is None or models[first_model_name].df_embed is None:
        return None, None

    if first_model_name not in _GLOBAL_MAP_CACHE:
        _GLOBAL_MAP_CACHE[first_model_name] = plot_global_map_static(models[first_model_name].df_embed)
    return _GLOBAL_MAP_CACHE[first_model_name]


# Fallback pixel bbox (x0, y_top, x1, y_bottom) for the map area within the
# 3500x1750 rendered PNG, matching the legacy hardcoded margins. Used only if
# the layout calibration in _map_axes_pixel_bbox fails.
_LEGACY_MAP_BBOX = (146.125, 87.5, 3353.875, 1691.375)


@lru_cache(maxsize=2)
def _map_axes_pixel_bbox(with_colorbar):
    """Compute the pixel bounding box of the map axes in the rendered PNG.

    Replicates the exact figure layout of visualize.plot_global_map_static
    (no colorbar) and visualize.plot_geographic_distribution (with colorbar):
    figsize 10x5 @ dpi 350 -> 3500x1750 px, PlateCarree extent [-180,180]x
    [-90,90], legend and optional colorbar. NaturalEarth features do not
    affect the layout and are omitted, so this works offline. The result is
    cached; on any failure the legacy hardcoded bbox is returned.

    Args:
        with_colorbar: True for search-result maps, False for the initial
            global map.

    Returns:
        tuple: (x0, y_top, x1, y_bottom) axes bounds in image pixels.
    """
    try:
        fig = Figure(figsize=(10, 5), dpi=350)
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        sc = ax.scatter(
            [0.0], [0.0], c=[0.5], cmap="Reds", s=0.35, alpha=0.8, transform=ccrs.PlateCarree(), label="Samples"
        )
        ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
        ax.axis("off")
        if with_colorbar:
            cbar = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.02)
            cbar.set_label("Similarity Score")
        ax.legend(
            loc="lower left", markerscale=3 if with_colorbar else 5, frameon=True, facecolor="white", framealpha=0.9
        )
        fig.tight_layout()
        # The draw applies cartopy's aspect adjustment to the axes position.
        fig.canvas.draw()
        pos = ax.get_position()
        width_px, height_px = fig.get_size_inches() * fig.dpi
        return (pos.x0 * width_px, (1 - pos.y1) * height_px, pos.x1 * width_px, (1 - pos.y0) * height_px)
    except Exception as e:
        print(f"⚠️ Map layout calibration failed ({e}); falling back to hardcoded map bbox.")
        return _LEGACY_MAP_BBOX


def get_initial_plot(models):
    """Find the first available embedding to plot the global map."""
    if models is None:
        print("Warning: models is None in get_initial_plot")
        return gr.update(visible=True), [], None

    img, df_vis = get_global_map(models)
    if img is None:
        print("No embedding data available for initial plot.")

    return gr.update(value=img, visible=True), [img] if img else [], df_vis


def handle_map_click(evt: gr.SelectData, df_vis):
    """Convert a map click (image pixel coordinates) to lat/lon and snap to the nearest sample.

    The pixel-to-geo mapping uses the real axes bounding box of the currently
    displayed map, calibrated from the same figure layout as visualize.py:
    search-result maps carry a "score" column in df_vis and are rendered with
    a colorbar, which shifts the axes relative to the initial global map.

    No-op / error paths return gr.update() for the coordinate outputs so
    existing user inputs are preserved (returning None would clear them).
    """
    if evt is None or evt.index is None:
        return gr.update(), gr.update(), gr.update(), "No point selected. Please click on the map."

    try:
        x, y = evt.index[0], evt.index[1]

        # plot_geographic_distribution adds a "score" column to its result df,
        # so its presence tells us the displayed map was rendered with a colorbar.
        with_colorbar = df_vis is not None and "score" in df_vis.columns
        x0, y_top, x1, y_bottom = _map_axes_pixel_bbox(with_colorbar)

        # Check if click is within the actual map bounds
        if x < x0 or x > x1 or y < y_top or y > y_bottom:
            return gr.update(), gr.update(), gr.update(), "Click outside map area. Please click on the map."

        # Calculate relative position within the map (0 to 1) and clamp
        x_rel = max(0.0, min(1.0, (x - x0) / (x1 - x0)))
        y_rel = max(0.0, min(1.0, (y - y_top) / (y_bottom - y_top)))

        # Convert to geographic coordinates
        lon = x_rel * 360 - 180
        lat = 90 - y_rel * 180

        # Find nearest point in df_vis if available
        pid = ""
        if df_vis is not None and not df_vis.empty:
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
        return gr.update(), gr.update(), gr.update(), f"Error: {e}"

    return lat, lon, pid, f"Selected Point: ({lat:.4f}, {lon:.4f})"


def download_image_by_location(lat, lon, pid, model_name, models):
    """Download and return the image at the specified location.

    For SatCLIP, automatically downloads multiband data and stores it in the
    returned tuple for subsequent encoding.

    Returns:
        (thumbnail_img, status_msg, multiband_array_or_None, image_metadata_or_None)
    """
    print(f"DEBUG download_image_by_location: lat={lat}, lon={lon}, model_name={model_name}")
    if lat is None or lon is None:
        return None, "Please specify coordinates first.", None, None

    model = models.get(model_name)
    if model is None:
        return None, f"Model {model_name} not loaded.", None, None

    try:
        # Convert to float to ensure proper formatting
        lat = float(lat)
        lon = float(lon)

        df = model.df_embed
        if df is None:
            return None, f"Model {model_name} embeddings not loaded (metadata missing).", None, None

        # Find Product ID if not provided
        if not pid:
            lats = pd.to_numeric(df["centre_lat"], errors="coerce")
            lons = pd.to_numeric(df["centre_lon"], errors="coerce")
            dists = (lats - lat) ** 2 + (lons - lon) ** 2
            nearest_idx = dists.idxmin()
            # Guard against silently downloading a far-away image when the
            # coordinates were not snapped to a sample on the map (same 5°
            # threshold as the map-click snap).
            if dists[nearest_idx] >= 25:
                dist_deg = float(dists[nearest_idx]) ** 0.5
                return (
                    None,
                    f"Nearest sample is {dist_deg:.1f}° away from ({lat:.4f}, {lon:.4f}) — too far to download. "
                    "Please click on the map to select a sample or refine the coordinates.",
                    None,
                    None,
                )
            pid = df.loc[nearest_idx, "product_id"]
        elif pid not in df["product_id"].values:
            # The pid was snapped from another model's map subset and does not
            # exist in this model's embedding table.
            return (
                None,
                f"Product ID '{pid}' not found in {model_name} embeddings. "
                "Please click the map again to re-select a sample for this model.",
                None,
                None,
            )

        source_row = df.loc[df["product_id"] == pid].iloc[0]
        image_metadata = {
            "product_id": pid,
            "timestamp": source_row.get("timestamp"),
        }

        # For multi-spectral models: download multiband for encoding; thumbnail for display
        needs_multiband = getattr(model, "requires_multiband", False)
        if needs_multiband:
            needs_clay_metadata = getattr(model, "supports_spatiotemporal_metadata", False)
            result = download_and_process_image(
                pid,
                df_source=model.df_embed,
                verbose=True,
                mode="multiband",
                return_metadata=needs_clay_metadata,
            )
            if needs_clay_metadata:
                img_384, _, multiband_array, clay_metadata = result
                if clay_metadata:
                    image_metadata.update(clay_metadata)
            else:
                img_384, _, multiband_array = result
            if img_384 is None:
                return None, f"Failed to download image for location ({lat:.4f}, {lon:.4f})", None, None
            metadata_status = f" {clay_metadata_status(image_metadata)}" if needs_clay_metadata else ""
            return (
                img_384,
                f"Downloaded image at ({lat:.4f}, {lon:.4f}) [multiband for {model_name}].{metadata_status}",
                multiband_array,
                image_metadata,
            )
        else:
            img_384, _ = download_and_process_image(pid, df_source=model.df_embed, verbose=True, mode="thumbnail")
            if img_384 is None:
                return None, f"Failed to download image for location ({lat:.4f}, {lon:.4f})", None, None
            return img_384, f"Downloaded image at ({lat:.4f}, {lon:.4f})", None, image_metadata

    except Exception as e:
        import traceback

        traceback.print_exc()
        return None, f"Error: {e!s}", None, None


def reset_to_global_map(models):
    """Reset the map to the initial global distribution view."""
    if models is None:
        print("Warning: models is None in reset_to_global_map")
        return gr.update(visible=True), [], None

    img, df_vis = get_global_map(models)
    if img is None:
        print("No embedding data available for initial plot.")

    return gr.update(value=img, visible=True), [img] if img else [], df_vis
