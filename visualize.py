import os
from io import BytesIO

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

# Use the NaturalEarth shapefiles bundled with the repo so cartopy never
# tries to download them from naturalearth S3 at runtime. On ModelScope
# Studio that download is slow/unreliable and used to stall the first map
# render (initial page load) and the first search for tens of seconds.
_BUNDLED_CARTOPY_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "cartopy_data")
if os.path.isdir(_BUNDLED_CARTOPY_DATA):
    cartopy.config["pre_existing_data_dir"] = _BUNDLED_CARTOPY_DATA

# Cache NaturalEarthFeature instances so the shapefiles are parsed only once
_FEATURE_CACHE = {}


def _get_features(scale):
    """Return cached (land, coastline) NaturalEarth features for a scale."""
    if scale not in _FEATURE_CACHE:
        land = cfeature.NaturalEarthFeature("physical", "land", scale)
        coastline = cfeature.NaturalEarthFeature("physical", "coastline", scale)
        _FEATURE_CACHE[scale] = (land, coastline)
    return _FEATURE_CACHE[scale]


def warm_up_map_data():
    """Force-load NaturalEarth geometries for all scales used by the app.

    Called once at startup so the first page load / first search doesn't pay
    the one-off shapefile loading cost.
    """
    for scale in ("50m", "10m"):
        for feature in _get_features(scale):
            list(feature.geometries())


def plot_global_map_static(df, lat_col="centre_lat", lon_col="centre_lon"):
    if df is None:
        return None, None

    # Ensure coordinates are numeric and drop NaNs
    df_clean = df.copy()
    df_clean[lat_col] = pd.to_numeric(df_clean[lat_col], errors="coerce")
    df_clean[lon_col] = pd.to_numeric(df_clean[lon_col], errors="coerce")
    df_clean = df_clean.dropna(subset=[lat_col, lon_col])

    # Sample to at most ~125k points so rendering stays fast for larger
    # datasets. df_vis is also used for map-click snapping
    # (ui/callbacks.handle_map_click), so do not lower this cap without
    # checking snap precision.
    if len(df_clean) > 250000:
        step = max(2, len(df_clean) // 125000)
        df_vis = df_clean.iloc[::step]
        print(f"Sampled {len(df_vis)} points from {len(df_clean)} total points (step={step}) for visualization.")
    else:
        df_vis = df_clean

    def _render(with_basemap):
        """Render the map; NaturalEarth features are optional (downloaded on demand)."""
        # Create static map using Matplotlib
        # Use a fixed size and DPI to make coordinate mapping easier
        # figsize 10x5 @ dpi 350 -> 3500x1750 px, aspect 2:1.
        # ui/callbacks._map_axes_pixel_bbox relies on this exact layout for click mapping.
        fig = Figure(figsize=(10, 5), dpi=350)
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

        if with_basemap:
            # Add land + coastline (Cartopy) - Use 50m resolution to show small islands
            land_50m, coastline_50m = _get_features("50m")
            ax.add_feature(land_50m, facecolor="lightgray", edgecolor="none", alpha=0.2)
            ax.add_feature(coastline_50m, facecolor="none", linewidth=0.8, alpha=0.5)

        # Plot points - Use blue to match user request
        ax.scatter(
            df_vis[lon_col],
            df_vis[lat_col],
            s=0.2,
            c="blue",
            marker="o",
            edgecolors="none",
            # alpha=0.6,
            transform=ccrs.PlateCarree(),
            label="Samples",
        )

        # Set limits to full world
        ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())

        # Remove axes and margins
        ax.axis("off")
        # fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        # Add Legend
        ax.legend(loc="lower left", markerscale=5, frameon=True, facecolor="white", framealpha=0.9)
        fig.tight_layout()

        # Save to PIL
        buf = BytesIO()
        fig.savefig(buf, format="png", facecolor="white")
        buf.seek(0)
        return Image.open(buf)

    try:
        img = _render(with_basemap=True)
    except Exception as e:
        # NaturalEarth features are downloaded at render time; when offline this
        # fails, so fall back to a plain scatter plot without the basemap.
        print(f"⚠️ Basemap unavailable ({e}); rendering map without NaturalEarth features.")
        img = _render(with_basemap=False)

    return img, df_vis


def plot_geographic_distribution(df, scores, lat_col="centre_lat", lon_col="centre_lon", title="Search Results"):
    if df is None or scores is None:
        return None, None

    df_vis = df.copy()
    df_vis["score"] = scores
    df_vis = df_vis.sort_values(by="score", ascending=False)

    # Show ALL filtered results (no additional threshold filtering)
    # The threshold was already applied in model.search() and apply_filters()
    df_filtered = df_vis

    def _render(with_basemap):
        """Render the distribution map; NaturalEarth features are optional (downloaded on demand)."""
        fig = Figure(figsize=(10, 5), dpi=350)
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

        if with_basemap:
            # Add land + coastline (Cartopy) - Use 10m resolution to show small islands
            land_10m, coastline_10m = _get_features("10m")
            ax.add_feature(land_10m, facecolor="lightgray", edgecolor="none", alpha=0.2)
            ax.add_feature(coastline_10m, facecolor="none", linewidth=0.8, alpha=0.5)

        # 2. Plot Search Results with color map
        label_text = f"{len(df_filtered)} Results"
        sc = ax.scatter(
            df_filtered[lon_col],
            df_filtered[lat_col],
            c=df_filtered["score"],
            cmap="Reds",
            s=0.35,
            alpha=0.8,
            transform=ccrs.PlateCarree(),
            label=label_text,
        )

        ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
        ax.axis("off")
        # fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        # Add Colorbar
        cbar = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label("Similarity Score")

        # Add Legend
        ax.legend(loc="lower left", markerscale=3, frameon=True, facecolor="white", framealpha=0.9)

        fig.tight_layout()

        # Add title (optional, might overlap)
        # ax.set_title(title)

        buf = BytesIO()
        fig.savefig(buf, format="png", facecolor="white")
        buf.seek(0)
        return Image.open(buf)

    try:
        img = _render(with_basemap=True)
    except Exception as e:
        # NaturalEarth features are downloaded at render time; when offline this
        # fails, so fall back to a plain scatter plot without the basemap.
        print(f"⚠️ Basemap unavailable ({e}); rendering map without NaturalEarth features.")
        img = _render(with_basemap=False)

    return img, df_filtered


def format_results_for_gallery(results):
    """
    Format results for Gradio Gallery.
    results: list of dicts
    Returns: list of (image, caption) tuples
    """
    gallery_items = []
    for res in results:
        # Use 384x384 image for gallery thumbnail/preview
        img = res.get("image_384")
        if img is None:
            continue

        caption = f"Score: {res['score']:.4f}\nLat: {res['lat']:.2f}, Lon: {res['lon']:.2f}\nID: {res['id']}"
        gallery_items.append((img, caption))

    return gallery_items


_OVERVIEW_TITLE_FONTSIZE = 12


def _format_acquisition_time(value):
    """Format a result timestamp compactly for the overview title."""
    if value is None:
        return "N/A"

    try:
        timestamp = pd.to_datetime(value, errors="coerce")
        if pd.isna(timestamp):
            return "N/A"
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return "N/A"


def _result_overview_title(rank, result):
    """Build the two-line title shown above a retrieved image."""
    acquired = _format_acquisition_time(result.get("timestamp"))
    return (
        f"Rank {rank}, Score: {result['score']:.4f}\n"
        f"{acquired} | ({result['lat']:.2f}, {result['lon']:.2f})"
    )


def _build_top5_figure(query_image, results, query_info="Query"):
    """Build the single-row result figure without serializing it."""
    top_k = len(results)
    if top_k == 0:
        return None

    has_query_image = query_image is not None
    cols = top_k + (1 if has_query_image else 0)

    fig = Figure(figsize=(4.5 * cols, 4))
    _canvas = FigureCanvasAgg(fig)

    if has_query_image:
        ax = fig.add_subplot(1, cols, 1)
        ax.imshow(query_image)
        ax.set_title(
            f"Query\n{query_info}",
            color="blue",
            fontweight="bold",
            fontsize=_OVERVIEW_TITLE_FONTSIZE,
        )
        ax.axis("off")
        start_col = 2
    else:
        start_col = 1

    for i, res in enumerate(results):
        ax1 = fig.add_subplot(1, cols, start_col + i)
        img_384 = res.get("image_384")
        if img_384 is not None:
            ax1.imshow(img_384)
            ax1.set_title(_result_overview_title(i + 1, res), fontsize=_OVERVIEW_TITLE_FONTSIZE)
        else:
            ax1.text(0.5, 0.5, "N/A", ha="center", va="center")
        ax1.axis("off")

    fig.tight_layout()
    return fig


def plot_top5_overview(query_image, results, query_info="Query"):
    """Render the query (when present) and 384px results in one row."""
    fig = _build_top5_figure(query_image, results, query_info)
    if fig is None:
        return None

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)

    return Image.open(buf)
