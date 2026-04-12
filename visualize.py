import json
import os
from io import BytesIO

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import plotly.graph_objects as go
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image


def get_background_map_trace():
    # Use absolute path relative to the script location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    geojson_path = os.path.join(base_dir, 'countries.geo.json')

    if not os.path.exists(geojson_path):
        print(f"Warning: Local GeoJSON not found at {geojson_path}")
        # Debugging info for remote deployment
        print(f"Current Working Directory: {os.getcwd()}")
        try:
            print(f"Files in {base_dir}: {os.listdir(base_dir)}")
        except Exception as e:
            print(f"Error listing files: {e}")
        return None

    try:
        with open(geojson_path, encoding='utf-8') as f:
            world_geojson = json.load(f)

        ids = [f['id'] for f in world_geojson['features'] if 'id' in f]
        print(f"DEBUG: Loaded {len(ids)} countries from {geojson_path}")

        if not ids:
            print("DEBUG: No IDs found in GeoJSON features")
            return None

        # Create a background map using Choropleth
        # We use a constant value for z to make all countries the same color
        bg_trace = go.Choropleth(
            geojson=world_geojson,
            locations=ids,
            z=[1]*len(ids), # Dummy value
            colorscale=[[0, 'rgb(243, 243, 243)'], [1, 'rgb(243, 243, 243)']], # Land color
            showscale=False,
            marker_line_color='rgb(204, 204, 204)', # Coastline color
            marker_line_width=0.5,
            hoverinfo='skip',
            name='Background'
        )
        return bg_trace
    except Exception as e:
        print(f"Error loading GeoJSON: {e}")
        return None


def plot_global_map_static(df, lat_col='centre_lat', lon_col='centre_lon'):
    if df is None:
        return None, None

    # Ensure coordinates are numeric and drop NaNs
    df_clean = df.copy()
    df_clean[lat_col] = pd.to_numeric(df_clean[lat_col], errors='coerce')
    df_clean[lon_col] = pd.to_numeric(df_clean[lon_col], errors='coerce')
    df_clean = df_clean.dropna(subset=[lat_col, lon_col])

    # Sample every 3rd item if too large
    if len(df_clean) > 250000:
        # Calculate step size to get approximately 50000 samples
        step = 2
        # step = max(1, len(df_clean) // 50000)
        df_vis = df_clean.iloc[::step]  # Take every 'step'-th row
        print(f"Sampled {len(df_vis)} points from {len(df_clean)} total points (step={step}) for visualization.")
    else:
        df_vis = df_clean

    # Create static map using Matplotlib
    # Use a fixed size and DPI to make coordinate mapping easier
    # Width=800px, Height=400px -> Aspect Ratio 2:1 (matches 360:180)
    # Increased DPI for better quality: 8x300 = 2400px width
    fig = Figure(figsize=(10, 5), dpi=300)
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

    # Add land + coastline (Cartopy)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, alpha=0.5)

    # Plot points - Use blue to match user request
    ax.scatter(
        df_vis[lon_col],
        df_vis[lat_col],
        s=0.2,
        c="blue",
        marker='o',
        edgecolors='none',
        # alpha=0.6,
        transform=ccrs.PlateCarree(),
        label='Samples',
    )

    # Set limits to full world
    ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())

    # Remove axes and margins
    ax.axis('off')
    # fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # Add Legend
    ax.legend(loc='lower left', markerscale=5, frameon=True, facecolor='white', framealpha=0.9)
    fig.tight_layout()

    # Save to PIL
    buf = BytesIO()
    fig.savefig(buf, format='png', facecolor='white')
    buf.seek(0)
    img = Image.open(buf)

    return img, df_vis

def plot_geographic_distribution(df, scores, threshold, lat_col='centre_lat', lon_col='centre_lon', title="Search Results"):
    if df is None or scores is None:
        return None, None

    df_vis = df.copy()
    df_vis['score'] = scores
    df_vis = df_vis.sort_values(by='score', ascending=False)

    # Top 1%
    top_n = int(len(df_vis) * threshold)
    if top_n < 1:
        top_n = 1
    # if top_n > 5000: top_n = 5000
    df_filtered = df_vis.head(top_n)

    fig = Figure(figsize=(10, 5), dpi=300)
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

    # Add land + coastline (Cartopy)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, alpha=0.5)

    # 2. Plot Search Results with color map
    label_text = f'Top {threshold * 1000:.0f}‰ Matches'
    sc = ax.scatter(
        df_filtered[lon_col],
        df_filtered[lat_col],
        c=df_filtered['score'],
        cmap='Reds',
        s=0.35,
        alpha=0.8,
        transform=ccrs.PlateCarree(),
        label=label_text,
    )

    ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
    ax.axis('off')
    # fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # Add Colorbar
    cbar = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label('Similarity Score')

    # Add Legend
    ax.legend(loc='lower left', markerscale=3, frameon=True, facecolor='white', framealpha=0.9)

    fig.tight_layout()

    # Add title (optional, might overlap)
    # ax.set_title(title)

    buf = BytesIO()
    fig.savefig(buf, format='png', facecolor='white')
    buf.seek(0)
    img = Image.open(buf)

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
        img = res.get('image_384')
        if img is None:
            continue

        caption = f"Score: {res['score']:.4f}\nLat: {res['lat']:.2f}, Lon: {res['lon']:.2f}\nID: {res['id']}"
        gallery_items.append((img, caption))

    return gallery_items


def plot_top5_overview(query_image, results, query_info="Query"):
    """
    Generates a matplotlib figure showing the query image and top retrieved images.
    Similar to the visualization in SigLIP_embdding.ipynb.
    Uses OO Matplotlib API for thread safety.
    """
    top_k = len(results)
    if top_k == 0:
        return None

    # Special case for Text Search (query_image is None) with 10 results
    # User requested: "Middle box top and bottom each 5 photos"
    if query_image is None and top_k == 10:
        cols = 5
        rows = 2
        fig = Figure(figsize=(4 * cols, 4 * rows)) # Square-ish aspect ratio per image
        FigureCanvasAgg(fig)

        for i, res in enumerate(results):
            # Calculate row and col
            _r = i // 5
            _c = i % 5

            # Add subplot (1-based index)
            ax = fig.add_subplot(rows, cols, i + 1)

            img_384 = res.get('image_384')
            if img_384:
                ax.imshow(img_384)
                ax.set_title(f"Rank {i+1}\nScore: {res['score']:.4f}\n({res['lat']:.2f}, {res['lon']:.2f})", fontsize=9)
            else:
                ax.text(0.5, 0.5, "N/A", ha='center', va='center')
            ax.axis('off')

        fig.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        return Image.open(buf)

    # Default behavior (for Image Search or other counts)
    # Layout:
    # If query_image exists:
    #   Row 1: Query Image (Left), Top-K 384x384 (Right)
    #   Row 2: Empty (Left), Top-K Original (Right)

    cols = top_k + (1 if query_image else 0)
    rows = 2

    fig = Figure(figsize=(4 * cols, 8))
    _canvas = FigureCanvasAgg(fig)

    # Plot Query Image
    if query_image:
        # Row 1, Col 1
        ax = fig.add_subplot(rows, cols, 1)
        ax.imshow(query_image)
        ax.set_title(f"Query\n{query_info}", color='blue', fontweight='bold')
        ax.axis('off')

        # Row 2, Col 1 (Empty or repeat?)
        # Let's leave it empty or show text
        ax = fig.add_subplot(rows, cols, cols + 1)
        ax.axis('off')

        start_col = 2
    else:
        start_col = 1

    # Plot Results
    for i, res in enumerate(results):
        # Row 1: 384x384
        ax1 = fig.add_subplot(rows, cols, start_col + i)
        img_384 = res.get('image_384')
        if img_384:
            ax1.imshow(img_384)
            ax1.set_title(f"Rank {i+1} (384)\nScore: {res['score']:.4f}\n({res['lat']:.2f}, {res['lon']:.2f})", fontsize=9)
        else:
            ax1.text(0.5, 0.5, "N/A", ha='center', va='center')
        ax1.axis('off')

        # Row 2: Full
        ax2 = fig.add_subplot(rows, cols, cols + start_col + i)
        img_full = res.get('image_full')
        if img_full:
            ax2.imshow(img_full)
            ax2.set_title("Original", fontsize=9)
        else:
            ax2.text(0.5, 0.5, "N/A", ha='center', va='center')
        ax2.axis('off')

    fig.tight_layout()

    # Save to buffer
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)

    return Image.open(buf)

def plot_location_distribution(df_all, query_lat, query_lon, results, query_info="Query"):
    """
    Generates a global distribution map for location search.
    Reference: improve2_satclip.ipynb
    """
    if df_all is None:
        return None

    fig = Figure(figsize=(8, 4), dpi=300)
    _canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

    # # 1. Background (All samples) - Sampled if too large
    # if len(df_all) > 300000:
    #     df_bg = df_all.sample(300000)
    # else:
    #     df_bg = df_all

    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, alpha=0.5)
    # ax.scatter(df_bg['centre_lon'], df_bg['centre_lat'], c='lightgray', s=1, alpha=0.3, label='All Samples')

    # 2. Query Location
    ax.scatter(query_lon, query_lat, c='red', s=150, marker='*', edgecolors='black', zorder=10, label='Input Coordinate')

    # 3. Retrieved Results
    res_lons = [r['lon'] for r in results]
    res_lats = [r['lat'] for r in results]
    ax.scatter(res_lons, res_lats, c='blue', s=50, marker='x', linewidths=2, label=f'Retrieved Top-{len(results)}')

    # 4. Connecting lines
    for r in results:
        ax.plot([query_lon, r['lon']], [query_lat, r['lat']], 'b--', alpha=0.2)

    ax.legend(loc='upper right')
    ax.set_title(f"Location of Top 5 Matched Images ({query_info})")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.2)

    # Save to buffer
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)

    return Image.open(buf)
