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
    fig = Figure(figsize=(10, 5), dpi=350)
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

    # Add land + coastline (Cartopy) - Use 50m resolution to show small islands
    land_50m = cfeature.NaturalEarthFeature('physical', 'land', '50m')
    coastline_50m = cfeature.NaturalEarthFeature('physical', 'coastline', '50m')
    ax.add_feature(land_50m, facecolor='lightgray', edgecolor='none', alpha=0.2)
    ax.add_feature(coastline_50m, facecolor='none', linewidth=0.8, alpha=0.5)

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

    # Show ALL filtered results (no additional threshold filtering)
    # The threshold was already applied in model.search() and apply_filters()
    df_filtered = df_vis

    fig = Figure(figsize=(10, 5), dpi=350)
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

    # Add land + coastline (Cartopy) - Use 10m resolution to show small islands
    land_10m = cfeature.NaturalEarthFeature('physical', 'land', '10m')
    coastline_10m = cfeature.NaturalEarthFeature('physical', 'coastline', '10m')
    ax.add_feature(land_10m, facecolor='lightgray', edgecolor='none', alpha=0.2)
    ax.add_feature(coastline_10m, facecolor='none', linewidth=0.8, alpha=0.5)

    # 2. Plot Search Results with color map
    label_text = f'{len(df_filtered)} Results'
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
