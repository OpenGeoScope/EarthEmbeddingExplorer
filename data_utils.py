import math
import os
from io import BytesIO

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import fsspec
import numpy as np
import pyarrow.parquet as pq
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw, ImageFont
from rasterio.io import MemoryFile


def crop_center(img_array, cropx, cropy):
    y, x, _c = img_array.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img_array[starty:starty+cropy, startx:startx+cropx]

def read_tif_bytes(tif_bytes):
    with MemoryFile(tif_bytes) as mem_f:
        with mem_f.open(driver='GTiff') as f:
            return f.read().squeeze()

def read_row_memory(row_dict, columns=None):
    if columns is None:
        columns = ["thumbnail"]
    url = row_dict['parquet_url']
    row_idx = row_dict['parquet_row']

    fs_options = {
        "cache_type": "readahead",
        "block_size": 5 * 1024 * 1024
    }

    with fsspec.open(url, mode='rb', **fs_options) as f:
        with pq.ParquetFile(f) as pf:
            table = pf.read_row_group(row_idx, columns=columns)

    row_output = {}
    for col in columns:
        col_data = table[col][0].as_py()

        if col != 'thumbnail':
            row_output[col] = read_tif_bytes(col_data)
        else:
            stream = BytesIO(col_data)
            row_output[col] = Image.open(stream)

    return row_output

def _prepare_row_dict(product_id, df_source, verbose=True):
    """Locate the product row and fix the parquet URL. Returns (row_dict, error_tuple)."""
    if df_source is None:
        if verbose:
            print("❌ Error: No DataFrame provided.")
        return None, (None, None)

    row_subset = df_source[df_source['product_id'] == product_id]
    if len(row_subset) == 0:
        if verbose:
            print(f"❌ Error: Product ID {product_id} not found in DataFrame.")
        return None, (None, None)

    row_dict = row_subset.iloc[0].to_dict()

    if 'parquet_url' in row_dict:
        url = row_dict['parquet_url']
        if 'huggingface.co' in url:
            row_dict['parquet_url'] = url.replace('https://huggingface.co', 'https://modelscope.cn').replace('resolve/main', 'resolve/master')
        elif 'hf-mirror.com' in url:
            row_dict['parquet_url'] = url.replace('https://hf-mirror.com', 'https://modelscope.cn').replace('resolve/main', 'resolve/master')
    else:
        if verbose:
            print("❌ Error: 'parquet_url' missing in metadata.")
        return None, (None, None)

    return row_dict, None


def _bands_to_rgb_pil(bands_data, verbose=True):
    """Stack B04/B03/B02 bands into a normalised RGB PIL Image pair (384-crop, full)."""
    rgb_img = np.stack([bands_data['B04'], bands_data['B03'], bands_data['B02']], axis=-1)

    if verbose:
        print(f"Raw RGB stats: Min={rgb_img.min()}, Max={rgb_img.max()}, Mean={rgb_img.mean()}, Dtype={rgb_img.dtype}")

    rgb_norm = (2.5 * (rgb_img.astype(float) / 10000.0)).clip(0, 1)
    rgb_uint8 = (rgb_norm * 255).astype(np.uint8)

    if verbose:
        print(f"Processed RGB stats: Min={rgb_uint8.min()}, Max={rgb_uint8.max()}, Mean={rgb_uint8.mean()}")

    img_full = Image.fromarray(rgb_uint8)

    if rgb_uint8.shape[0] >= 384 and rgb_uint8.shape[1] >= 384:
        cropped_array = crop_center(rgb_uint8, 384, 384)
        img_384 = Image.fromarray(cropped_array)
    else:
        if verbose:
            print(f"⚠️ Image too small {rgb_uint8.shape}, resizing to 384x384.")
        img_384 = img_full.resize((384, 384))

    return img_384, img_full


def _thumbnail_to_pil(thumb_img, verbose=True):
    """Convert a thumbnail PIL Image to a (384-crop/resize, full) pair."""
    img_full = thumb_img.convert("RGB")
    w, h = img_full.size
    if w >= 384 and h >= 384:
        arr = np.array(img_full)
        cropped = crop_center(arr, 384, 384)
        img_384 = Image.fromarray(cropped)
    else:
        if verbose:
            print(f"⚠️ Thumbnail too small ({w}x{h}), resizing to 384x384.")
        img_384 = img_full.resize((384, 384))
    return img_384, img_full


# All 12 Sentinel-2 bands available in MajorTOM parquet files
MULTIBAND_COLUMNS = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']


def download_and_process_image(product_id, df_source=None, verbose=True, mode="thumbnail"):
    """
    Download and process a MajorTOM image.

    Args:
        product_id: The product identifier in df_source.
        df_source: DataFrame with metadata (product_id, parquet_url, parquet_row, …).
        verbose: Print progress / debug info.
        mode: Download mode — one of:
            "thumbnail" (default) — read the pre-rendered thumbnail column (fastest).
            "rgb"                 — read B04/B03/B02 bands and compose true-color RGB.
            "multiband"           — read all 12 S2 bands + thumbnail for preview.

    Returns:
        mode="thumbnail" → (img_384, img_full)          — PIL Images from thumbnail.
        mode="rgb"       → (img_384, img_full)          — PIL Images from RGB bands.
        mode="multiband" → (img_384, img_full, bands)   — thumbnail preview + np.ndarray (H, W, 12) uint16.
    """
    if os.path.exists("./configs/modelscope_ai.yaml"):
        os.environ["MODEL_DOMAIN"] = "modelscope.cn"
    else:
        os.environ["MODEL_DOMAIN"] = "modelscope.cn"
    row_dict, _err = _prepare_row_dict(product_id, df_source, verbose)
    if row_dict is None:
        return (None, None) if mode != "multiband" else (None, None, None)

    if verbose:
        print(f"⬇️ Fetching data for {product_id} [mode={mode}] from {row_dict['parquet_url']}...")

    try:
        # ---- thumbnail mode ----
        if mode == "thumbnail":
            data = read_row_memory(row_dict, columns=['thumbnail'])
            if 'thumbnail' not in data or data['thumbnail'] is None:
                if verbose:
                    print("⚠️ Thumbnail unavailable, falling back to rgb mode.")
                return download_and_process_image(product_id, df_source, verbose, mode="rgb")
            img_384, img_full = _thumbnail_to_pil(data['thumbnail'], verbose)
            if verbose:
                print(f"✅ Successfully processed {product_id} (thumbnail)")
            return img_384, img_full

        # ---- rgb mode ----
        elif mode == "rgb":
            bands_data = read_row_memory(row_dict, columns=['B04', 'B03', 'B02'])
            if not all(b in bands_data for b in ['B04', 'B03', 'B02']):
                if verbose:
                    print(f"❌ Error: Missing bands in fetched data for {product_id}")
                return None, None
            img_384, img_full = _bands_to_rgb_pil(bands_data, verbose)
            if verbose:
                print(f"✅ Successfully processed {product_id} (rgb)")
            return img_384, img_full

        # ---- multiband mode ----
        elif mode == "multiband":
            columns_to_read = ['thumbnail', *MULTIBAND_COLUMNS]
            data = read_row_memory(row_dict, columns=columns_to_read)

            # Preview from thumbnail (fallback to RGB composite)
            if 'thumbnail' in data and data['thumbnail'] is not None:
                img_384, img_full = _thumbnail_to_pil(data['thumbnail'], verbose)
            elif all(b in data for b in ['B04', 'B03', 'B02']):
                img_384, img_full = _bands_to_rgb_pil(data, verbose)
            else:
                img_384, img_full = None, None

            # Stack all 12 bands → (H, W, 12)
            # Determine reference shape from 10m bands (B04/B03/B02) for consistent dimensions
            ref_bands_10m = ['B04', 'B03', 'B02']
            ref_shape = None
            for rb in ref_bands_10m:
                if rb in data and data[rb] is not None:
                    ref_shape = data[rb].shape[:2]  # (H, W)
                    break
            if ref_shape is None:
                ref_shape = next((data[b].shape[:2] for b in MULTIBAND_COLUMNS if b in data and data[b] is not None), (224, 224))

            band_arrays = []
            for band_name in MULTIBAND_COLUMNS:
                if band_name not in data or data[band_name] is None:
                    if verbose:
                        print(f"⚠️ Band {band_name} missing, filling with zeros.")
                    band_arrays.append(np.zeros(ref_shape, dtype=np.uint16))
                else:
                    arr = data[band_name]
                    # Resize bands with different spatial resolution to the reference shape
                    if arr.shape[:2] != ref_shape:
                        if verbose:
                            print(f"⚠️ Band {band_name} shape {arr.shape} != ref {ref_shape}, resizing.")
                        arr_pil = Image.fromarray(arr)
                        arr_pil = arr_pil.resize((ref_shape[1], ref_shape[0]), resample=Image.BICUBIC)
                        arr = np.array(arr_pil)
                    band_arrays.append(arr)
            multiband_array = np.stack(band_arrays, axis=-1)  # (H, W, 12)

            if verbose:
                print(f"✅ Successfully processed {product_id} (multiband {multiband_array.shape})")
            return img_384, img_full, multiband_array

        else:
            if verbose:
                print(f"❌ Unknown mode: {mode}")
            return None, None

    except Exception as e:
        if verbose:
            print(f"❌ Error processing {product_id}: {e}")
        import traceback
        traceback.print_exc()
        return (None, None) if mode != "multiband" else (None, None, None)

# Define Esri Imagery Class
class EsriImagery(cimgt.GoogleTiles):
    def _image_url(self, tile):
        x, y, z = tile
        return f'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'


def get_placeholder_image(text="Image Unavailable", size=(384, 384)):
    img = Image.new('RGB', size, color=(200, 200, 200))
    d = ImageDraw.Draw(img)
    try:
        # Try to load a default font
        font = ImageFont.load_default()
    except Exception:
        font = None

    # Draw text in center (rough approximation)
    # For better centering we would need font metrics, but simple is fine here
    d.text((20, size[1]//2), text, fill=(0, 0, 0), font=font)
    return img

def get_esri_satellite_image(lat, lon, score=None, rank=None, query=None):
    """
    Generates a satellite image visualization using Esri World Imagery via Cartopy.
    Matches the style of the provided notebook.
    Uses OO Matplotlib API for thread safety.
    """
    try:
        imagery = EsriImagery()

        # Create figure using OO API
        fig = Figure(figsize=(5, 5), dpi=100)
        _canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(1, 1, 1, projection=imagery.crs)

        # Set extent to approx 10km x 10km around the point
        extent_deg = 0.05
        ax.set_extent([lon - extent_deg, lon + extent_deg, lat - extent_deg, lat + extent_deg], crs=ccrs.PlateCarree())

        # Add the imagery
        ax.add_image(imagery, 14)

        # Add a marker for the center
        ax.plot(lon, lat, marker='+', color='yellow', markersize=12, markeredgewidth=2, transform=ccrs.PlateCarree())

        # Add Bounding Box (3840m x 3840m)
        box_size_m = 384 * 10 # 3840m

        # Convert meters to degrees (approx)
        # 1 deg lat = 111320m
        # 1 deg lon = 111320m * cos(lat)
        dlat = (box_size_m / 111320)
        dlon = (box_size_m / (111320 * math.cos(math.radians(lat))))

        # Bottom-Left corner
        rect_lon = lon - dlon / 2
        rect_lat = lat - dlat / 2

        # Add Rectangle
        rect = Rectangle((rect_lon, rect_lat), dlon, dlat,
                        linewidth=2, edgecolor='red', facecolor='none', transform=ccrs.PlateCarree())
        ax.add_patch(rect)

        # Title
        title_parts = []
        if query:
            title_parts.append(f"{query}")
        if rank is not None:
            title_parts.append(f"Rank {rank}")
        if score is not None:
            title_parts.append(f"Score: {score:.4f}")

        ax.set_title("\n".join(title_parts), fontsize=10)

        # Save to buffer
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)

        return Image.open(buf)

    except Exception as e:
        # Suppress full traceback for network errors to avoid log spam
        error_msg = str(e)
        if "Connection reset by peer" in error_msg or "Network is unreachable" in error_msg or "urlopen error" in error_msg:
            print(f"⚠️ Network warning: Could not fetch Esri satellite map for ({lat:.4f}, {lon:.4f}). Server might be offline.")
        else:
            print(f"Error generating Esri image for {lat}, {lon}: {e}")
            # Only print traceback for non-network errors
            # import traceback
            # traceback.print_exc()

        # Return a placeholder image with text
        return get_placeholder_image(f"Map Unavailable\n({lat:.2f}, {lon:.2f})")

def get_esri_satellite_image_url(lat, lon, zoom=14):
    """
    Returns the URL for the Esri World Imagery tile at the given location.
    """
    try:
        # imagery = EsriImagery()
        # Calculate tile coordinates
        # This is a simplification, cimgt handles this internally usually
        # But for direct URL we might need more logic or just use the static map approach above
        # For now, let's stick to the static map generation which works
        pass
    except Exception:
        pass
    return None
