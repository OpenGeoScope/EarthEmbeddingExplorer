import os
from io import BytesIO

import fsspec
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont
from rasterio.io import MemoryFile


def preprocess_s2_true_color(rgb_array):
    """
    Normalize raw Sentinel-2 RGB bands to true-color values for display.

    Applies the standard true-color normalization: divide by 10,000 and
    scale by 2.5, clipping to the range [0, 1].

    Args:
        rgb_array (np.ndarray): Raw Sentinel-2 RGB array (H, W, 3) in uint16.

    Returns:
        np.ndarray: Normalized true-color array in range [0, 1] (float32).
    """
    return (2.5 * (rgb_array.astype(np.float32) / 10000.0)).clip(0, 1)


def crop_center(img_array, cropx, cropy):
    y, x, _c = img_array.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img_array[starty : starty + cropy, startx : startx + cropx]


def read_tif_bytes(tif_bytes):
    with MemoryFile(tif_bytes) as mem_f:
        with mem_f.open(driver="GTiff") as f:
            return f.read().squeeze()


def _fsspec_options_for(url):
    """Build fsspec open() options for a parquet URL.

    Local parquet files (e.g. MajorTOM source images on disk) do not need
    HTTP timeouts or read-ahead caching; fsspec handles local paths fine.
    Remote HTTP reads from ModelScope/HuggingFace can be slow and hit the
    default aiohttp timeout, so extend it and use a modest block cache.
    """
    if os.path.isfile(url):
        return {}
    try:
        import aiohttp

        timeout = aiohttp.ClientTimeout(total=300, connect=30)
        return {
            "cache_type": "readahead",
            "block_size": 1 * 1024 * 1024,
            "client_kwargs": {"timeout": timeout},
        }
    except Exception:
        return {"cache_type": "readahead", "block_size": 1 * 1024 * 1024}


def _parquet_has_column(row_dict, column):
    """Check whether the parquet file referenced by row_dict contains a column.

    Reads only the Arrow schema metadata (no data pages).
    """
    url = row_dict["parquet_url"]
    try:
        with fsspec.open(url, mode="rb", **_fsspec_options_for(url)) as f:
            with pq.ParquetFile(f) as pf:
                return column in pf.schema_arrow.names
    except Exception as e:
        print(f"⚠️ Could not inspect parquet schema for {url}: {e}")
        return False


def read_row_memory(row_dict, columns=None):
    if columns is None:
        columns = ["thumbnail"]
    url = row_dict["parquet_url"]
    row_idx = row_dict["parquet_row"]

    fs_options = _fsspec_options_for(url)

    with fsspec.open(url, mode="rb", **fs_options) as f:
        with pq.ParquetFile(f) as pf:
            table = pf.read_row_group(row_idx, columns=columns)

    row_output = {}
    for col in columns:
        col_data = table[col][0].as_py()

        if col != "thumbnail":
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

    row_subset = df_source[df_source["product_id"] == product_id]
    if len(row_subset) == 0:
        if verbose:
            print(f"❌ Error: Product ID {product_id} not found in DataFrame.")
        return None, (None, None)

    row_dict = row_subset.iloc[0].to_dict()

    if "parquet_url" in row_dict:
        url = row_dict["parquet_url"]
        if "huggingface.co" in url:
            row_dict["parquet_url"] = url.replace("https://huggingface.co", "https://modelscope.cn").replace(
                "resolve/main", "resolve/master"
            )

        # Use a locally available MajorTOM source-image mirror if it exists.
        # Set MAJOR_TOM_LOCAL_IMAGES to the directory containing images_249k/.
        local_root = os.getenv("MAJOR_TOM_LOCAL_IMAGES")
        if local_root:
            parsed = url.split("/")
            if "images_249k" in parsed:
                filename = parsed[-1]
                local_path = os.path.join(local_root, "images_249k", filename)
                if os.path.exists(local_path):
                    row_dict["parquet_url"] = local_path
    else:
        if verbose:
            print("❌ Error: 'parquet_url' missing in metadata.")
        return None, (None, None)

    return row_dict, None


def _bands_to_rgb_pil(bands_data, verbose=True, normalize=True):
    """
    Stack B04/B03/B02 bands into a RGB PIL Image pair (384-crop, full).

    Args:
        bands_data (dict): Dictionary with 'B04', 'B03', 'B02' band arrays.
        verbose (bool): Whether to print debug info.
        normalize (bool): If True, apply true-color normalization (2.5 * value / 1e4).
                          If False, return raw values directly converted to uint8
                          (values > 255 will be clamped).

    Returns:
        tuple: (img_384, img_full) as PIL Images.
    """
    rgb_img = np.stack([bands_data["B04"], bands_data["B03"], bands_data["B02"]], axis=-1)

    if verbose:
        print(f"Raw RGB stats: Min={rgb_img.min()}, Max={rgb_img.max()}, Mean={rgb_img.mean()}, Dtype={rgb_img.dtype}")

    if normalize:
        rgb_norm = preprocess_s2_true_color(rgb_img)
        rgb_uint8 = (rgb_norm * 255).astype(np.uint8)
    else:
        rgb_uint8 = rgb_img.clip(0, 255).astype(np.uint8)

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
MULTIBAND_COLUMNS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]


def reorder_multiband(multiband_array, target_bands, source_bands=None):
    """
    Reorder a multiband array from source band order to target band order.

    This is the single source of truth for mapping between the 12-band
    MajorTOM format and any model-specific band subset/order.

    Args:
        multiband_array (np.ndarray): Array of shape [..., C] where C matches
            len(source_bands).  Typically [H, W, 12] from download_and_process_image.
        target_bands (list[str]): Band names the model expects, e.g.
            ['B02','B03',...] for Clay or ['B01',...,'B12'] for SatCLIP.
        source_bands (list[str] | None): Band names present in multiband_array.
            Defaults to MULTIBAND_COLUMNS.

    Returns:
        np.ndarray: Array reordered to target_bands, shape [..., len(target_bands)].

    Raises:
        ValueError: If a band in target_bands is not found in source_bands.
    """
    if source_bands is None:
        source_bands = MULTIBAND_COLUMNS

    # Fast path: no reordering needed
    if len(target_bands) == len(source_bands) and list(target_bands) == list(source_bands):
        return multiband_array

    band_map = {name: i for i, name in enumerate(source_bands)}
    missing = [b for b in target_bands if b not in band_map]
    if missing:
        raise ValueError(
            f"Target bands not found in source: {missing}. Source has {source_bands}, target asked for {target_bands}."
        )
    indices = [band_map[b] for b in target_bands]
    return multiband_array[..., indices]


def download_and_process_image(product_id, df_source=None, verbose=True, mode="thumbnail", normalize=True):
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
        normalize: For mode="rgb", whether to apply true-color normalization.
                   Set to False if you need raw band values for model preprocessing.

    Returns:
        mode="thumbnail" → (img_384, img_full)          — PIL Images from thumbnail.
        mode="rgb"       → (img_384, img_full)          — PIL Images from RGB bands.
        mode="multiband" → (img_384, img_full, bands)   — thumbnail preview + np.ndarray (H, W, 12) uint16.
    """
    os.environ.setdefault("MODEL_DOMAIN", "modelscope.cn")
    row_dict, _err = _prepare_row_dict(product_id, df_source, verbose)
    if row_dict is None:
        return (None, None) if mode != "multiband" else (None, None, None)

    if verbose:
        print(f"⬇️ Fetching data for {product_id} [mode={mode}] from {row_dict['parquet_url']}...")

    try:
        # ---- thumbnail mode ----
        if mode == "thumbnail":
            if not _parquet_has_column(row_dict, "thumbnail"):
                if verbose:
                    print("⚠️ Parquet has no thumbnail column, falling back to rgb mode.")
                return download_and_process_image(product_id, df_source, verbose, mode="rgb", normalize=normalize)
            data = read_row_memory(row_dict, columns=["thumbnail"])
            if "thumbnail" not in data or data["thumbnail"] is None:
                if verbose:
                    print("⚠️ Thumbnail unavailable, falling back to rgb mode.")
                return download_and_process_image(product_id, df_source, verbose, mode="rgb", normalize=normalize)
            img_384, img_full = _thumbnail_to_pil(data["thumbnail"], verbose)
            if verbose:
                print(f"✅ Successfully processed {product_id} (thumbnail)")
            return img_384, img_full

        # ---- rgb mode ----
        elif mode == "rgb":
            bands_data = read_row_memory(row_dict, columns=["B04", "B03", "B02"])
            if not all(b in bands_data for b in ["B04", "B03", "B02"]):
                if verbose:
                    print(f"❌ Error: Missing bands in fetched data for {product_id}")
                return None, None
            img_384, img_full = _bands_to_rgb_pil(bands_data, verbose, normalize=normalize)
            if verbose:
                print(f"✅ Successfully processed {product_id} (rgb)")
            return img_384, img_full

        # ---- multiband mode ----
        elif mode == "multiband":
            columns_to_read = ["thumbnail", *MULTIBAND_COLUMNS]
            data = read_row_memory(row_dict, columns=columns_to_read)

            # Preview from thumbnail (fallback to RGB composite)
            if "thumbnail" in data and data["thumbnail"] is not None:
                img_384, img_full = _thumbnail_to_pil(data["thumbnail"], verbose)
            elif all(b in data for b in ["B04", "B03", "B02"]):
                img_384, img_full = _bands_to_rgb_pil(data, verbose)
            else:
                img_384, img_full = None, None

            # Stack all 12 bands → (H, W, 12)
            # Determine reference shape from 10m bands (B04/B03/B02) for consistent dimensions
            ref_bands_10m = ["B04", "B03", "B02"]
            ref_shape = None
            for rb in ref_bands_10m:
                if rb in data and data[rb] is not None:
                    ref_shape = data[rb].shape[:2]  # (H, W)
                    break
            if ref_shape is None:
                ref_shape = next(
                    (data[b].shape[:2] for b in MULTIBAND_COLUMNS if b in data and data[b] is not None), None
                )
            if ref_shape is None:
                if verbose:
                    print(f"❌ No usable band data for {product_id}; cannot build multiband array.")
                return img_384, img_full, None

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


def get_placeholder_image(text="Image Unavailable", size=(384, 384)):
    img = Image.new("RGB", size, color=(200, 200, 200))
    d = ImageDraw.Draw(img)
    try:
        # Try to load a default font
        font = ImageFont.load_default()
    except Exception:
        font = None

    # Draw text in center (rough approximation)
    # For better centering we would need font metrics, but simple is fine here
    d.text((20, size[1] // 2), text, fill=(0, 0, 0), font=font)
    return img
