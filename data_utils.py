import os
from io import BytesIO

import cv2
import fsspec
import numpy as np
import pyarrow.parquet as pq
import rasterio
from PIL import Image, ImageDraw, ImageFont
from rasterio.io import MemoryFile

from clay_metadata import resolve_clay_metadata, timestamp_from_tiff_tags, wgs84_centroid


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


def read_tif_bytes(tif_bytes, return_metadata=False):
    with MemoryFile(tif_bytes) as mem_f:
        with mem_f.open(driver="GTiff") as f:
            array = f.read().squeeze()
            if not return_metadata:
                return array
            bounds = tuple(f.bounds)
            crs = f.crs.to_string() if f.crs is not None else None
            tags = f.tags()
            centroid = wgs84_centroid(bounds, crs)
            metadata = {
                "bounds": bounds,
                "crs": crs,
                "tags": tags,
                "tiff_timestamp": timestamp_from_tiff_tags(tags),
                "centroid": centroid,
            }
            return array, metadata


def load_multispectral_geotiff(path):
    """Load a local 12-band example GeoTIFF and resolve its model metadata."""
    with rasterio.open(path) as dataset:
        if dataset.count != len(MULTIBAND_COLUMNS):
            raise ValueError(f"Expected {len(MULTIBAND_COLUMNS)} bands, found {dataset.count}")
        data = dataset.read()
        descriptions = list(dataset.descriptions)
        if all(name in descriptions for name in MULTIBAND_COLUMNS):
            indices = [descriptions.index(name) for name in MULTIBAND_COLUMNS]
            data = data[indices]
        elif any(description is not None for description in descriptions):
            raise ValueError(f"Unexpected GeoTIFF band descriptions: {descriptions}")

        tags = dataset.tags()
        bounds = tuple(dataset.bounds)
        crs = dataset.crs.to_string() if dataset.crs is not None else None

    centroid = wgs84_centroid(bounds, crs)
    latlon_candidates = []
    if centroid is not None:
        latlon_candidates.append((*centroid, "tiff_bounds"))
    latlon_candidates.append((tags.get("centre_lat"), tags.get("centre_lon"), "tiff_tag"))
    metadata = resolve_clay_metadata(
        time_candidates=[
            (tags.get("product_datetime"), "tiff_tag"),
            (tags.get("timestamp"), "tiff_tag"),
            (timestamp_from_tiff_tags(tags), "tiff_tag"),
        ],
        latlon_candidates=latlon_candidates,
    )
    metadata.update(
        {
            "product_id": tags.get("product_id"),
            "product_datetime": tags.get("product_datetime"),
            "raster_crs": crs,
            "raster_bounds": bounds,
            "band_names": list(MULTIBAND_COLUMNS),
        }
    )
    return data.transpose(1, 2, 0), metadata


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


def read_row_memory(row_dict, columns=None, include_raster_metadata=False):
    if columns is None:
        columns = ["thumbnail"]
    url = row_dict["parquet_url"]
    row_idx = row_dict["parquet_row"]

    fs_options = _fsspec_options_for(url)

    with fsspec.open(url, mode="rb", **fs_options) as f:
        with pq.ParquetFile(f) as pf:
            available_columns = set(pf.schema_arrow.names)
            table = pf.read_row_group(row_idx, columns=[column for column in columns if column in available_columns])

    row_output = {}
    for col in columns:
        if col not in table.column_names:
            continue
        col_data = table[col][0].as_py()

        if col == "thumbnail":
            stream = BytesIO(col_data)
            row_output[col] = Image.open(stream)
        elif col in {*MULTIBAND_COLUMNS, "cloud_mask"}:
            if include_raster_metadata and "raster_metadata" not in row_output:
                row_output[col], row_output["raster_metadata"] = read_tif_bytes(col_data, return_metadata=True)
            else:
                row_output[col] = read_tif_bytes(col_data)
        else:
            row_output[col] = col_data

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
        endpoint = os.getenv("DOWNLOAD_ENDPOINT", "modelscope.cn")
        preferred_host = "www.modelscope.ai" if endpoint in ("modelscope.ai", "ai") else "www.modelscope.cn"
        if "huggingface.co" in url:
            url = url.replace("https://huggingface.co", f"https://{preferred_host}").replace(
                "resolve/main", "resolve/master"
            )
        elif "modelscope.cn" in url or "modelscope.ai" in url:
            url = url.replace("www.modelscope.cn", preferred_host).replace("www.modelscope.ai", preferred_host)
            url = url.replace("https://modelscope.cn", f"https://{preferred_host}").replace(
                "https://modelscope.ai", f"https://{preferred_host}"
            )
        row_dict["parquet_url"] = url

        # Use a locally available MajorTOM source-image mirror if it exists.
        # Set MAJOR_TOM_LOCAL_IMAGES to the directory containing images_249k/.
        local_root = os.getenv("MAJOR_TOM_LOCAL_IMAGES")
        if local_root:
            parsed = row_dict["parquet_url"].split("/")
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


def download_and_process_image(
    product_id, df_source=None, verbose=True, mode="thumbnail", normalize=True, return_metadata=False
):
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
        return_metadata: For multiband mode, append resolved TIFF/source-row
            metadata without changing the default return tuple.

    Returns:
        mode="thumbnail" → (img_384, img_full)          — PIL Images from thumbnail.
        mode="rgb"       → (img_384, img_full)          — PIL Images from RGB bands.
        mode="multiband" → (img_384, img_full, bands)   — thumbnail preview + np.ndarray (H, W, 12) uint16.
        mode="multiband", return_metadata=True → (..., metadata).
    """
    os.environ.setdefault("MODEL_DOMAIN", "modelscope.cn")
    row_dict, _err = _prepare_row_dict(product_id, df_source, verbose)
    if row_dict is None:
        if mode != "multiband":
            return None, None
        return (None, None, None, None) if return_metadata else (None, None, None)

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
            columns_to_read = ["product_id", "thumbnail", "product_datetime", *MULTIBAND_COLUMNS]
            data = read_row_memory(row_dict, columns=columns_to_read, include_raster_metadata=return_metadata)
            downloaded_product_id = data.get("product_id")
            if downloaded_product_id is not None and downloaded_product_id != product_id:
                raise ValueError(
                    f"Source row product mismatch: requested {product_id}, downloaded {downloaded_product_id}"
                )

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
                if return_metadata:
                    return img_384, img_full, None, None
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
                        # Match MajorTOM_Embedder._read_image so offline index
                        # generation and online query encoding see identical pixels.
                        arr = cv2.resize(arr, (ref_shape[1], ref_shape[0]), interpolation=cv2.INTER_NEAREST)
                    band_arrays.append(arr)
            multiband_array = np.stack(band_arrays, axis=-1)  # (H, W, 12)

            if verbose:
                print(f"✅ Successfully processed {product_id} (multiband {multiband_array.shape})")
            if return_metadata:
                raster_metadata = data.get("raster_metadata") or {}
                centroid = raster_metadata.get("centroid")
                latlon_candidates = []
                if centroid is not None:
                    latlon_candidates.append((*centroid, "tiff_bounds"))
                latlon_candidates.append((row_dict.get("centre_lat"), row_dict.get("centre_lon"), "embedding_center"))
                metadata = resolve_clay_metadata(
                    time_candidates=[
                        (raster_metadata.get("tiff_timestamp"), "tiff_tag"),
                        (data.get("product_datetime"), "parquet_product_datetime"),
                        (row_dict.get("timestamp"), "embedding_timestamp"),
                    ],
                    latlon_candidates=latlon_candidates,
                )
                metadata.update(
                    {
                        "product_id": product_id,
                        "product_datetime": data.get("product_datetime"),
                        "raster_crs": raster_metadata.get("crs"),
                        "raster_bounds": raster_metadata.get("bounds"),
                    }
                )
                return img_384, img_full, multiband_array, metadata
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
        if mode != "multiband":
            return None, None
        return (None, None, None, None) if return_metadata else (None, None, None)


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
