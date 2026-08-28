#!/usr/bin/env python3
"""
Generate embedding datasets from MajorTOM parquet files.

This script replicates the functionality of the
05-Generate-Major-TOM-Embeddings.ipynb notebook. It loads a chosen model,
wraps it with MajorTOM_Embedder, processes each row group in the input
parquet(s), and writes a GeoParquet file containing the embeddings and
spatial metadata.

Example:
python generate_embeddings.py \
    --model_name dinov2 \
    --meta_path /data384/datasets/Core-S2L2A/metadata.parquet \
    --parquet_input /data384/datasets/Core-S2L2A/images/part_00001.parquet \
    --output_path /data384/datasets/embeddings_test/dinov2_test.parquet \
    --fragment_size 384
"""

import argparse
import hashlib
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from fsspec.parquet import open_parquet_file
from pyproj import CRS, Transformer
from shapely.ops import transform as shapely_transform

from clay_metadata import resolve_clay_metadata, wgs84_centroid

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MajorTOM.embedder.MajorTOM_Embedder import MajorTOM_Embedder
from models.load_config import load_config


def _load_model_class(model_name):
    """Lazily import the model class requested by --model_name."""
    if model_name == "dinov2":
        from models.dinov2_model import DINOv2Model

        return DINOv2Model
    if model_name == "siglip":
        from models.siglip_model import SigLIPModel

        return SigLIPModel
    if model_name == "farslip":
        from models.farslip_model import FarSLIPModel

        return FarSLIPModel
    if model_name == "tipsv2":
        from models.tipsv2_model import TIPSv2Model

        return TIPSv2Model
    if model_name == "satclip":
        from models.satclip_model import SatCLIPModel

        return SatCLIPModel
    if model_name == "clay":
        from models.clay_model import ClayModel

        return ClayModel
    if model_name == "olmoearth":
        from models.olmoearth_model import OlmoEarthModel

        return OlmoEarthModel
    if model_name == "qwen3vl":
        from models.qwen3vl_embedding_model import Qwen3VLEmbeddingModel

        return Qwen3VLEmbeddingModel
    raise ValueError(f"Unknown model: {model_name}")


MODEL_MAP = {
    "dinov2": "dinov2",
    "siglip": "siglip",
    "farslip": "farslip",
    "tipsv2": "tipsv2",
    "satclip": "satclip",
    "clay": "clay",
    "olmoearth": "olmoearth",
    "qwen3vl": "qwen3vl",
}


def get_model_kwargs(model_name, device):
    """Build model kwargs from config.yaml or defaults."""
    kwargs = {"device": device}
    config = load_config()
    if config and model_name in config:
        model_cfg = config[model_name]
        if "ckpt_path" in model_cfg:
            kwargs["ckpt_path"] = model_cfg["ckpt_path"]
        if "model_name" in model_cfg:
            kwargs["model_name"] = model_cfg["model_name"]
        if "tokenizer_path" in model_cfg:
            kwargs["tokenizer_path"] = model_cfg["tokenizer_path"]
        if "model_size" in model_cfg:
            kwargs["model_size"] = model_cfg["model_size"]
        if "model_version" in model_cfg:
            kwargs["model_version"] = model_cfg["model_version"]
        if "image_size" in model_cfg:
            kwargs["image_size"] = model_cfg["image_size"]
        if "repo_path" in model_cfg:
            kwargs["repo_path"] = model_cfg["repo_path"]
        if "warmup_runs" in model_cfg:
            kwargs["warmup_runs"] = model_cfg["warmup_runs"]
        if "warmup_batch" in model_cfg:
            kwargs["warmup_batch"] = model_cfg["warmup_batch"]
    return kwargs


def get_parquet_files(parquet_input):
    """Return a list of parquet file paths from a file or directory."""
    if os.path.isfile(parquet_input):
        return [parquet_input]
    elif os.path.isdir(parquet_input):
        files = []
        for fname in sorted(os.listdir(parquet_input)):
            if fname.endswith(".parquet"):
                files.append(os.path.join(parquet_input, fname))
        return files
    else:
        raise ValueError(f"parquet_input must be a file or directory: {parquet_input}")


def resolve_meta_url(meta_path, parquet_file_path):
    """
    Resolve metadata path. If meta_path is relative and parquet_file_path
    points to a local directory, try to locate metadata relative to the
    parquet directory.
    """
    if os.path.isabs(meta_path) or os.path.exists(meta_path):
        return meta_path

    # If parquet is local, try resolving relative to its parent
    if os.path.isfile(parquet_file_path):
        base_dir = os.path.dirname(os.path.dirname(parquet_file_path))
        candidate = os.path.join(base_dir, meta_path)
        if os.path.exists(candidate):
            return candidate
    return meta_path


def _rewrite_parquet_url_for_subset(parquet_url, local_pf_path):
    """
    Rewrite a full Core-S2L2A remote URL to the Core-S2L2A-249k subset URL.

    When the local input parquet comes from the Core-S2L2A-249k subset
    (e.g. downloaded from ModelScope), the metadata may still point to the
    full Core-S2L2A dataset whose row ordering differs from the subset.
    The caller must also replace parquet_row with the local subset row-group
    index because the full and subset datasets use different row ordering.
    """
    local_pf_path = local_pf_path or ""
    # Only rewrite when the local source is the 249k subset.
    if not ("Core-S2L2A-249k" in local_pf_path or "images_249k" in local_pf_path):
        return parquet_url

    match = re.search(r"part_(\d+)\.parquet", os.path.basename(local_pf_path))
    if not match:
        return parquet_url

    part_num = match.group(1)
    return (
        f"https://www.modelscope.cn/datasets/Major-TOM/Core-S2L2A-249k/"
        f"resolve/master/images_249k/part_{part_num}.parquet"
    )


def _rewrite_row_meta_parquet_location(row_meta, pf_path, row_idx):
    """Rewrite parquet_url and parquet_row for a local 249k subset row.

    Only local Core-S2L2A-249k subset files trigger a rewrite;
    _rewrite_parquet_url_for_subset has an internal path whitelist, so
    non-249k inputs are returned unchanged.
    """
    if not os.path.isfile(pf_path):
        return row_meta
    original_url = row_meta["parquet_url"].iloc[0] if "parquet_url" in row_meta.columns else None
    rewritten_url = _rewrite_parquet_url_for_subset(original_url, pf_path)
    if rewritten_url == original_url:
        return row_meta

    row_meta = row_meta.copy()
    row_meta["parquet_url"] = rewritten_url
    row_meta["parquet_row"] = row_idx
    return row_meta


def _metadata_lookup(meta_index, grid_cell, product_id):
    """Return a single-row metadata DataFrame for a parquet row."""
    try:
        row_meta = meta_index.loc[(grid_cell, product_id)]
    except KeyError:
        return None

    if isinstance(row_meta, pd.Series):
        row_meta = row_meta.to_frame().T
    else:
        row_meta = row_meta.head(1)
    return row_meta


def _first_value(row_meta, column):
    """Extract a scalar from a one-row metadata DataFrame."""
    return row_meta[column].iloc[0]


def _prepare_single_fragment_image(img, fragment_size):
    """Resize a pre-cropped image to fragment_size and return CHW tensor."""
    h, w, _c = img.shape
    if h != fragment_size or w != fragment_size:
        img_np = img.numpy() if torch.is_tensor(img) else np.array(img)
        img = torch.from_numpy(cv2.resize(img_np, (fragment_size, fragment_size), interpolation=cv2.INTER_NEAREST))
    elif not torch.is_tensor(img):
        img = torch.from_numpy(np.array(img))
    return img.permute(2, 0, 1)


def _decode_single_fragment_row(row, row_meta, fragment_size, embedder):
    """Decode one parquet row for single-fragment embedding.

    Returns either:
      - A dict with keys ``image``, ``row_meta``, ``footprint``, ``crs`` for
        normal single-fragment samples.
      - A tuple ``("large", row, row_meta, img, footprint, crs)`` when the
        image is larger than ``fragment_size`` and needs the original tiling
        path.
      - ``None`` when metadata is missing.
    """
    if row_meta is None or row_meta.empty:
        return None

    img, footprint, crs, raster_metadata = embedder._read_image(row, return_metadata=True)
    h, w = img.shape[:2]
    if h <= fragment_size and w <= fragment_size:
        centroid = wgs84_centroid(tuple(footprint.bounds), crs)
        latlon_candidates = []
        if centroid is not None:
            latlon_candidates.append((*centroid, "tiff_bounds"))
        latlon_candidates.append(
            (
                _first_value(row_meta, "centre_lat") if "centre_lat" in row_meta.columns else None,
                _first_value(row_meta, "centre_lon") if "centre_lon" in row_meta.columns else None,
                "embedding_center",
            )
        )
        product_datetime = row["product_datetime"][0].as_py() if "product_datetime" in row.column_names else None
        clay_metadata = None
        if getattr(embedder.embedder, "supports_spatiotemporal_metadata", False):
            clay_metadata = resolve_clay_metadata(
                time_candidates=[
                    ((raster_metadata or {}).get("tiff_timestamp"), "tiff_tag"),
                    (product_datetime, "parquet_product_datetime"),
                    (_first_value(row_meta, "timestamp"), "embedding_timestamp"),
                ],
                latlon_candidates=latlon_candidates,
            )
        return {
            "image": _prepare_single_fragment_image(img, fragment_size),
            "row_meta": row_meta,
            "footprint": footprint,
            "crs": crs,
            "clay_metadata": clay_metadata,
        }
    return ("large", row, row_meta, img, footprint, crs)


def _build_single_fragment_rows(batch_items, embeddings, fragment_size):
    """Build output rows for a batch of one-fragment images."""
    rows = []
    column_types = {
        "grid_row_u": "int16",
        "grid_col_r": "int16",
        "centre_lat": "float32",
        "centre_lon": "float32",
    }

    for item, embedding_tensor in zip(batch_items, embeddings, strict=True):
        row_meta = item["row_meta"]
        embedding = embedding_tensor.detach().cpu().numpy()
        transformer = Transformer.from_crs(item["crs"], CRS.from_epsg(4326), always_xy=True)
        geometry = shapely_transform(transformer.transform, item["footprint"])
        centre_lon, centre_lat = geometry.centroid.coords[0]

        timestamp = _first_value(row_meta, "timestamp")
        product_id = _first_value(row_meta, "product_id")
        combined = f"{geometry}_{timestamp}_{product_id}_{embedding}"
        unique_id = hashlib.sha256(combined.encode()).hexdigest()

        row_dict = {
            "unique_id": unique_id,
            "embedding": embedding,
            "timestamp": timestamp,
            "product_id": product_id,
            "grid_cell": _first_value(row_meta, "grid_cell"),
            "grid_row_u": _first_value(row_meta, "grid_row_u"),
            "grid_col_r": _first_value(row_meta, "grid_col_r"),
            "geometry": geometry,
            "centre_lat": centre_lat,
            "centre_lon": centre_lon,
            "utm_footprint": item["footprint"].wkt,
            "utm_crs": item["crs"].to_string(),
            "pixel_bbox": [0, 0, fragment_size, fragment_size],
            "parquet_row": _first_value(row_meta, "parquet_row") if "parquet_row" in row_meta.columns else None,
            "parquet_url": _first_value(row_meta, "parquet_url") if "parquet_url" in row_meta.columns else None,
        }
        if item["clay_metadata"] is not None:
            row_dict.update(
                {
                    "clay_time_input": item["clay_metadata"]["clay_time_input"],
                    "clay_latlon_input": item["clay_metadata"]["clay_latlon_input"],
                    "clay_time_input_source": item["clay_metadata"]["clay_time_input_source"],
                    "clay_latlon_input_source": item["clay_metadata"]["clay_latlon_input_source"],
                }
            )
        rows.append(row_dict)

    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326").astype(column_types)


def _flush_single_fragment_batch(embedder, batch_items, device, fragment_size):
    """Encode a pending batch of single-fragment images."""
    if not batch_items:
        return None

    image_batch = torch.stack([item["image"] for item in batch_items], dim=0).to(device, non_blocking=True)
    timestamps = [_first_value(item["row_meta"], "timestamp") for item in batch_items]
    metadata = [item["clay_metadata"] for item in batch_items]
    if not any(item is not None for item in metadata):
        metadata = None
    with torch.no_grad():
        embeddings = embedder._encode_images(image_batch, timestamps=timestamps, metadata=metadata)
    return _build_single_fragment_rows(batch_items, embeddings, fragment_size)


def generate_embeddings(
    model_name,
    meta_path,
    parquet_input,
    output_path,
    device=None,
    max_row_groups=None,
    fragment_size=None,
    batch_size=16,
    preload_parquet=False,
    num_workers=None,
    num_shards=1,
    shard_index=0,
):
    """Main embedding generation logic."""
    if model_name not in MODEL_MAP:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODEL_MAP.keys())}")
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError(f"Expected 0 <= shard_index < num_shards, got {shard_index=} and {num_shards=}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Loading {model_name} model...")

    # Load model (no embedding file needed)
    model_cls = _load_model_class(model_name)
    model_kwargs = get_model_kwargs(model_name, device)
    model = model_cls(**model_kwargs)

    print(f"Model bands: {model.bands}")
    print(f"Model input size: {model.size}")

    # Wrap with MajorTOM_Embedder
    embedder = MajorTOM_Embedder(model)
    embedder.to(device)

    # Override fragment_size if specified (e.g. for pre-cropped 384x384 imagery)
    if fragment_size is not None:
        embedder.frag_params["fragment_size"] = fragment_size
        print(f"Override fragment_size to {fragment_size}")

    use_single_fragment = fragment_size is not None
    if num_workers is None:
        num_workers = batch_size if use_single_fragment else 1

    all_parquet_files = get_parquet_files(parquet_input)
    parquet_files = all_parquet_files[shard_index::num_shards]
    print(
        f"Found {len(all_parquet_files)} parquet file(s); "
        f"shard {shard_index}/{num_shards} will process {len(parquet_files)}."
    )
    print(f"Single-fragment decoding workers: {num_workers}")

    embed_frames = []
    meta_cache = {}

    for pf_path in parquet_files:
        print(f"\nProcessing {pf_path} ...")

        resolved_meta = resolve_meta_url(meta_path, pf_path)
        if resolved_meta not in meta_cache:
            print(f"Loading metadata from {resolved_meta} ...")
            meta_df = pd.read_parquet(resolved_meta)
            meta_cache[resolved_meta] = meta_df.set_index(["grid_cell", "product_id"], drop=False).sort_index()
        else:
            print(f"Reusing metadata from {resolved_meta} ...")
        meta_index = meta_cache[resolved_meta]

        bands = embedder.bands()
        columns = [*bands, "product_id", "grid_cell"]
        if getattr(model, "supports_spatiotemporal_metadata", False):
            columns.append("product_datetime")

        # Open parquet file
        f = None
        if os.path.isfile(pf_path):
            # Local file
            pf = pq.ParquetFile(pf_path)
        else:
            # Remote file via fsspec
            f = open_parquet_file(pf_path, columns=columns)
            pf = pq.ParquetFile(f)

        try:
            preloaded_table = None
            if preload_parquet and use_single_fragment:
                if os.path.isfile(pf_path):
                    print("Preloading parquet columns into memory ...")
                    preloaded_table = pq.read_table(pf_path, columns=columns)
                else:
                    print("Skipping parquet preload for non-local parquet input.")

            num_row_groups = pf.num_row_groups if max_row_groups is None else min(pf.num_row_groups, max_row_groups)
            if preloaded_table is not None:
                num_row_groups = min(preloaded_table.num_rows, num_row_groups)

            batch_items = []
            embed_count = sum(len(frame) for frame in embed_frames if frame is not None)

            def process_row(
                row, row_idx, batch_items, meta_index=meta_index, num_row_groups=num_row_groups, pf_path=pf_path
            ):
                nonlocal embed_count
                grid_cell = row["grid_cell"][0].as_py()
                product_id = row["product_id"][0].as_py()

                row_meta = _metadata_lookup(meta_index, grid_cell, product_id)

                if row_meta is None or row_meta.empty:
                    print(f"  ⚠️ Metadata not found for {product_id} / {grid_cell}, skipping.")
                    return batch_items

                # The metadata's parquet_url usually points to the full Core-S2L2A
                # dataset, but when we process the local Core-S2L2A-249k subset the
                # row numbers correspond to the subset ordering. Rewrite both the
                # URL and row index so downstream readers fetch the same product.
                row_meta = _rewrite_row_meta_parquet_location(row_meta, pf_path, row_idx)

                if use_single_fragment:
                    img, footprint, crs, raster_metadata = embedder._read_image(row, return_metadata=True)
                    h, w = img.shape[:2]
                    if h <= fragment_size and w <= fragment_size:
                        centroid = wgs84_centroid(tuple(footprint.bounds), crs)
                        latlon_candidates = []
                        if centroid is not None:
                            latlon_candidates.append((*centroid, "tiff_bounds"))
                        latlon_candidates.append(
                            (
                                _first_value(row_meta, "centre_lat") if "centre_lat" in row_meta.columns else None,
                                _first_value(row_meta, "centre_lon") if "centre_lon" in row_meta.columns else None,
                                "embedding_center",
                            )
                        )
                        product_datetime = (
                            row["product_datetime"][0].as_py() if "product_datetime" in row.column_names else None
                        )
                        batch_items.append(
                            {
                                "image": _prepare_single_fragment_image(img, fragment_size),
                                "row_meta": row_meta,
                                "footprint": footprint,
                                "crs": crs,
                                "clay_metadata": (
                                    resolve_clay_metadata(
                                        time_candidates=[
                                            ((raster_metadata or {}).get("tiff_timestamp"), "tiff_tag"),
                                            (product_datetime, "parquet_product_datetime"),
                                            (_first_value(row_meta, "timestamp"), "embedding_timestamp"),
                                        ],
                                        latlon_candidates=latlon_candidates,
                                    )
                                    if getattr(model, "supports_spatiotemporal_metadata", False)
                                    else None
                                ),
                            }
                        )
                        if len(batch_items) >= batch_size:
                            batch_frame = _flush_single_fragment_batch(embedder, batch_items, device, fragment_size)
                            embed_frames.append(batch_frame)
                            embed_count += len(batch_frame)
                            batch_items = []
                    else:
                        batch_frame = _flush_single_fragment_batch(embedder, batch_items, device, fragment_size)
                        if batch_frame is not None:
                            embed_frames.append(batch_frame)
                            embed_count += len(batch_frame)
                        batch_items = []
                        embed_dict = embedder(row, row_meta, device=device)
                        embed_frames.append(embed_dict)
                        embed_count += len(embed_dict)
                else:
                    embed_dict = embedder(row, row_meta, device=device)
                    embed_frames.append(embed_dict)
                    embed_count += len(embed_dict)

                if (row_idx + 1) % 10 == 0 or row_idx == num_row_groups - 1:
                    print(
                        f"  Processed {row_idx + 1}/{num_row_groups} row groups, "
                        f"total embeddings: {embed_count + len(batch_items)}"
                    )

                return batch_items

            if use_single_fragment and preloaded_table is not None:
                for batch_start in range(0, num_row_groups, batch_size):
                    batch_end = min(batch_start + batch_size, num_row_groups)
                    row_table = preloaded_table.slice(batch_start, batch_end - batch_start)
                    for row_pos, row_idx in enumerate(range(batch_start, batch_end)):
                        batch_items = process_row(row_table.slice(row_pos, 1), row_idx, batch_items)
            elif use_single_fragment:
                for batch_start in range(0, num_row_groups, batch_size):
                    row_group_indices = list(range(batch_start, min(batch_start + batch_size, num_row_groups)))
                    row_table = pf.read_row_groups(row_group_indices, columns=columns)

                    rows_to_decode = [row_table.slice(row_pos, 1) for row_pos in range(len(row_group_indices))]
                    row_metas = []
                    for row, row_idx in zip(rows_to_decode, row_group_indices, strict=True):
                        grid_cell = row["grid_cell"][0].as_py()
                        product_id = row["product_id"][0].as_py()
                        row_meta = _metadata_lookup(meta_index, grid_cell, product_id)
                        if row_meta is not None and not row_meta.empty:
                            row_meta = _rewrite_row_meta_parquet_location(row_meta, pf_path, row_idx)
                        row_metas.append(row_meta)

                    if num_workers > 1:
                        with ThreadPoolExecutor(max_workers=num_workers) as executor:
                            futures = [
                                executor.submit(
                                    _decode_single_fragment_row,
                                    row,
                                    row_meta,
                                    fragment_size,
                                    embedder,
                                )
                                for row, row_meta in zip(rows_to_decode, row_metas, strict=True)
                            ]
                            decoded = [future.result() for future in futures]
                    else:
                        decoded = [
                            _decode_single_fragment_row(row, row_meta, fragment_size, embedder)
                            for row, row_meta in zip(rows_to_decode, row_metas, strict=True)
                        ]

                    for row_pos, row_idx in enumerate(row_group_indices):
                        result = decoded[row_pos]
                        if result is None:
                            continue

                        if isinstance(result, tuple) and result[0] == "large":
                            batch_frame = _flush_single_fragment_batch(embedder, batch_items, device, fragment_size)
                            if batch_frame is not None:
                                embed_frames.append(batch_frame)
                                embed_count += len(batch_frame)
                            batch_items = []
                            _, row, row_meta, _img, _footprint, _crs = result
                            embed_dict = embedder(row, row_meta, device=device)
                            embed_frames.append(embed_dict)
                            embed_count += len(embed_dict)
                        else:
                            batch_items.append(result)
                            if len(batch_items) >= batch_size:
                                batch_frame = _flush_single_fragment_batch(embedder, batch_items, device, fragment_size)
                                embed_frames.append(batch_frame)
                                embed_count += len(batch_frame)
                                batch_items = []

                        if (row_idx + 1) % 10 == 0 or row_idx == num_row_groups - 1:
                            print(
                                f"  Processed {row_idx + 1}/{num_row_groups} row groups, "
                                f"total embeddings: {embed_count + len(batch_items)}"
                            )

            else:
                for row_idx in range(num_row_groups):
                    row = pf.read_row_group(row_idx, columns=columns)
                    batch_items = process_row(row, row_idx, batch_items)

            batch_frame = _flush_single_fragment_batch(embedder, batch_items, device, fragment_size)
            if batch_frame is not None:
                embed_frames.append(batch_frame)

        finally:
            pf.close()
            if f is not None:
                f.close()

    if not embed_frames:
        print("No embeddings were generated.")
        return

    embed_df = pd.concat(embed_frames, ignore_index=True)
    if embed_df.empty:
        print("No embeddings were generated.")
        return

    embed_df = embed_df.reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    embed_df.to_parquet(output_path)
    print(f"\n✅ Saved {len(embed_df)} embeddings to {output_path}")

    # Sanity check
    sanity = pd.read_parquet(output_path)
    print("Sanity check columns:", sanity.columns.tolist())
    print(sanity.head())


def main():
    parser = argparse.ArgumentParser(description="Generate MajorTOM embeddings")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=["dinov2", "siglip", "farslip", "tipsv2", "satclip", "clay", "olmoearth", "qwen3vl"],
        help="Model to use for embedding generation",
    )
    parser.add_argument("--meta_path", type=str, required=True, help="Path to metadata.parquet")
    parser.add_argument(
        "--parquet_input", type=str, required=True, help="Path to a parquet file or directory containing parquet files"
    )
    parser.add_argument("--output_path", type=str, required=True, help="Output GeoParquet file path")
    parser.add_argument(
        "--device", type=str, default=None, help="Device to run on (cuda/cpu). Auto-detected if omitted."
    )
    parser.add_argument(
        "--max_row_groups",
        type=int,
        default=None,
        help="Maximum number of row groups to process per parquet file (default: all).",
    )
    parser.add_argument(
        "--fragment_size",
        type=int,
        default=None,
        help=(
            "Override the default fragment size (model input size). "
            "Useful for pre-cropped imagery (e.g. 384x384) where each image "
            "should produce a single embedding instead of multiple fragments."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of pre-cropped single-fragment images to encode per model call.",
    )
    parser.add_argument(
        "--preload_parquet",
        action="store_true",
        help="Preload local parquet columns into memory before single-fragment embedding generation.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help=(
            "Number of parallel workers for decoding single-fragment parquet rows. "
            "Defaults to batch_size when fragment_size is set."
        ),
    )
    parser.add_argument("--num_shards", type=int, default=1, help="Split input parquet files into this many shards.")
    parser.add_argument("--shard_index", type=int, default=0, help="Zero-based shard to process.")

    args = parser.parse_args()
    generate_embeddings(
        model_name=args.model_name,
        meta_path=args.meta_path,
        parquet_input=args.parquet_input,
        output_path=args.output_path,
        device=args.device,
        max_row_groups=args.max_row_groups,
        fragment_size=args.fragment_size,
        batch_size=args.batch_size,
        preload_parquet=args.preload_parquet,
        num_workers=args.num_workers,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )


if __name__ == "__main__":
    main()
