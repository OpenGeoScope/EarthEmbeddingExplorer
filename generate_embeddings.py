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
import sys

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from fsspec.parquet import open_parquet_file
from pyproj import CRS, Transformer
from shapely.ops import transform as shapely_transform

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MajorTOM.embedder.MajorTOM_Embedder import MajorTOM_Embedder
from models.clay_model import ClayModel
from models.dinov2_model import DINOv2Model
from models.farslip_model import FarSLIPModel
from models.load_config import load_config
from models.olmoearth_model import OlmoEarthModel
from models.satclip_model import SatCLIPModel
from models.siglip_model import SigLIPModel

MODEL_MAP = {
    "dinov2": DINOv2Model,
    "siglip": SigLIPModel,
    "farslip": FarSLIPModel,
    "satclip": SatCLIPModel,
    "clay": ClayModel,
    "olmoearth": OlmoEarthModel,
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


def _embed_single_fragment(embedder, row, row_meta, device, fragment_size, img=None, footprint=None, crs=None):
    """
    Embed a pre-cropped image as a single fragment (no tiling).

    Reads the image bands (or uses pre-read ones), optionally resizes to
    fragment_size, encodes the whole image with the model, and returns a
    GeoDataFrame with a single row.
    """
    if img is None:
        img, footprint, crs = embedder._read_image(row)
    h, w, _c = img.shape

    # Resize to target fragment_size if image is not exactly fragment_size
    if h != fragment_size or w != fragment_size:
        img_np = img.numpy() if torch.is_tensor(img) else np.array(img)
        img_resized = cv2.resize(img_np, (fragment_size, fragment_size), interpolation=cv2.INTER_NEAREST)
        img = torch.from_numpy(img_resized)
    else:
        img = img if torch.is_tensor(img) else torch.from_numpy(np.array(img))

    # Encode whole image: (H,W,C) -> (1,C,H,W)
    img_tensor = img.permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = embedder.embedder(img_tensor).cpu().numpy()[0]

    pixel_bbox = [0, 0, fragment_size, fragment_size]
    utm_footprint = footprint
    transformer = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)
    geometry = shapely_transform(transformer.transform, utm_footprint)
    centre_lon, centre_lat = geometry.centroid.coords[0]

    combined = f"{geometry}_{row_meta.timestamp.item()}_{row_meta.product_id.item()}_{embedding}"
    unique_id = hashlib.sha256(combined.encode()).hexdigest()

    row_dict = {
        "unique_id": unique_id,
        "embedding": embedding,
        "timestamp": row_meta.timestamp.item(),
        "product_id": row_meta.product_id.item(),
        "grid_cell": row_meta.grid_cell.item(),
        "grid_row_u": row_meta.grid_row_u.item(),
        "grid_col_r": row_meta.grid_col_r.item(),
        "geometry": geometry,
        "centre_lat": centre_lat,
        "centre_lon": centre_lon,
        "utm_footprint": utm_footprint.wkt,
        "utm_crs": crs.to_string(),
        "pixel_bbox": pixel_bbox,
        "parquet_row": row_meta.parquet_row.item() if "parquet_row" in row_meta.columns else None,
        "parquet_url": row_meta.parquet_url.item() if "parquet_url" in row_meta.columns else None,
    }

    gdf = gpd.GeoDataFrame([row_dict])
    column_types = {
        "grid_row_u": "int16",
        "grid_col_r": "int16",
        "centre_lat": "float32",
        "centre_lon": "float32",
    }
    return gdf.astype(column_types)


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

        rows.append(
            {
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
        )

    return gpd.GeoDataFrame(rows).astype(column_types)


def _flush_single_fragment_batch(embedder, batch_items, device, fragment_size):
    """Encode a pending batch of single-fragment images."""
    if not batch_items:
        return None

    image_batch = torch.stack([item["image"] for item in batch_items], dim=0).to(device, non_blocking=True)
    with torch.no_grad():
        embeddings = embedder.embedder(image_batch)
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
):
    """Main embedding generation logic."""
    if model_name not in MODEL_MAP:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODEL_MAP.keys())}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Loading {model_name} model...")

    # Load model (no embedding file needed)
    model_cls = MODEL_MAP[model_name]
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

    parquet_files = get_parquet_files(parquet_input)
    print(f"Found {len(parquet_files)} parquet file(s) to process.")

    embed_frames = []
    meta_cache = {}

    for pf_path in parquet_files:
        print(f"\nProcessing {pf_path} ...")

        resolved_meta = resolve_meta_url(meta_path, pf_path)
        if resolved_meta not in meta_cache:
            print(f"Loading metadata from {resolved_meta} ...")
            meta_df = pd.read_parquet(resolved_meta)
            meta_cache[resolved_meta] = meta_df.set_index(["grid_cell", "product_id"], drop=False)
        else:
            print(f"Reusing metadata from {resolved_meta} ...")
        meta_index = meta_cache[resolved_meta]

        bands = embedder.bands()
        columns = [*bands, "product_id", "grid_cell"]

        # Open parquet file
        if os.path.isfile(pf_path):
            # Local file
            pf = pq.ParquetFile(pf_path)
        else:
            # Remote file via fsspec
            f = open_parquet_file(pf_path, columns=columns)
            pf = pq.ParquetFile(f)

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

        def process_row(row, row_idx, batch_items, meta_index=meta_index, num_row_groups=num_row_groups):
            nonlocal embed_count
            grid_cell = row["grid_cell"][0].as_py()
            product_id = row["product_id"][0].as_py()

            row_meta = _metadata_lookup(meta_index, grid_cell, product_id)

            if row_meta is None or row_meta.empty:
                print(f"  ⚠️ Metadata not found for {product_id} / {grid_cell}, skipping.")
                return batch_items

            if use_single_fragment:
                img, footprint, crs = embedder._read_image(row)
                h, w = img.shape[:2]
                if h <= fragment_size and w <= fragment_size:
                    batch_items.append(
                        {
                            "image": _prepare_single_fragment_image(img, fragment_size),
                            "row_meta": row_meta,
                            "footprint": footprint,
                            "crs": crs,
                        }
                    )
                    if len(batch_items) >= batch_size:
                        embed_frames.append(_flush_single_fragment_batch(embedder, batch_items, device, fragment_size))
                        batch_items = []
                    embed_count = sum(len(frame) for frame in embed_frames if frame is not None) + len(batch_items)
                else:
                    batch_frame = _flush_single_fragment_batch(embedder, batch_items, device, fragment_size)
                    if batch_frame is not None:
                        embed_frames.append(batch_frame)
                    batch_items = []
                    embed_dict = embedder(row, row_meta, device=device)
                    embed_frames.append(embed_dict)
                    embed_count = sum(len(frame) for frame in embed_frames if frame is not None)
            else:
                embed_dict = embedder(row, row_meta, device=device)
                embed_frames.append(embed_dict)
                embed_count = sum(len(frame) for frame in embed_frames if frame is not None)

            if (row_idx + 1) % 10 == 0 or row_idx == num_row_groups - 1:
                print(f"  Processed {row_idx + 1}/{num_row_groups} row groups, total embeddings: {embed_count}")

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
                for row_pos, row_idx in enumerate(row_group_indices):
                    batch_items = process_row(row_table.slice(row_pos, 1), row_idx, batch_items)
        else:
            for row_idx in range(num_row_groups):
                row = pf.read_row_group(row_idx, columns=columns)
                batch_items = process_row(row, row_idx, batch_items)

        batch_frame = _flush_single_fragment_batch(embedder, batch_items, device, fragment_size)
        if batch_frame is not None:
            embed_frames.append(batch_frame)

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
        choices=["dinov2", "siglip", "farslip", "satclip", "clay", "olmoearth"],
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
    )


if __name__ == "__main__":
    main()
