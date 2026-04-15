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
        --output_path /data384/datasets/embeddings_test/dinov2_test.parquet
"""

import argparse
import os
import sys

import pandas as pd
import pyarrow.parquet as pq
import torch
from fsspec.parquet import open_parquet_file

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MajorTOM.embedder.MajorTOM_Embedder import MajorTOM_Embedder
from models.dinov2_model import DINOv2Model
from models.farslip_model import FarSLIPModel
from models.satclip_model import SatCLIPModel
from models.siglip_model import SigLIPModel


MODEL_MAP = {
    "dinov2": DINOv2Model,
    "siglip": SigLIPModel,
    "farslip": FarSLIPModel,
    "satclip": SatCLIPModel,
}


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


def generate_embeddings(model_name, meta_path, parquet_input, output_path, device=None, max_row_groups=None):
    """Main embedding generation logic."""
    if model_name not in MODEL_MAP:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODEL_MAP.keys())}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Loading {model_name} model...")

    # Load model (no embedding file needed)
    model_cls = MODEL_MAP[model_name]
    model = model_cls(device=device)

    print(f"Model bands: {model.bands}")
    print(f"Model input size: {model.size}")

    # Wrap with MajorTOM_Embedder
    embedder = MajorTOM_Embedder(model)
    embedder.to(device)

    parquet_files = get_parquet_files(parquet_input)
    print(f"Found {len(parquet_files)} parquet file(s) to process.")

    embed_df = None

    for pf_path in parquet_files:
        print(f"\nProcessing {pf_path} ...")

        resolved_meta = resolve_meta_url(meta_path, pf_path)
        print(f"Loading metadata from {resolved_meta} ...")
        meta_df = pd.read_parquet(resolved_meta)

        bands = embedder.bands()
        columns = list(bands) + ["product_id", "grid_cell", "timestamp"]

        # Open parquet file
        if os.path.isfile(pf_path):
            # Local file
            pf = pq.ParquetFile(pf_path)
        else:
            # Remote file via fsspec
            f = open_parquet_file(pf_path, columns=columns)
            pf = pq.ParquetFile(f)

        num_row_groups = pf.num_row_groups if max_row_groups is None else min(pf.num_row_groups, max_row_groups)

        for row_idx in range(num_row_groups):
            row = pf.read_row_group(row_idx, columns=columns)

            grid_cell = row["grid_cell"][0].as_py()
            product_id = row["product_id"][0].as_py()

            row_meta = meta_df[
                (meta_df["grid_cell"] == grid_cell) & (meta_df["product_id"] == product_id)
            ].head(1)

            if row_meta.empty:
                print(f"  ⚠️ Metadata not found for {product_id} / {grid_cell}, skipping.")
                continue

            embed_dict = embedder(row, row_meta, device=device)

            if embed_df is None:
                embed_df = embed_dict
            else:
                embed_df = pd.concat([embed_df, embed_dict], ignore_index=True)

            if (row_idx + 1) % 10 == 0 or row_idx == num_row_groups - 1:
                print(f"  Processed {row_idx + 1}/{num_row_groups} row groups, total embeddings: {len(embed_df)}")

    if embed_df is None or embed_df.empty:
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
    parser.add_argument("--model_name", type=str, required=True,
                        choices=["dinov2", "siglip", "farslip", "satclip"],
                        help="Model to use for embedding generation")
    parser.add_argument("--meta_path", type=str, required=True,
                        help="Path to metadata.parquet")
    parser.add_argument("--parquet_input", type=str, required=True,
                        help="Path to a parquet file or directory containing parquet files")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output GeoParquet file path")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run on (cuda/cpu). Auto-detected if omitted.")
    parser.add_argument("--max_row_groups", type=int, default=None,
                        help="Maximum number of row groups to process per parquet file (default: all).")

    args = parser.parse_args()
    generate_embeddings(
        model_name=args.model_name,
        meta_path=args.meta_path,
        parquet_input=args.parquet_input,
        output_path=args.output_path,
        device=args.device,
        max_row_groups=args.max_row_groups,
    )


if __name__ == "__main__":
    main()
