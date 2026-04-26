#!/usr/bin/env python3
"""
Test image search for all models in EarthEmbeddingExplorer.

This script tests the full pipeline for image-based retrieval:
1. Find nearest product_id for a given lat/lon
2. Download the image (multiband for SatCLIP/Clay, thumbnail for RGB models)
3. Preprocess and encode the image
4. Search against pre-computed embeddings
5. Verify results are reasonable

Usage:
    python tests/test_image_search.py --model Clay
    python tests/test_image_search.py --model all
    python tests/test_image_search.py --model SatCLIP --lat -3 --lon -63
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils import download_and_process_image, reorder_multiband
from models.load_config import load_and_process_config

# Import all model classes
from models.clay_model import ClayModel
from models.dinov2_model import DINOv2Model
from models.farslip_model import FarSLIPModel
from models.satclip_model import SatCLIPModel
from models.siglip_model import SigLIPModel

MODEL_CLASS_MAP = {
    "Clay": ClayModel,
    "DINOv2": DINOv2Model,
    "FarSLIP": FarSLIPModel,
    "SatCLIP": SatCLIPModel,
    "SigLIP": SigLIPModel,
}


# Default test coordinates (Amazon region)
DEFAULT_LAT = -3.0
DEFAULT_LON = -63.0

# All supported models — no hard-coded spectral categories.
# A model declares itself as multi-spectral via `requires_multiband = True`.
ALL_MODELS = {"SigLIP", "FarSLIP", "SatCLIP", "DINOv2", "Clay"}


def find_nearest_product_id(model, lat, lon):
    """Find the nearest product_id to the given lat/lon in the model's embeddings."""
    df = model.df_embed
    if df is None or df.empty:
        raise ValueError(f"Model has no embeddings loaded")
    lats = pd.to_numeric(df["centre_lat"], errors="coerce")
    lons = pd.to_numeric(df["centre_lon"], errors="coerce")
    dists = (lats - lat) ** 2 + (lons - lon) ** 2
    nearest_idx = dists.idxmin()
    pid = df.loc[nearest_idx, "product_id"]
    actual_lat = df.loc[nearest_idx, "centre_lat"]
    actual_lon = df.loc[nearest_idx, "centre_lon"]
    return pid, actual_lat, actual_lon


def download_image_for_model(model, pid, df_embed):
    """Download image appropriate for the model type.

    Multi-spectral models (declared via model.requires_multiband) receive the
    12-band numpy array; RGB models receive the PIL thumbnail.
    """
    needs_multiband = getattr(model, 'requires_multiband', False)
    if needs_multiband:
        result = download_and_process_image(pid, df_source=df_embed, verbose=False, mode="multiband")
        img_384, _, multiband = result
        return img_384, multiband
    else:
        img_384, img_full = download_and_process_image(pid, df_source=df_embed, verbose=False, mode="thumbnail")
        return img_384, None


def test_model_image_search(model_manager, model_name, lat, lon, model=None):
    """Test image search for a single model."""
    print(f"\n{'='*60}")
    print(f"Testing model: {model_name}")
    print(f"Query location: ({lat}, {lon})")
    print(f"{'='*60}")

    if model is None and model_manager is not None:
        model, error = model_manager.get_model(model_name)
        if error:
            print(f"❌ Model not available: {error}")
            return False
    elif model is None:
        print(f"❌ Model not provided")
        return False

    try:
        # 1. Find nearest product_id
        t0 = time.time()
        pid, actual_lat, actual_lon = find_nearest_product_id(model, lat, lon)
        print(f"📍 Nearest product: {pid} at ({actual_lat:.4f}, {actual_lon:.4f})")

        # 2. Download image
        img_384, multiband = download_image_for_model(model, pid, model.df_embed)
        if img_384 is None:
            print(f"❌ Failed to download image for {pid}")
            return False
        print(f"🖼️  Downloaded image: {img_384.size}, mode={img_384.mode}")
        if multiband is not None:
            print(f"📊 Multiband shape: {multiband.shape}, dtype: {multiband.dtype}")

        # 3. Encode image
        needs_multiband = getattr(model, 'requires_multiband', False)
        if needs_multiband:
            # Reorder from generic 12-band MajorTOM format to model-specific bands
            multiband = reorder_multiband(multiband, model.bands)
            print(f"📊 Reordered multiband to {model.bands} -> shape {multiband.shape}")
            image_features = model.encode_image(multiband)
        else:
            image_features = model.encode_image(img_384)

        if image_features is None:
            print(f"❌ Image encoding returned None")
            return False

        print(f"🔢 Embedding shape: {image_features.shape}, device: {image_features.device}")
        print(f"🔢 Embedding norm: {image_features.norm().item():.4f}")

        # 4. Search
        probs, filtered_indices, top_indices = model.search(image_features, top_k=5, top_percent=0.01)

        if probs is None or len(top_indices) == 0:
            print(f"❌ Search returned no results")
            return False

        print(f"🔍 Top-5 results:")
        for i, idx in enumerate(top_indices):
            row = model.df_embed.iloc[idx]
            print(f"   {i+1}. {row['product_id']} ({row['centre_lat']:.4f}, {row['centre_lon']:.4f}) score={probs[idx]:.4f}")

        # 5. Verify results
        assert image_features.shape[-1] > 0, "Empty embedding"
        assert len(top_indices) == 5, f"Expected 5 top results, got {len(top_indices)}"
        assert probs[top_indices[0]] >= probs[top_indices[-1]], "Scores not sorted"

        elapsed = time.time() - t0
        print(f"✅ {model_name} test passed in {elapsed:.2f}s")
        return True

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ {model_name} test failed: {e}")
        return False


def load_single_model(model_name, device="cuda"):
    """Load a single model without using ModelManager (saves GPU memory)."""
    config = load_and_process_config()
    model_cfg = config.get(model_name.lower(), {}) if config else {}
    model_cls = MODEL_CLASS_MAP[model_name]

    kwargs = {"device": device}
    if "ckpt_path" in model_cfg:
        kwargs["ckpt_path"] = model_cfg["ckpt_path"]
    if "model_name" in model_cfg:
        kwargs["model_name"] = model_cfg["model_name"]
    if "tokenizer_path" in model_cfg:
        kwargs["tokenizer_path"] = model_cfg["tokenizer_path"]
    if "embedding_path" in model_cfg:
        kwargs["embedding_path"] = model_cfg["embedding_path"]

    print(f"Loading {model_name}...")
    model = model_cls(**kwargs)
    return model


def main():
    parser = argparse.ArgumentParser(description="Test image search for EarthEmbeddingExplorer models")
    parser.add_argument("--model", type=str, default="all",
                        choices=list(ALL_MODELS) + ["all"],
                        help="Model to test (default: all)")
    parser.add_argument("--lat", type=float, default=DEFAULT_LAT, help="Latitude (default: -3)")
    parser.add_argument("--lon", type=float, default=DEFAULT_LON, help="Longitude (default: -63)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()

    if args.model == "all":
        models_to_test = sorted(list(ALL_MODELS))
    else:
        models_to_test = [args.model]

    print(f"Models to test: {models_to_test}")
    print(f"Query location: ({args.lat}, {args.lon})")
    print(f"Device: {args.device}")

    results = {}
    for model_name in models_to_test:
        # Free GPU memory between models
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

        model = load_single_model(model_name, device=args.device)
        results[model_name] = test_model_image_search(None, model_name, args.lat, args.lon, model=model)

        # Clean up
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for model_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {model_name}: {status}")

    all_passed = all(results.values())
    if all_passed:
        print(f"\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  Some tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
