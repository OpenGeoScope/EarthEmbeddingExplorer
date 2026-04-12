import os
import tempfile
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import gradio as gr
import numpy as np
import pandas as pd
import torch
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont

from data_utils import download_and_process_image, get_esri_satellite_image, get_placeholder_image
from models.dinov2_model import DINOv2Model
from models.farslip_model import FarSLIPModel
from models.load_config import load_and_process_config
from models.satclip_model import SatCLIPModel

# Import custom modules
from models.siglip_model import SigLIPModel
from visualize import (
    format_results_for_gallery,
    plot_geographic_distribution,
    plot_global_map_static,
    plot_top5_overview,
)

# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {device}")

# Load and process configuration
config = load_and_process_config()
print(config)

# Initialize Models
print("Initializing models...")
models = {}

# DINOv2
try:
    if config and "dinov2" in config:
        models["DINOv2"] = DINOv2Model(
            ckpt_path=config["dinov2"].get("ckpt_path"),
            embedding_path=config["dinov2"].get("embedding_path"),
            device=device,
        )
    else:
        models["DINOv2"] = DINOv2Model(device=device)
except Exception as e:
    print(f"Failed to load DINOv2: {e}")

# SigLIP
try:
    if config and "siglip" in config:
        models["SigLIP"] = SigLIPModel(
            ckpt_path=config["siglip"].get("ckpt_path"),
            tokenizer_path=config["siglip"].get("tokenizer_path"),
            embedding_path=config["siglip"].get("embedding_path"),
            device=device,
        )
    else:
        models["SigLIP"] = SigLIPModel(device=device)
except Exception as e:
    print(f"Failed to load SigLIP: {e}")

# SatCLIP
try:
    if config and "satclip" in config:
        models["SatCLIP"] = SatCLIPModel(
            ckpt_path=config["satclip"].get("ckpt_path"),
            embedding_path=config["satclip"].get("embedding_path"),
            device=device,
        )
    else:
        models["SatCLIP"] = SatCLIPModel(device=device)
except Exception as e:
    print(f"Failed to load SatCLIP: {e}")

# FarSLIP
try:
    if config and "farslip" in config:
        models["FarSLIP"] = FarSLIPModel(
            ckpt_path=config["farslip"].get("ckpt_path"),
            model_name=config["farslip"].get("model_name"),
            embedding_path=config["farslip"].get("embedding_path"),
            device=device,
        )
    else:
        models["FarSLIP"] = FarSLIPModel(device=device)
except Exception as e:
    print(f"Failed to load FarSLIP: {e}")


def get_active_model(model_name):
    if model_name not in models:
        return None, f"Model {model_name} not loaded."
    return models[model_name], None


def build_filter_options(
    enable_time=False,
    start_date="2016-01-01",
    end_date="2024-12-31",
    enable_geo=False,
    lat_min=-90,
    lat_max=90,
    lon_min=-180,
    lon_max=180,
):
    """Pack UI filter controls into a single dict for search functions."""
    return {
        "time": {"enabled": enable_time, "start": start_date, "end": end_date},
        "geo": {"enabled": enable_geo, "lat_min": lat_min, "lat_max": lat_max, "lon_min": lon_min, "lon_max": lon_max},
        # Future: "polygon": {"enabled": False, "coordinates": []}
    }


def apply_filters(df_embed, probs, filtered_indices, top_indices, filter_options):
    """
    Apply post-search filters (time range, geo bounding box, etc.) to retrieval results.
    Returns: (new_filtered_indices, new_top_indices, df_for_geo, probs_for_geo)
    """
    if not filter_options:
        return filtered_indices, top_indices, df_embed, probs

    global_mask = np.ones(len(df_embed), dtype=bool)

    # --- Time filter ---
    time_opts = filter_options.get("time", {})
    if time_opts.get("enabled"):
        try:
            timestamps = pd.to_datetime(df_embed["timestamp"])
            start = pd.to_datetime(time_opts["start"])
            end = pd.to_datetime(time_opts["end"]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            global_mask &= ((timestamps >= start) & (timestamps <= end)).values
        except Exception as e:
            print(f"Time filter parse error, skipping: {e}")

    # --- Geo bounding box filter ---
    geo_opts = filter_options.get("geo", {})
    if geo_opts.get("enabled"):
        try:
            lats = pd.to_numeric(df_embed["centre_lat"], errors="coerce").values
            lons = pd.to_numeric(df_embed["centre_lon"], errors="coerce").values
            g_lat_min, g_lat_max = float(geo_opts.get("lat_min", -90)), float(geo_opts.get("lat_max", 90))
            g_lon_min, g_lon_max = float(geo_opts.get("lon_min", -180)), float(geo_opts.get("lon_max", 180))
            global_mask &= (lats >= g_lat_min) & (lats <= g_lat_max) & (lons >= g_lon_min) & (lons <= g_lon_max)
        except Exception as e:
            print(f"Geo filter error, skipping: {e}")

    # --- Polygon filter (future) ---
    # poly_opts = filter_options.get("polygon", {})
    # if poly_opts.get("enabled"):
    #     from shapely.geometry import Point, Polygon
    #     poly = Polygon(poly_opts["coordinates"])
    #     global_mask &= np.array([poly.contains(Point(lo, la)) for la, lo in zip(lats, lons)])

    # If nothing was filtered out, return as-is
    if global_mask.all():
        return filtered_indices, top_indices, df_embed, probs

    # Re-filter indices
    new_filtered = filtered_indices[global_mask[filtered_indices]]

    # Re-rank top indices from all in-range data by score descending
    in_range_idx = np.where(global_mask)[0]
    in_range_scores = probs[in_range_idx]
    reranked = in_range_idx[np.argsort(in_range_scores)[::-1]]

    # Subset for geographic distribution plot
    df_for_geo = df_embed[global_mask].reset_index(drop=True)
    probs_for_geo = probs[global_mask]

    return new_filtered, reranked, df_for_geo, probs_for_geo


def combine_images(img1, img2):
    if img1 is None:
        return img2
    if img2 is None:
        return img1

    # Resize to match width
    w1, h1 = img1.size
    w2, h2 = img2.size

    new_w = max(w1, w2)
    new_h1 = int(h1 * new_w / w1)
    new_h2 = int(h2 * new_w / w2)

    img1 = img1.resize((new_w, new_h1))
    img2 = img2.resize((new_w, new_h2))

    dst = PILImage.new("RGB", (new_w, new_h1 + new_h2), (255, 255, 255))
    dst.paste(img1, (0, 0))
    dst.paste(img2, (0, new_h1))
    return dst


def create_text_image(text, size=(384, 384)):
    img = PILImage.new("RGB", size, color=(240, 240, 240))
    d = ImageDraw.Draw(img)

    # Try to load a font, fallback to default
    try:
        # Try to find a font that supports larger size
        font = ImageFont.truetype("DejaVuSans.ttf", 40)
    except Exception:
        font = ImageFont.load_default()

    # Wrap text simply
    margin = 20
    offset = 100
    for line in text.split(","):
        d.text((margin, offset), line.strip(), font=font, fill=(0, 0, 0))
        offset += 50

    d.text((margin, offset + 50), "Text Query", font=font, fill=(0, 0, 255))
    return img


def fetch_top_k_images(top_indices, probs, df_embed, query_text=None):
    """
    Fetches top-k thumbnail images for display (Gallery / Results plot).
    Always uses mode='thumbnail' for speed.
    """
    results = []

    # We can run this in parallel
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_idx = {}
        for _i, idx in enumerate(top_indices):
            row = df_embed.iloc[idx]
            pid = row["product_id"]

            future = executor.submit(
                download_and_process_image, pid, df_source=df_embed, verbose=False, mode="thumbnail"
            )
            future_to_idx[future] = idx

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                img_384, img_full = future.result()

                if img_384 is None:
                    # Fallback to Esri if download fails
                    print(f"Download failed for idx {idx}, falling back to Esri...")
                    row = df_embed.iloc[idx]
                    img_384 = get_esri_satellite_image(
                        row["centre_lat"], row["centre_lon"], score=probs[idx], rank=0, query=query_text
                    )
                    img_full = img_384

                row = df_embed.iloc[idx]
                results.append(
                    {
                        "image_384": img_384,
                        "image_full": img_full,
                        "score": probs[idx],
                        "lat": row["centre_lat"],
                        "lon": row["centre_lon"],
                        "id": row["product_id"],
                    }
                )
            except Exception as e:
                print(f"Error fetching image for idx {idx}: {e}")

    # Sort results by score descending (since futures complete in random order)
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def get_all_results_metadata(model, filtered_indices, probs):
    if len(filtered_indices) == 0:
        return []

    # Sort by score descending
    filtered_scores = probs[filtered_indices]
    sorted_order = np.argsort(filtered_scores)[::-1]
    sorted_indices = filtered_indices[sorted_order]

    # Extract from DataFrame
    df_results = model.df_embed.iloc[sorted_indices].copy()
    df_results["score"] = probs[sorted_indices]

    # Rename columns
    df_results = df_results.rename(columns={"product_id": "id", "centre_lat": "lat", "centre_lon": "lon"})

    # Convert to list of dicts
    return df_results[["id", "lat", "lon", "score"]].to_dict("records")


def search_text(query, threshold, model_name, filter_options=None):
    model, error = get_active_model(model_name)
    if error:
        yield None, None, error, None, None, None, None
        return

    if not query:
        yield None, None, "Please enter a query.", None, None, None, None
        return

    try:
        timings = {}

        # 1. Encode Text
        yield None, None, "Encoding text...", None, None, None, None
        t0 = time.time()
        text_features = model.encode_text(query)
        timings["Encoding"] = time.time() - t0

        if text_features is None:
            yield None, None, "Model does not support text encoding or is not initialized.", None, None, None, None
            return

        # 2. Search
        yield None, None, "Encoding text... ✓\nRetrieving similar images...", None, None, None, None
        t0 = time.time()
        probs, filtered_indices, top_indices = model.search(text_features, top_percent=threshold / 1000.0)
        timings["Retrieval"] = time.time() - t0

        if probs is None:
            yield None, None, "Search failed (embeddings missing?).", None, None, None, None
            return

        # Apply post-search filters (time range, geo, etc.)
        df_embed = model.df_embed
        filtered_indices, top_indices, df_for_geo, probs_for_geo = apply_filters(
            df_embed, probs, filtered_indices, top_indices, filter_options
        )

        # Show geographic distribution (not timed)
        geo_dist_map, df_filtered = plot_geographic_distribution(
            df_for_geo, probs_for_geo, threshold / 1000.0, title=f'Similarity to "{query}" ({model_name})'
        )

        # Handle 0 results after filtering
        if len(top_indices) == 0:
            status_msg = (
                "No results found with current filter settings.\nTry relaxing the filters or adjusting the threshold."
            )
            yield (
                gr.update(visible=False),
                [],
                status_msg,
                None,
                [geo_dist_map],
                df_filtered,
                gr.update(value=geo_dist_map, visible=True),
            )
            return

        # 3. Download Images (display always uses thumbnail for gallery)
        yield (
            gr.update(visible=False),
            None,
            "Encoding text... ✓\nRetrieving similar images... ✓\nDownloading images...",
            None,
            None,
            df_filtered,
            gr.update(value=geo_dist_map, visible=True),
        )
        t0 = time.time()
        top_indices = top_indices[:10]
        results = fetch_top_k_images(top_indices, probs, df_embed, query_text=query)
        timings["Download"] = time.time() - t0

        # 4. Visualize - keep geo_dist_map visible
        yield (
            gr.update(visible=False),
            None,
            "Encoding text... ✓\nRetrieving similar images... ✓\nDownloading images... ✓\nGenerating visualizations...",
            None,
            None,
            df_filtered,
            gr.update(value=geo_dist_map, visible=True),
        )
        t0 = time.time()
        fig_results = plot_top5_overview(None, results, query_info=query)
        gallery_items = format_results_for_gallery(results)
        timings["Visualization"] = time.time() - t0

        # 5. Generate Final Status
        timing_str = f"Encoding {timings['Encoding']:.1f}s, Retrieval {timings['Retrieval']:.1f}s, Download {timings['Download']:.1f}s, Visualization {timings['Visualization']:.1f}s\n\n"
        status_msg = timing_str + generate_status_msg(len(filtered_indices), threshold / 100.0, results)

        all_results = get_all_results_metadata(model, filtered_indices, probs)
        results_txt = format_results_to_text(all_results)

        # current_fig: [map, results_img, text, results_meta_for_download]
        top_results_meta = [{"id": r["id"], "lat": r["lat"], "lon": r["lon"], "score": r["score"]} for r in results]
        yield (
            gr.update(visible=False),
            gallery_items,
            status_msg,
            fig_results,
            [geo_dist_map, fig_results, results_txt, top_results_meta, model_name],
            df_filtered,
            gr.update(value=geo_dist_map, visible=True),
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        yield None, None, f"Error: {e!s}", None, None, None, None


def search_image(image_input, threshold, model_name, filter_options=None, multiband_data=None):
    model, error = get_active_model(model_name)
    if error:
        yield None, None, error, None, None, None, None
        return

    if image_input is None:
        yield None, None, "Please upload an image.", None, None, None, None
        return

    try:
        timings = {}

        # 1. Encode Image
        # For SatCLIP: require multiband data (12-band numpy array)
        yield None, None, "Encoding image...", None, None, None, None
        t0 = time.time()
        if model_name == "SatCLIP":
            if multiband_data is not None:
                print(f"SatCLIP: encoding with multiband data {multiband_data.shape}")
                image_features = model.encode_image(multiband_data)
            else:
                yield (
                    None,
                    None,
                    (
                        "⚠️ SatCLIP requires multi-spectral Sentinel-2 input (12/13 bands).\n\n"
                        "RGB images are NOT compatible with SatCLIP image retrieval.\n"
                        "Please use 'Download Image by Geolocation' to obtain a multi-band image first,\n"
                        "or switch to DINOv2 / SigLIP / FarSLIP for RGB image retrieval."
                    ),
                    None,
                    None,
                    None,
                    None,
                )
                return
        else:
            image_features = model.encode_image(image_input)
        timings["Encoding"] = time.time() - t0

        if image_features is None:
            yield None, None, "Model does not support image encoding.", None, None, None, None
            return

        # 2. Search
        yield None, None, "Encoding image... ✓\nRetrieving similar images...", None, None, None, None
        t0 = time.time()
        probs, filtered_indices, top_indices = model.search(image_features, top_percent=threshold / 1000.0)
        timings["Retrieval"] = time.time() - t0

        # Apply post-search filters (time range, geo, etc.)
        df_embed = model.df_embed
        filtered_indices, top_indices, df_for_geo, probs_for_geo = apply_filters(
            df_embed, probs, filtered_indices, top_indices, filter_options
        )

        # Show geographic distribution (not timed)
        geo_dist_map, df_filtered = plot_geographic_distribution(
            df_for_geo, probs_for_geo, threshold / 1000.0, title=f"Similarity to Input Image ({model_name})"
        )

        # Handle 0 results after filtering
        if len(top_indices) == 0:
            status_msg = (
                "No results found with current filter settings.\nTry relaxing the filters or adjusting the threshold."
            )
            yield (
                gr.update(visible=False),
                [],
                status_msg,
                None,
                [geo_dist_map],
                df_filtered,
                gr.update(value=geo_dist_map, visible=True),
            )
            return

        # 3. Download Images (display always uses thumbnail for gallery)
        yield (
            gr.update(visible=False),
            None,
            "Encoding image... ✓\nRetrieving similar images... ✓\nDownloading images...",
            None,
            None,
            df_filtered,
            gr.update(value=geo_dist_map, visible=True),
        )
        t0 = time.time()
        top_indices = top_indices[:6]
        results = fetch_top_k_images(top_indices, probs, df_embed, query_text="Image Query")
        timings["Download"] = time.time() - t0

        # 4. Visualize - keep geo_dist_map visible
        yield (
            gr.update(visible=False),
            None,
            "Encoding image... ✓\nRetrieving similar images... ✓\nDownloading images... ✓\nGenerating visualizations...",
            None,
            None,
            df_filtered,
            gr.update(value=geo_dist_map, visible=True),
        )
        t0 = time.time()
        fig_results = plot_top5_overview(image_input, results, query_info="Image Query")
        gallery_items = format_results_for_gallery(results)
        timings["Visualization"] = time.time() - t0

        # 5. Generate Final Status
        timing_str = f"Encoding {timings['Encoding']:.1f}s, Retrieval {timings['Retrieval']:.1f}s, Download {timings['Download']:.1f}s, Visualization {timings['Visualization']:.1f}s\n\n"
        status_msg = timing_str + generate_status_msg(len(filtered_indices), threshold / 100.0, results)

        all_results = get_all_results_metadata(model, filtered_indices, probs)
        results_txt = format_results_to_text(all_results[:50])

        # current_fig: [map, results_img, text, results_meta_for_download]
        top_results_meta = [{"id": r["id"], "lat": r["lat"], "lon": r["lon"], "score": r["score"]} for r in results]
        yield (
            gr.update(visible=False),
            gallery_items,
            status_msg,
            fig_results,
            [geo_dist_map, fig_results, results_txt, top_results_meta, model_name],
            df_filtered,
            gr.update(value=geo_dist_map, visible=True),
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        yield None, None, f"Error: {e!s}", None, None, None, None


def search_location(lat, lon, threshold, filter_options=None):
    model_name = "SatCLIP"
    model, error = get_active_model(model_name)
    if error:
        yield None, None, error, None, None, None, None
        return

    try:
        timings = {}

        # 1. Encode Location
        yield None, None, "Encoding location...", None, None, None, None
        t0 = time.time()
        loc_features = model.encode_location(float(lat), float(lon))
        timings["Encoding"] = time.time() - t0

        if loc_features is None:
            yield None, None, "Location encoding failed.", None, None, None, None
            return

        # 2. Search
        yield None, None, "Encoding location... ✓\nRetrieving similar images...", None, None, None, None
        t0 = time.time()
        probs, filtered_indices, top_indices = model.search(loc_features, top_percent=threshold / 100.0)
        timings["Retrieval"] = time.time() - t0

        # Apply post-search filters (time range, geo, etc.)
        df_embed = model.df_embed
        filtered_indices, top_indices, df_for_geo, probs_for_geo = apply_filters(
            df_embed, probs, filtered_indices, top_indices, filter_options
        )

        # 3. Generate Distribution Map (not timed for location distribution)
        yield (
            None,
            None,
            "Encoding location... ✓\nRetrieving similar images... ✓\nGenerating distribution map...",
            None,
            None,
            None,
            None,
        )
        top_10_indices = top_indices[:10]
        top_10_results = []
        for idx in top_10_indices:
            row = df_embed.iloc[idx]
            top_10_results.append({"lat": row["centre_lat"], "lon": row["centre_lon"]})

        # Show geographic distribution (not timed)
        geo_dist_map, df_filtered = plot_geographic_distribution(
            df_for_geo, probs_for_geo, threshold / 1000.0, title=f"Similarity to Location ({lat}, {lon})"
        )

        # Handle 0 results after filtering
        if len(top_indices) == 0:
            status_msg = (
                "No results found with current filter settings.\nTry relaxing the filters or adjusting the threshold."
            )
            yield (
                gr.update(visible=False),
                [],
                status_msg,
                None,
                [geo_dist_map],
                df_filtered,
                gr.update(value=geo_dist_map, visible=True),
            )
            return

        # 4. Download Images
        yield (
            gr.update(visible=False),
            None,
            "Encoding location... ✓\nRetrieving similar images... ✓\nGenerating distribution map... ✓\nDownloading images...",
            None,
            None,
            df_filtered,
            gr.update(value=geo_dist_map, visible=True),
        )
        t0 = time.time()
        top_6_indices = top_indices[:6]
        results = fetch_top_k_images(top_6_indices, probs, df_embed, query_text=f"Loc: {lat},{lon}")

        # Get query tile
        query_tile = None
        try:
            lats = pd.to_numeric(df_embed["centre_lat"], errors="coerce")
            lons = pd.to_numeric(df_embed["centre_lon"], errors="coerce")
            dists = (lats - float(lat)) ** 2 + (lons - float(lon)) ** 2
            nearest_idx = dists.idxmin()
            pid = df_embed.loc[nearest_idx, "product_id"]
            query_tile, _ = download_and_process_image(pid, df_source=df_embed, verbose=False, mode="thumbnail")
        except Exception as e:
            print(f"Error fetching nearest MajorTOM image: {e}")
        if query_tile is None:
            query_tile = get_placeholder_image(f"Query Location\n({lat}, {lon})")
        timings["Download"] = time.time() - t0

        # 5. Visualize - keep geo_dist_map visible
        yield (
            gr.update(visible=False),
            None,
            "Encoding location... ✓\nRetrieving similar images... ✓\nGenerating distribution map... ✓\nDownloading images... ✓\nGenerating visualizations...",
            None,
            None,
            df_filtered,
            gr.update(value=geo_dist_map, visible=True),
        )
        t0 = time.time()
        fig_results = plot_top5_overview(query_tile, results, query_info=f"Loc: {lat},{lon}")
        gallery_items = format_results_for_gallery(results)
        timings["Visualization"] = time.time() - t0

        # 6. Generate Final Status
        timing_str = f"Encoding {timings['Encoding']:.1f}s, Retrieval {timings['Retrieval']:.1f}s, Download {timings['Download']:.1f}s, Visualization {timings['Visualization']:.1f}s\n\n"
        status_msg = timing_str + generate_status_msg(len(filtered_indices), threshold / 100.0, results)

        all_results = get_all_results_metadata(model, filtered_indices, probs)
        results_txt = format_results_to_text(all_results)

        # current_fig: [map, results_img, text, results_meta_for_download]
        top_results_meta = [{"id": r["id"], "lat": r["lat"], "lon": r["lon"], "score": r["score"]} for r in results]
        yield (
            gr.update(visible=False),
            gallery_items,
            status_msg,
            fig_results,
            [geo_dist_map, fig_results, results_txt, top_results_meta, model_name],
            df_filtered,
            gr.update(value=geo_dist_map, visible=True),
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        yield None, None, f"Error: {e!s}", None, None, None, None


def generate_status_msg(count, threshold, results):
    status_msg = f"Found {count} matches in top {threshold * 100:.0f}‰.\n\nTop {len(results)} similar images:\n"
    for i, res in enumerate(results[:3]):
        status_msg += f"{i + 1}. Product ID: {res['id']}, Location: ({res['lat']:.4f}, {res['lon']:.4f}), Score: {res['score']:.4f}\n"
    return status_msg


def normalize_scores(scores):
    """Min-max normalize scores to [0, 1] range."""
    s_min, s_max = scores.min(), scores.max()
    if s_max - s_min < 1e-9:
        return np.zeros_like(scores)
    return (scores - s_min) / (s_max - s_min)


def search_mixed(
    query_text,
    query_image,
    lat,
    lon,
    weight_text,
    weight_image,
    weight_location,
    threshold,
    model_name,
    filter_options=None,
    multiband_data=None,
):
    """
    Mixed search combining text, image, and location modalities.

    Uses score-level fusion: final_score = w_t * s_text + w_i * s_image + w_l * s_location
    Text/Image use the selected model (FarSLIP/SigLIP), Location uses SatCLIP.
    """
    try:
        timings = {}

        # Determine which modalities are active (weight > 0 and input provided)
        use_text = weight_text > 0 and query_text and query_text.strip()
        use_image = weight_image > 0 and query_image is not None
        use_location = weight_location > 0 and lat is not None and lon is not None

        if not use_text and not use_image and not use_location:
            yield (
                None,
                None,
                "Please provide at least one query (text, image, or location) with weight > 0.",
                None,
                None,
                None,
                None,
            )
            return

        # Get models
        text_image_model, error = get_active_model(model_name)
        if error and (use_text or use_image):
            yield None, None, error, None, None, None, None
            return

        satclip_model, error = get_active_model("SatCLIP")
        if error and use_location:
            yield None, None, f"SatCLIP required for location search: {error}", None, None, None, None
            return

        # Determine the reference df_embed (use the one with most samples or text_image_model's)
        if use_text or use_image:
            df_ref = text_image_model.df_embed
            ref_model_name = model_name
        else:
            df_ref = satclip_model.df_embed
            ref_model_name = "SatCLIP"

        # If using location AND (text or image), we need to align product_ids
        need_alignment = use_location and (use_text or use_image)

        if need_alignment:
            # Find intersection of product_ids
            pids_ti = set(text_image_model.df_embed["product_id"].values)
            pids_loc = set(satclip_model.df_embed["product_id"].values)
            common_pids = pids_ti & pids_loc

            if len(common_pids) == 0:
                yield (
                    None,
                    None,
                    "No common product IDs between models. Cannot perform mixed search.",
                    None,
                    None,
                    None,
                    None,
                )
                return

            # Create aligned indices
            df_ti = text_image_model.df_embed
            df_loc = satclip_model.df_embed

            # Build pid to index mapping
            ti_pid_to_idx = {pid: idx for idx, pid in enumerate(df_ti["product_id"].values)}
            loc_pid_to_idx = {pid: idx for idx, pid in enumerate(df_loc["product_id"].values)}

            common_pids_list = list(common_pids)
            ti_indices = np.array([ti_pid_to_idx[pid] for pid in common_pids_list])
            loc_indices = np.array([loc_pid_to_idx[pid] for pid in common_pids_list])

            # Use text_image_model's df for result display
            df_ref = df_ti.iloc[ti_indices].reset_index(drop=True)
            ref_model_name = model_name

        # Initialize scores array
        n_samples = len(df_ref)
        final_scores = np.zeros(n_samples, dtype=np.float32)

        # Normalize weights
        total_weight = 0
        if use_text:
            total_weight += weight_text
        if use_image:
            total_weight += weight_image
        if use_location:
            total_weight += weight_location

        w_text = weight_text / total_weight if use_text else 0
        w_image = weight_image / total_weight if use_image else 0
        w_location = weight_location / total_weight if use_location else 0

        status_parts = []

        # --- Encode and compute text scores ---
        if use_text:
            yield None, None, "Encoding text...", None, None, None, None
            t0 = time.time()
            text_features = text_image_model.encode_text(query_text)
            timings["Text Encoding"] = time.time() - t0

            if text_features is None:
                yield None, None, f"Model {model_name} does not support text encoding.", None, None, None, None
                return

            # Compute similarity
            embeddings = text_image_model.image_embeddings.cpu().numpy()
            if need_alignment:
                embeddings = embeddings[ti_indices]
            text_scores = (embeddings @ text_features.cpu().numpy().T).squeeze()
            text_scores = normalize_scores(text_scores)
            final_scores += w_text * text_scores
            status_parts.append(f"Text (w={w_text:.2f})")

        # --- Encode and compute image scores ---
        if use_image:
            status_msg = "Encoding text... ✓\n" if use_text else ""
            yield None, None, status_msg + "Encoding image...", None, None, None, None
            t0 = time.time()
            image_features = text_image_model.encode_image(query_image)
            timings["Image Encoding"] = time.time() - t0

            if image_features is None:
                yield None, None, f"Model {model_name} does not support image encoding.", None, None, None, None
                return

            # Compute similarity
            embeddings = text_image_model.image_embeddings.cpu().numpy()
            if need_alignment:
                embeddings = embeddings[ti_indices]
            image_scores = (embeddings @ image_features.cpu().numpy().T).squeeze()
            image_scores = normalize_scores(image_scores)
            final_scores += w_image * image_scores
            status_parts.append(f"Image (w={w_image:.2f})")

        # --- Encode and compute location scores ---
        if use_location:
            status_msg = ""
            if use_text:
                status_msg += "Encoding text... ✓\n"
            if use_image:
                status_msg += "Encoding image... ✓\n"
            yield None, None, status_msg + "Encoding location...", None, None, None, None
            t0 = time.time()
            loc_features = satclip_model.encode_location(float(lat), float(lon))
            timings["Location Encoding"] = time.time() - t0

            if loc_features is None:
                yield None, None, "Location encoding failed.", None, None, None, None
                return

            # Compute similarity
            embeddings = satclip_model.image_embeddings.cpu().numpy()
            if need_alignment:
                embeddings = embeddings[loc_indices]
            loc_scores = (embeddings @ loc_features.cpu().numpy().T).squeeze()
            loc_scores = normalize_scores(loc_scores)
            final_scores += w_location * loc_scores
            status_parts.append(f"Location (w={w_location:.2f})")

        # --- Retrieve top results ---
        status_msg = ""
        if use_text:
            status_msg += "Encoding text... ✓\n"
        if use_image:
            status_msg += "Encoding image... ✓\n"
        if use_location:
            status_msg += "Encoding location... ✓\n"
        yield None, None, status_msg + "Retrieving similar images...", None, None, None, None

        t0 = time.time()
        # Apply threshold
        top_percent = threshold / 1000.0
        threshold_value = np.percentile(final_scores, 100 * (1 - top_percent))
        filtered_mask = final_scores >= threshold_value
        filtered_indices = np.where(filtered_mask)[0]

        # Sort by score descending
        sorted_order = np.argsort(final_scores)[::-1]
        top_indices = sorted_order[: max(10, int(len(final_scores) * top_percent))]
        timings["Retrieval"] = time.time() - t0

        # Apply post-search filters
        filtered_indices, top_indices, df_for_geo, probs_for_geo = apply_filters(
            df_ref, final_scores, filtered_indices, top_indices, filter_options
        )

        # Generate geographic distribution map
        query_info = " + ".join(status_parts)
        geo_dist_map, df_filtered = plot_geographic_distribution(
            df_for_geo, probs_for_geo, threshold / 1000.0, title=f"Mixed Search: {query_info}"
        )

        # Handle 0 results after filtering
        if len(top_indices) == 0:
            status_msg = (
                "No results found with current filter settings.\nTry relaxing the filters or adjusting the threshold."
            )
            yield (
                gr.update(visible=False),
                [],
                status_msg,
                None,
                [geo_dist_map],
                df_filtered,
                gr.update(value=geo_dist_map, visible=True),
            )
            return

        # --- Download images ---
        yield (
            gr.update(visible=False),
            None,
            status_msg + "Retrieving similar images... ✓\nDownloading images...",
            None,
            None,
            df_filtered,
            gr.update(value=geo_dist_map, visible=True),
        )
        t0 = time.time()
        top_indices = top_indices[:10]
        results = fetch_top_k_images(top_indices, final_scores, df_ref, query_text=query_info)
        timings["Download"] = time.time() - t0

        # --- Visualize ---
        yield (
            gr.update(visible=False),
            None,
            status_msg + "Retrieving similar images... ✓\nDownloading images... ✓\nGenerating visualizations...",
            None,
            None,
            df_filtered,
            gr.update(value=geo_dist_map, visible=True),
        )
        t0 = time.time()

        # Create query visualization
        query_vis = None
        if use_image:
            query_vis = query_image
        elif use_text:
            query_vis = create_text_image(query_text)

        fig_results = plot_top5_overview(query_vis, results, query_info=f"Mixed: {query_info}")
        gallery_items = format_results_for_gallery(results)
        timings["Visualization"] = time.time() - t0

        # --- Generate final status ---
        timing_parts = [f"{k} {v:.1f}s" for k, v in timings.items()]
        timing_str = ", ".join(timing_parts) + "\n\n"

        weight_info = f"Weights: Text={w_text:.2f}, Image={w_image:.2f}, Location={w_location:.2f}\n"
        status_msg = timing_str + weight_info + generate_status_msg(len(filtered_indices), threshold / 100.0, results)

        # Prepare results for download
        all_results = []
        for idx in filtered_indices:
            row = df_ref.iloc[idx]
            all_results.append(
                {
                    "id": row["product_id"],
                    "lat": row["centre_lat"],
                    "lon": row["centre_lon"],
                    "score": final_scores[idx],
                }
            )
        all_results.sort(key=lambda x: x["score"], reverse=True)
        results_txt = format_results_to_text(all_results[:50])

        top_results_meta = [{"id": r["id"], "lat": r["lat"], "lon": r["lon"], "score": r["score"]} for r in results]
        yield (
            gr.update(visible=False),
            gallery_items,
            status_msg,
            fig_results,
            [geo_dist_map, fig_results, results_txt, top_results_meta, ref_model_name],
            df_filtered,
            gr.update(value=geo_dist_map, visible=True),
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        yield None, None, f"Error: {e!s}", None, None, None, None


def get_initial_plot():
    # Find the first available embedding to plot the global map
    df_vis = None
    img = None
    first_model_name = next(iter(models), None)
    if first_model_name is not None and models[first_model_name].df_embed is not None:
        img, df_vis = plot_global_map_static(models[first_model_name].df_embed)
    else:
        print("No embedding data available for initial plot.")
        img, df_vis = None, None

    return gr.update(value=img, visible=True), [img], df_vis, gr.update(visible=False)


def handle_map_click(evt: gr.SelectData, df_vis):
    if evt is None:
        return None, None, None, "No point selected."

    try:
        x, y = evt.index[0], evt.index[1]

        # Image dimensions (New)
        img_width = 3000
        img_height = 1500

        # Scaled Margins (Proportional to 4000x2000)
        left_margin = 110 * 0.75
        right_margin = 110 * 0.75
        top_margin = 100 * 0.75
        bottom_margin = 67 * 0.75

        plot_width = img_width - left_margin - right_margin
        plot_height = img_height - top_margin - bottom_margin

        # Adjust for aspect ratio preservation
        map_aspect = 360.0 / 180.0  # 2.0
        plot_aspect = plot_width / plot_height

        if plot_aspect > map_aspect:
            actual_map_width = plot_height * map_aspect
            actual_map_height = plot_height
            h_offset = (plot_width - actual_map_width) / 2
            v_offset = 0
        else:
            actual_map_width = plot_width
            actual_map_height = plot_width / map_aspect
            h_offset = 0
            v_offset = (plot_height - actual_map_height) / 2

        # Calculate relative position within the plot area
        x_in_plot = x - left_margin
        y_in_plot = y - top_margin

        # Check if click is within the actual map bounds
        if (
            x_in_plot < h_offset
            or x_in_plot > h_offset + actual_map_width
            or y_in_plot < v_offset
            or y_in_plot > v_offset + actual_map_height
        ):
            return None, None, None, "Click outside map area. Please click on the map."

        # Calculate relative position within the map (0 to 1)
        x_rel = (x_in_plot - h_offset) / actual_map_width
        y_rel = (y_in_plot - v_offset) / actual_map_height

        # Clamp to [0, 1]
        x_rel = max(0, min(1, x_rel))
        y_rel = max(0, min(1, y_rel))

        # Convert to geographic coordinates
        lon = x_rel * 360 - 180
        lat = 90 - y_rel * 180

        # Find nearest point in df_vis if available
        pid = ""
        if df_vis is not None:
            dists = (df_vis["centre_lat"] - lat) ** 2 + (df_vis["centre_lon"] - lon) ** 2
            min_idx = dists.idxmin()
            nearest_row = df_vis.loc[min_idx]

            if dists[min_idx] < 25:
                lat = nearest_row["centre_lat"]
                lon = nearest_row["centre_lon"]
                pid = nearest_row["product_id"]

    except Exception as e:
        print(f"Error handling click: {e}")
        import traceback

        traceback.print_exc()
        return None, None, None, f"Error: {e}"

    return lat, lon, pid, f"Selected Point: ({lat:.4f}, {lon:.4f})"


def download_image_by_location(lat, lon, pid, model_name):
    """Download and return the image at the specified location.

    For SatCLIP, automatically downloads multiband data and stores it in the
    returned tuple for subsequent encoding.

    Returns:
        (thumbnail_img, status_msg, multiband_array_or_None)
    """
    if lat is None or lon is None:
        return None, "Please specify coordinates first.", None

    model, error = get_active_model(model_name)
    if error:
        return None, error, None

    try:
        # Convert to float to ensure proper formatting
        lat = float(lat)
        lon = float(lon)

        # Find Product ID if not provided
        if not pid:
            df = model.df_embed
            lats = pd.to_numeric(df["centre_lat"], errors="coerce")
            lons = pd.to_numeric(df["centre_lon"], errors="coerce")
            dists = (lats - lat) ** 2 + (lons - lon) ** 2
            nearest_idx = dists.idxmin()
            pid = df.loc[nearest_idx, "product_id"]

        # For SatCLIP: download multiband for encoding; thumbnail for display
        multiband_array = None
        if model_name == "SatCLIP":
            result = download_and_process_image(pid, df_source=model.df_embed, verbose=True, mode="multiband")
            img_384, _, multiband_array = result
            if img_384 is None:
                return None, f"Failed to download image for location ({lat:.4f}, {lon:.4f})", None
            return img_384, f"Downloaded image at ({lat:.4f}, {lon:.4f}) [multiband for SatCLIP]", multiband_array
        else:
            img_384, _ = download_and_process_image(pid, df_source=model.df_embed, verbose=True, mode="thumbnail")
            if img_384 is None:
                return None, f"Failed to download image for location ({lat:.4f}, {lon:.4f})", None
            return img_384, f"Downloaded image at ({lat:.4f}, {lon:.4f})", None

    except Exception as e:
        import traceback

        traceback.print_exc()
        return None, f"Error: {e!s}", None


def reset_to_global_map():
    """Reset the map to the initial global distribution view"""
    img = None
    df_vis = None
    first_model_name = next(iter(models), None)
    if first_model_name is not None and models[first_model_name].df_embed is not None:
        img, df_vis = plot_global_map_static(models[first_model_name].df_embed)
    else:
        img, df_vis = None, None

    return gr.update(value=img, visible=True), [img], df_vis


def format_results_to_text(results):
    if not results:
        return "No results found."

    txt = f"Top {len(results)} Retrieval Results\n"
    txt += "=" * 30 + "\n\n"
    for i, res in enumerate(results):
        txt += f"Rank: {i + 1}\n"
        txt += f"Product ID: {res['id']}\n"
        txt += f"Location: Latitude {res['lat']:.6f}, Longitude {res['lon']:.6f}\n"
        txt += f"Similarity Score: {res['score']:.6f}\n"
        txt += "-" * 30 + "\n"
    return txt


def save_plot(figs, download_mode="thumbnail"):
    """
    Save results as a downloadable zip file.

    download_mode controls what image data is included for top results:
      - "thumbnail": save the thumbnail images (fast, default)
      - "rgb":       re-download B04/B03/B02 composites and save as PNG
      - "multiband": re-download all 12 S2 bands and save as .npy per image
    """
    if figs is None:
        return None

    temp_dir = tempfile.gettempdir()

    def unique_temp_path(prefix, suffix):
        return os.path.join(temp_dir, f"{prefix}_{uuid.uuid4().hex}{suffix}")

    def save_pil_image(image_obj, prefix, suffix=".png"):
        path = unique_temp_path(prefix, suffix)
        image_obj.save(path)
        return path

    def add_file(zipf, path, arcname):
        zipf.write(path, arcname=arcname)
        return path

    try:
        # Single image: return a standalone PNG
        if isinstance(figs, PILImage.Image):
            return save_pil_image(figs, "earth_explorer_map")

        # Single image inside a list
        if isinstance(figs, (list, tuple)) and len(figs) == 1 and isinstance(figs[0], PILImage.Image):
            return save_pil_image(figs[0], "earth_explorer_map")

        # Plotly fallback
        if not isinstance(figs, (list, tuple)):
            path = unique_temp_path("earth_explorer_plot", ".html")
            figs.write_html(path)
            return path

        zip_path = unique_temp_path("earth_explorer_results", ".zip")

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
            # Map image
            if len(figs) > 0 and figs[0] is not None:
                map_path = save_pil_image(figs[0], "map_distribution")
                add_file(zipf, map_path, "map_distribution.png")

            # Retrieval overview
            if len(figs) > 1 and figs[1] is not None:
                res_path = save_pil_image(figs[1], "retrieval_results")
                add_file(zipf, res_path, "retrieval_results.png")

            # Text report
            if len(figs) > 2 and figs[2] is not None:
                txt_path = unique_temp_path("results", ".txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(figs[2])
                add_file(zipf, txt_path, "results.txt")

            results_meta = figs[3] if len(figs) > 3 else None
            model_name = figs[4] if len(figs) > 4 else None
            df_source = models[model_name].df_embed if model_name in models else None

            if results_meta and isinstance(results_meta, list):
                for rank, res in enumerate(results_meta, start=1):
                    pid = res["id"]

                    try:
                        if download_mode == "multiband" and df_source is not None:
                            result = download_and_process_image(
                                pid, df_source=df_source, verbose=False, mode="multiband"
                            )

                            if result[2] is not None:
                                npy_path = unique_temp_path(f"rank{rank}_{pid}_12bands", ".npy")
                                np.save(npy_path, result[2])
                                add_file(zipf, npy_path, f"images/rank{rank}_{pid}_12bands.npy")

                            if result[0] is not None:
                                preview_path = save_pil_image(result[0], f"rank{rank}_{pid}_preview")
                                add_file(zipf, preview_path, f"images/rank{rank}_{pid}_preview.png")

                        elif download_mode == "rgb" and df_source is not None:
                            _, img_full = download_and_process_image(
                                pid, df_source=df_source, verbose=False, mode="rgb"
                            )
                            if img_full is not None:
                                rgb_path = save_pil_image(img_full, f"rank{rank}_{pid}_rgb")
                                add_file(zipf, rgb_path, f"images/rank{rank}_{pid}_rgb.png")

                        else:
                            _, img_full = download_and_process_image(
                                pid, df_source=df_source, verbose=False, mode="thumbnail"
                            )
                            if img_full is not None:
                                thumb_path = save_pil_image(img_full, f"rank{rank}_{pid}_thumbnail")
                                add_file(zipf, thumb_path, f"images/rank{rank}_{pid}_thumbnail.png")

                    except Exception as e:
                        print(f"Error downloading result image {pid}: {e}")

        return zip_path

    except Exception as e:
        print(f"Error saving: {e}")
        return None


# Gradio Blocks Interface
with gr.Blocks(
    title="EarthEmbeddingExplorer",
    css="""
.filter-checkbox {
    background: transparent !important;
    border: 1px solid #d1d5db !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
    margin: 4px 0 !important;
    box-shadow: none !important;
    outline: none !important;
}
.filter-checkbox > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}
.filter-checkbox label {
    background: transparent !important;
    font-weight: bold !important;
}
.filter-checkbox label span {
    background: transparent !important;
}
/* Remove gray border from Gradio form group wrapping the filter checkboxes */
.form:has(> .filter-checkbox),
div.form:has(.filter-checkbox) {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
}
""",
) as demo:
    gr.Markdown("# EarthEmbeddingExplorer")
    gr.HTML("""
    <div style="font-size: 1.2em;">
    EarthEmbeddingExplorer 是一款工具跨模态遥感图像检索工具，允许您使用自然语言描述、图像、地理位置或简单地在地图上点击来搜索地球的卫星图像。例如，您可以输入“热带雨林”或“有城市的海岸线”，系统就会找到地球上与您描述相符的位置。然后，它会在世界地图上可视化这些位置的卫星图像嵌入和您的输入嵌入的相似度，并显示最相似的图像。您可以下载检索结果和最相似的图像。<br>
    EarthEmbeddingExplorer is a tool that allows you to search for satellite images of the Earth using natural language descriptions, images, geolocations, or a simple a click on the map. For example, you can type "tropical rainforest" or "coastline with a city," and the system will find locations on Earth that match your description. It then visualizes these locations on a world map and displays the top matching images.
    </div>

    <div style="display: flex; gap: 0.2em; align-items: center; justify-content: center;">
        <a href="https://modelscope.cn/studios/Major-TOM/EarthEmbeddingExplorer/"><img src="https://img.shields.io/badge/Open in ModelScope.cn-xGPU-624aff"></a>
        <a href="https://modelscope.ai/studios/Major-TOM/EarthEmbeddingExplorer/"><img src="https://img.shields.io/badge/Open in ModelScope.ai-xGPU-624aff"></a>
        <a href="https://modelscope.cn/datasets/VoyagerX/EarthEmbeddings"><img src="https://img.shields.io/badge/👾 MS-Dataset-624aff"></a>
        <a href="https://huggingface.co/datasets/ML4RS-Anonymous/EarthEmbeddings/tree/main"><img src="https://img.shields.io/badge/🤗 HF-Dataset-FFD21E"></a>
        <a href="https://modelscope.ai/studios/Major-TOM/EarthEmbeddingExplorer/file/view/master/doc.md?status=1"> <img src="https://img.shields.io/badge/Document-📖-007bff"> </a>
        <a href="https://modelscope.cn/studios/Major-TOM/EarthEmbeddingExplorer/file/view/master/doc_zh.md?status=1"> <img src="https://img.shields.io/badge/中文文档-📖-007bff"> </a>
        <a href="https://openreview.net/forum?id=LSsEenJVqD"> <img src="https://img.shields.io/badge/Tutorial-@ICLR26📖-007bff"> </a>
    </div>

    """)

    with gr.Row():
        with gr.Column(scale=4):
            with gr.Tabs():
                with gr.TabItem("Text Search") as tab_text:
                    model_selector_text = gr.Dropdown(choices=["SigLIP", "FarSLIP"], value="FarSLIP", label="Model")
                    query_input = gr.Textbox(label="Query", placeholder="e.g., rainforest, glacier")

                    gr.Examples(
                        examples=[
                            ["a satellite image of a river around a city"],
                            ["a satellite image of a rainforest"],
                            ["a satellite image of a slum"],
                            ["a satellite image of a glacier"],
                            ["a satellite image of snow covered mountains"],
                        ],
                        inputs=[query_input],
                        label="Text Examples",
                    )

                    search_btn = gr.Button("Search by Text", variant="primary")

                with gr.TabItem("Image Search") as tab_image:
                    model_selector_img = gr.Dropdown(
                        choices=["SigLIP", "FarSLIP", "SatCLIP", "DINOv2"], value="FarSLIP", label="Model"
                    )

                    gr.Markdown("### Option 1: Upload or Select Image")
                    image_input = gr.Image(type="pil", label="Upload Image")

                    gr.Examples(
                        examples=[
                            ["./examples/example1.png"],
                            ["./examples/example2.png"],
                            ["./examples/example3.png"],
                        ],
                        inputs=[image_input],
                        label="Image Examples",
                    )

                    gr.Markdown("### Option 2: Click Map or Enter Coordinates")
                    btn_reset_map_img = gr.Button("🔄 Reset Map to Global View", variant="secondary", size="sm")

                    with gr.Row():
                        img_lat = gr.Number(label="Latitude", interactive=True)
                        img_lon = gr.Number(label="Longitude", interactive=True)

                    img_pid = gr.Textbox(label="Product ID (auto-filled)", visible=False)
                    img_click_status = gr.Markdown("")

                    btn_download_img = gr.Button("Download Image by Geolocation", variant="secondary")

                    search_img_btn = gr.Button("Search by Image", variant="primary")

                with gr.TabItem("Location Search") as tab_location:
                    gr.Markdown("Search using **SatCLIP** location encoder.")

                    gr.Markdown("### Click Map or Enter Coordinates")
                    btn_reset_map_loc = gr.Button("🔄 Reset Map to Global View", variant="secondary", size="sm")

                    with gr.Row():
                        lat_input = gr.Number(label="Latitude", value=30.0, interactive=True)
                        lon_input = gr.Number(label="Longitude", value=120.0, interactive=True)

                    loc_pid = gr.Textbox(label="Product ID (auto-filled)", visible=False)
                    loc_click_status = gr.Markdown("")

                    gr.Examples(
                        examples=[
                            [30.32, 120.15],
                            [40.7128, -74.0060],
                            [24.65, 46.71],
                            [-3.4653, -62.2159],
                            [64.4, 16.8],
                        ],
                        inputs=[lat_input, lon_input],
                        label="Location Examples",
                    )

                    search_loc_btn = gr.Button("Search by Location", variant="primary")

                with gr.TabItem("Mixed Search") as tab_mixed:
                    gr.Markdown("""
                    ### Multi-Modal Fusion Search
                    Combine **Text**, **Image**, and **Location** queries with adjustable weights.
                    Text/Image use FarSLIP or SigLIP; Location uses SatCLIP. Scores are normalized and fused.
                    """)

                    model_selector_mixed = gr.Dropdown(
                        choices=["FarSLIP", "SigLIP"], value="FarSLIP", label="Model for Text/Image"
                    )

                    gr.Markdown("#### 📝 Text Query")
                    mixed_text_input = gr.Textbox(
                        label="Text Query (optional)", placeholder="e.g., tropical rainforest, glacier, urban area"
                    )

                    gr.Examples(
                        examples=[
                            ["a satellite image of a river around a city"],
                            ["a satellite image of a rainforest"],
                            ["a satellite image of a glacier"],
                            ["a satellite image of snow covered mountains"],
                        ],
                        inputs=[mixed_text_input],
                        label="Text Examples",
                    )

                    gr.Markdown("#### 🖼️ Image Query")
                    mixed_image_input = gr.Image(type="pil", label="Upload Image (optional)")

                    gr.Examples(
                        examples=[
                            ["./examples/example1.png"],
                            ["./examples/example2.png"],
                            ["./examples/example3.png"],
                        ],
                        inputs=[mixed_image_input],
                        label="Image Examples",
                    )

                    gr.Markdown("#### 📍 Location Query")
                    btn_reset_map_mixed = gr.Button("🔄 Reset Map to Global View", variant="secondary", size="sm")
                    with gr.Row():
                        mixed_lat = gr.Number(label="Latitude", interactive=True)
                        mixed_lon = gr.Number(label="Longitude", interactive=True)
                    mixed_pid = gr.Textbox(label="Product ID (auto-filled)", visible=False)
                    mixed_click_status = gr.Markdown("")

                    gr.Markdown("#### ⚖️ Fusion Weights")
                    gr.Markdown("_Weights are auto-normalized. Set weight to 0 to disable a modality._")
                    with gr.Row():
                        weight_text = gr.Slider(minimum=0, maximum=1, value=0.33, step=0.01, label="Text Weight")
                        weight_image = gr.Slider(minimum=0, maximum=1, value=0.33, step=0.01, label="Image Weight")
                        weight_location = gr.Slider(
                            minimum=0, maximum=1, value=0.33, step=0.01, label="Location Weight"
                        )

                    search_mixed_btn = gr.Button("🔍 Mixed Search", variant="primary")

            threshold_slider = gr.Slider(minimum=1, maximum=30, value=7, step=1, label="Top Percentage (‰)")

            # Filter controls
            enable_time_filter = gr.Checkbox(
                label="📅 Enable Time Filter", value=False, elem_classes=["filter-checkbox"]
            )
            with gr.Row():
                time_start = gr.Textbox(label="Start Date", placeholder="YYYY-MM-DD", value="2016-01-01", visible=False)
                time_end = gr.Textbox(label="End Date", placeholder="YYYY-MM-DD", value="2024-12-31", visible=False)
            enable_geo_filter = gr.Checkbox(
                label="🌍 Enable Geo Filter (Bounding Box)", value=False, elem_classes=["filter-checkbox"]
            )
            with gr.Row():
                geo_lat_min = gr.Number(label="Lat Min", value=-90, visible=False)
                geo_lat_max = gr.Number(label="Lat Max", value=90, visible=False)
            with gr.Row():
                geo_lon_min = gr.Number(label="Lon Min", value=-180, visible=False)
                geo_lon_max = gr.Number(label="Lon Max", value=180, visible=False)

            status_output = gr.Textbox(label="Status", lines=10)
            download_mode = gr.Dropdown(
                choices=["thumbnail", "rgb", "multiband"],
                value="thumbnail",
                label="Image Download Mode",
                info="thumbnail: fast preview | rgb: B04/B03/B02 composite | multiband: all 12 S2 bands",
            )
            save_btn = gr.Button("Download Result")
            download_file = gr.File(label="Zipped Results", height=40)

        with gr.Column(scale=6):
            plot_map = gr.Image(
                label="Geographical Distribution", type="pil", interactive=False, height=400, width=800, visible=True
            )
            plot_map_interactive = gr.Plot(label="Geographical Distribution (Interactive)", visible=False)
            results_plot = gr.Image(label="Top 5 Matched Images", type="pil")
            gallery_images = gr.Gallery(label="Top Retrieved Images (Zoom)", columns=3, height="auto")

    current_fig = gr.State()
    map_data_state = gr.State()
    multiband_state = gr.State(value=None)  # Stores 12-band numpy array for SatCLIP encoding
    image_source = gr.State(value="upload")  # Tracks whether image came from "upload" or "download"

    # Clear multiband state only when user uploads a new image manually,
    # NOT when the image was programmatically set by the download button.
    def _clear_multiband_on_upload(img, source):
        if source == "download":
            # Image was set by the download button — keep multiband, reset source flag
            return gr.update(), "upload"
        # User manually uploaded/changed image — discard stale multiband data
        return None, "upload"

    image_input.change(
        fn=_clear_multiband_on_upload, inputs=[image_input, image_source], outputs=[multiband_state, image_source]
    )

    # Initial Load
    demo.load(fn=get_initial_plot, outputs=[plot_map, current_fig, map_data_state, plot_map_interactive])

    # Reset Map Buttons
    btn_reset_map_img.click(fn=reset_to_global_map, outputs=[plot_map, current_fig, map_data_state])

    btn_reset_map_loc.click(fn=reset_to_global_map, outputs=[plot_map, current_fig, map_data_state])

    btn_reset_map_mixed.click(fn=reset_to_global_map, outputs=[plot_map, current_fig, map_data_state])

    # Map Click Event - updates Image Search coordinates
    plot_map.select(fn=handle_map_click, inputs=[map_data_state], outputs=[img_lat, img_lon, img_pid, img_click_status])

    # Map Click Event - also updates Location Search coordinates
    plot_map.select(
        fn=handle_map_click, inputs=[map_data_state], outputs=[lat_input, lon_input, loc_pid, loc_click_status]
    )

    # Map Click Event - also updates Mixed Search coordinates
    plot_map.select(
        fn=handle_map_click, inputs=[map_data_state], outputs=[mixed_lat, mixed_lon, mixed_pid, mixed_click_status]
    )

    # Download Image by Geolocation
    def _download_and_mark_source(lat, lon, pid, model_name):
        img, status, multiband = download_image_by_location(lat, lon, pid, model_name)
        return img, status, multiband, "download"  # Mark source so change handler won't clear multiband

    btn_download_img.click(
        fn=_download_and_mark_source,
        inputs=[img_lat, img_lon, img_pid, model_selector_img],
        outputs=[image_input, img_click_status, multiband_state, image_source],
    )

    # Filter toggle events
    def toggle_time_filter(enabled):
        return gr.update(visible=enabled), gr.update(visible=enabled)

    def toggle_geo_filter(enabled):
        return (
            gr.update(visible=enabled),
            gr.update(visible=enabled),
            gr.update(visible=enabled),
            gr.update(visible=enabled),
        )

    enable_time_filter.change(fn=toggle_time_filter, inputs=[enable_time_filter], outputs=[time_start, time_end])

    enable_geo_filter.change(
        fn=toggle_geo_filter, inputs=[enable_geo_filter], outputs=[geo_lat_min, geo_lat_max, geo_lon_min, geo_lon_max]
    )

    # Wrapper functions: pack UI filter controls into filter_options dict
    _filter_inputs = [
        enable_time_filter,
        time_start,
        time_end,
        enable_geo_filter,
        geo_lat_min,
        geo_lat_max,
        geo_lon_min,
        geo_lon_max,
    ]

    def _wrap_search_text(
        query, threshold, model_name, enable_time, start_date, end_date, enable_geo, lat_min, lat_max, lon_min, lon_max
    ):
        fo = build_filter_options(enable_time, start_date, end_date, enable_geo, lat_min, lat_max, lon_min, lon_max)
        yield from search_text(query, threshold, model_name, fo)

    def _wrap_search_image(
        image_input,
        threshold,
        model_name,
        enable_time,
        start_date,
        end_date,
        enable_geo,
        lat_min,
        lat_max,
        lon_min,
        lon_max,
        multiband_data=None,
    ):
        fo = build_filter_options(enable_time, start_date, end_date, enable_geo, lat_min, lat_max, lon_min, lon_max)
        yield from search_image(image_input, threshold, model_name, fo, multiband_data=multiband_data)

    def _wrap_search_location(
        lat, lon, threshold, enable_time, start_date, end_date, enable_geo, f_lat_min, f_lat_max, f_lon_min, f_lon_max
    ):
        fo = build_filter_options(
            enable_time, start_date, end_date, enable_geo, f_lat_min, f_lat_max, f_lon_min, f_lon_max
        )
        yield from search_location(lat, lon, threshold, fo)

    def _wrap_search_mixed(
        query_text,
        query_image,
        lat,
        lon,
        w_text,
        w_image,
        w_location,
        threshold,
        model_name,
        enable_time,
        start_date,
        end_date,
        enable_geo,
        f_lat_min,
        f_lat_max,
        f_lon_min,
        f_lon_max,
    ):
        fo = build_filter_options(
            enable_time, start_date, end_date, enable_geo, f_lat_min, f_lat_max, f_lon_min, f_lon_max
        )
        yield from search_mixed(
            query_text, query_image, lat, lon, w_text, w_image, w_location, threshold, model_name, fo
        )

    # Search Event (Text)
    search_btn.click(
        fn=_wrap_search_text,
        inputs=[
            query_input,
            threshold_slider,
            model_selector_text,
            *_filter_inputs,
        ],
        outputs=[
            plot_map_interactive,
            gallery_images,
            status_output,
            results_plot,
            current_fig,
            map_data_state,
            plot_map,
        ],
    )

    # Search Event (Image)
    search_img_btn.click(
        fn=_wrap_search_image,
        inputs=[image_input, threshold_slider, model_selector_img, *_filter_inputs, multiband_state],
        outputs=[
            plot_map_interactive,
            gallery_images,
            status_output,
            results_plot,
            current_fig,
            map_data_state,
            plot_map,
        ],
    )

    # Search Event (Location)
    search_loc_btn.click(
        fn=_wrap_search_location,
        inputs=[lat_input, lon_input, threshold_slider, *_filter_inputs],
        outputs=[
            plot_map_interactive,
            gallery_images,
            status_output,
            results_plot,
            current_fig,
            map_data_state,
            plot_map,
        ],
    )

    # Search Event (Mixed)
    search_mixed_btn.click(
        fn=_wrap_search_mixed,
        inputs=[
            mixed_text_input,
            mixed_image_input,
            mixed_lat,
            mixed_lon,
            weight_text,
            weight_image,
            weight_location,
            threshold_slider,
            model_selector_mixed,
            *_filter_inputs,
        ],
        outputs=[
            plot_map_interactive,
            gallery_images,
            status_output,
            results_plot,
            current_fig,
            map_data_state,
            plot_map,
        ],
    )

    # Save Event — download_mode controls the image format in the exported zip
    save_btn.click(fn=save_plot, inputs=[current_fig, download_mode], outputs=[download_file])

    # Tab Selection Events
    def show_static_map():
        return gr.update(visible=True), gr.update(visible=False)

    tab_text.select(fn=show_static_map, outputs=[plot_map, plot_map_interactive])
    tab_image.select(fn=show_static_map, outputs=[plot_map, plot_map_interactive])
    tab_location.select(fn=show_static_map, outputs=[plot_map, plot_map_interactive])
    tab_mixed.select(fn=show_static_map, outputs=[plot_map, plot_map_interactive])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
