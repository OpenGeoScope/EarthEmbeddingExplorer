"""Search engine for text, image, location, and mixed modalities."""

import time

import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image as PILImage

from data_utils import download_and_process_image, get_placeholder_image
from visualize import format_results_for_gallery, plot_geographic_distribution, plot_top5_overview

from .filters import apply_filters


def _get_model_and_error(model_manager, model_name):
    """Helper to get model from ModelManager."""
    return model_manager.get_model(model_name)


def search_text(model_manager, query, threshold, model_name, filter_options=None):
    """Search satellite imagery using text query."""
    model, error = _get_model_and_error(model_manager, model_name)
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
        results = _fetch_top_k_images(top_indices, probs, df_embed, query_text=query)
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
        status_msg = timing_str + _generate_status_msg(len(filtered_indices), threshold / 100.0, results)

        all_results = _get_all_results_metadata(model, filtered_indices, probs)
        results_txt = _format_results_to_text(all_results)

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


def search_image(model_manager, image_input, threshold, model_name, filter_options=None, multiband_data=None):
    """Search satellite imagery using image query."""
    model, error = _get_model_and_error(model_manager, model_name)
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
        results = _fetch_top_k_images(top_indices, probs, df_embed, query_text="Image Query")
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
        status_msg = timing_str + _generate_status_msg(len(filtered_indices), threshold / 100.0, results)

        all_results = _get_all_results_metadata(model, filtered_indices, probs)
        results_txt = _format_results_to_text(all_results[:50])

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


def search_location(model_manager, lat, lon, threshold, filter_options=None):
    """Search satellite imagery using geographic location."""
    model_name = "SatCLIP"
    model, error = _get_model_and_error(model_manager, model_name)
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
        results = _fetch_top_k_images(top_6_indices, probs, df_embed, query_text=f"Loc: {lat},{lon}")

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
        status_msg = timing_str + _generate_status_msg(len(filtered_indices), threshold / 100.0, results)

        all_results = _get_all_results_metadata(model, filtered_indices, probs)
        results_txt = _format_results_to_text(all_results)

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


def _normalize_scores(scores):
    """Min-max normalize scores to [0, 1] range."""
    s_min, s_max = scores.min(), scores.max()
    if s_max - s_min < 1e-9:
        return np.zeros_like(scores)
    return (scores - s_min) / (s_max - s_min)


def search_mixed(
    model_manager,
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
    """Mixed search combining text, image, and location modalities.

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
        text_image_model, error = _get_model_and_error(model_manager, model_name)
        if error and (use_text or use_image):
            yield None, None, error, None, None, None, None
            return

        satclip_model, error = _get_model_and_error(model_manager, "SatCLIP")
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
            text_scores = _normalize_scores(text_scores)
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
            image_scores = _normalize_scores(image_scores)
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
            loc_scores = _normalize_scores(loc_scores)
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
        results = _fetch_top_k_images(top_indices, final_scores, df_ref, query_text=query_info)
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
            query_vis = _create_text_image(query_text)

        fig_results = plot_top5_overview(query_vis, results, query_info=f"Mixed: {query_info}")
        gallery_items = format_results_for_gallery(results)
        timings["Visualization"] = time.time() - t0

        # --- Generate final status ---
        timing_parts = [f"{k} {v:.1f}s" for k, v in timings.items()]
        timing_str = ", ".join(timing_parts) + "\n\n"

        weight_info = f"Weights: Text={w_text:.2f}, Image={w_image:.2f}, Location={w_location:.2f}\n"
        status_msg = timing_str + weight_info + _generate_status_msg(len(filtered_indices), threshold / 100.0, results)

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
        results_txt = _format_results_to_text(all_results[:50])

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


# Helper functions (moved from app.py)


def _fetch_top_k_images(top_indices, probs, df_embed, query_text=None):
    """Download and process top-K images for display."""
    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_idx = {}
        for _i, idx in enumerate(top_indices):
            row = df_embed.iloc[idx]
            pid = row["product_id"]

            future = executor.submit(download_and_process_image, pid, df_source=df_embed, verbose=False, mode="thumbnail")
            future_to_idx[future] = idx

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                img_384, img_full = future.result()
                if img_384 is None:
                    continue

                row = df_embed.iloc[idx]
                results.append(
                    {
                        "id": row["product_id"],
                        "lat": row["centre_lat"],
                        "lon": row["centre_lon"],
                        "score": probs[idx],
                        "image_384": img_384,
                        "image_full": img_full,
                    }
                )
            except Exception as e:
                print(f"Error processing image at index {idx}: {e}")

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def _get_all_results_metadata(model, filtered_indices, probs):
    """Get metadata for all filtered results."""
    all_results = []
    for idx in filtered_indices:
        row = model.df_embed.iloc[idx]
        all_results.append(
            {
                "id": row["product_id"],
                "lat": row["centre_lat"],
                "lon": row["centre_lon"],
                "score": probs[idx],
            }
        )
    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results


def _generate_status_msg(count, threshold, results):
    """Generate status message with top results summary."""
    status_msg = f"Found {count} matches in top {threshold * 100:.0f}‰.\n\nTop {len(results)} similar images:\n"
    for i, res in enumerate(results[:3]):
        status_msg += f"{i + 1}. Product ID: {res['id']}, Location: ({res['lat']:.4f}, {res['lon']:.4f}), Score: {res['score']:.4f}\n"
    return status_msg


def _create_text_image(text="Image Unavailable", size=(384, 384)):
    """Create a text placeholder image."""
    from PIL import ImageDraw, ImageFont

    img = PILImage.new("RGB", size, color=(200, 200, 200))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 40)
    except Exception:
        font = ImageFont.load_default()

    margin = 20
    offset = margin
    for line in text.split("\n"):
        d.text((margin, offset), line.strip(), font=font, fill=(0, 0, 0))
        offset += 50

    d.text((margin, offset + 50), "Text Query", font=font, fill=(0, 0, 255))
    return img


def _format_results_to_text(results):
    """Format search results to text report."""
    if not results:
        return "No results found."

    lines = ["Search Results Report", "=" * 50, ""]
    for i, res in enumerate(results, 1):
        lines.append(f"Rank #{i}")
        lines.append(f"  Product ID: {res['id']}")
        lines.append(f"  Location: ({res['lat']:.4f}, {res['lon']:.4f})")
        lines.append(f"  Similarity Score: {res['score']:.4f}")
        lines.append("")

    return "\n".join(lines)


# Need to import these at the top
from concurrent.futures import ThreadPoolExecutor, as_completed
