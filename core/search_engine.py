"""Search engine for text, image, location, and mixed modalities."""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image as PILImage

from data_utils import download_and_process_image, get_placeholder_image, reorder_multiband
from visualize import format_results_for_gallery, plot_geographic_distribution, plot_top5_overview

from .filters import apply_filters

DISPLAY_TOP_K = 5


def _get_model_and_error(model_manager, model_name):
    """Helper to get model from ModelManager."""
    return model_manager.get_model(model_name)


def _model_supports_native_joint_encoding(model):
    """Check whether a model supports native text+image joint encoding."""
    return hasattr(model, "encode_text_and_image") and callable(getattr(model, "encode_text_and_image"))


def _validate_lat_lon(lat, lon):
    """Validate latitude/longitude inputs.

    Returns:
        str | None: A user-facing error message if the input is missing, not
            numeric, or out of range; None if the location is valid.
    """
    if lat is None or lon is None:
        return "Please provide both latitude and longitude."
    try:
        lat_f, lon_f = float(lat), float(lon)
    except (TypeError, ValueError):
        return f"Invalid location input (lat={lat!r}, lon={lon!r}). Latitude and longitude must be numbers."
    if not (-90 <= lat_f <= 90):
        return f"Invalid latitude {lat_f}: must be within [-90, 90]."
    if not (-180 <= lon_f <= 180):
        return f"Invalid longitude {lon_f}: must be within [-180, 180]."
    return None


def _align_on_grid_cell(df_a, df_b):
    """Align two embedding DataFrames on their unique ``grid_cell`` column.

    ``product_id`` is not unique within an embedding table (one satellite
    product can cover multiple grid cells), whereas ``grid_cell`` is unique
    per row and consistent across model embedding sets, so it is the correct
    key for cross-model alignment. The intersection is sorted to keep the
    alignment deterministic across processes.

    Returns:
        tuple: (indices_a, indices_b) as np.ndarray, row-aligned on the sorted
            intersection of grid cells.

    Raises:
        ValueError: If either DataFrame has no ``grid_cell`` column or the
            intersection is empty.
    """
    if "grid_cell" not in df_a.columns or "grid_cell" not in df_b.columns:
        raise ValueError("Embedding metadata is missing the 'grid_cell' column. Cannot perform mixed search.")
    cells_a = df_a["grid_cell"].values
    cells_b = df_b["grid_cell"].values
    common = sorted(set(cells_a) & set(cells_b))
    if len(common) == 0:
        raise ValueError("No common grid cells between models. Cannot perform mixed search.")
    a_cell_to_idx = {cell: idx for idx, cell in enumerate(cells_a)}
    b_cell_to_idx = {cell: idx for idx, cell in enumerate(cells_b)}
    indices_a = np.array([a_cell_to_idx[cell] for cell in common])
    indices_b = np.array([b_cell_to_idx[cell] for cell in common])
    return indices_a, indices_b


def _append_warnings(status_msg, warnings):
    """Append human-readable warnings to a status message (one per line)."""
    if not warnings:
        return status_msg
    return status_msg + "\n" + "\n".join(f"⚠️ {w}" for w in warnings)


def _safe_plot_geographic_distribution(df_for_geo, probs_for_geo, title):
    """Plot the geographic distribution, degrading gracefully on failure.

    Returns:
        tuple: (geo_dist_map, df_filtered, warning). If rendering fails, the
            map is None, ``df_filtered`` falls back to the input DataFrame so
            search results are still returned, and ``warning`` describes the
            issue for the status bar.
    """
    try:
        geo_dist_map, df_filtered = plot_geographic_distribution(df_for_geo, probs_for_geo, title=title)
        return geo_dist_map, df_filtered, None
    except Exception as e:
        print(f"⚠️ Geographic distribution map failed: {e}")
        return None, df_for_geo, "Map visualization unavailable."


def search_text(model_manager, query, threshold, model_name, filter_options=None):
    """Search satellite imagery using text query."""
    model, error = _get_model_and_error(model_manager, model_name)
    if error:
        yield gr.update(), error, gr.update(), gr.update(), gr.update(), gr.update()
        return

    if not query:
        yield gr.update(), "Please enter a query.", gr.update(), gr.update(), gr.update(), gr.update()
        return

    try:
        timings = {}

        # 1. Encode Text
        yield gr.update(), "Encoding text...", gr.update(), gr.update(), gr.update(), gr.update()
        t0 = time.time()
        text_features = model.encode_text(query)
        timings["Encoding"] = time.time() - t0

        if text_features is None:
            yield gr.update(), "Model does not support text encoding or is not initialized.", gr.update(), gr.update(), gr.update(), gr.update()
            return

        # 2. Search
        yield gr.update(), "Encoding text... ✓\nRetrieving similar images...", gr.update(), gr.update(), gr.update(), gr.update()
        t0 = time.time()
        probs, filtered_indices, top_indices = model.search(text_features, top_percent=threshold / 1000.0)
        timings["Retrieval"] = time.time() - t0

        if probs is None:
            yield gr.update(), "Search failed (embeddings missing?).", gr.update(), gr.update(), gr.update(), gr.update()
            return

        # Apply post-search filters (time range, geo, etc.)
        df_embed = model.df_embed
        filtered_indices, top_indices, df_for_geo, probs_for_geo, extra_warnings = apply_filters(
            df_embed, probs, filtered_indices, top_indices, filter_options
        )

        # Generate geographic distribution (not timed); yield a status update
        # first so the UI shows this stage while the map renders.
        yield (
            gr.update(),
            "Encoding text... ✓\nRetrieving similar images... ✓\nGenerating distribution map...",
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )
        geo_dist_map, df_filtered, map_warning = _safe_plot_geographic_distribution(
            df_for_geo, probs_for_geo, title=f'Similarity to "{query}" ({model_name})'
        )
        if map_warning:
            extra_warnings.append(map_warning)

        # Handle 0 results after filtering
        if len(top_indices) == 0:
            status_msg = _append_warnings(
                "No results found with current filter settings.\nTry relaxing the filters or adjusting the threshold.",
                extra_warnings,
            )
            yield (
                [],
                status_msg,
                None,
                [geo_dist_map],
                df_filtered,
                gr.update(value=geo_dist_map, visible=geo_dist_map is not None),
            )
            return

        # 3. Download Images (display always uses thumbnail for gallery)
        yield (
            None,
            "Encoding text... ✓\nRetrieving similar images... ✓\nGenerating distribution map... ✓\nDownloading images...",
            None,
            None,
            df_filtered,
            gr.update(value=geo_dist_map, visible=geo_dist_map is not None),
        )
        t0 = time.time()
        display_indices = top_indices[:DISPLAY_TOP_K]
        results = _fetch_top_k_images(display_indices, probs, df_embed, query_text=query)
        timings["Download"] = time.time() - t0

        # 4. Visualize - keep geo_dist_map visible
        yield (
            None,
            "Encoding text... ✓\nRetrieving similar images... ✓\nGenerating distribution map... ✓\nDownloading images... ✓\nGenerating visualizations...",
            None,
            None,
            df_filtered,
            gr.update(value=geo_dist_map, visible=geo_dist_map is not None),
        )
        t0 = time.time()
        fig_results = plot_top5_overview(None, results, query_info=query)
        gallery_items = format_results_for_gallery(results)
        timings["Visualization"] = time.time() - t0

        # 5. Generate Final Status
        timing_str = f"Encoding {timings['Encoding']:.1f}s, Retrieval {timings['Retrieval']:.1f}s, Download {timings['Download']:.1f}s, Visualization {timings['Visualization']:.1f}s\n\n"
        status_msg = _append_warnings(
            timing_str + _generate_status_msg(len(filtered_indices), threshold / 100.0, results), extra_warnings
        )

        all_results = _get_all_results_metadata(model, filtered_indices, probs)
        results_txt = _format_results_to_text(all_results)

        # current_fig: [map, results_img, text, results_meta_for_download]
        top_results_meta = [{"id": r["id"], "lat": r["lat"], "lon": r["lon"], "score": r["score"]} for r in results]
        yield (
            gallery_items,
            status_msg,
            fig_results,
            [geo_dist_map, fig_results, results_txt, top_results_meta, model_name],
            df_filtered,
            gr.update(value=geo_dist_map, visible=geo_dist_map is not None),
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        yield gr.update(), f"Error: {e!s}", gr.update(), gr.update(), gr.update(), gr.update()


def search_image(model_manager, image_input, threshold, model_name, filter_options=None, multiband_data=None):
    """Search satellite imagery using image query."""
    model, error = _get_model_and_error(model_manager, model_name)
    if error:
        yield gr.update(), error, gr.update(), gr.update(), gr.update(), gr.update()
        return

    if image_input is None:
        yield gr.update(), "Please upload an image.", gr.update(), gr.update(), gr.update(), gr.update()
        return

    try:
        timings = {}

        # 1. Encode Image
        # For multi-spectral models: require multiband data
        yield gr.update(), "Encoding image...", gr.update(), gr.update(), gr.update(), gr.update()
        t0 = time.time()
        # Determine if the model needs multi-spectral input
        needs_multiband = getattr(model, "requires_multiband", False)

        if needs_multiband:
            if multiband_data is not None:
                print(f"{model_name}: encoding with multiband data {multiband_data.shape}")
                # Reorder bands from the generic 12-band MajorTOM format to what
                # this model expects (no-op if model.bands == MULTIBAND_COLUMNS).
                multiband_data = reorder_multiband(multiband_data, model.bands)
                print(f"{model_name}: reordered to bands {model.bands} -> shape {multiband_data.shape}")
                image_features = model.encode_image(multiband_data)
            else:
                yield (
                    gr.update(),
                    (
                        f"⚠️ {model_name} requires multi-spectral Sentinel-2 input.\n\n"
                        "RGB images are NOT compatible with this model's image retrieval.\n"
                        "Please use 'Download Image by Geolocation' to obtain a multi-band image first,\n"
                        "or switch to an RGB-capable model for image retrieval."
                    ),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                )
                return
        else:
            image_features = model.encode_image(image_input)
        timings["Encoding"] = time.time() - t0

        if image_features is None:
            yield gr.update(), "Model does not support image encoding.", gr.update(), gr.update(), gr.update(), gr.update()
            return

        # 2. Search
        yield gr.update(), "Encoding image... ✓\nRetrieving similar images...", gr.update(), gr.update(), gr.update(), gr.update()
        t0 = time.time()
        probs, filtered_indices, top_indices = model.search(image_features, top_percent=threshold / 1000.0)
        timings["Retrieval"] = time.time() - t0

        if probs is None:
            yield gr.update(), "Search failed (embeddings missing?).", gr.update(), gr.update(), gr.update(), gr.update()
            return

        # Apply post-search filters (time range, geo, etc.)
        df_embed = model.df_embed
        filtered_indices, top_indices, df_for_geo, probs_for_geo, extra_warnings = apply_filters(
            df_embed, probs, filtered_indices, top_indices, filter_options
        )

        # Generate geographic distribution (not timed); yield a status update
        # first so the UI shows this stage while the map renders.
        yield (
            gr.update(),
            "Encoding image... ✓\nRetrieving similar images... ✓\nGenerating distribution map...",
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )
        geo_dist_map, df_filtered, map_warning = _safe_plot_geographic_distribution(
            df_for_geo, probs_for_geo, title=f"Similarity to Input Image ({model_name})"
        )
        if map_warning:
            extra_warnings.append(map_warning)

        # Handle 0 results after filtering
        if len(top_indices) == 0:
            status_msg = _append_warnings(
                "No results found with current filter settings.\nTry relaxing the filters or adjusting the threshold.",
                extra_warnings,
            )
            yield (
                [],
                status_msg,
                None,
                [geo_dist_map],
                df_filtered,
                gr.update(value=geo_dist_map, visible=geo_dist_map is not None),
            )
            return

        # 3. Download Images (display always uses thumbnail for gallery)
        yield (
            None,
            "Encoding image... ✓\nRetrieving similar images... ✓\nGenerating distribution map... ✓\nDownloading images...",
            None,
            None,
            df_filtered,
            gr.update(value=geo_dist_map, visible=geo_dist_map is not None),
        )
        t0 = time.time()
        display_indices = top_indices[:DISPLAY_TOP_K]
        results = _fetch_top_k_images(display_indices, probs, df_embed, query_text="Image Query")
        timings["Download"] = time.time() - t0

        # 4. Visualize - keep geo_dist_map visible
        yield (
            None,
            "Encoding image... ✓\nRetrieving similar images... ✓\nGenerating distribution map... ✓\nDownloading images... ✓\nGenerating visualizations...",
            None,
            None,
            df_filtered,
            gr.update(value=geo_dist_map, visible=geo_dist_map is not None),
        )
        t0 = time.time()
        fig_results = plot_top5_overview(image_input, results, query_info="Image Query")
        gallery_items = format_results_for_gallery(results)
        timings["Visualization"] = time.time() - t0

        # 5. Generate Final Status
        timing_str = f"Encoding {timings['Encoding']:.1f}s, Retrieval {timings['Retrieval']:.1f}s, Download {timings['Download']:.1f}s, Visualization {timings['Visualization']:.1f}s\n\n"
        status_msg = _append_warnings(
            timing_str + _generate_status_msg(len(filtered_indices), threshold / 100.0, results), extra_warnings
        )

        all_results = _get_all_results_metadata(model, filtered_indices, probs)
        results_txt = _format_results_to_text(all_results)

        # current_fig: [map, results_img, text, results_meta_for_download]
        top_results_meta = [{"id": r["id"], "lat": r["lat"], "lon": r["lon"], "score": r["score"]} for r in results]
        yield (
            gallery_items,
            status_msg,
            fig_results,
            [geo_dist_map, fig_results, results_txt, top_results_meta, model_name],
            df_filtered,
            gr.update(value=geo_dist_map, visible=geo_dist_map is not None),
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        yield gr.update(), f"Error: {e!s}", gr.update(), gr.update(), gr.update(), gr.update()


def search_location(model_manager, lat, lon, threshold, filter_options=None):
    """Search satellite imagery using geographic location."""
    model_name = "SatCLIP"
    model, error = _get_model_and_error(model_manager, model_name)
    if error:
        yield gr.update(), error, gr.update(), gr.update(), gr.update(), gr.update()
        return

    try:
        timings = {}

        # Validate location input before encoding (avoids raw float(None) TypeError)
        loc_error = _validate_lat_lon(lat, lon)
        if loc_error:
            yield gr.update(), loc_error, gr.update(), gr.update(), gr.update(), gr.update()
            return

        # 1. Encode Location
        yield gr.update(), "Encoding location...", gr.update(), gr.update(), gr.update(), gr.update()
        t0 = time.time()
        loc_features = model.encode_location(float(lat), float(lon))
        timings["Encoding"] = time.time() - t0

        if loc_features is None:
            yield gr.update(), "Location encoding failed.", gr.update(), gr.update(), gr.update(), gr.update()
            return

        # 2. Search
        yield gr.update(), "Encoding location... ✓\nRetrieving similar images...", gr.update(), gr.update(), gr.update(), gr.update()
        t0 = time.time()
        probs, filtered_indices, top_indices = model.search(loc_features, top_percent=threshold / 1000.0)
        timings["Retrieval"] = time.time() - t0

        if probs is None:
            yield gr.update(), "Search failed (embeddings missing?).", gr.update(), gr.update(), gr.update(), gr.update()
            return

        # Apply post-search filters (time range, geo, etc.)
        df_embed = model.df_embed
        filtered_indices, top_indices, df_for_geo, probs_for_geo, extra_warnings = apply_filters(
            df_embed, probs, filtered_indices, top_indices, filter_options
        )

        # 3. Generate Distribution Map (not timed for location distribution)
        yield (
            gr.update(),
            "Encoding location... ✓\nRetrieving similar images... ✓\nGenerating distribution map...",
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )

        # Show geographic distribution (not timed)
        geo_dist_map, df_filtered, map_warning = _safe_plot_geographic_distribution(
            df_for_geo, probs_for_geo, title=f"Similarity to Location ({lat}, {lon})"
        )
        if map_warning:
            extra_warnings.append(map_warning)

        # Handle 0 results after filtering
        if len(top_indices) == 0:
            status_msg = _append_warnings(
                "No results found with current filter settings.\nTry relaxing the filters or adjusting the threshold.",
                extra_warnings,
            )
            yield (
                [],
                status_msg,
                None,
                [geo_dist_map],
                df_filtered,
                gr.update(value=geo_dist_map, visible=geo_dist_map is not None),
            )
            return

        # 4. Download Images
        yield (
            None,
            "Encoding location... ✓\nRetrieving similar images... ✓\nGenerating distribution map... ✓\nDownloading images...",
            None,
            None,
            df_filtered,
            gr.update(value=geo_dist_map, visible=geo_dist_map is not None),
        )
        t0 = time.time()
        display_indices = top_indices[:DISPLAY_TOP_K]
        results = _fetch_top_k_images(display_indices, probs, df_embed, query_text=f"Loc: {lat},{lon}")

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
            None,
            "Encoding location... ✓\nRetrieving similar images... ✓\nGenerating distribution map... ✓\nDownloading images... ✓\nGenerating visualizations...",
            None,
            None,
            df_filtered,
            gr.update(value=geo_dist_map, visible=geo_dist_map is not None),
        )
        t0 = time.time()
        fig_results = plot_top5_overview(query_tile, results, query_info=f"Loc: {lat},{lon}")
        gallery_items = format_results_for_gallery(results)
        timings["Visualization"] = time.time() - t0

        # 6. Generate Final Status
        timing_str = f"Encoding {timings['Encoding']:.1f}s, Retrieval {timings['Retrieval']:.1f}s, Download {timings['Download']:.1f}s, Visualization {timings['Visualization']:.1f}s\n\n"
        status_msg = _append_warnings(
            timing_str + _generate_status_msg(len(filtered_indices), threshold / 100.0, results), extra_warnings
        )

        all_results = _get_all_results_metadata(model, filtered_indices, probs)
        results_txt = _format_results_to_text(all_results)

        # current_fig: [map, results_img, text, results_meta_for_download]
        top_results_meta = [{"id": r["id"], "lat": r["lat"], "lon": r["lon"], "score": r["score"]} for r in results]
        yield (
            gallery_items,
            status_msg,
            fig_results,
            [geo_dist_map, fig_results, results_txt, top_results_meta, model_name],
            df_filtered,
            gr.update(value=geo_dist_map, visible=geo_dist_map is not None),
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        yield gr.update(), f"Error: {e!s}", gr.update(), gr.update(), gr.update(), gr.update()


def _normalize_scores(scores, modality=None):
    """Min-max normalize scores to [0, 1] range.

    Returns:
        tuple: (normalized_scores, warning). ``warning`` is a human-readable
            string when all scores are (near-)constant — in that degenerate
            case the modality contributes no discrimination and zeros are
            returned — otherwise None.
    """
    s_min, s_max = scores.min(), scores.max()
    if s_max - s_min < 1e-9:
        label = modality or "Modality"
        warning = f"Warning: {label} scores are constant, contributing no discrimination."
        return np.zeros_like(scores), warning
    return (scores - s_min) / (s_max - s_min), None


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
    use_native_joint=True,
):
    """Mixed search combining text, image, and location modalities.

    Uses score-level fusion: final_score = w_t * s_text + w_i * s_image + w_l * s_location
    Text/Image use the selected model (FarSLIP/SigLIP), Location uses SatCLIP.
    """
    try:
        timings = {}

        # Determine which modalities are active (weight > 0 and input provided)
        use_text = bool(weight_text > 0 and query_text and query_text.strip())
        use_image = weight_image > 0 and query_image is not None
        use_location = weight_location > 0 and lat is not None and lon is not None

        if not use_text and not use_image and not use_location:
            yield (
                gr.update(),
                "Please provide at least one query (text, image, or location) with weight > 0.",
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
            )
            return

        # Get models
        text_image_model, error = _get_model_and_error(model_manager, model_name)
        if error and (use_text or use_image):
            yield gr.update(), error, gr.update(), gr.update(), gr.update(), gr.update()
            return

        satclip_model, error = _get_model_and_error(model_manager, "SatCLIP")
        if error and use_location:
            yield gr.update(), f"SatCLIP required for location search: {error}", gr.update(), gr.update(), gr.update(), gr.update()
            return

        # Determine the reference df_embed (use the one with most samples or text_image_model's)
        if use_text or use_image:
            if text_image_model.df_embed is None:
                yield gr.update(), f"Model {model_name} embeddings not loaded (metadata missing). Cannot perform mixed search.", gr.update(), gr.update(), gr.update(), gr.update()
                return
            df_ref = text_image_model.df_embed
            ref_model_name = model_name
        else:
            if satclip_model.df_embed is None:
                yield gr.update(), "SatCLIP embeddings not loaded (metadata missing). Cannot perform mixed search.", gr.update(), gr.update(), gr.update(), gr.update()
                return
            df_ref = satclip_model.df_embed
            ref_model_name = "SatCLIP"

        # If using location AND (text or image), we need to align on grid cells
        need_alignment = use_location and (use_text or use_image)

        if need_alignment:
            df_ti = text_image_model.df_embed
            df_loc = satclip_model.df_embed
            if df_ti is None or df_loc is None:
                missing = model_name if df_ti is None else "SatCLIP"
                yield gr.update(), f"Model {missing} embeddings not loaded (metadata missing). Cannot perform mixed search.", gr.update(), gr.update(), gr.update(), gr.update()
                return

            # Align on grid_cell: unique per row within each embedding table and
            # consistent across model sets (product_id has ~13% duplicates).
            try:
                ti_indices, loc_indices = _align_on_grid_cell(df_ti, df_loc)
            except ValueError as e:
                yield gr.update(), str(e), gr.update(), gr.update(), gr.update(), gr.update()
                return

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
        score_warnings = []

        # --- Native text+image joint encoding (Qwen3VL style) ---
        use_native_joint = (
            use_native_joint
            and use_text
            and use_image
            and not use_location
            and _model_supports_native_joint_encoding(text_image_model)
        )

        if use_native_joint:
            yield gr.update(), "Encoding text and image jointly...", gr.update(), gr.update(), gr.update(), gr.update()
            t0 = time.time()
            joint_features = text_image_model.encode_text_and_image(query_text, query_image)
            timings["Joint Encoding"] = time.time() - t0

            if joint_features is None:
                yield gr.update(), f"Model {model_name} does not support joint text+image encoding.", gr.update(), gr.update(), gr.update(), gr.update()
                return

            if text_image_model.image_embeddings is None:
                yield gr.update(), f"Model {model_name} image embeddings not loaded. Cannot perform mixed search.", gr.update(), gr.update(), gr.update(), gr.update()
                return
            embeddings = text_image_model.image_embeddings.cpu().numpy()
            joint_scores = (embeddings @ joint_features.cpu().numpy().T).ravel()
            final_scores = joint_scores
            status_parts.append("Text+Image (joint)")
        else:
            # --- Encode and compute text scores ---
            if use_text:
                yield gr.update(), "Encoding text...", gr.update(), gr.update(), gr.update(), gr.update()
                t0 = time.time()
                text_features = text_image_model.encode_text(query_text)
                timings["Text Encoding"] = time.time() - t0

                if text_features is None:
                    yield gr.update(), f"Model {model_name} does not support text encoding.", gr.update(), gr.update(), gr.update(), gr.update()
                    return

                # Compute similarity
                if text_image_model.image_embeddings is None:
                    yield gr.update(), f"Model {model_name} image embeddings not loaded. Cannot perform mixed search.", gr.update(), gr.update(), gr.update(), gr.update()
                    return
                embeddings = text_image_model.image_embeddings.cpu().numpy()
                if need_alignment:
                    embeddings = embeddings[ti_indices]
                text_scores = (embeddings @ text_features.cpu().numpy().T).ravel()
                text_scores, warn = _normalize_scores(text_scores, modality="Text")
                if warn:
                    score_warnings.append(warn)
                final_scores += w_text * text_scores
                status_parts.append(f"Text (w={w_text:.2f})")

            # --- Encode and compute image scores ---
            if use_image:
                status_msg = "Encoding text... ✓\n" if use_text else ""
                yield gr.update(), status_msg + "Encoding image...", gr.update(), gr.update(), gr.update(), gr.update()
                t0 = time.time()
                image_features = text_image_model.encode_image(query_image)
                timings["Image Encoding"] = time.time() - t0

                if image_features is None:
                    yield gr.update(), f"Model {model_name} does not support image encoding.", gr.update(), gr.update(), gr.update(), gr.update()
                    return

                # Compute similarity
                if text_image_model.image_embeddings is None:
                    yield gr.update(), f"Model {model_name} image embeddings not loaded. Cannot perform mixed search.", gr.update(), gr.update(), gr.update(), gr.update()
                    return
                embeddings = text_image_model.image_embeddings.cpu().numpy()
                if need_alignment:
                    embeddings = embeddings[ti_indices]
                image_scores = (embeddings @ image_features.cpu().numpy().T).ravel()
                image_scores, warn = _normalize_scores(image_scores, modality="Image")
                if warn:
                    score_warnings.append(warn)
                final_scores += w_image * image_scores
                status_parts.append(f"Image (w={w_image:.2f})")

        # --- Encode and compute location scores ---
        if use_location:
            loc_error = _validate_lat_lon(lat, lon)
            if loc_error:
                yield gr.update(), loc_error, gr.update(), gr.update(), gr.update(), gr.update()
                return

            status_msg = ""
            if use_text:
                status_msg += "Encoding text... ✓\n"
            if use_image:
                status_msg += "Encoding image... ✓\n"
            yield gr.update(), status_msg + "Encoding location...", gr.update(), gr.update(), gr.update(), gr.update()
            t0 = time.time()
            loc_features = satclip_model.encode_location(float(lat), float(lon))
            timings["Location Encoding"] = time.time() - t0

            if loc_features is None:
                yield gr.update(), "Location encoding failed.", gr.update(), gr.update(), gr.update(), gr.update()
                return

            # Compute similarity
            if satclip_model.image_embeddings is None:
                yield gr.update(), "SatCLIP image embeddings not loaded. Cannot perform mixed search.", gr.update(), gr.update(), gr.update(), gr.update()
                return
            embeddings = satclip_model.image_embeddings.cpu().numpy()
            if need_alignment:
                embeddings = embeddings[loc_indices]
            loc_scores = (embeddings @ loc_features.cpu().numpy().T).ravel()
            loc_scores, warn = _normalize_scores(loc_scores, modality="Location")
            if warn:
                score_warnings.append(warn)
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
        yield gr.update(), status_msg + "Retrieving similar images...", gr.update(), gr.update(), gr.update(), gr.update()

        t0 = time.time()
        # Apply top-percentage threshold for the map candidate pool.
        top_percent = threshold / 1000.0
        sorted_order = np.argsort(final_scores)[::-1]
        k = max(1, int(len(final_scores) * top_percent))
        filtered_indices = sorted_order[:k]
        top_indices = filtered_indices
        timings["Retrieval"] = time.time() - t0

        # Apply post-search filters
        filtered_indices, top_indices, df_for_geo, probs_for_geo, extra_warnings = apply_filters(
            df_ref, final_scores, filtered_indices, top_indices, filter_options
        )
        extra_warnings = score_warnings + extra_warnings

        # Generate geographic distribution map; yield a status update first so
        # the UI shows this stage while the map renders.
        query_info = " + ".join(status_parts)
        yield (
            gr.update(),
            status_msg + "Retrieving similar images... ✓\nGenerating distribution map...",
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )
        geo_dist_map, df_filtered, map_warning = _safe_plot_geographic_distribution(
            df_for_geo, probs_for_geo, title=f"Mixed Search: {query_info}"
        )
        if map_warning:
            extra_warnings.append(map_warning)

        # Handle 0 results after filtering
        if len(top_indices) == 0:
            status_msg = _append_warnings(
                "No results found with current filter settings.\nTry relaxing the filters or adjusting the threshold.",
                extra_warnings,
            )
            yield (
                [],
                status_msg,
                None,
                [geo_dist_map],
                df_filtered,
                gr.update(value=geo_dist_map, visible=geo_dist_map is not None),
            )
            return

        # --- Download images ---
        yield (
            None,
            status_msg + "Retrieving similar images... ✓\nGenerating distribution map... ✓\nDownloading images...",
            None,
            None,
            df_filtered,
            gr.update(value=geo_dist_map, visible=geo_dist_map is not None),
        )
        t0 = time.time()
        display_indices = top_indices[:DISPLAY_TOP_K]
        results = _fetch_top_k_images(display_indices, final_scores, df_ref, query_text=query_info)
        timings["Download"] = time.time() - t0

        # --- Visualize ---
        yield (
            None,
            status_msg
            + "Retrieving similar images... ✓\nGenerating distribution map... ✓\nDownloading images... ✓\nGenerating visualizations...",
            None,
            None,
            df_filtered,
            gr.update(value=geo_dist_map, visible=geo_dist_map is not None),
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

        if use_native_joint:
            # Native joint encoding ignores w_text/w_image; scores are raw cosine similarity.
            score_info = "Native joint encoding (weights not applied); scores are raw cosine similarity.\n"
        else:
            score_info = (
                f"Weights: Text={w_text:.2f}, Image={w_image:.2f}, Location={w_location:.2f}\n"
                "Scores are min-max normalized fused scores.\n"
            )
        status_msg = _append_warnings(
            timing_str + score_info + _generate_status_msg(len(filtered_indices), threshold / 100.0, results),
            extra_warnings,
        )

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
        results_txt = _format_results_to_text(all_results)

        top_results_meta = [{"id": r["id"], "lat": r["lat"], "lon": r["lon"], "score": r["score"]} for r in results]
        yield (
            gallery_items,
            status_msg,
            fig_results,
            [geo_dist_map, fig_results, results_txt, top_results_meta, ref_model_name],
            df_filtered,
            gr.update(value=geo_dist_map, visible=geo_dist_map is not None),
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        yield gr.update(), f"Error: {e!s}", gr.update(), gr.update(), gr.update(), gr.update()


# Helper functions (moved from app.py)


def _fetch_top_k_images(top_indices, probs, df_embed, query_text=None):
    """Download and process top-K images for display."""
    results = []
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
