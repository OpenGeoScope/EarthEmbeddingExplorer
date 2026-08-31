import os
import re
import tempfile
import time
import uuid
import zipfile
from datetime import datetime, timezone
from io import BytesIO

import gradio as gr
import numpy as np
from PIL import Image as PILImage

from data_utils import download_and_process_image

# Prefixes of temporary files created by this app (see save_plot).
_TEMP_FILE_PREFIXES = (
    "earth_embedding_explorer_",
    "earth_explorer_map_",
    "earth_explorer_plot_",
    "map_distribution_",
    "retrieval_results_",
    "rank",
)


def _safe_filename_part(value, fallback):
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").lower()
    return slug[:48] or fallback


def _save_all_model_archive(bundle, models, download_mode):
    mode = bundle.get("query_mode", "search")
    query_slug = _safe_filename_part(bundle.get("query_label"), "query")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    zip_name = f"earth_embedding_explorer_{mode}_all_models_{query_slug}_{timestamp}.zip"
    zip_path = os.path.join(tempfile.gettempdir(), zip_name)
    temporary_archives = []

    try:
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as destination:
            model_names = [artifact["model_name"] for artifact in bundle.get("models", [])]
            destination.writestr(
                "manifest.txt",
                "EarthEmbeddingExplorer all-model export\n"
                f"Mode: {mode}\n"
                f"Query: {bundle.get('query_label', '')}\n"
                f"Models: {', '.join(model_names)}\n"
                f"Created (UTC): {timestamp}\n",
            )

            for index, artifact in enumerate(bundle.get("models", []), start=1):
                model_name = artifact["model_name"]
                model_slug = _safe_filename_part(model_name, f"model_{index}")
                folder = f"{index:02d}_{model_slug}"
                package = [
                    artifact.get("distribution"),
                    artifact.get("overview"),
                    artifact.get("results_text"),
                    artifact.get("results_meta"),
                    model_name,
                ]
                model_archive = save_plot(package, models, download_mode)
                if model_archive is None or not zipfile.is_zipfile(model_archive):
                    continue
                temporary_archives.append(model_archive)

                with zipfile.ZipFile(model_archive) as source:
                    for member in source.namelist():
                        if member == "map_distribution.png":
                            target = f"{folder}/{model_slug}_{mode}_distribution.png"
                        elif member == "retrieval_results.png":
                            target = f"{folder}/{model_slug}_{mode}_top5.png"
                        elif member == "results.txt":
                            target = f"{folder}/{model_slug}_{mode}_results.txt"
                        elif member.startswith("images/"):
                            target = f"{folder}/{member}"
                        else:
                            target = f"{folder}/{member}"
                        destination.writestr(target, source.read(member))

        return zip_path
    finally:
        for path in temporary_archives:
            try:
                os.remove(path)
            except OSError:
                pass


def _write_png(zip_file, image, filename):
    if image is None:
        return False
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    zip_file.writestr(filename, buffer.getvalue())
    return True


def save_comparison_figures(figs):
    """Export cached distribution and Top-5 figures without downloading imagery."""
    _cleanup_stale_temp_files()
    if figs is None:
        gr.Warning("Nothing to download yet — run a search first.")
        return None

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if isinstance(figs, dict) and figs.get("kind") == "all_models":
        mode = figs.get("query_mode", "search")
        archive_name = f"earth_embedding_explorer_{mode}_comparison_figures_{timestamp}.zip"
        entries = [
            (artifact["model_name"], artifact.get("distribution"), artifact.get("overview"))
            for artifact in figs.get("models", [])
        ]
    elif isinstance(figs, (list, tuple)) and len(figs) >= 5:
        model_name = figs[4] or "model"
        archive_name = f"earth_embedding_explorer_{_safe_filename_part(model_name, 'model')}_figures_{timestamp}.zip"
        entries = [(model_name, figs[0], figs[1])]
    else:
        gr.Warning("No distribution or Top-5 figures are available.")
        return None

    archive_path = os.path.join(tempfile.gettempdir(), archive_name)
    written = False
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for model_name, distribution, top5 in entries:
            safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", str(model_name)).strip("_") or "model"
            written |= _write_png(zip_file, distribution, f"{safe_name}_distribution.png")
            written |= _write_png(zip_file, top5, f"{safe_name}_Top5.png")

    if not written:
        try:
            os.remove(archive_path)
        except OSError:
            pass
        gr.Warning("No distribution or Top-5 figures are available.")
        return None
    return archive_path


def _cleanup_stale_temp_files(max_age_seconds=24 * 3600):
    """Best-effort removal of this app's temp files older than max_age_seconds."""
    try:
        temp_dir = tempfile.gettempdir()
        now = time.time()
        for fname in os.listdir(temp_dir):
            if not fname.startswith(_TEMP_FILE_PREFIXES):
                continue
            fpath = os.path.join(temp_dir, fname)
            try:
                if os.path.isfile(fpath) and now - os.path.getmtime(fpath) > max_age_seconds:
                    os.remove(fpath)
            except OSError:
                pass
    except Exception:
        pass


def save_plot(figs, models, download_mode="thumbnail"):
    """
    Save results as a downloadable zip file.

    download_mode controls what image data is included for top results:
      - "thumbnail": save the thumbnail images (fast, default)
      - "rgb":       re-download B04/B03/B02 composites and save as PNG
      - "multiband": re-download all 12 S2 bands and save as .npy per image
    """
    _cleanup_stale_temp_files()

    if isinstance(figs, dict) and figs.get("kind") == "all_models":
        return _save_all_model_archive(figs, models, download_mode)

    if figs is None or (isinstance(figs, (list, tuple)) and len(figs) == 0):
        gr.Warning("Nothing to download yet — run a search first.")
        return None

    temp_dir = tempfile.gettempdir()
    intermediate_files = []

    def unique_temp_path(prefix, suffix):
        return os.path.join(temp_dir, f"{prefix}_{uuid.uuid4().hex}{suffix}")

    def save_pil_image(image_obj, prefix, suffix=".png"):
        path = unique_temp_path(prefix, suffix)
        image_obj.save(path)
        return path

    def add_file(zipf, path, arcname):
        zipf.write(path, arcname=arcname)
        intermediate_files.append(path)
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

        zip_path = unique_temp_path("earth_embedding_explorer_results", ".zip")

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

            download_total = 0
            download_failures = 0

            if results_meta and isinstance(results_meta, list):
                for rank, res in enumerate(results_meta, start=1):
                    pid = res["id"]
                    download_total += 1
                    produced_file = False

                    try:
                        if download_mode == "multiband" and df_source is not None:
                            result = download_and_process_image(
                                pid, df_source=df_source, verbose=False, mode="multiband"
                            )

                            if result[2] is not None:
                                npy_path = unique_temp_path(f"rank{rank}_{pid}_12bands", ".npy")
                                np.save(npy_path, result[2])
                                add_file(zipf, npy_path, f"images/rank{rank}_{pid}_12bands.npy")
                                produced_file = True

                            if result[0] is not None:
                                preview_path = save_pil_image(result[0], f"rank{rank}_{pid}_preview")
                                add_file(zipf, preview_path, f"images/rank{rank}_{pid}_preview.png")
                                produced_file = True

                        elif download_mode == "rgb" and df_source is not None:
                            _, img_full = download_and_process_image(
                                pid, df_source=df_source, verbose=False, mode="rgb"
                            )
                            if img_full is not None:
                                rgb_path = save_pil_image(img_full, f"rank{rank}_{pid}_rgb")
                                add_file(zipf, rgb_path, f"images/rank{rank}_{pid}_rgb.png")
                                produced_file = True

                        else:
                            _, img_full = download_and_process_image(
                                pid, df_source=df_source, verbose=False, mode="thumbnail"
                            )
                            if img_full is not None:
                                thumb_path = save_pil_image(img_full, f"rank{rank}_{pid}_thumbnail")
                                add_file(zipf, thumb_path, f"images/rank{rank}_{pid}_thumbnail.png")
                                produced_file = True

                    except Exception as e:
                        print(f"Error downloading result image {pid}: {e}")

                    if not produced_file:
                        download_failures += 1

            if download_total and download_failures == download_total:
                gr.Warning("All result images failed to download. The ZIP contains only the map and report.")
            elif download_failures:
                gr.Warning(
                    f"{download_failures} of {download_total} result images failed to download and were skipped."
                )

            zip_is_empty = not zipf.namelist()

        # zipfile.write copies file contents into the archive, so once the
        # ZipFile is closed the intermediate files are safe to delete. The
        # returned zip_path (and the single-image PNG/HTML paths above) must
        # stay on disk: Gradio's File component reads them on download.
        for path in intermediate_files:
            try:
                os.remove(path)
            except OSError:
                pass

        if zip_is_empty:
            try:
                os.remove(zip_path)
            except OSError:
                pass
            gr.Warning("Nothing to export.")
            return None

        return zip_path

    except Exception as e:
        print(f"Error saving: {e}")
        return None
