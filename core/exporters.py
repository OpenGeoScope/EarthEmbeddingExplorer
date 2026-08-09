import os
import tempfile
import time
import uuid
import zipfile

import gradio as gr
import numpy as np
from PIL import Image as PILImage

from data_utils import download_and_process_image

# Prefixes of temporary files created by this app (see save_plot).
_TEMP_FILE_PREFIXES = (
    "earth_embedding_explorer_results_",
    "earth_explorer_map_",
    "earth_explorer_plot_",
    "map_distribution_",
    "retrieval_results_",
    "rank",
)


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
