import os
import tempfile
import uuid
import zipfile

import numpy as np
from PIL import Image as PILImage

from data_utils import download_and_process_image


def save_plot(figs, models, download_mode="thumbnail"):
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
