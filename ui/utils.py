import numpy as np
import pandas as pd
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont

from data_utils import download_and_process_image, get_esri_satellite_image
from visualize import format_results_for_gallery, plot_top5_overview


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
    from concurrent.futures import ThreadPoolExecutor, as_completed

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
