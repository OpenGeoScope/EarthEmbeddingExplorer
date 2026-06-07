"""Pretrained checkpoint helpers for the vendored FarSLIP open_clip fork.

This project loads FarSLIP from an explicit checkpoint path. The upstream
open_clip pretrained registry is intentionally omitted here so ModelScope
Studio does not auto-associate unrelated public CLIP repositories from static
`hf_hub` strings.
"""

import hashlib
import os
import urllib
import warnings
from functools import partial
from typing import Dict, Iterable, Optional

from tqdm import tqdm

from .constants import HF_SAFE_WEIGHTS_NAME, HF_WEIGHTS_NAME
from .version import __version__

try:
    import safetensors.torch  # noqa: F401

    _has_safetensors = True
except ImportError:
    _has_safetensors = False

try:
    from huggingface_hub import hf_hub_download

    hf_hub_download = partial(hf_hub_download, library_name="open_clip", library_version=__version__)
    _has_hf_hub = True
except ImportError:
    hf_hub_download = None
    _has_hf_hub = False


_PRETRAINED = {}


def _clean_tag(tag: str):
    """Normalize pretrained tags."""
    return tag.lower().replace("-", "_")


def list_pretrained(as_str: bool = False):
    """Return built-in pretrained registry entries.

    FarSLIP runtime uses explicit checkpoint paths, so this registry is empty
    by design.
    """
    return [":".join([k, t]) if as_str else (k, t) for k in _PRETRAINED for t in _PRETRAINED[k]]


def list_pretrained_models_by_tag(tag: str):
    """Return all built-in models having the specified pretrained tag."""
    tag = _clean_tag(tag)
    return [model for model, tags in _PRETRAINED.items() if tag in tags]


def list_pretrained_tags_by_model(model: str):
    """Return built-in pretrained tags for the specified model architecture."""
    return list(_PRETRAINED.get(model, {}).keys())


def is_pretrained_cfg(model: str, tag: str):
    """Return whether a built-in pretrained config exists."""
    return _clean_tag(tag) in _PRETRAINED.get(model, {})


def get_pretrained_cfg(model: str, tag: str):
    """Return a built-in pretrained config, if present."""
    return _PRETRAINED.get(model, {}).get(_clean_tag(tag), {})


def get_pretrained_url(model: str, tag: str):
    """Return a built-in pretrained URL, if present."""
    return get_pretrained_cfg(model, tag).get("url", "")


def download_pretrained_from_url(
    url: str,
    cache_dir: Optional[str] = None,
):
    if not cache_dir:
        cache_dir = os.path.expanduser("~/.cache/clip")
    os.makedirs(cache_dir, exist_ok=True)
    filename = os.path.basename(url)

    if "openaipublic" in url:
        expected_sha256 = url.split("/")[-2]
    elif "mlfoundations" in url:
        expected_sha256 = os.path.splitext(filename)[0].split("-")[-1]
    else:
        expected_sha256 = ""

    download_target = os.path.join(cache_dir, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if expected_sha256:
            if hashlib.sha256(open(download_target, "rb").read()).hexdigest().startswith(expected_sha256):
                return download_target
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")
        else:
            return download_target

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.headers.get("Content-Length")), ncols=80, unit="iB", unit_scale=True) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if expected_sha256 and not hashlib.sha256(open(download_target, "rb").read()).hexdigest().startswith(
        expected_sha256
    ):
        raise RuntimeError("Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


def has_hf_hub(necessary=False):
    if not _has_hf_hub and necessary:
        raise RuntimeError("Hugging Face hub model specified but package not installed. Run `pip install huggingface_hub`.")
    return _has_hf_hub


def _get_safe_alternatives(filename: str) -> Iterable[str]:
    """Return potential safetensors alternatives for a Hugging Face weight file."""
    if filename == HF_WEIGHTS_NAME:
        yield HF_SAFE_WEIGHTS_NAME

    if filename not in (HF_WEIGHTS_NAME,) and (filename.endswith(".bin") or filename.endswith(".pth")):
        yield filename[:-4] + ".safetensors"


def download_pretrained_from_hf(
    model_id: str,
    filename: Optional[str] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
):
    has_hf_hub(True)

    filename = filename or HF_WEIGHTS_NAME

    if _has_safetensors:
        for safe_filename in _get_safe_alternatives(filename):
            try:
                return hf_hub_download(
                    repo_id=model_id,
                    filename=safe_filename,
                    revision=revision,
                    cache_dir=cache_dir,
                )
            except Exception:
                pass

    try:
        return hf_hub_download(
            repo_id=model_id,
            filename=filename,
            revision=revision,
            cache_dir=cache_dir,
        )
    except Exception as e:
        raise FileNotFoundError(f"Failed to download file ({filename}) for {model_id}. Last error: {e}") from e


def download_pretrained(
    cfg: Dict,
    prefer_hf_hub: bool = True,
    cache_dir: Optional[str] = None,
):
    if not cfg:
        return ""

    if "file" in cfg:
        return cfg["file"]

    has_hub = has_hf_hub()
    download_url = cfg.get("url", "")
    download_hf_hub = cfg.get("hf_hub", "")
    if has_hub and prefer_hf_hub and download_hf_hub:
        download_url = ""

    if download_url:
        return download_pretrained_from_url(download_url, cache_dir=cache_dir)

    if download_hf_hub:
        has_hf_hub(True)
        model_id, filename = os.path.split(download_hf_hub)
        if filename:
            return download_pretrained_from_hf(model_id, filename=filename, cache_dir=cache_dir)
        return download_pretrained_from_hf(model_id, cache_dir=cache_dir)

    return ""
