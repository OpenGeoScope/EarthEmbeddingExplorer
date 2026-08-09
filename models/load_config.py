import os

import yaml

if os.getenv("DOWNLOAD_ENDPOINT", "") == "modelscope.ai":
    os.environ["MODELSCOPE_DOMAIN"] = "www.modelscope.ai"

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(config_path=None):
    """Load configuration from config.yaml or fall back to legacy configs.

    Config candidates are resolved relative to the current working directory
    first, then relative to configs/ under the project root. Other relative
    paths inside the config (e.g. ckpt_path) stay CWD-relative and are not
    rewritten here.
    """
    if config_path is None:
        candidates = [
            "./configs/config_local.yaml",
            "./configs/config.yaml",
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                config_path = candidate
                break
        if config_path is None:
            # CWD-independent fallback: configs/ under the project root.
            for candidate in candidates:
                root_candidate = os.path.join(_PROJECT_ROOT, candidate)
                if os.path.exists(root_candidate):
                    config_path = root_candidate
                    break

    if config_path is None or not os.path.exists(config_path):
        print("No config file found, using default configurations")
        return None

    print(f"Loading configuration from {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_path(path_str):
    """Resolve hf:// or ms:// prefixed paths by downloading from remote hubs.

    Raises RuntimeError (carrying the original path and the reason) when a
    remote path is malformed or the download fails, instead of silently
    returning the unresolved hf:// / ms:// string.
    """
    if path_str is None or not isinstance(path_str, str):
        return path_str

    # Normalize multiple slashes after protocol
    if path_str.startswith("hf://"):
        # Strip leading slashes after protocol
        rest = path_str[5:].lstrip("/")
        parts = rest.split("/", 2)
        if len(parts) < 3:
            raise RuntimeError(f"Invalid HuggingFace path format: {path_str}")
        repo_id = f"{parts[0]}/{parts[1]}"
        filename = parts[2]
        print(f"Downloading from HuggingFace: {repo_id}/{filename}")
        try:
            from huggingface_hub import hf_hub_download

            return hf_hub_download(repo_id, filename, repo_type="dataset")
        except Exception as e:
            raise RuntimeError(f"Failed to resolve HuggingFace path {path_str}: {e}") from e

    elif path_str.startswith("ms://"):
        # Strip leading slashes after protocol
        rest = path_str[5:].lstrip("/")
        parts = rest.split("/", 2)
        if len(parts) < 3:
            raise RuntimeError(f"Invalid ModelScope path format: {path_str}")
        repo_id = f"{parts[0]}/{parts[1]}"
        filename = parts[2]
        print(f"Downloading from ModelScope: {repo_id}/{filename}")
        try:
            from modelscope.hub.snapshot_download import snapshot_download

            cache_dir = snapshot_download(repo_id, repo_type="dataset", allow_file_pattern=filename)
            downloaded_file = os.path.join(cache_dir, filename)
            if os.path.exists(downloaded_file):
                return downloaded_file
            raise RuntimeError(f"File not found after ModelScope download: {downloaded_file} (from {path_str})")
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to resolve ModelScope path {path_str}: {e}") from e

    return path_str


def load_and_process_config(config_path=None):
    """Load config and resolve embedding paths with local-first priority."""
    config = load_config(config_path)
    if config is None:
        return None
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping of model name to settings, got {type(config).__name__}")

    processed = {}
    for model_name, model_config in config.items():
        if not isinstance(model_config, dict):
            raise ValueError(
                f"Config section for model '{model_name}' must be a mapping, got {type(model_config).__name__}"
            )
        processed[model_name] = {}
        for key, value in model_config.items():
            if key == "embedding_path":
                # Local-first: use local path if it exists. Relative paths are
                # tried against the CWD first, then the project root.
                if value and isinstance(value, str) and not value.startswith(("hf://", "ms://")):
                    local_candidate = value
                    if not os.path.exists(local_candidate):
                        root_candidate = os.path.join(_PROJECT_ROOT, value)
                        if os.path.exists(root_candidate):
                            local_candidate = root_candidate
                    if os.path.exists(local_candidate):
                        print(f"Using local embedding: {local_candidate}")
                        processed[model_name][key] = local_candidate
                        continue
                # Fallback: resolve ms:// / hf:// or keep plain paths as-is.
                # Resolution failure disables embeddings for this model only;
                # model wrappers tolerate embedding_path=None.
                try:
                    processed[model_name][key] = resolve_path(value)
                except RuntimeError as e:
                    print(
                        f"⚠️ {model_name}: embedding path resolution failed ({e}); "
                        "embeddings disabled for this model."
                    )
                    processed[model_name][key] = None
            else:
                processed[model_name][key] = value

    return processed
