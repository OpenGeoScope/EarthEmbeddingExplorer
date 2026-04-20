import os

import yaml

if os.getenv("DOWNLOAD_ENDPOINT", "") == "modelscope.ai":
    os.environ["MODELSCOPE_DOMAIN"] = "www.modelscope.ai"
def load_config(config_path=None):
    """Load configuration from config.yaml or fall back to legacy configs."""
    if config_path is None:
        candidates = [
            "./configs/config_local.yaml",
            "./configs/config.yaml",
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                config_path = candidate
                break

    if config_path is None or not os.path.exists(config_path):
        print("No config file found, using default configurations")
        return None

    print(f"Loading configuration from {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_path(path_str):
    """Resolve hf:// or ms:// prefixed paths by downloading from remote hubs."""
    if path_str is None or not isinstance(path_str, str):
        return path_str

    # Normalize multiple slashes after protocol
    if path_str.startswith("hf://"):
        try:
            from huggingface_hub import hf_hub_download
            # Strip leading slashes after protocol
            rest = path_str[5:].lstrip('/')
            parts = rest.split('/', 2)
            if len(parts) >= 3:
                repo_id = f"{parts[0]}/{parts[1]}"
                filename = parts[2]
                print(f"Downloading from HuggingFace: {repo_id}/{filename}")
                return hf_hub_download(repo_id, filename, repo_type='dataset')
            else:
                print(f"Invalid HuggingFace path format: {path_str}")
        except Exception as e:
            print(f"Error downloading from HuggingFace: {e}")
        return path_str

    elif path_str.startswith("ms://"):
        try:
            from modelscope.hub.snapshot_download import snapshot_download
            # Strip leading slashes after protocol
            rest = path_str[5:].lstrip('/')
            parts = rest.split('/', 2)
            if len(parts) >= 3:
                repo_id = f"{parts[0]}/{parts[1]}"
                filename = parts[2]
                print(f"Downloading from ModelScope: {repo_id}/{filename}")
                cache_dir = snapshot_download(repo_id, repo_type='dataset', allow_file_pattern=filename)
                downloaded_file = os.path.join(cache_dir, filename)
                if os.path.exists(downloaded_file):
                    return downloaded_file
                else:
                    print(f"File not found after download: {downloaded_file}")
            else:
                print(f"Invalid ModelScope path format: {path_str}")
        except Exception as e:
            print(f"Error downloading from ModelScope: {e}")
        return path_str

    return path_str


def load_and_process_config(config_path=None):
    """Load config and resolve embedding paths with local-first priority."""
    config = load_config(config_path)
    if config is None:
        return None

    processed = {}
    for model_name, model_config in config.items():
        processed[model_name] = {}
        for key, value in model_config.items():
            if key == "embedding_path":
                # Local-first: use local path if it exists
                if value and os.path.exists(value):
                    print(f"Using local embedding: {value}")
                    processed[model_name][key] = value
                else:
                    # Fallback: resolve ms:// / hf:// or return as-is
                    processed[model_name][key] = resolve_path(value)
            else:
                processed[model_name][key] = value

    return processed
