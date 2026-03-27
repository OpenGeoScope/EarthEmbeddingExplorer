import os
import yaml

def load_config():
    """Load configuration from local.yaml if exists"""
    
    if os.path.exists("./configs/local.yaml"):
        config_path = "./configs/local.yaml"
    elif os.path.exists("./configs/modelscope_cn.yaml"):
        config_path = "./configs/modelscope_cn.yaml"
    elif os.path.exists("./configs/modelscope_ai.yaml"):
        config_path = "./configs/modelscope_ai.yaml"
        os.environ["MODELSCOPE_DOMAIN"] = "modelscope.ai"
    elif os.path.exists("./configs/huggingface.yaml"):
        config_path = "./configs/huggingface.yaml"
    else:
        print("No local.yaml found, using default configurations")
        return None
    print(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
def resolve_path(path_str):
    """
    Resolve path string, supporting HuggingFace Hub downloads and ModelScope downloads.
    Dataset Format: 
      - hf://repo_owner/repo_name/path/to/file (HuggingFace)
      - ms://repo_owner/repo_name/path/to/file (ModelScope)
    
    Note: repo_id contains '/', e.g., VoyagerX/EarthEmbeddings or ML4Sustain/EarthEmbeddings
    """
    if path_str is None:
        return None
    
    if isinstance(path_str, str):
        if path_str.startswith("hf://"):
            try:
                from huggingface_hub import hf_hub_download
                # Parse: hf://owner/repo/path/to/file
                # Split into at most 3 parts: owner, repo, filename
                path_without_prefix = path_str[5:]  # Remove "hf://"
                parts = path_without_prefix.split('/', 2)  # Split into owner, repo, filename
                
                if len(parts) >= 3:
                    repo_id = f"{parts[0]}/{parts[1]}"  # owner/repo
                    filename = parts[2]  # path/to/file
                    print(f"Downloading from HuggingFace: {repo_id}/{filename}")
                    return hf_hub_download(repo_id, filename, repo_type='dataset')
                else:
                    print(f"Invalid HuggingFace path format: {path_str}")
                    print(f"Expected format: hf://owner/repo/path/to/file")
                    return None
            except ImportError:
                print("huggingface_hub not installed, cannot download from HuggingFace")
                return None
            except Exception as e:
                print(f"Error downloading from HuggingFace: {e}")
                return None
        
        elif path_str.startswith("ms://"):
            try:
                from modelscope.hub.snapshot_download import snapshot_download
                # Parse: ms://owner/repo/path/to/file
                path_without_prefix = path_str[5:]  # Remove "ms://"
                parts = path_without_prefix.split('/', 2)  # Split into owner, repo, filename
                
                if len(parts) >= 3:
                    repo_id = f"{parts[0]}/{parts[1]}"  # owner/repo
                    filename = parts[2]  # path/to/file
                    print(f"Downloading from ModelScope: {repo_id}/{filename}")
                    # Use snapshot_download with allow_file_pattern to download single file
                    cache_dir = snapshot_download(
                        repo_id=repo_id,
                        repo_type='dataset',
                        allow_file_pattern=filename  # Only download this specific file
                    )
                    # Return the full path to the downloaded file
                    downloaded_file = os.path.join(cache_dir, filename)
                    if os.path.exists(downloaded_file):
                        return downloaded_file
                    else:
                        print(f"File not found after download: {downloaded_file}")
                        return None
                else:
                    print(f"Invalid ModelScope path format: {path_str}")
                    print(f"Expected format: ms://owner/repo/path/to/file")
                    return None
            except Exception as e:
                print(f"Error downloading from ModelScope: {e}")
                import traceback
                traceback.print_exc()
                return None
    
    return path_str

def process_config(config):
    """Process config to resolve all paths"""
    if config is None:
        return None
    
    processed = {}
    for model_name, model_config in config.items():
        processed[model_name] = {}
        for key, value in model_config.items():
            if key.endswith('_path'):
                processed[model_name][key] = resolve_path(value)
            else:
                processed[model_name][key] = value
    
    return processed

def load_and_process_config():
    """Load and process configuration in one step"""
    config = load_config()
    return process_config(config)
