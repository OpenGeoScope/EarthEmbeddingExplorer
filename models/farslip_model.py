import os
import inspect
import torch
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch.nn.functional as F
from PIL import Image
from huggingface_hub import hf_hub_download
from modelscope.hub.snapshot_download import snapshot_download

try:
    # FarSLIP must use its vendored open_clip fork (checkpoint format / model defs differ).
    from .FarSLIP.open_clip.factory import create_model_and_transforms, get_tokenizer
    _OPENCLIP_BACKEND = "vendored_farslip"
    print("Successfully imported FarSLIP vendored open_clip.")
except ImportError as e:
    raise ImportError(
        "Failed to import FarSLIP vendored open_clip from 'models/FarSLIP/open_clip'. "
    ) from e

    
class FarSLIPModel:
    def __init__(self, 
                 ckpt_path="./checkpoints/FarSLIP/FarSLIP2_ViT-B-16.pt", 
                 model_name="ViT-B-16",
                 embedding_path="./embedding_datasets/10percent_farslip_encoded/all_farslip_embeddings.parquet",
                 device=None):
        
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        if 'hf' in ckpt_path:
            ckpt_path = hf_hub_download("ZhenShiL/FarSLIP", "FarSLIP2_ViT-B-16.pt")
        elif 'ms' in ckpt_path:
            cache_dir = snapshot_download(
                repo_id='VoyagerX/FarSLIP',
                allow_file_pattern="FarSLIP2_ViT-B-16.pt"
            )
            ckpt_path = os.path.join(cache_dir, "FarSLIP2_ViT-B-16.pt")
        
        self.ckpt_path = ckpt_path
        self.embedding_path = embedding_path
        
        self.model = None
        self.tokenizer = None
        self.preprocess = None
        self.df_embed = None
        self.image_embeddings = None
        
        # Force setup path and reload open_clip for FarSLIP
        # self.setup_path_and_reload()
        self.load_model()
        if self.embedding_path:
            self.load_embeddings()


    def load_model(self):
        print(f"Loading FarSLIP model from {self.ckpt_path}...")
        try:          
            if not os.path.exists(self.ckpt_path):
                print(f"Warning: Checkpoint not found at {self.ckpt_path}")
                # Try downloading? (Skipping for now as per instructions to use local)
            
            # Different open_clip variants expose slightly different factory signatures.
            # Build kwargs and filter by the actual callable signature (no sys.path hacks).
            factory_kwargs = {
                "model_name": self.model_name,
                "pretrained": self.ckpt_path,
                "precision": "amp",
                "device": self.device,
                "output_dict": True,
                "force_quick_gelu": False,
                "long_clip": "load_from_scratch",
            }

            sig = inspect.signature(create_model_and_transforms)
            supported = set(sig.parameters.keys())
            # Some variants take model_name as positional first arg; keep both styles working.
            if "model_name" in supported:
                call_kwargs = {k: v for k, v in factory_kwargs.items() if k in supported}
                self.model, _, self.preprocess = create_model_and_transforms(**call_kwargs)
            else:
                # Positional model name
                call_kwargs = {k: v for k, v in factory_kwargs.items() if k in supported and k != "model_name"}
                self.model, _, self.preprocess = create_model_and_transforms(self.model_name, **call_kwargs)
            
            self.tokenizer = get_tokenizer(self.model_name)
            self.model.eval()
            print(f"FarSLIP model loaded on {self.device} (backend={_OPENCLIP_BACKEND})")
        except Exception as e:
            print(f"Error loading FarSLIP model: {e}")

    def load_embeddings(self):
        print(f"Loading FarSLIP embeddings from {self.embedding_path}...")
        try:
            if not os.path.exists(self.embedding_path):
                print(f"Warning: Embedding file not found at {self.embedding_path}")
                return

            self.df_embed = pq.read_table(self.embedding_path).to_pandas()
            
            image_embeddings_np = np.stack(self.df_embed['embedding'].values)
            self.image_embeddings = torch.from_numpy(image_embeddings_np).to(self.device).float()
            self.image_embeddings = F.normalize(self.image_embeddings, dim=-1)
            print(f"FarSLIP Data loaded: {len(self.df_embed)} records")
        except Exception as e:
            print(f"Error loading FarSLIP embeddings: {e}")

    def encode_text(self, text):
        if self.model is None or self.tokenizer is None:
            return None
        
        text_tokens = self.tokenizer([text], context_length=self.model.context_length).to(self.device)
        
        with torch.no_grad():
            if self.device == "cuda":
                with torch.amp.autocast('cuda'):
                    text_features = self.model.encode_text(text_tokens)
            else:
                text_features = self.model.encode_text(text_tokens)
            
            text_features = F.normalize(text_features, dim=-1)
        return text_features

    def encode_image(self, image):
        if self.model is None:
            return None
        
        if isinstance(image, Image.Image):
            image = image.convert("RGB")
        
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.device == "cuda":
                with torch.amp.autocast('cuda'):
                    image_features = self.model.encode_image(image_tensor)
            else:
                image_features = self.model.encode_image(image_tensor)
            
            image_features = F.normalize(image_features, dim=-1)
        return image_features

    def search(self, query_features, top_k=5, top_percent=None, threshold=0.0):
        if self.image_embeddings is None:
            return None, None, None

        query_features = query_features.float()
        
        # Similarity calculation
        # FarSLIP might use different scaling, but usually dot product for normalized vectors
        probs = (self.image_embeddings @ query_features.T).detach().cpu().numpy().flatten()
        
        if top_percent is not None:
            k = int(len(probs) * top_percent)
            if k < 1: k = 1
            threshold = np.partition(probs, -k)[-k]

        mask = probs >= threshold
        filtered_indices = np.where(mask)[0]
        
        top_indices = np.argsort(probs)[-top_k:][::-1]
        
        return probs, filtered_indices, top_indices
