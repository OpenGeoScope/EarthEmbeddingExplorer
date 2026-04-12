import os

import numpy as np
import open_clip
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from open_clip.tokenizer import HFTokenizer
from PIL import Image


class SigLIPModel:
    def __init__(self,
                 ckpt_path="./checkpoints/ViT-SO400M-14-SigLIP-384/open_clip_pytorch_model.bin",
                 model_name="ViT-SO400M-14-SigLIP-384",
                 tokenizer_path="./checkpoints/ViT-SO400M-14-SigLIP-384",
                 embedding_path="./embedding_datasets/10percent_siglip_encoded/all_siglip_embeddings.parquet",
                 device=None):

        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.ckpt_path = ckpt_path
        self.tokenizer_path = tokenizer_path
        self.embedding_path = embedding_path

        self.model = None
        self.tokenizer = None
        self.preprocess = None
        self.df_embed = None
        self.image_embeddings = None

        self.load_model()
        self.load_embeddings()

    def load_model(self):
        print(f"Loading SigLIP model from {self.ckpt_path}...")
        try:
            # Check if paths exist, if not try relative paths or raise warning
            if not os.path.exists(self.ckpt_path):
                print(f"Warning: Checkpoint not found at {self.ckpt_path}")

            if 'hf' in self.ckpt_path:
                from huggingface_hub import snapshot_download
                cache_dir = snapshot_download(repo_id="timm/ViT-SO400M-14-SigLIP-384")
                self.tokenizer_path = cache_dir
                self.ckpt_path = os.path.join(cache_dir, "open_clip_pytorch_model.bin")

            elif 'ms' in self.ckpt_path:
                from modelscope.hub.snapshot_download import snapshot_download
                if self.ckpt_path.endswith("cn"):
                    repo_id = "timm/ViT-SO400M-14-SigLIP-384"
                else:
                    repo_id = "VoyagerX/ViT-SO400M-14-SigLIP-384"
                cache_dir = snapshot_download(repo_id=repo_id)
                self.tokenizer_path = cache_dir
                self.ckpt_path = os.path.join(cache_dir, "open_clip_pytorch_model.bin")

            self.tokenizer = HFTokenizer(self.tokenizer_path)
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.ckpt_path
            )
            self.model = self.model.to(self.device)
            self.model.eval()

            print(f"SigLIP model loaded on {self.device}")
        except Exception as e:
            print(f"Error loading SigLIP model: {e}")

    def load_embeddings(self):
        print(f"Loading SigLIP embeddings from {self.embedding_path}...")
        try:
            if not os.path.exists(self.embedding_path):
                print(f"Warning: Embedding file not found at {self.embedding_path}")
                return

            self.df_embed = pq.read_table(self.embedding_path).to_pandas()

            # Pre-compute image embeddings tensor
            image_embeddings_np = np.stack(self.df_embed['embedding'].values)
            self.image_embeddings = torch.from_numpy(image_embeddings_np).to(self.device).float()
            self.image_embeddings = F.normalize(self.image_embeddings, dim=-1)
            print(f"SigLIP Data loaded: {len(self.df_embed)} records")
        except Exception as e:
            print(f"Error loading SigLIP embeddings: {e}")

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

        # Ensure RGB
        if isinstance(image, Image.Image):
            image = image.convert("RGB")

        # Preprocess
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

        # Ensure query_features is float32
        query_features = query_features.float()

        # Similarity calculation
        # Logits: (N_images, 1)
        # logits = self.image_embeddings @ query_features.T * self.model.logit_scale.exp() + self.model.logit_bias
        # probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()

        # Use Cosine Similarity directly (aligned with SigLIP_embdding.ipynb)
        similarity = (self.image_embeddings @ query_features.T).squeeze()
        probs = similarity.detach().cpu().numpy()

        if top_percent is not None:
            k = int(len(probs) * top_percent)
            if k < 1:
                k = 1
            threshold = np.partition(probs, -k)[-k]

        # Filter by threshold
        mask = probs >= threshold
        filtered_indices = np.where(mask)[0]

        # Get top k
        top_indices = np.argsort(probs)[-top_k:][::-1]

        return probs, filtered_indices, top_indices
