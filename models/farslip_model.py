import inspect
import os

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image

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
    """
    FarSLIP model wrapper for Sentinel-2 RGB data embedding and search.

    This class provides a unified interface for:
    - Loading FarSLIP models from local checkpoint or remote repositories
    - Encoding images and text into embeddings
    - Loading pre-computed embeddings
    - Searching similar images using cosine similarity
    """

    def __init__(self,
                 ckpt_path=None,
                 model_name="ViT-B-16",
                 embedding_path=None,
                 device=None):
        """
        Initialize the FarSLIPModel.

        Args:
            ckpt_path (str): Path to local checkpoint. If None or not found,
                downloaded according to DOWNLOAD_ENDPOINT env var.
            model_name (str): Model architecture name.
            embedding_path (str): Path to pre-computed embeddings parquet file.
            device (str): Device to use ('cuda', 'cpu', or None for auto-detection).
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.ckpt_path = ckpt_path
        self.embedding_path = embedding_path

        self.model = None
        self.tokenizer = None
        self.preprocess = None
        self.df_embed = None
        self.image_embeddings = None

        # Define the RGB bands for Sentinel-2 (B04, B03, B02)
        self.bands = ['B04', 'B03', 'B02']
        self.size = (224, 224)

        self.load_model()
        if self.embedding_path is not None:
            self.load_embeddings()

    def load_model(self):
        """Load FarSLIP model, tokenizer, and preprocessing pipeline."""
        endpoint = os.getenv("DOWNLOAD_ENDPOINT", "modelscope.cn").lower()

        if self.ckpt_path is not None and os.path.exists(self.ckpt_path):
            print(f"Loading FarSLIP model from local path: {self.ckpt_path}")
        elif endpoint in ("huggingface", "hf"):
            print("Loading FarSLIP model from HuggingFace...")
            from huggingface_hub import hf_hub_download
            self.ckpt_path = hf_hub_download("ZhenShiL/FarSLIP", "FarSLIP2_ViT-B-16.pt")
        else:
            print("Loading FarSLIP model from ModelScope...")
            from modelscope.hub.snapshot_download import snapshot_download
            cache_dir = snapshot_download(
                repo_id='VoyagerX/FarSLIP',
                allow_file_pattern="FarSLIP2_ViT-B-16.pt"
            )
            self.ckpt_path = os.path.join(cache_dir, "FarSLIP2_ViT-B-16.pt")

        try:
            if not os.path.exists(self.ckpt_path):
                print(f"Warning: Checkpoint not found at {self.ckpt_path}")

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
        """Load pre-computed embeddings from parquet file."""
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
        """
        Encode a text query into a feature embedding.

        Args:
            text (str): Input text query.

        Returns:
            torch.Tensor: Normalized text embedding vector.
        """
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

    def preprocess_s2(self, input_data):
        """
        Preprocess Sentinel-2 RGB data for FarSLIP model input.

        Converts raw Sentinel-2 reflectance values to normalized true-color values
        by dividing by 10,000 and scaling by 2.5, clipping to the range [0, 1].

        Args:
            input_data (torch.Tensor or np.ndarray): Raw Sentinel-2 image data.

        Returns:
            torch.Tensor or np.ndarray: Preprocessed image data in range [0, 1].
        """
        return (2.5 * (input_data / 1e4)).clip(0, 1)

    def _tensor_to_pil_batch(self, image):
        """Convert a torch.Tensor batch to a list of PIL Images."""
        pil_images = []
        for i in range(image.shape[0]):
            img = image[i]
            if img.shape[0] == 3:  # [C, H, W]
                img = img.permute(1, 2, 0)
            img_np = img.detach().cpu().numpy()
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)
            pil_images.append(Image.fromarray(img_np, mode='RGB'))
        return pil_images

    def encode_image(self, image, preprocess_s2=True, normalize=True):
        """
        Encode an image into a feature embedding.

        Args:
            image (PIL.Image, torch.Tensor, or np.ndarray): Input image.
                - PIL.Image: RGB image.
                - torch.Tensor: Image tensor with shape [C, H, W] or [N, C, H, W].
                - np.ndarray: Image array with shape [H, W, C] or [N, H, W, C].
            preprocess_s2 (bool): Whether to apply Sentinel-2 preprocessing.
            normalize (bool): Whether to scale to [0, 255] and convert to PIL Image.
                For FarSLIP, the open_clip preprocess pipeline expects PIL Images,
                so tensor inputs are always converted to PIL internally.

        Returns:
            torch.Tensor: Normalized embedding vector with shape [embedding_dim] or
                [N, embedding_dim] for batched input.
        """
        if self.model is None:
            return None

        if isinstance(image, torch.Tensor):
            if preprocess_s2:
                image = self.preprocess_s2(image)
            if image.dim() == 3:
                image = image.unsqueeze(0)
            pil_images = self._tensor_to_pil_batch(image)
            image_tensors = torch.stack([self.preprocess(img) for img in pil_images]).to(self.device)
            with torch.no_grad():
                if self.device == "cuda":
                    with torch.amp.autocast('cuda'):
                        image_features = self.model.encode_image(image_tensors)
                else:
                    image_features = self.model.encode_image(image_tensors)
                image_features = F.normalize(image_features, dim=-1)
            return image_features

        elif isinstance(image, np.ndarray):
            if preprocess_s2:
                image = self.preprocess_s2(image)
            if image.ndim == 3:
                image = image[np.newaxis, ...]
            pil_images = []
            for i in range(image.shape[0]):
                img = image[i]
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
                pil_images.append(Image.fromarray(img, mode='RGB'))
            image_tensors = torch.stack([self.preprocess(img) for img in pil_images]).to(self.device)
            with torch.no_grad():
                if self.device == "cuda":
                    with torch.amp.autocast('cuda'):
                        image_features = self.model.encode_image(image_tensors)
                else:
                    image_features = self.model.encode_image(image_tensors)
                image_features = F.normalize(image_features, dim=-1)
            return image_features

        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.device == "cuda":
                with torch.amp.autocast('cuda'):
                    image_features = self.model.encode_image(image_tensor)
            else:
                image_features = self.model.encode_image(image_tensor)

            image_features = F.normalize(image_features, dim=-1)
        return image_features

    def __call__(self, input):
        """
        Callable wrapper that delegates to forward().

        Args:
            input (torch.Tensor): Raw Sentinel-2 image tensor.

        Returns:
            torch.Tensor: Normalized embedding vector.
        """
        return self.forward(input)

    def forward(self, input):
        """
        Forward pass for compatibility with MajorTOM_Embedder.

        Applies Sentinel-2 preprocessing and generates embeddings directly from
        a raw image tensor. This method is used by MajorTOM_Embedder during
        embedding generation.

        Args:
            input (torch.Tensor): Raw Sentinel-2 image tensor with shape
                [N, C, H, W] or [C, H, W], where C=3 (RGB channels).

        Returns:
            torch.Tensor: Normalized embedding vector with shape [N, embedding_dim]
                or [embedding_dim].
        """
        return self.encode_image(input, preprocess_s2=True, normalize=False)

    def search(self, query_features, top_k=5, top_percent=None, threshold=0.0):
        """
        Search for similar images using cosine similarity.

        Args:
            query_features (torch.Tensor): Query embedding vector.
            top_k (int): Number of top results to return.
            top_percent (float): If set, use top percentage instead of top_k.
            threshold (float): Minimum similarity threshold.

        Returns:
            tuple: (similarities, filtered_indices, top_indices)
                - similarities: Similarity scores for all images.
                - filtered_indices: Indices of images above threshold.
                - top_indices: Indices of top-k results.
        """
        if self.image_embeddings is None:
            return None, None, None

        query_features = query_features.float()

        # Similarity calculation
        probs = (self.image_embeddings @ query_features.T).detach().cpu().numpy().flatten()

        if top_percent is not None:
            k = int(len(probs) * top_percent)
            if k < 1:
                k = 1
            threshold = np.partition(probs, -k)[-k]

        mask = probs >= threshold
        filtered_indices = np.where(mask)[0]

        top_indices = np.argsort(probs)[-top_k:][::-1]

        return probs, filtered_indices, top_indices
