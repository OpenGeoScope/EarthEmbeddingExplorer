import os
import warnings

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image

# Attempt to import get_satclip, but handle potential issues gracefully

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from models.SatCLIP.satclip.load import get_satclip
    print("Successfully imported models.SatCLIP.satclip.load.get_satclip.")


class SatCLIPModel:
    """
    SatCLIP model wrapper for multi-spectral Sentinel-2 data embedding and search.

    This class provides a unified interface for:
    - Loading SatCLIP models from local checkpoint or remote repositories
    - Encoding images and geographic locations into embeddings
    - Loading pre-computed embeddings
    - Searching similar images using cosine similarity
    """

    def __init__(self,
                 ckpt_path=None,
                 embedding_path=None,
                 device=None):
        """
        Initialize the SatCLIPModel.

        Args:
            ckpt_path (str): Path to local checkpoint. If None or not found,
                downloaded according to DOWNLOAD_ENDPOINT env var.
            embedding_path (str): Path to pre-computed embeddings parquet file.
            device (str): Device to use ('cuda', 'cpu', or None for auto-detection).
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt_path = ckpt_path
        self.embedding_path = embedding_path

        self.model = None
        self.df_embed = None
        self.image_embeddings = None

        # Define the 12 Sentinel-2 bands for SatCLIP input
        self.bands = [
            'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07',
            'B08', 'B8A', 'B09', 'B11', 'B12'
        ]
        self.size = (224, 224)

        self.load_model()
        if self.embedding_path is not None:
            self.load_embeddings()

    def load_model(self):
        """Load SatCLIP model and visual/location encoders."""
        if get_satclip is None:
            print("Error: SatCLIP functionality is not available.")
            return

        endpoint = os.getenv("DOWNLOAD_ENDPOINT", "modelscope.cn")

        if self.ckpt_path is not None and os.path.exists(self.ckpt_path):
            print(f"Loading SatCLIP model from local path: {self.ckpt_path}")
        elif endpoint in ("huggingface"):
            print("Loading SatCLIP model from HuggingFace...")
            from huggingface_hub import hf_hub_download
            self.ckpt_path = hf_hub_download("microsoft/SatCLIP-ViT16-L40", "satclip-vit16-l40.ckpt")
        elif endpoint in ("modelscope.ai"):
            print("Loading SatCLIP model from ModelScope (modelscope.ai)...")
            os.environ["MODELSCOPE_DOMAIN"] = "www.modelscope.ai"
            from modelscope.hub.snapshot_download import snapshot_download
            cache_dir = snapshot_download(
                repo_id="VoyagerX/SatCLIP-ViT16-L40",
                allow_file_pattern="satclip-vit16-l40.ckpt"
            )
            self.ckpt_path = os.path.join(cache_dir, "satclip-vit16-l40.ckpt")
        else:
            print("Loading SatCLIP model from ModelScope (modelscope.cn)...")
            from modelscope.hub.snapshot_download import snapshot_download
            cache_dir = snapshot_download(
                repo_id="microsoft/SatCLIP-ViT16-L40",
                allow_file_pattern="satclip-vit16-l40.ckpt"
            )
            self.ckpt_path = os.path.join(cache_dir, "satclip-vit16-l40.ckpt")

        try:
            if not os.path.exists(self.ckpt_path):
                print(f"Warning: Checkpoint not found at {self.ckpt_path}")
                return

            # Load model using get_satclip
            # return_all=True to get both visual and location encoders
            self.model = get_satclip(self.ckpt_path, self.device, return_all=True)
            self.model.eval()
            print(f"SatCLIP model loaded on {self.device}")
        except Exception as e:
            print(f"Error loading SatCLIP model: {e}")

    def load_embeddings(self):
        """Load pre-computed embeddings from parquet file."""
        print(f"Loading SatCLIP embeddings from {self.embedding_path}...")
        try:
            if not os.path.exists(self.embedding_path):
                print(f"Warning: Embedding file not found at {self.embedding_path}")
                return

            self.df_embed = pq.read_table(self.embedding_path).to_pandas()

            # Pre-compute image embeddings tensor
            image_embeddings_np = np.stack(self.df_embed['embedding'].values)
            self.image_embeddings = torch.from_numpy(image_embeddings_np).to(self.device).float()
            self.image_embeddings = F.normalize(self.image_embeddings, dim=-1)
            print(f"SatCLIP Data loaded: {len(self.df_embed)} records")
        except Exception as e:
            print(f"Error loading SatCLIP embeddings: {e}")

    def encode_location(self, lat, lon):
        """
        Encode a (latitude, longitude) pair into a vector.

        Args:
            lat (float): Latitude.
            lon (float): Longitude.

        Returns:
            torch.Tensor: Normalized location embedding vector.
        """
        if self.model is None:
            return None

        # SatCLIP expects input shape (N, 2) -> (lon, lat)
        # Note: SatCLIP usually uses (lon, lat) order.
        # Use double precision as per notebook reference
        coords = torch.tensor([[lon, lat]], dtype=torch.double).to(self.device)

        with torch.no_grad():
            # Use model.encode_location instead of model.location_encoder
            # And normalize as per notebook: x / x.norm()
            loc_features = self.model.encode_location(coords).float()
            loc_features = loc_features / loc_features.norm(dim=1, keepdim=True)

        return loc_features

    def preprocess_s2(self, input_data):
        """
        Preprocess Sentinel-2 multi-spectral data for SatCLIP model input.

        Normalizes raw Sentinel-2 reflectance values by dividing by 10,000.

        Args:
            input_data (torch.Tensor or np.ndarray): Raw Sentinel-2 image data.

        Returns:
            torch.Tensor or np.ndarray: Preprocessed image data.
        """
        return input_data / 1e4

    def _prepare_satclip_tensor(self, tensor):
        """
        Prepare a torch.Tensor for SatCLIP encoding.

        Pads B10 zeros if input has 12 channels, and resizes to 224x224.

        Args:
            tensor (torch.Tensor): Input tensor with shape [N, C, H, W].

        Returns:
            torch.Tensor: Prepared tensor with shape [N, 13, 224, 224].
        """
        if tensor.shape[1] == 12:
            zeros = torch.zeros(tensor.shape[0], 1, tensor.shape[2], tensor.shape[3],
                                dtype=tensor.dtype, device=tensor.device)
            tensor = torch.cat([tensor[:, :10], zeros, tensor[:, 10:]], dim=1)
        elif tensor.shape[1] != 13:
            raise ValueError(f"Expected 12 or 13 channels, got {tensor.shape[1]}")
        tensor = F.interpolate(tensor, size=(224, 224), mode='bicubic', align_corners=False)
        return tensor

    def encode_image(self, image, preprocess_s2=True, normalize=True):
        """
        Encode an image into a vector using SatCLIP visual encoder.

        Supports:
        - torch.Tensor with shape [N, 12, H, W] or [N, 13, H, W] (MajorTOM format).
        - np.ndarray with shape [H, W, 12] or [N, H, W, 12] (raw Sentinel-2).
        - PIL Image (RGB): adapted to 13 channels (R->B04, G->B03, B->B02, rest zeros).

        Args:
            image: Input image (torch.Tensor, np.ndarray, or PIL.Image).
            preprocess_s2 (bool): Whether to apply Sentinel-2 preprocessing (divide by 1e4).
            normalize (bool): Unused for SatCLIP tensor inputs, kept for API consistency.

        Returns:
            torch.Tensor: Normalized image embedding vector.
        """
        if self.model is None:
            return None

        try:
            if isinstance(image, torch.Tensor):
                if image.dim() == 3:
                    image = image.unsqueeze(0)
                if preprocess_s2:
                    image = image.float() / 10000.0
                image = self._prepare_satclip_tensor(image)
                with torch.no_grad():
                    img_feature = self.model.encode_image(image)
                    img_feature = img_feature / img_feature.norm(dim=1, keepdim=True)
                return img_feature

            elif isinstance(image, np.ndarray):
                if image.ndim == 3 and image.shape[-1] == 12:
                    if preprocess_s2:
                        img = image.astype(np.float32) / 10000.0
                    else:
                        img = image.astype(np.float32)
                    img = img.transpose(2, 0, 1)
                    _b10 = np.zeros((1, img.shape[1], img.shape[2]), dtype=img.dtype)
                    img_13 = np.concatenate([img[:10], _b10, img[10:]], axis=0)
                    input_tensor = torch.from_numpy(img_13).unsqueeze(0)
                    input_tensor = F.interpolate(input_tensor, size=(224, 224), mode='bicubic', align_corners=False)
                    input_tensor = input_tensor.to(self.device)
                    with torch.no_grad():
                        img_feature = self.model.encode_image(input_tensor)
                        img_feature = img_feature / img_feature.norm(dim=1, keepdim=True)
                    return img_feature

                elif image.ndim == 4 and image.shape[-1] == 12:
                    if preprocess_s2:
                        img = image.astype(np.float32) / 10000.0
                    else:
                        img = image.astype(np.float32)
                    features = []
                    for i in range(img.shape[0]):
                        arr = img[i].transpose(2, 0, 1)
                        _b10 = np.zeros((1, arr.shape[1], arr.shape[2]), dtype=arr.dtype)
                        arr_13 = np.concatenate([arr[:10], _b10, arr[10:]], axis=0)
                        t = torch.from_numpy(arr_13).unsqueeze(0)
                        t = F.interpolate(t, size=(224, 224), mode='bicubic', align_corners=False)
                        t = t.to(self.device)
                        with torch.no_grad():
                            f = self.model.encode_image(t)
                            f = f / f.norm(dim=1, keepdim=True)
                        features.append(f)
                    return torch.cat(features, dim=0)
                else:
                    print(f"Unsupported ndarray shape for SatCLIP: {image.shape}")
                    return None

            elif isinstance(image, Image.Image):
                image = image.convert("RGB")
                image = image.resize((224, 224))
                img_np = np.array(image).astype(np.float32) / 255.0

                # Construct 13 channels
                # S2 bands: B01, B02(B), B03(G), B04(R), B05, B06, B07, B08, B8A, B09, B10, B11, B12
                input_tensor = np.zeros((13, 224, 224), dtype=np.float32)
                input_tensor[1] = img_np[:, :, 2]  # Blue  -> B02
                input_tensor[2] = img_np[:, :, 1]  # Green -> B03
                input_tensor[3] = img_np[:, :, 0]  # Red   -> B04

                input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    img_feature = self.model.encode_image(input_tensor)
                    img_feature = img_feature / img_feature.norm(dim=1, keepdim=True)
                return img_feature

            else:
                print(f"Unsupported image type for SatCLIP: {type(image)}")
                return None

        except Exception as e:
            print(f"Error encoding image in SatCLIP: {e}")
            import traceback
            traceback.print_exc()
            return None

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
        a raw 12-band image tensor. This method is used by MajorTOM_Embedder
        during embedding generation.

        Args:
            input (torch.Tensor): Raw Sentinel-2 image tensor with shape
                [N, C, H, W] or [C, H, W], where C=12.

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

        # Similarity calculation (Cosine similarity)
        # SatCLIP embeddings are normalized, so dot product is cosine similarity
        probs = (self.image_embeddings @ query_features.T).detach().cpu().numpy().flatten()

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
