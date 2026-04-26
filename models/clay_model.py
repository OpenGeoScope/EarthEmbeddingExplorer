import math
import os
import warnings

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from .Clay.claymodel.finetune.embedder.factory import Embedder


class ClayModel:
    """
    Clay v1.5 model wrapper for Sentinel-2 multi-spectral data embedding and search.

    This class provides a unified interface for:
    - Loading Clay models from local checkpoint
    - Encoding images into embeddings
    - Loading pre-computed embeddings
    - Searching similar images using cosine similarity
    """

    def __init__(self,
                 ckpt_path=None,
                 embedding_path=None,
                 device=None):
        """
        Initialize the ClayModel.

        Args:
            ckpt_path (str): Path to local checkpoint (clay-v1.5.ckpt).
                If None, defaults to /data384/checkpoints/Clay/v1.5/clay-v1.5.ckpt.
            embedding_path (str): Path to pre-computed embeddings parquet file.
            device (str): Device to use ('cuda', 'cpu', or None for auto-detection).
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        if ckpt_path is None:
            ckpt_path = "/data384/checkpoints/Clay/v1.5/clay-v1.5.ckpt"
        self.ckpt_path = ckpt_path
        self.embedding_path = embedding_path

        self.model = None
        self.df_embed = None
        self.image_embeddings = None

        # Clay Sentinel-2 L2A bands (10 bands)
        self.bands = [
            'B02', 'B03', 'B04', 'B05', 'B06', 'B07',
            'B08', 'B8A', 'B11', 'B12'
        ]
        self.requires_multiband = True  # Model needs multi-spectral Sentinel-2 input
        self.size = (384, 384)

        # Clay metadata for Sentinel-2 normalization
        self.clay_mean = np.array(
            [1105., 1355., 1552., 1887., 2422., 2630., 2743., 2785., 2388., 1835.],
            dtype=np.float32,
        )
        self.clay_std = np.array(
            [1809., 1757., 1888., 1870., 1732., 1697., 1742., 1648., 1470., 1379.],
            dtype=np.float32,
        )
        self.clay_waves = torch.tensor(
            [0.493, 0.56, 0.665, 0.704, 0.74, 0.783, 0.842, 0.865, 1.61, 2.19],
            dtype=torch.float32,
        )
        self.clay_gsd = torch.tensor([10.0], dtype=torch.float32)

        self.load_model()
        if self.embedding_path is not None:
            self.load_embeddings()

    def load_model(self):
        """Load Clay model and visual encoder."""
        endpoint = os.getenv("DOWNLOAD_ENDPOINT", "modelscope.cn")

        if self.ckpt_path is not None and os.path.exists(self.ckpt_path):
            print(f"Loading Clay model from local path: {self.ckpt_path}")
        elif endpoint in ("huggingface"):
            print("Loading Clay model from HuggingFace...")
            from huggingface_hub import snapshot_download
            cache_dir = snapshot_download(repo_id="made-with-clay/Clay")
            self.ckpt_path = os.path.join(cache_dir, "v1.5", "clay-v1.5.ckpt")
        elif endpoint in ("modelscope.ai"):
            print("Loading Clay model from ModelScope (modelscope.ai)...")
            os.environ["MODELSCOPE_DOMAIN"] = "www.modelscope.ai"
            from modelscope.hub.snapshot_download import snapshot_download
            cache_dir = snapshot_download(repo_id="VoyagerX/Clay")
            self.ckpt_path = os.path.join(cache_dir, "v1.5", "clay-v1.5.ckpt")
        else:
            print("Loading Clay model from ModelScope (modelscope.cn)...")
            from modelscope.hub.snapshot_download import snapshot_download
            cache_dir = snapshot_download(repo_id="VoyagerX/Clay")
            self.ckpt_path = os.path.join(cache_dir, "v1.5", "clay-v1.5.ckpt")

        try:
            if not os.path.exists(self.ckpt_path):
                print(f"Warning: Checkpoint not found at {self.ckpt_path}")
                return

            self.model = Embedder(
                img_size=self.size[0],
                ckpt_path=self.ckpt_path,
                device=self.device,
            )
            self.model.eval()
            print(f"Clay model loaded on {self.device}")
        except Exception as e:
            print(f"Error loading Clay model: {e}")
            import traceback
            traceback.print_exc()

    def load_embeddings(self):
        """Load pre-computed embeddings from parquet file."""
        print(f"Loading Clay embeddings from {self.embedding_path}...")
        try:
            if not os.path.exists(self.embedding_path):
                print(f"Warning: Embedding file not found at {self.embedding_path}")
                return

            self.df_embed = pq.read_table(self.embedding_path).to_pandas()

            image_embeddings_np = np.stack(self.df_embed['embedding'].values)
            self.image_embeddings = torch.from_numpy(image_embeddings_np).to(self.device).float()
            self.image_embeddings = F.normalize(self.image_embeddings, dim=-1)
            print(f"Clay Data loaded: {len(self.df_embed)} records")
        except Exception as e:
            print(f"Error loading Clay embeddings: {e}")

    @staticmethod
    def _normalize_latlon(lat, lon):
        """Normalize latitude/longitude to sin/cos encoding for Clay."""
        lat_rad = lat * np.pi / 180
        lon_rad = lon * np.pi / 180
        lat_norm = (math.sin(lat_rad), math.cos(lat_rad))
        lon_norm = (math.sin(lon_rad), math.cos(lon_rad))
        return lat_norm, lon_norm

    @staticmethod
    def _normalize_time(week, hour):
        """Normalize week/hour to sin/cos encoding for Clay."""
        week_rad = week * 2 * np.pi / 52
        hour_rad = hour * 2 * np.pi / 24
        week_norm = (math.sin(week_rad), math.cos(week_rad))
        hour_norm = (math.sin(hour_rad), math.cos(hour_rad))
        return week_norm, hour_norm

    def preprocess_s2(self, input_data):
        """
        Preprocess Sentinel-2 multi-spectral data for Clay model input.

        Applies Clay-specific normalization using mean and std values
        derived from the Clay training corpus.

        Args:
            input_data (torch.Tensor or np.ndarray): Raw Sentinel-2 image data.
                Expected shape for torch.Tensor: [..., C, H, W].
                Expected shape for np.ndarray: [..., H, W, C].

        Returns:
            torch.Tensor or np.ndarray: Normalized image data.
        """
        if isinstance(input_data, torch.Tensor):
            # Assume [..., C, H, W]
            shape = [1] * input_data.dim()
            shape[-3] = -1
            mean = torch.from_numpy(self.clay_mean).to(input_data.device).view(*shape)
            std = torch.from_numpy(self.clay_std).to(input_data.device).view(*shape)
            return (input_data - mean) / std
        else:
            # np.ndarray, assume [..., H, W, C]
            return (input_data - self.clay_mean) / self.clay_std

    def _build_datacube(self, image_tensor, latlon=None, time=None):
        """
        Build a Clay datacube dictionary from an image tensor.

        Args:
            image_tensor (torch.Tensor): Image tensor with shape [B, C, H, W].
            latlon (torch.Tensor, optional): Lat/lon encoding [B, 4]. Defaults to zeros.
            time (torch.Tensor, optional): Time encoding [B, 4]. Defaults to zeros.

        Returns:
            dict: Clay datacube dictionary.
        """
        B = image_tensor.shape[0]
        device = image_tensor.device

        if latlon is None:
            latlon = torch.zeros(B, 4, dtype=torch.float32, device=device)
        if time is None:
            time = torch.zeros(B, 4, dtype=torch.float32, device=device)

        waves = self.clay_waves.to(device)
        gsd = self.clay_gsd.to(device)

        return {
            "pixels": image_tensor.float(),
            "time": time,
            "latlon": latlon,
            "waves": waves,
            "gsd": gsd,
        }

    def encode_image(self, image, preprocess_s2=True, normalize=True, latlon=None, time=None):
        """
        Encode an image into a feature embedding using Clay encoder.

        Supports:
        - torch.Tensor with shape [N, 10, H, W] (MajorTOM/Clay format).
        - np.ndarray with shape [H, W, 10] or [N, H, W, 10].
        - PIL Image (RGB): adapted to 10 channels (rest zeros).

        Args:
            image: Input image (PIL.Image, torch.Tensor, or np.ndarray).
            preprocess_s2 (bool): Whether to apply Clay Sentinel-2 normalization.
            normalize (bool): Unused for Clay tensor inputs, kept for API consistency.
            latlon (torch.Tensor, optional): [B, 4] lat/lon encoding.
            time (torch.Tensor, optional): [B, 4] time encoding.

        Returns:
            torch.Tensor: Normalized image embedding vector.
        """
        if self.model is None:
            return None

        try:
            if isinstance(image, torch.Tensor):
                if image.dim() == 3:
                    image = image.unsqueeze(0)
                # Resize to Clay's expected input size if needed
                if image.shape[2:] != self.size:
                    image = F.interpolate(image.float(), size=self.size, mode='bicubic', align_corners=False)
                if preprocess_s2:
                    image = self.preprocess_s2(image)
                datacube = self._build_datacube(image, latlon=latlon, time=time)
                with torch.no_grad():
                    img_feature = self.model(datacube)
                    img_feature = F.normalize(img_feature, dim=-1)
                return img_feature

            elif isinstance(image, np.ndarray):
                if image.ndim == 3:
                    image = image[np.newaxis, ...]
                # Assume [N, H, W, C]
                if image.shape[-1] == len(self.bands):
                    image = image.transpose(0, 3, 1, 2)  # [N, C, H, W]
                image_tensor = torch.from_numpy(image).to(self.device)
                # Resize to Clay's expected input size if needed
                if image_tensor.shape[2:] != self.size:
                    image_tensor = F.interpolate(image_tensor.float(), size=self.size, mode='bicubic', align_corners=False)
                if preprocess_s2:
                    image_tensor = self.preprocess_s2(image_tensor)
                datacube = self._build_datacube(image_tensor, latlon=latlon, time=time)
                with torch.no_grad():
                    img_feature = self.model(datacube)
                    img_feature = F.normalize(img_feature, dim=-1)
                return img_feature

            elif isinstance(image, Image.Image):
                image = image.convert("RGB")
                image = image.resize(self.size)
                img_np = np.array(image).astype(np.float32)
                # Map RGB to Clay bands (B04=red, B03=green, B02=blue)
                # Fill remaining bands with zeros
                input_tensor = np.zeros((len(self.bands), self.size[0], self.size[1]), dtype=np.float32)
                input_tensor[2] = img_np[:, :, 0]  # Red -> B04
                input_tensor[1] = img_np[:, :, 1]  # Green -> B03
                input_tensor[0] = img_np[:, :, 2]  # Blue -> B02
                input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).to(self.device)
                if preprocess_s2:
                    input_tensor = self.preprocess_s2(input_tensor)
                datacube = self._build_datacube(input_tensor, latlon=latlon, time=time)
                with torch.no_grad():
                    img_feature = self.model(datacube)
                    img_feature = F.normalize(img_feature, dim=-1)
                return img_feature

            else:
                print(f"Unsupported image type for Clay: {type(image)}")
                return None

        except Exception as e:
            print(f"Error encoding image in Clay: {e}")
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

        Applies Clay Sentinel-2 preprocessing and generates embeddings directly from
        a raw image tensor. This method is used by MajorTOM_Embedder during
        embedding generation.

        Args:
            input (torch.Tensor): Raw Sentinel-2 image tensor with shape
                [N, C, H, W] or [C, H, W], where C=10 (Clay bands).

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
