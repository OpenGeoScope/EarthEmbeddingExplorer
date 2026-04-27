import os

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn.flexi_vit import PoolingType
from PIL import Image


class OlmoEarthModel:
    """
    OlmoEarth model wrapper for Sentinel-2 multi-spectral data embedding and search.

    This class provides a unified interface for:
    - Loading OlmoEarth models from HuggingFace (Nano/Tiny/Base/Large)
    - Encoding images into embeddings using the OlmoEarth encoder
    - Loading pre-computed embeddings
    - Searching similar images using cosine similarity

    OlmoEarth is a multi-modal, spatio-temporal foundation model. This wrapper
    adapts it for single-timestep (T=1) Sentinel-2 L2A inputs to be compatible
    with the MajorTOM Core-S2L2A-249k dataset.
    """

    # MajorTOM band order -> OlmoEarth band order reorder indices
    # MajorTOM:  [B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12]
    # OlmoEarth: [B02, B03, B04, B08, B05, B06, B07, B8A, B11, B12, B01, B09]
    _BAND_REORDER = (1, 2, 3, 7, 4, 5, 6, 8, 10, 11, 0, 9)

    def __init__(
        self,
        ckpt_path=None,
        model_size="nano",
        embedding_path=None,
        device=None,
    ):
        """
        Initialize the OlmoEarthModel.

        Args:
            ckpt_path (str): Ignored for OlmoEarth; weights are auto-downloaded
                from HuggingFace via olmoearth-pretrain-minimal.
            model_size (str): One of "nano", "tiny", "base", "large".
            embedding_path (str): Path to pre-computed embeddings parquet file.
            device (str): Device to use ('cuda', 'cpu', or None for auto-detection).
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_size = model_size.lower()
        self.ckpt_path = ckpt_path
        self.embedding_path = embedding_path

        self.model = None
        self.normalizer = None
        self.df_embed = None
        self.image_embeddings = None

        # Sentinel-2 L2A bands (MajorTOM order)
        self.bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
        self.requires_multiband = True  # Model needs multi-spectral Sentinel-2 input
        self.size = (128, 128)

        self.load_model()
        if self.embedding_path is not None:
            self.load_embeddings()

    def load_model(self):
        """Load OlmoEarth model from HuggingFace via olmoearth-pretrain-minimal."""
        try:
            from olmoearth_pretrain_minimal import ModelID, Normalizer, load_model_from_id
            from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import Modality

            size_to_id = {
                "nano": ModelID.OLMOEARTH_V1_NANO,
                "tiny": ModelID.OLMOEARTH_V1_TINY,
                "base": ModelID.OLMOEARTH_V1_BASE,
                "large": ModelID.OLMOEARTH_V1_LARGE,
            }

            if self.model_size not in size_to_id:
                raise ValueError(
                    f"Unknown OlmoEarth model_size: {self.model_size}. "
                    f"Choose from {list(size_to_id.keys())}"
                )

            model_id = size_to_id[self.model_size]
            print(f"Loading OlmoEarth {self.model_size} from HuggingFace...")
            self.model = load_model_from_id(model_id, load_weights=True)
            self.model = self.model.to(self.device)
            self.model.eval()

            self.normalizer = Normalizer(std_multiplier=2.0)
            # Cache Modality enum for reuse
            self._modality = Modality.SENTINEL2_L2A

            print(f"OlmoEarth {self.model_size} loaded on {self.device}")
        except Exception as e:
            print(f"Error loading OlmoEarth model: {e}")

    def load_embeddings(self):
        """Load pre-computed embeddings from parquet file."""
        print(f"Loading OlmoEarth embeddings from {self.embedding_path}...")
        try:
            if not os.path.exists(self.embedding_path):
                print(f"Warning: Embedding file not found at {self.embedding_path}")
                return

            self.df_embed = pq.read_table(self.embedding_path).to_pandas()

            image_embeddings_np = np.stack(self.df_embed["embedding"].values)
            self.image_embeddings = (
                torch.from_numpy(image_embeddings_np).to(self.device).float()
            )
            # NOTE: Official tutorial does NOT L2-normalize MEAN-pooled embeddings.
            # Keeping raw dot-product for search consistency with allenai/olmoearth_ml4rs_tutorial.
            print(f"OlmoEarth Data loaded: {len(self.df_embed)} records")
        except Exception as e:
            print(f"Error loading OlmoEarth embeddings: {e}")

    def _read_multiband_from_tiff(self, tiff_path):
        """Read a multi-band GeoTIFF file and return (C, H, W) torch tensor."""
        import rasterio
        with rasterio.open(tiff_path) as src:
            data = src.read().astype(np.float32)  # (C, H, W)
            if data.shape[0] != 12:
                print(f"Warning: Expected 12 bands, got {data.shape[0]} in {tiff_path}")
            if data.shape[0] < 12:
                padded = np.zeros((12, data.shape[1], data.shape[2]), dtype=np.float32)
                padded[:data.shape[0]] = data
                data = padded
            elif data.shape[0] > 12:
                data = data[:12]
        return torch.from_numpy(data).unsqueeze(0)  # (1, C, H, W)

    def _read_multiband_from_dir(self, dir_path):
        """Read single-band GeoTIFFs from directory (B01.tif, B02.tif, ...) and stack."""
        import rasterio
        bands = self.bands
        img = []
        for band in bands:
            band_path = os.path.join(dir_path, f"{band}.tif")
            if not os.path.exists(band_path):
                band_path_alt = os.path.join(dir_path, f"{band.lower()}.tif")
                if os.path.exists(band_path_alt):
                    band_path = band_path_alt
                else:
                    raise FileNotFoundError(f"Band file not found: {band_path}")
            with rasterio.open(band_path) as src:
                img.append(src.read()[0].astype(np.float32))
        data = np.stack(img, axis=0)  # (12, H, W)
        return torch.from_numpy(data).unsqueeze(0)  # (1, C, H, W)

    def _prepare_input(self, tensor):
        """
        Convert a torch.Tensor from MajorTOM format to OlmoEarth format.

        Args:
            tensor (torch.Tensor): Shape (N, C, H, W) where C=12 in MajorTOM order.

        Returns:
            torch.Tensor: Normalized tensor of shape (N, H, W, T=1, C=12) in
                OlmoEarth band order.
        """
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)

        # Reorder bands: MajorTOM -> OlmoEarth
        tensor = tensor[:, self._BAND_REORDER, :, :]

        # Convert to (N, H, W, C) numpy for normalizer
        np_tensor = tensor.permute(0, 2, 3, 1).cpu().numpy()
        # Add time dimension: (N, H, W, T=1, C)
        np_tensor = np_tensor.reshape(
            np_tensor.shape[0], np_tensor.shape[1], np_tensor.shape[2], 1, np_tensor.shape[3]
        )

        # Normalize
        normalized = self.normalizer.normalize(self._modality, np_tensor)
        return torch.from_numpy(normalized).float()

    def _create_sample(self, normalized_tensor):
        """
        Build MaskedOlmoEarthSample from normalized tensor.

        Args:
            normalized_tensor (torch.Tensor): Shape (N, H, W, T=1, C=12).

        Returns:
            MaskedOlmoEarthSample
        """
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import (
            MaskedOlmoEarthSample,
        )

        batch_size = normalized_tensor.shape[0]
        h, w = normalized_tensor.shape[1], normalized_tensor.shape[2]
        num_bandsets = 3  # Determined empirically from OlmoEarth tokenization config

        timestamps = torch.zeros(batch_size, 1, 3, dtype=torch.long, device=self.device)
        # Use a default month index (e.g., 6 for July) since single-timestep
        timestamps[:, 0, 1] = 6

        return MaskedOlmoEarthSample(
            timestamps=timestamps,
            sentinel2_l2a=normalized_tensor.to(self.device),
            sentinel2_l2a_mask=torch.zeros(
                batch_size, h, w, 1, num_bandsets, dtype=torch.long, device=self.device
            ),
        )

    def encode_image(self, image, preprocess_s2=True, normalize=True):
        """
        Encode an image into a feature embedding.

        Args:
            image (str, Path, PIL.Image, torch.Tensor, or np.ndarray): Input image.
                - str / Path: Path to a GeoTIFF file (single-band or multi-band).
                    For multi-band GeoTIFF with 12 bands, bands are read in file order
                    and assumed to be in MajorTOM order [B01..B12].
                    For single-band files, a directory path can be provided containing
                    files named B01.tif, B02.tif, etc.
                - PIL.Image: RGB image; adapted to 12 bands (R->B04, G->B03, B->B02).
                - torch.Tensor: Image tensor with shape [C, H, W] or [N, C, H, W].
                - np.ndarray: Image array with shape [H, W, C] or [N, H, W, C].
            preprocess_s2 (bool): Ignored for OlmoEarth; kept for API consistency.
            normalize (bool): Ignored for OlmoEarth; kept for API consistency.

        Returns:
            torch.Tensor: Normalized embedding vector with shape [embedding_dim] or
                [N, embedding_dim] for batched input.
        """
        if self.model is None:
            return None

        try:
            if isinstance(image, (str, os.PathLike)):
                image_path = str(image)
                if os.path.isdir(image_path):
                    # Directory of single-band GeoTIFFs
                    img = self._read_multiband_from_dir(image_path)
                else:
                    # Single multi-band GeoTIFF
                    img = self._read_multiband_from_tiff(image_path)
                return self.encode_image(img)

            if isinstance(image, torch.Tensor):
                normalized = self._prepare_input(image)
                sample = self._create_sample(normalized)
                with torch.no_grad():
                    output = self.model.encoder(
                        sample, patch_size=8, input_res=10, fast_pass=True
                    )
                embedding = output["tokens_and_masks"].pool_unmasked_tokens(
                    pooling_type=PoolingType.MEAN
                )
                return embedding

            elif isinstance(image, np.ndarray):
                # Convert to torch tensor first
                if image.ndim == 3:
                    image = image.transpose(2, 0, 1)  # HWC -> CHW
                    image = torch.from_numpy(image).unsqueeze(0)
                elif image.ndim == 4:
                    image = image.transpose(0, 3, 1, 2)  # NHWC -> NCHW
                    image = torch.from_numpy(image)
                else:
                    raise ValueError(f"Unsupported ndarray shape: {image.shape}")
                return self.encode_image(image)

            elif isinstance(image, Image.Image):
                image = image.convert("RGB")
                # Resize to model input size
                image = image.resize(self.size)
                img_np = np.array(image).astype(np.float32)  # (H, W, 3)

                # Construct 12 channels in MajorTOM order.
                # _prepare_input will reorder to OlmoEarth format.
                # MajorTOM: [B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12]
                # RGB maps to B04(R), B03(G), B02(B)
                input_tensor = np.zeros(
                    (12, self.size[0], self.size[1]), dtype=np.float32
                )
                input_tensor[1] = img_np[:, :, 2]  # Blue -> B02 (MajorTOM index 1)
                input_tensor[2] = img_np[:, :, 1]  # Green -> B03 (MajorTOM index 2)
                input_tensor[3] = img_np[:, :, 0]  # Red -> B04 (MajorTOM index 3)

                input_tensor = torch.from_numpy(input_tensor).unsqueeze(0)
                return self.encode_image(input_tensor)
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")

        except Exception as e:
            print(f"Error encoding image in OlmoEarth: {e}")
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

        Args:
            input (torch.Tensor): Raw Sentinel-2 image tensor with shape
                [N, C, H, W] or [C, H, W], where C=12 (MajorTOM band order).

        Returns:
            torch.Tensor: Normalized embedding vector with shape [N, embedding_dim]
                or [embedding_dim].
        """
        return self.encode_image(input, preprocess_s2=True, normalize=False)

    def encode_text(self, text):
        """
        Encode a text query into a feature embedding.

        OlmoEarth does not support text encoding.
        """
        return None

    def encode_location(self, lat, lon):
        """
        Encode a (latitude, longitude) pair into a vector.

        OlmoEarth does not support location encoding.
        """
        return None

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
        """
        if self.image_embeddings is None:
            print("Embeddings not loaded!")
            return None, None, None

        try:
            query_features = query_features.float().to(self.device)
            # NOTE: Official tutorial uses raw dot-product for classification.
            # For retrieval, we L2-normalize both embeddings and query to compute
            # cosine similarity, eliminating geographic bias from embedding norm
            # variations (e.g. polar regions have systematically higher norms).
            image_embeddings_norm = F.normalize(self.image_embeddings, dim=-1)
            query_features_norm = F.normalize(query_features, dim=-1)

            similarity = (image_embeddings_norm @ query_features_norm.T).squeeze()
            similarities = similarity.detach().cpu().numpy()

            if top_percent is not None:
                k = int(len(similarities) * top_percent)
                if k < 1:
                    k = 1
                threshold = np.partition(similarities, -k)[-k]

            mask = similarities >= threshold
            filtered_indices = np.where(mask)[0]
            top_indices = np.argsort(similarities)[-top_k:][::-1]

            return similarities, filtered_indices, top_indices

        except Exception as e:
            print(f"Error during search: {e}")
            return None, None, None
