import os
import re
from datetime import date, datetime

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image


class OlmoEarthModel:
    """
    OlmoEarth model wrapper for Sentinel-2 multi-spectral data embedding and search.

    This class provides a unified interface for:
    - Loading OlmoEarth v1.2 models from HuggingFace or ModelScope
    - Encoding images into embeddings using the OlmoEarth encoder
    - Loading pre-computed embeddings
    - Searching similar images using cosine similarity

    OlmoEarth is a multi-modal, spatio-temporal foundation model. This wrapper
    adapts it for single-timestep (T=1) Sentinel-2 L2A inputs and preserves the
    acquisition timestamp when it is available in MajorTOM metadata.
    """

    DEFAULT_TIMESTAMP = datetime(2020, 7, 1)

    def __init__(
        self,
        ckpt_path=None,
        model_size="base",
        model_version="v1_2",
        embedding_path=None,
        device=None,
    ):
        """
        Initialize the OlmoEarthModel.

        Args:
            ckpt_path (str): Optional local model directory. Otherwise weights
                are downloaded from the configured endpoint.
            model_size (str): One of "nano", "tiny", "small", "base".
            model_version (str): OlmoEarth release. Defaults to "v1_2".
            embedding_path (str): Path to pre-computed embeddings parquet file.
            device (str): Device to use ('cuda', 'cpu', or None for auto-detection).
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_size = model_size.lower()
        self.model_version = model_version.lower().replace(".", "_")
        self.ckpt_path = ckpt_path
        self.embedding_path = embedding_path

        self.model = None
        self.normalizer = None
        self.df_embed = None
        self.image_embeddings = None

        # OlmoEarth expected band order (used by reorder_multiband in search_engine / callbacks)
        self.bands = ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"]
        self.requires_multiband = True  # Model needs multi-spectral Sentinel-2 input
        self.supports_timestamps = True
        self.size = (128, 128)
        self.native_input_res = 10.0

        self.load_model()
        if self.embedding_path is not None:
            self.load_embeddings()

    def load_model(self):
        """Load OlmoEarth model respecting DOWNLOAD_ENDPOINT."""
        match = re.match(r"^(\d+)\.(\d+)", torch.__version__)
        torch_version = tuple(int(part) for part in match.groups()) if match else (0, 0)
        if torch_version < (2, 7):
            print(f"OlmoEarth requires torch>=2.7, found {torch.__version__}. Skipping model load.")
            return

        if self.model_version != "v1_2":
            raise ValueError(f"Unsupported OlmoEarth version: {self.model_version}; expected v1_2")

        endpoint = os.getenv("DOWNLOAD_ENDPOINT", "modelscope.cn")

        # Determine model source
        if self.ckpt_path is not None and os.path.exists(self.ckpt_path):
            model_path = self.ckpt_path
            print(f"Loading OlmoEarth {self.model_version} {self.model_size} from local path: {model_path}")
        elif endpoint in ("huggingface", "hf"):
            print(f"Loading OlmoEarth {self.model_version} {self.model_size} from HuggingFace...")
            try:
                from olmoearth_pretrain_minimal import ModelID, Normalizer, load_model_from_id
                from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import Modality

                size_to_id = {
                    "nano": ModelID.OLMOEARTH_V1_2_NANO,
                    "tiny": ModelID.OLMOEARTH_V1_2_TINY,
                    "small": ModelID.OLMOEARTH_V1_2_SMALL,
                    "base": ModelID.OLMOEARTH_V1_2_BASE,
                }

                if self.model_size not in size_to_id:
                    raise ValueError(
                        f"Unknown OlmoEarth model_size: {self.model_size}. Choose from {list(size_to_id.keys())}"
                    )

                model_id = size_to_id[self.model_size]
                self.model = load_model_from_id(model_id, load_weights=True)
                self.model = self.model.to(self.device)
                self.model.eval()

                self.normalizer = Normalizer(std_multiplier=2.0)
                self._modality = Modality.SENTINEL2_L2A

                print(f"OlmoEarth {self.model_version} {self.model_size} loaded on {self.device}")
            except Exception as e:
                print(f"Error loading OlmoEarth model: {e}")
            return
        else:
            if endpoint in ("modelscope.ai", "ai"):
                print(f"Loading OlmoEarth {self.model_version} {self.model_size} from ModelScope (modelscope.ai)...")
                os.environ["MODELSCOPE_DOMAIN"] = "www.modelscope.ai"
            else:
                print(f"Loading OlmoEarth {self.model_version} {self.model_size} from ModelScope (modelscope.cn)...")
            repo_id = self._modelscope_repo_id(endpoint)

            from modelscope.hub.snapshot_download import snapshot_download

            model_path = snapshot_download(repo_id=repo_id)
            print(f"OlmoEarth weights cached at: {model_path}")

        # Load from local path (covers local ckpt_path and ModelScope cache)
        try:
            from olmoearth_pretrain_minimal import Normalizer, load_model_from_path
            from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import Modality

            self.model = load_model_from_path(model_path, load_weights=True)
            self.model = self.model.to(self.device)
            self.model.eval()

            self.normalizer = Normalizer(std_multiplier=2.0)
            self._modality = Modality.SENTINEL2_L2A

            print(f"OlmoEarth {self.model_version} {self.model_size} loaded on {self.device}")
        except Exception as e:
            print(f"Error loading OlmoEarth model: {e}")

    def _modelscope_repo_id(self, endpoint):
        namespace = "Major-TOM" if endpoint in ("modelscope.ai", "ai") else "allenai"
        return f"{namespace}/OlmoEarth-{self.model_version}-{self.model_size.capitalize()}"

    def load_embeddings(self):
        """Load pre-computed embeddings from parquet file."""
        print(f"Loading OlmoEarth embeddings from {self.embedding_path}...")
        try:
            if not os.path.exists(self.embedding_path):
                print(f"Warning: Embedding file not found at {self.embedding_path}")
                return

            self.df_embed = pq.read_table(self.embedding_path).to_pandas()

            image_embeddings_np = np.stack(self.df_embed["embedding"].values)
            self.image_embeddings = torch.from_numpy(image_embeddings_np).to(self.device).float()
            # NOTE: Official tutorial does NOT L2-normalize MEAN-pooled embeddings.
            # Keeping raw dot-product for search consistency with allenai/olmoearth_ml4rs_tutorial.
            print(f"OlmoEarth Data loaded: {len(self.df_embed)} records")
        except Exception as e:
            print(f"Error loading OlmoEarth embeddings: {e}")

    def _prepare_input(self, tensor):
        """
        Convert an OlmoEarth-ordered torch.Tensor to normalized model input.

        Args:
            tensor (torch.Tensor): Shape (N, C, H, W), with C=12 ordered as
                ``self.bands``.

        Returns:
            torch.Tensor: Normalized tensor of shape (N, H, W, T=1, C=12) in
                OlmoEarth band order.
        """
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)

        # Convert to float32 to avoid dtype issues (e.g. UInt16 from Sentinel-2 raw data)
        tensor = tensor.float()

        # Convert to (N, H, W, C) numpy for normalizer
        np_tensor = tensor.permute(0, 2, 3, 1).cpu().numpy()
        # Add time dimension: (N, H, W, T=1, C)
        np_tensor = np_tensor.reshape(np_tensor.shape[0], np_tensor.shape[1], np_tensor.shape[2], 1, np_tensor.shape[3])

        # Normalize
        normalized = self.normalizer.normalize(self._modality, np_tensor)
        return torch.from_numpy(normalized).float()

    def _effective_input_res(self, height, width):
        """Return GSD after resizing a source chip to the encoder input size."""
        scale_h = height / self.size[0]
        scale_w = width / self.size[1]
        return self.native_input_res * (scale_h + scale_w) / 2

    @classmethod
    def _parse_timestamp(cls, value):
        """Convert common MajorTOM timestamps to (day, zero-based month, year)."""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            parsed = cls.DEFAULT_TIMESTAMP
        elif isinstance(value, datetime):
            parsed = value
        elif isinstance(value, date):
            parsed = datetime.combine(value, datetime.min.time())
        elif isinstance(value, np.datetime64):
            parsed = (
                cls.DEFAULT_TIMESTAMP
                if np.isnat(value)
                else datetime.fromisoformat(np.datetime_as_string(value, unit="s"))
            )
        elif hasattr(value, "to_pydatetime"):
            candidate = value.to_pydatetime()
            parsed = candidate if isinstance(candidate, datetime) else cls.DEFAULT_TIMESTAMP
        else:
            text = value.decode() if isinstance(value, bytes) else str(value).strip()
            parsed = None
            if not text or text.lower() in {"nat", "nan", "none"}:
                return cls._parse_timestamp(None)
            for fmt in ("%Y%m%dT%H%M%S", "%Y%m%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
                try:
                    parsed = datetime.strptime(text[: len(datetime.now().strftime(fmt))], fmt)
                    break
                except ValueError:
                    continue
            if parsed is None:
                try:
                    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
                except ValueError:
                    parsed = cls.DEFAULT_TIMESTAMP

        return parsed.day, parsed.month - 1, parsed.year

    @classmethod
    def _prepare_timestamps(cls, timestamps, batch_size, device):
        """Build a [B, 1, 3] OlmoEarth timestamp tensor."""
        if torch.is_tensor(timestamps):
            tensor = timestamps.to(device=device, dtype=torch.long)
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(1)
            if tensor.shape != (batch_size, 1, 3):
                raise ValueError(f"Expected timestamps shape {(batch_size, 1, 3)}, got {tuple(tensor.shape)}")
            return tensor

        if timestamps is None or isinstance(timestamps, (str, bytes, datetime, date, np.datetime64)):
            values = [timestamps] * batch_size
        else:
            values = list(timestamps)
            if len(values) != batch_size:
                raise ValueError(f"Expected {batch_size} timestamps, got {len(values)}")

        components = [cls._parse_timestamp(value) for value in values]
        return torch.tensor(components, dtype=torch.long, device=device).unsqueeze(1)

    def _create_sample(self, normalized_tensor, timestamps=None):
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
        num_bandsets = self.model.encoder.patch_embeddings.tokenization_config.get_num_bandsets(self._modality.name)
        timestamp_tensor = self._prepare_timestamps(timestamps, batch_size, self.device)

        return MaskedOlmoEarthSample(
            timestamps=timestamp_tensor,
            sentinel2_l2a=normalized_tensor.to(self.device),
            sentinel2_l2a_mask=torch.zeros(batch_size, h, w, 1, num_bandsets, dtype=torch.long, device=self.device),
        )

    def encode_image(self, image, preprocess_s2=True, normalize=True, timestamp=None):
        """
        Encode an image into a feature embedding.

        Args:
            image (PIL.Image, torch.Tensor, or np.ndarray): Input image.
                - PIL.Image: RGB image; adapted to 12 bands (R->B04, G->B03, B->B02).
                - torch.Tensor: Image tensor with shape [C, H, W] or [N, C, H, W].
                - np.ndarray: Image array with shape [H, W, C] or [N, H, W, C].
            preprocess_s2 (bool): Ignored for OlmoEarth; kept for API consistency.
            normalize (bool): Ignored for OlmoEarth; kept for API consistency.
            timestamp: Scalar acquisition time or one timestamp per batch item.
                Missing values use 2020-07-01, the midpoint of the pretraining period.

        Returns:
            torch.Tensor: Embedding vector with shape [embedding_dim] or
                [N, embedding_dim] for batched input.
        """
        if self.model is None:
            return None

        try:
            if isinstance(image, torch.Tensor):
                if image.dim() == 3:
                    image = image.unsqueeze(0)
                source_h, source_w = image.shape[-2:]
                input_res = self._effective_input_res(source_h, source_w)
                # Resize to model input size if needed
                if image.shape[-2:] != self.size:
                    image = F.interpolate(image.float(), size=self.size, mode="bilinear", align_corners=False)
                normalized = self._prepare_input(image)
                sample = self._create_sample(normalized, timestamps=timestamp)
                from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn.flexi_vit import PoolingType

                with torch.no_grad():
                    output = self.model.encoder(sample, patch_size=8, input_res=input_res, fast_pass=True)
                embedding = output["tokens_and_masks"].pool_unmasked_tokens(pooling_type=PoolingType.MEAN)
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
                return self.encode_image(image, timestamp=timestamp)

            elif isinstance(image, Image.Image):
                image = image.convert("RGB")
                img_np = np.array(image).astype(np.float32)  # (H, W, 3)

                # Construct 12 channels directly in OlmoEarth order.
                input_tensor = np.zeros((12, img_np.shape[0], img_np.shape[1]), dtype=np.float32)
                input_tensor[0] = img_np[:, :, 2]  # Blue -> B02
                input_tensor[1] = img_np[:, :, 1]  # Green -> B03
                input_tensor[2] = img_np[:, :, 0]  # Red -> B04

                input_tensor = torch.from_numpy(input_tensor).unsqueeze(0)
                return self.encode_image(input_tensor, timestamp=timestamp)
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")

        except Exception as e:
            print(f"Error encoding image in OlmoEarth: {e}")
            import traceback

            traceback.print_exc()
            return None

    def __call__(self, input, timestamps=None):
        """
        Callable wrapper that delegates to forward().

        Args:
            input (torch.Tensor): Raw Sentinel-2 image tensor.

        Returns:
            torch.Tensor: Normalized embedding vector.
        """
        return self.forward(input, timestamps=timestamps)

    def forward(self, input, timestamps=None):
        """
        Forward pass for compatibility with MajorTOM_Embedder.

        Args:
            input (torch.Tensor): Raw Sentinel-2 image tensor with shape
                [N, C, H, W] or [C, H, W], where C=12 in ``self.bands`` order.

        Returns:
            torch.Tensor: Normalized embedding vector with shape [N, embedding_dim]
                or [embedding_dim].
        """
        return self.encode_image(input, preprocess_s2=True, normalize=False, timestamp=timestamps)

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

            similarity = (image_embeddings_norm @ query_features_norm.T).squeeze(-1)
            similarities = similarity.detach().cpu().numpy()

            sorted_indices = np.argsort(similarities)[::-1]
            if top_percent is not None:
                k = max(1, int(len(similarities) * top_percent))
                filtered_indices = sorted_indices[:k]
                top_indices = filtered_indices
            else:
                mask = similarities >= threshold
                filtered_indices = sorted_indices[mask[sorted_indices]]
                top_indices = filtered_indices[:top_k]

            return similarities, filtered_indices, top_indices

        except Exception as e:
            print(f"Error during search: {e}")
            return None, None, None
