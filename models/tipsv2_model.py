import os

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import AutoModel


def _patch_hf_hub_download_for_local_repo():
    """
    TIPSv2's trust_remote_code module calls hf_hub_download(repo_id, *.py) to
    load sibling files. When repo_id is a local directory, the default
    hf_hub_download rejects it. This patch redirects those calls to the local
    files if they exist.
    """
    try:
        import huggingface_hub
    except Exception:
        return

    if getattr(huggingface_hub, "_tipsv2_download_patched", False):
        return

    _orig = huggingface_hub.hf_hub_download

    def _patched_hf_hub_download(repo_id, filename, **kwargs):
        if isinstance(filename, str):
            candidates = [repo_id]
            if not os.path.isabs(repo_id):
                # If repo_id is relative, also try resolving against the working directory.
                candidates.append(os.path.join(os.getcwd(), repo_id))
            for base in candidates:
                if os.path.isdir(base):
                    local_path = os.path.join(base, filename)
                    if os.path.exists(local_path):
                        return local_path
        return _orig(repo_id, filename, **kwargs)

    huggingface_hub.hf_hub_download = _patched_hf_hub_download
    huggingface_hub._tipsv2_download_patched = True


def _ensure_tipsv2_local_tokenizer():
    """Ensure TIPSv2's trust_remote_code module uses the patched hf_hub_download.

    The module may cache the downloader at import time, so this forces the patched
    version into any already-loaded TIPSv2 modules before encode_text is called.
    """
    try:
        import sys

        import huggingface_hub

        downloader = huggingface_hub.hf_hub_download
        for name, mod in list(sys.modules.items()):
            if "tips" in name.lower():
                if hasattr(mod, "hf_hub_download"):
                    mod.hf_hub_download = downloader
    except Exception:
        pass


class TIPSv2Model:
    """
    Google TIPSv2 wrapper for EarthEmbeddingExplorer.

    Assumes the official Hugging Face model interface:
        model = AutoModel.from_pretrained(..., trust_remote_code=True)
        out = model.encode_image(pixel_values)  # out.cls_token[:, 0, :] for global embedding
        text_features = model.encode_text([text])

    Image preprocessing:
        - PIL / numpy inputs are treated as 8-bit RGB and scaled to [0, 1].
        - torch.Tensor inputs are treated as Sentinel-2 reflectance (B04, B03, B02)
          and converted with the same convention as SigLIP/FarSLIP:
              (2.5 * reflectance / 10000).clip(0, 1)
        - No ImageNet normalization is applied (TIPSv2 official requirement).
    """

    requires_multiband = False

    def __init__(
        self,
        ckpt_path: str | None = None,
        model_name: str = "google/tipsv2-b14",
        embedding_path: str | None = None,
        device: str | None = None,
        revision: str | None = None,
        image_size: int = 448,
        use_fp16: bool = True,
    ):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt_path = ckpt_path
        self.model_name = model_name
        self.embedding_path = embedding_path
        self.revision = revision
        self.image_size = image_size
        self.use_fp16 = use_fp16 and self.device.startswith("cuda")

        # EEE-compatible metadata
        self.bands = ["B04", "B03", "B02"]
        self.size = (self.image_size, self.image_size)

        self.model = None
        self.df_embed = None
        self.image_embeddings = None

        # PIL preprocessing: resize -> [0, 1], no ImageNet normalization
        self._pil_preprocess = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ]
        )

        self.load_model()
        if self.embedding_path is not None:
            self.load_embeddings()

    def _resolve_source(self) -> str:
        """Return a local weights directory, downloading first if needed.

        A valid local ``ckpt_path`` wins. Otherwise weights are fetched
        according to the DOWNLOAD_ENDPOINT env var (same pattern as the other
        model wrappers): Hugging Face returns the repo id for
        ``from_pretrained``; ModelScope endpoints snapshot the repo and return
        the local cache dir.
        """
        if self.ckpt_path and os.path.exists(self.ckpt_path):
            return self.ckpt_path

        endpoint = os.getenv("DOWNLOAD_ENDPOINT", "modelscope.cn")
        if endpoint in ("huggingface", "hf"):
            return self.model_name
        if endpoint in ("modelscope.ai", "ai"):
            os.environ["MODELSCOPE_DOMAIN"] = "www.modelscope.ai"
        from modelscope.hub.snapshot_download import snapshot_download

        print(f"Downloading TIPSv2 weights from ModelScope ({endpoint})...")
        return snapshot_download(repo_id=self.model_name)

    def load_model(self):
        """Load the TIPSv2 model from a local directory or Hugging Face repo_id."""
        source = self._resolve_source()

        kwargs = {"trust_remote_code": True}
        if self.revision:
            kwargs["revision"] = self.revision
        if self.use_fp16:
            kwargs["dtype"] = torch.float16

        # Allow loading from a local directory that contains trust_remote_code files.
        _patch_hf_hub_download_for_local_repo()

        try:
            self.model = AutoModel.from_pretrained(source, **kwargs)
            self.model = self.model.to(self.device)
            self.model.eval()
            print(f"TIPSv2 model loaded from {source} on {self.device}")
        except Exception as e:
            print(f"Error loading TIPSv2 model from {source}: {e}")
            raise

    def load_embeddings(self):
        """Load pre-computed embeddings from a parquet file."""
        print(f"Loading TIPSv2 embeddings from {self.embedding_path} ...")
        try:
            if not os.path.exists(self.embedding_path):
                print(f"Warning: Embedding file not found at {self.embedding_path}")
                return

            self.df_embed = pq.read_table(self.embedding_path).to_pandas()
            image_embeddings_np = np.stack(self.df_embed["embedding"].values)
            self.image_embeddings = torch.from_numpy(image_embeddings_np).to(self.device).float()
            self.image_embeddings = F.normalize(self.image_embeddings, dim=-1)
            print(f"TIPSv2 Data loaded: {len(self.df_embed)} records")
        except Exception as e:
            print(f"Error loading TIPSv2 embeddings: {e}")

    def preprocess_s2(self, input_data: torch.Tensor) -> torch.Tensor:
        """Convert raw Sentinel-2 reflectance to [0, 1] RGB."""
        return (2.5 * (input_data / 1e4)).clip(0, 1)

    def prepare_index_aligned_image(self, image: np.ndarray) -> np.ndarray:
        """Match the fragment resize used to build the published 448px index.

        ``generate_embeddings._prepare_single_fragment_image`` uses OpenCV
        nearest-neighbor interpolation when a 384px source chip is expanded to
        the configured 448px TIPSv2 fragment.  Apply the same operation to raw
        online RGB queries before the model's tensor preprocessing.
        """
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(f"Expected an HWC RGB query, got shape {array.shape}")
        if tuple(array.shape[:2]) == self.size:
            return array
        return cv2.resize(array, (self.size[1], self.size[0]), interpolation=cv2.INTER_NEAREST)

    def _prepare_tensor(self, image: torch.Tensor, preprocess_s2: bool = True) -> torch.Tensor:
        """Prepare a torch.Tensor (CHW or NCHW) for TIPSv2 encoding."""
        tensor = image.float().to(self.device)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        # MajorTOM gives B04/B03/B02 as channels; ensure CHW order.
        if tensor.shape[-1] == 3 and tensor.shape[1] != 3:
            tensor = tensor.permute(0, 3, 1, 2)

        if tensor.shape[1] != 3:
            raise ValueError(f"TIPSv2 expects 3 RGB channels, got shape {tuple(tensor.shape)}")

        # Heuristic: if max > 1, assume raw Sentinel-2 DN/reflectance needs scaling.
        if preprocess_s2 and tensor.max() > 1.0:
            tensor = self.preprocess_s2(tensor)
        elif tensor.max() > 1.0:
            tensor = tensor / 255.0

        # Resize to target size if needed.
        if tuple(tensor.shape[-2:]) != self.size:
            tensor = F.interpolate(
                tensor,
                size=self.size,
                mode="bilinear",
                align_corners=False,
            )

        return tensor

    def _prepare_pil(self, image: Image.Image) -> torch.Tensor:
        """Prepare a PIL Image for TIPSv2 encoding."""
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL Image, got {type(image)}")
        return self._pil_preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)

    def _prepare_numpy(self, image: np.ndarray) -> torch.Tensor:
        """Prepare a numpy array (H,W,C or N,H,W,C) for TIPSv2 encoding."""
        arr = np.asarray(image)
        if arr.ndim == 3:
            arr = arr[np.newaxis, ...]
        if arr.shape[-1] != 3:
            raise ValueError(f"TIPSv2 expects H,W,C RGB array, got shape {arr.shape}")
        # ToTensor-like scaling: uint8 -> [0, 1]
        tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).float().to(self.device)
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        if tuple(tensor.shape[-2:]) != self.size:
            tensor = F.interpolate(
                tensor,
                size=self.size,
                mode="bilinear",
                align_corners=False,
            )
        return tensor

    def encode_text(self, text: str) -> torch.Tensor | None:
        """Encode a text query into a normalized feature embedding."""
        if self.model is None or not text:
            return None

        _ensure_tipsv2_local_tokenizer()
        with torch.inference_mode():
            text_features = self.model.encode_text([text])
            if not isinstance(text_features, torch.Tensor):
                text_features = torch.as_tensor(text_features)
            text_features = text_features.to(self.device).float()
            text_features = F.normalize(text_features, dim=-1)
        return text_features

    def encode_image(
        self,
        image: Image.Image | np.ndarray | torch.Tensor,
        preprocess_s2: bool = True,
        normalize: bool = True,
    ) -> torch.Tensor | None:
        """Encode an image into a normalized feature embedding."""
        if self.model is None:
            return None

        if isinstance(image, Image.Image):
            pixel_values = self._prepare_pil(image)
        elif isinstance(image, torch.Tensor):
            pixel_values = self._prepare_tensor(image, preprocess_s2=preprocess_s2)
        elif isinstance(image, np.ndarray):
            pixel_values = self._prepare_numpy(image)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        with torch.inference_mode():
            # Match model dtype when using FP16.
            pixel_values = pixel_values.to(self.model.dtype)
            out = self.model.encode_image(pixel_values)
            # TIPSv2 returns cls_token as global image embedding.
            image_features = out.cls_token[:, 0, :]
            image_features = image_features.float()
            image_features = F.normalize(image_features, dim=-1)

        return image_features

    def __call__(self, input):
        return self.forward(input)

    def forward(self, input):
        """Callable used by MajorTOM_Embedder (expects tensor input)."""
        return self.encode_image(input, preprocess_s2=True)

    def search(self, query_features, top_k=5, top_percent=None, threshold=0.0):
        """Search pre-computed image embeddings using cosine similarity."""
        if self.image_embeddings is None:
            return None, None, None

        query_features = query_features.float()
        probs = (self.image_embeddings @ query_features.T).detach().cpu().numpy().flatten()
        sorted_indices = np.argsort(probs)[::-1]

        if top_percent is not None:
            k = max(1, int(len(probs) * top_percent))
            top_indices = sorted_indices[:k]
            filtered_indices = top_indices
        else:
            mask = probs >= threshold
            filtered_indices = sorted_indices[mask[sorted_indices]]
            top_indices = filtered_indices[:top_k]

        return probs, filtered_indices, top_indices
