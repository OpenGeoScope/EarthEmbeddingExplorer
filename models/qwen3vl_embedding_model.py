import importlib.util
import os
from typing import List, Optional, Union

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image


# Fixed English retrieval instruction for text queries (instruction-aware embedding).
# Image documents use the embedder's own default instruction ("Represent the user's input.")
# so that offline (generation) and online (query) image embeddings stay consistent.
DEFAULT_TEXT_INSTRUCTION = (
    "Retrieve Sentinel-2 satellite image patches that match the user's "
    "natural-language description of Earth observation scenes."
)


def _load_embedder_class(scripts_path: str):
    """Load the official ``Qwen3VLEmbedder`` from a bundled ``qwen3_vl_embedding.py``.

    The model snapshot ships the official embedder code under ``<ckpt>/scripts/``,
    so the wrapper stays self-contained inside the model cache and does not depend
    on an externally cloned repo at runtime.
    """
    file_path = os.path.join(scripts_path, "qwen3_vl_embedding.py")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"qwen3_vl_embedding.py not found at {file_path}")
    spec = importlib.util.spec_from_file_location("qwen3_vl_embedding_eee", file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Qwen3VLEmbedder


class Qwen3VLEmbeddingModel:
    """Qwen3-VL-Embedding-2B wrapper for EarthEmbeddingExplorer (RGB, first version).

    A general multimodal embedding model used here as an RGB image-text model:
        - text query  -> encode_text(str)  -> [1, 2048] (L2-normalized)
        - image       -> encode_image(PIL/ndarray/tensor) -> [N, 2048] (L2-normalized)

    Image preprocessing (kept identical to SigLIP/TIPSv2 for a fair comparison):
        - torch.Tensor inputs are treated as Sentinel-2 reflectance (B04, B03, B02)
          and converted with ``(2.5 * reflectance / 10000).clip(0, 1)`` then to uint8 RGB.
        - PIL / numpy inputs are treated as 8-bit RGB.
        - The Qwen3-VL processor performs its own internal resize (min/max pixels).

    Does NOT support location encoding (returns ``None``); mixed search is out of scope
    for the first version.
    """

    requires_multiband = False

    def __init__(
        self,
        ckpt_path: Optional[str] = None,
        model_name: str = "Qwen/Qwen3-VL-Embedding-2B",
        embedding_path: Optional[str] = None,
        device: Optional[str] = None,
        image_size: int = 384,
        repo_path: Optional[str] = None,
        text_instruction: Optional[str] = None,
        image_instruction: Optional[str] = None,
        use_bf16: bool = True,
        warmup_runs: int = 1,
        warmup_batch: int = 8,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
    ):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt_path = ckpt_path
        self.model_name = model_name
        self.embedding_path = embedding_path
        self.image_size = image_size
        self.repo_path = repo_path
        self.use_bf16 = use_bf16 and str(self.device).startswith("cuda")
        self.warmup_runs = warmup_runs
        self.warmup_batch = warmup_batch
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.text_instruction = text_instruction or DEFAULT_TEXT_INSTRUCTION
        # None -> use the embedder default ("Represent the user's input.")
        self.image_instruction = image_instruction

        # EEE-compatible metadata
        self.bands = ["B04", "B03", "B02"]
        self.size = (self.image_size, self.image_size)
        self.embedding_dim = 2048

        self.model = None
        self.df_embed = None
        self.image_embeddings = None

        self.load_model()
        if self.embedding_path is not None:
            self.load_embeddings()

    def _resolve_source(self) -> str:
        """Return local model directory or HF/ModelScope repo id."""
        if self.ckpt_path:
            return self.ckpt_path
        return self.model_name

    def _resolve_scripts_path(self) -> str:
        """Locate the directory that contains the official ``qwen3_vl_embedding.py``."""
        if self.repo_path:
            return self.repo_path
        if self.ckpt_path:
            bundled = os.path.join(self.ckpt_path, "scripts")
            if os.path.exists(os.path.join(bundled, "qwen3_vl_embedding.py")):
                return bundled
        raise FileNotFoundError(
            "Cannot locate Qwen3VLEmbedder code. Set 'repo_path' in the config to the "
            "directory containing qwen3_vl_embedding.py."
        )

    def load_model(self):
        """Load the Qwen3-VL-Embedding model via the official ``Qwen3VLEmbedder``."""
        source = self._resolve_source()
        scripts_path = self._resolve_scripts_path()
        embedder_cls = _load_embedder_class(scripts_path)

        kwargs = {}
        if self.use_bf16:
            kwargs["torch_dtype"] = torch.bfloat16
        if self.min_pixels is not None:
            kwargs["min_pixels"] = self.min_pixels
        if self.max_pixels is not None:
            kwargs["max_pixels"] = self.max_pixels

        try:
            self.model = embedder_cls(model_name_or_path=source, **kwargs)
            self._warmup_cudnn()
            print(f"Qwen3VL model loaded from {source} on {self.device}")
        except Exception as e:
            print(f"Error loading Qwen3VL model from {source}: {e}")
            raise

    def _warmup_cudnn(self):
        """Warm up cuDNN with a fixed-resolution dummy batch.

        Qwen3-VL's 3D patch-embedding conv needs several runs at a fixed input
        shape before cuDNN caches a fast algorithm. Without warmup, the first
        few real batches can be 5-10x slower, dominating embedding generation.
        The batch size and run count are configurable because the optimal
        warmup cost/speed trade-off depends on the use case (online vs batch).
        """
        try:
            from PIL import Image
            b = max(1, self.warmup_batch)
            dummy = [Image.new("RGB", (self.image_size, self.image_size), color=(128, 128, 128))] * b
            inputs = [{"image": im} for im in dummy]
            with torch.inference_mode():
                for _ in range(max(0, self.warmup_runs)):
                    _ = self.model.process(inputs)
            torch.cuda.synchronize() if self.device.startswith("cuda") else None
            print(f"Qwen3VL cuDNN warmup done (batch={b}, runs={self.warmup_runs})")
        except Exception as e:
            print(f"Qwen3VL warmup skipped: {e}")

    def load_embeddings(self):
        """Load pre-computed embeddings from a parquet file."""
        print(f"Loading Qwen3VL embeddings from {self.embedding_path} ...")
        try:
            if not os.path.exists(self.embedding_path):
                print(f"Warning: Embedding file not found at {self.embedding_path}")
                return

            self.df_embed = pq.read_table(self.embedding_path).to_pandas()
            image_embeddings_np = np.stack(self.df_embed["embedding"].values)
            self.image_embeddings = (
                torch.from_numpy(image_embeddings_np).to(self.device).float()
            )
            self.image_embeddings = F.normalize(self.image_embeddings, dim=-1)
            print(f"Qwen3VL Data loaded: {len(self.df_embed)} records")
        except Exception as e:
            print(f"Error loading Qwen3VL embeddings: {e}")

    def preprocess_s2(self, input_data: torch.Tensor) -> torch.Tensor:
        """Convert raw Sentinel-2 reflectance to [0, 1] RGB (same as SigLIP/TIPSv2)."""
        return (2.5 * (input_data / 1e4)).clip(0, 1)

    def _tensor_to_pil_list(self, image: torch.Tensor, preprocess_s2: bool = True) -> List[Image.Image]:
        """Prepare a torch.Tensor (CHW or NCHW) into a list of uint8 RGB PIL images."""
        tensor = image.float()
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        # MajorTOM gives B04/B03/B02 as channels; ensure NCHW order.
        if tensor.shape[-1] == 3 and tensor.shape[1] != 3:
            tensor = tensor.permute(0, 3, 1, 2)
        if tensor.shape[1] != 3:
            raise ValueError(f"Qwen3VL expects 3 RGB channels, got shape {tuple(tensor.shape)}")

        # Heuristic: values > 1 mean raw reflectance/DN (needs scaling) vs already [0, 1].
        if preprocess_s2 and tensor.max() > 1.0:
            tensor = self.preprocess_s2(tensor)
        elif tensor.max() > 1.0:
            tensor = tensor / 255.0
        tensor = tensor.clamp(0, 1)

        arr = (tensor * 255).round().to(torch.uint8).cpu().numpy()  # N, C, H, W
        return [Image.fromarray(np.transpose(a, (1, 2, 0))).convert("RGB") for a in arr]

    def _to_pil_rgb(self, image: Union[Image.Image, np.ndarray]) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, np.ndarray):
            arr = image
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return Image.fromarray(arr).convert("RGB")
        raise TypeError(f"Unsupported image type: {type(image)}")

    def _process(self, inputs: List[dict]) -> torch.Tensor:
        """Run the embedder and return a float, L2-normalized tensor on self.device."""
        emb = self.model.process(inputs)  # [N, 2048], already L2-normalized
        if not isinstance(emb, torch.Tensor):
            emb = torch.as_tensor(emb)
        emb = emb.float().to(self.device)
        if emb.ndim == 1:
            emb = emb.unsqueeze(0)
        if emb.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Qwen3VL embedding dim {emb.shape[-1]} != expected {self.embedding_dim}"
            )
        return F.normalize(emb, dim=-1)

    def encode_text(self, text: str) -> Optional[torch.Tensor]:
        """Encode a text query into a normalized [1, 2048] feature embedding."""
        if self.model is None or not text:
            return None
        with torch.inference_mode():
            return self._process([{"text": text, "instruction": self.text_instruction}])

    def encode_image(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor, list, tuple],
        preprocess_s2: bool = True,
        normalize: bool = True,
    ) -> Optional[torch.Tensor]:
        """Encode image(s) into normalized [N, 2048] feature embeddings."""
        if self.model is None:
            return None

        if isinstance(image, torch.Tensor):
            pil_list = self._tensor_to_pil_list(image, preprocess_s2=preprocess_s2)
        elif isinstance(image, Image.Image):
            pil_list = [image.convert("RGB")]
        elif isinstance(image, np.ndarray):
            pil_list = [self._to_pil_rgb(image)]
        elif isinstance(image, (list, tuple)):
            pil_list = [self._to_pil_rgb(im) for im in image]
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        if self.image_instruction:
            inputs = [{"image": im, "instruction": self.image_instruction} for im in pil_list]
        else:
            inputs = [{"image": im} for im in pil_list]

        with torch.inference_mode():
            emb = self._process(inputs)
        # normalize flag kept for interface parity; _process already normalizes.
        return emb if normalize else emb

    def encode_text_and_image(
        self,
        text: str,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        preprocess_s2: bool = True,
    ) -> Optional[torch.Tensor]:
        """Encode a text+image pair into a single joint [1, 2048] embedding.

        This uses Qwen3-VL-Embedding's native mixed-modal input capability:
            model.process([{"text": ..., "image": ..., "instruction": ...}])

        Args:
            text: Text query string.
            image: PIL Image, numpy array, or torch tensor.
            preprocess_s2: Whether to apply Sentinel-2 reflectance scaling for
                torch.Tensor inputs (MajorTOM B04/B03/B02 reflectance).

        Returns:
            L2-normalized [1, 2048] tensor, or None if model not loaded.
        """
        if self.model is None or not text:
            return None

        # Convert image to the same PIL list format used by encode_image.
        if isinstance(image, torch.Tensor):
            pil_list = self._tensor_to_pil_list(image, preprocess_s2=preprocess_s2)
        elif isinstance(image, Image.Image):
            pil_list = [image.convert("RGB")]
        elif isinstance(image, np.ndarray):
            pil_list = [self._to_pil_rgb(image)]
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        if len(pil_list) != 1:
            raise ValueError("encode_text_and_image only supports a single image.")

        inputs = [
            {
                "text": text,
                "image": pil_list[0],
                "instruction": self.text_instruction,
            }
        ]

        with torch.inference_mode():
            return self._process(inputs)

    def encode_location(self, *args, **kwargs):
        """Qwen3-VL-Embedding-2B does not support location encoding."""
        return None

    def __call__(self, input):
        return self.forward(input)

    def forward(self, input):
        """Callable used by MajorTOM_Embedder (expects reflectance tensor input)."""
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
