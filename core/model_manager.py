"""Model initialization and management for EarthEmbeddingExplorer."""

from typing import ClassVar

import torch

from models.clay_model import ClayModel
from models.dinov2_model import DINOv2Model
from models.farslip_model import FarSLIPModel
from models.load_config import load_and_process_config
from models.olmoearth_model import OlmoEarthModel
from models.qwen3vl_embedding_model import Qwen3VLEmbeddingModel
from models.satclip_model import SatCLIPModel
from models.siglip_model import SigLIPModel
from models.tipsv2_model import TIPSv2Model


class ModelManager:
    """Manages model loading and retrieval."""

    MODEL_LOAD_ORDER: ClassVar[tuple[str, ...]] = (
        "DINOv2",
        "SigLIP",
        "TIPSv2",
        "SatCLIP",
        "FarSLIP",
        "Clay",
        "OlmoEarth",
        "Qwen3VL",
    )
    _MODEL_ALIASES: ClassVar[dict[str, str]] = {model_name.lower(): model_name for model_name in MODEL_LOAD_ORDER}

    def __init__(self, device=None, selected_models=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running on device: {self.device}")

        self.config = load_and_process_config()
        print(self.config)

        self.selected_models = self._normalize_selected_models(selected_models)
        self.models = {}
        self._load_all_models()

    @classmethod
    def parse_model_list(cls, value):
        """Parse a comma-separated model list.

        Returns None for empty/all values so callers can keep the default
        behavior of loading every supported model.
        """
        if value is None:
            return None

        if isinstance(value, str):
            parts = [part.strip() for part in value.replace(";", ",").split(",")]
        else:
            parts = [str(part).strip() for part in value]

        requested = [part for part in parts if part]
        if not requested or any(part.lower() == "all" for part in requested):
            return None

        return requested

    @classmethod
    def _normalize_selected_models(cls, selected_models):
        requested = cls.parse_model_list(selected_models)
        if requested is None:
            return list(cls.MODEL_LOAD_ORDER)

        selected = []
        invalid = []
        for model_name in requested:
            canonical_name = cls._MODEL_ALIASES.get(model_name.lower())
            if canonical_name is None:
                invalid.append(model_name)
                continue
            if canonical_name not in selected:
                selected.append(canonical_name)

        if invalid:
            valid = ", ".join(cls.MODEL_LOAD_ORDER)
            invalid_str = ", ".join(invalid)
            raise ValueError(f"Unknown model(s): {invalid_str}. Valid models: {valid}.")

        return selected

    def _load_all_models(self):
        """Load the selected embedding models."""
        print(f"Initializing models: {', '.join(self.selected_models)}")

        loaders = {
            "DINOv2": self._load_dinov2,
            "SigLIP": self._load_siglip,
            "TIPSv2": self._load_tipsv2,
            "SatCLIP": self._load_satclip,
            "FarSLIP": self._load_farslip,
            "Clay": self._load_clay,
            "OlmoEarth": self._load_olmoearth,
            "Qwen3VL": self._load_qwen3vl,
        }
        for model_name in self.selected_models:
            loaders[model_name]()

    def _load_dinov2(self):
        """Load DINOv2 model."""
        try:
            if self.config and "dinov2" in self.config:
                self.models["DINOv2"] = DINOv2Model(
                    ckpt_path=self.config["dinov2"].get("ckpt_path"),
                    embedding_path=self.config["dinov2"].get("embedding_path"),
                    device=self.device,
                )
            else:
                self.models["DINOv2"] = DINOv2Model(device=self.device)
        except Exception as e:
            print(f"Failed to load DINOv2: {e}")

    def _load_siglip(self):
        """Load SigLIP model."""
        try:
            if self.config and "siglip" in self.config:
                self.models["SigLIP"] = SigLIPModel(
                    ckpt_path=self.config["siglip"].get("ckpt_path"),
                    tokenizer_path=self.config["siglip"].get("tokenizer_path"),
                    embedding_path=self.config["siglip"].get("embedding_path"),
                    device=self.device,
                )
            else:
                self.models["SigLIP"] = SigLIPModel(device=self.device)
        except Exception as e:
            print(f"Failed to load SigLIP: {e}")

    def _load_tipsv2(self):
        """Load TIPSv2 model."""
        try:
            if self.config and "tipsv2" in self.config:
                self.models["TIPSv2"] = TIPSv2Model(
                    ckpt_path=self.config["tipsv2"].get("ckpt_path"),
                    model_name=self.config["tipsv2"].get("model_name", "google/tipsv2-b14"),
                    embedding_path=self.config["tipsv2"].get("embedding_path"),
                    revision=self.config["tipsv2"].get("revision"),
                    image_size=self.config["tipsv2"].get("image_size", 448),
                    device=self.device,
                )
            else:
                self.models["TIPSv2"] = TIPSv2Model(device=self.device)
        except Exception as e:
            print(f"Failed to load TIPSv2: {e}")

    def _load_qwen3vl(self):
        """Load Qwen3-VL-Embedding-2B model."""
        try:
            if self.config and "qwen3vl" in self.config:
                self.models["Qwen3VL"] = Qwen3VLEmbeddingModel(
                    ckpt_path=self.config["qwen3vl"].get("ckpt_path"),
                    model_name=self.config["qwen3vl"].get("model_name", "Qwen/Qwen3-VL-Embedding-2B"),
                    embedding_path=self.config["qwen3vl"].get("embedding_path"),
                    image_size=self.config["qwen3vl"].get("image_size", 384),
                    repo_path=self.config["qwen3vl"].get("repo_path"),
                    device=self.device,
                    warmup_runs=self.config["qwen3vl"].get("warmup_runs", 1),
                    warmup_batch=self.config["qwen3vl"].get("warmup_batch", 8),
                )
            else:
                self.models["Qwen3VL"] = Qwen3VLEmbeddingModel(device=self.device)
        except Exception as e:
            print(f"Failed to load Qwen3VL: {e}")

    def _load_satclip(self):
        """Load SatCLIP model."""
        try:
            if self.config and "satclip" in self.config:
                self.models["SatCLIP"] = SatCLIPModel(
                    ckpt_path=self.config["satclip"].get("ckpt_path"),
                    embedding_path=self.config["satclip"].get("embedding_path"),
                    device=self.device,
                )
            else:
                self.models["SatCLIP"] = SatCLIPModel(device=self.device)
        except Exception as e:
            print(f"Failed to load SatCLIP: {e}")

    def _load_farslip(self):
        """Load FarSLIP model."""
        try:
            if self.config and "farslip" in self.config:
                self.models["FarSLIP"] = FarSLIPModel(
                    ckpt_path=self.config["farslip"].get("ckpt_path"),
                    model_name=self.config["farslip"].get("model_name"),
                    embedding_path=self.config["farslip"].get("embedding_path"),
                    device=self.device,
                )
            else:
                self.models["FarSLIP"] = FarSLIPModel(device=self.device)
        except Exception as e:
            print(f"Failed to load FarSLIP: {e}")

    def _load_clay(self):
        """Load Clay model."""
        try:
            if self.config and "clay" in self.config:
                self.models["Clay"] = ClayModel(
                    ckpt_path=self.config["clay"].get("ckpt_path"),
                    embedding_path=self.config["clay"].get("embedding_path"),
                    device=self.device,
                )
            else:
                self.models["Clay"] = ClayModel(device=self.device)
        except Exception as e:
            print(f"Failed to load Clay: {e}")

    def _load_olmoearth(self):
        """Load OlmoEarth model."""
        try:
            if self.config and "olmoearth" in self.config:
                self.models["OlmoEarth"] = OlmoEarthModel(
                    ckpt_path=self.config["olmoearth"].get("ckpt_path"),
                    model_size=self.config["olmoearth"].get("model_size", "nano"),
                    embedding_path=self.config["olmoearth"].get("embedding_path"),
                    device=self.device,
                )
            else:
                self.models["OlmoEarth"] = OlmoEarthModel(device=self.device)
        except Exception as e:
            print(f"Failed to load OlmoEarth: {e}")

    def get_model(self, model_name):
        """Get a loaded model by name.

        Returns:
            tuple: (model_instance, error_message)
        """
        if model_name not in self.models:
            return None, f"Model {model_name} not loaded."
        return self.models[model_name], None

    def get_available_models(self):
        """Get list of available model names."""
        return list(self.models.keys())
