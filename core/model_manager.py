"""Model initialization and management for EarthEmbeddingExplorer."""

import torch

from models.clay_model import ClayModel
from models.dinov2_model import DINOv2Model
from models.farslip_model import FarSLIPModel
from models.load_config import load_and_process_config
from models.olmoearth_model import OlmoEarthModel
from models.satclip_model import SatCLIPModel
from models.siglip_model import SigLIPModel


class ModelManager:
    """Manages model loading and retrieval."""

    def __init__(self, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running on device: {self.device}")

        self.config = load_and_process_config()
        print(self.config)

        self.models = {}
        self._load_all_models()

    def _load_all_models(self):
        """Load all available embedding models."""
        print("Initializing models...")

        self._load_dinov2()
        self._load_siglip()
        self._load_satclip()
        self._load_farslip()
        self._load_clay()
        self._load_olmoearth()

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
