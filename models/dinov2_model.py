import os

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

class DINOv2Model:
    """
    DINOv2 model wrapper for Sentinel-2 RGB data embedding and search.

    This class provides a unified interface for:
    - Loading DINOv2 models from local checkpoint or HuggingFace
    - Encoding images into embeddings
    - Loading pre-computed embeddings
    - Searching similar images using cosine similarity

    The model processes Sentinel-2 RGB data by normalizing it to true-color values
    and generating feature embeddings using the DINOv2 architecture.
    """

    def __init__(self,
                 ckpt_path=None,
                 model_name="facebook/dinov2-large",
                 embedding_path=None,
                 device=None):
        """
        Initialize the DINOv2Model.

        Args:
            ckpt_path (str): Path to local checkpoint directory. If None or not found,
                the model is downloaded according to DOWNLOAD_ENDPOINT env var.
            model_name (str): Model identifier used for remote download.
            embedding_path (str): Path to pre-computed embeddings parquet file.
            device (str): Device to use ('cuda', 'cpu', or None for auto-detection).
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.ckpt_path = ckpt_path
        self.embedding_path = embedding_path

        self.model = None
        self.processor = None
        self.df_embed = None
        self.image_embeddings = None

        # Define the RGB bands for Sentinel-2 (B04, B03, B02)
        self.bands = ['B04', 'B03', 'B02']
        self.size = None

        self.load_model()
        if self.embedding_path is not None:
            self.load_embeddings()


    def load_model(self):
        """Load DINOv2 model and processor from local path or remote repository."""
        endpoint = os.getenv("DOWNLOAD_ENDPOINT").lower()

        if self.ckpt_path is not None and os.path.exists(self.ckpt_path):
            print(f"Loading DINOv2 model from local path: {self.ckpt_path}")
            load_source = self.ckpt_path
        elif endpoint == "huggingface":
            print(f"Loading DINOv2 model from HuggingFace: {self.model_name}")
            load_source = self.model_name
        elif endpoint == "modelscope.cn":
            print(f"Loading DINOv2 model from ModelScope: {self.model_name}")
            load_source = self.model_name
        elif endpoint == "modelscope.ai":
            print(f"Loading DINOv2 model from ModelScope: VoyagerX/dinov2-large")
            load_source = "VoyagerX/dinov2-large"
            os.environ["MODELSCOPE_DOMAIN"] = "www.modelscope.ai"
        else:
            print(f"Unknown DOWNLOAD_ENDPOINT '{endpoint}', defaulting to modelscope.cn")
            load_source = self.model_name

        try:
            if endpoint == "huggingface":
                self.processor = AutoImageProcessor.from_pretrained(load_source)
                self.model = AutoModel.from_pretrained(load_source)
            else:
                import modelscope
                self.processor = modelscope.AutoImageProcessor.from_pretrained(load_source)
                self.model = modelscope.AutoModel.from_pretrained(load_source)

            self.model = self.model.to(self.device)
            self.model.eval()

            # Extract the input size from the processor settings
            if hasattr(self.processor, 'crop_size'):
                self.size = (self.processor.crop_size['height'], self.processor.crop_size['width'])
            elif hasattr(self.processor, 'size'):
                if isinstance(self.processor.size, dict):
                    self.size = (self.processor.size.get('height', 224), self.processor.size.get('width', 224))
                else:
                    self.size = (self.processor.size, self.processor.size)
            else:
                self.size = (224, 224)

            print(f"DINOv2 model loaded on {self.device}, input size: {self.size}")
        except Exception as e:
            print(f"Error loading DINOv2 model: {e}")

    def load_embeddings(self):
        """Load pre-computed embeddings from parquet file."""
        print(f"Loading DINOv2 embeddings from {self.embedding_path}...")
        try:
            if not os.path.exists(self.embedding_path):
                print(f"Warning: Embedding file not found at {self.embedding_path}")
                return

            self.df_embed = pq.read_table(self.embedding_path).to_pandas()

            # Pre-compute image embeddings tensor
            image_embeddings_np = np.stack(self.df_embed['embedding'].values)
            self.image_embeddings = torch.from_numpy(image_embeddings_np).to(self.device).float()
            self.image_embeddings = F.normalize(self.image_embeddings, dim=-1)
            print(f"DINOv2 Data loaded: {len(self.df_embed)} records")
        except Exception as e:
            print(f"Error loading DINOv2 embeddings: {e}")

    def preprocess_s2(self, input_data):
        """
        Preprocess Sentinel-2 RGB data for DINOv2 model input.

        Converts raw Sentinel-2 reflectance values to normalized true-color values
        suitable for the DINOv2 model by dividing by 10,000 and scaling by 2.5,
        clipping to the range [0, 1].

        Args:
            input_data (torch.Tensor or np.ndarray): Raw Sentinel-2 image data.

        Returns:
            torch.Tensor or np.ndarray: Preprocessed image data in range [0, 1].
        """
        return (2.5 * (input_data / 1e4)).clip(0, 1)

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
                If False, the preprocessed tensor/array is passed directly to the processor.

        Returns:
            torch.Tensor: Normalized embedding vector with shape [embedding_dim] or
                [N, embedding_dim] for batched input.
        """
        if self.model is None or self.processor is None:
            print("Model not loaded!")
            return None

        # Convert to PIL Image if needed
        if isinstance(image, torch.Tensor):
            if preprocess_s2:
                image = self.preprocess_s2(image)
            if normalize:
                # Convert to [H, W, C] and then to numpy
                if image.dim() == 4:
                    # Batch processing: [N, C, H, W]
                    features = []
                    for i in range(image.shape[0]):
                        img = image[i]
                        if img.shape[0] == 3:  # [C, H, W]
                            img = img.permute(1, 2, 0)
                        img_np = (img.detach().cpu().numpy() * 255).astype(np.uint8)
                        img_pil = Image.fromarray(img_np, mode='RGB')
                        inputs = self.processor(images=img_pil, return_tensors="pt")
                        pixel_values = inputs['pixel_values'].to(self.device)
                        with torch.no_grad():
                            outputs = self.model(pixel_values)
                            feat = outputs.last_hidden_state.mean(dim=1)
                            feat = F.normalize(feat, dim=-1)
                        features.append(feat)
                    return torch.cat(features, dim=0)
                else:
                    if image.shape[0] == 3:  # [C, H, W]
                        image = image.permute(1, 2, 0)
                    image_np = (image.detach().cpu().numpy() * 255).astype(np.uint8)
                    image = Image.fromarray(image_np, mode='RGB')
            else:
                # Pass tensor directly to processor
                if image.dim() == 3:
                    image = image.unsqueeze(0)
                inputs = self.processor(images=image, return_tensors="pt")
                pixel_values = inputs['pixel_values'].to(self.device)
                with torch.no_grad():
                    outputs = self.model(pixel_values)
                    image_features = outputs.last_hidden_state.mean(dim=1)
                    image_features = F.normalize(image_features, dim=-1)
                return image_features

        elif isinstance(image, np.ndarray):
            if preprocess_s2:
                image = self.preprocess_s2(image)
            if normalize:
                # Assume [H, W, C] format
                if image.ndim == 4:
                    features = []
                    for i in range(image.shape[0]):
                        img = image[i]
                        if img.max() <= 1.0:
                            img = (img * 255).astype(np.uint8)
                        else:
                            img = img.astype(np.uint8)
                        img_pil = Image.fromarray(img, mode='RGB')
                        inputs = self.processor(images=img_pil, return_tensors="pt")
                        pixel_values = inputs['pixel_values'].to(self.device)
                        with torch.no_grad():
                            outputs = self.model(pixel_values)
                            feat = outputs.last_hidden_state.mean(dim=1)
                            feat = F.normalize(feat, dim=-1)
                        features.append(feat)
                    return torch.cat(features, dim=0)
                else:
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)
                    image = Image.fromarray(image, mode='RGB')
            else:
                image = torch.from_numpy(image)
                if image.dim() == 3:
                    image = image.unsqueeze(0)
                inputs = self.processor(images=image, return_tensors="pt")
                pixel_values = inputs['pixel_values'].to(self.device)
                with torch.no_grad():
                    outputs = self.model(pixel_values)
                    image_features = outputs.last_hidden_state.mean(dim=1)
                    image_features = F.normalize(image_features, dim=-1)
                return image_features

        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # Process image
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(self.device)

        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(pixel_values)
            image_features = outputs.last_hidden_state.mean(dim=1)

            # Normalize
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
            query_features (torch.Tensor): Query embedding vector
            top_k (int): Number of top results to return
            top_percent (float): If set, use top percentage instead of top_k
            threshold (float): Minimum similarity threshold

        Returns:
            tuple: (similarities, filtered_indices, top_indices)
                - similarities: Similarity scores for all images
                - filtered_indices: Indices of images above threshold
                - top_indices: Indices of top-k results
        """
        if self.image_embeddings is None:
            print("Embeddings not loaded!")
            return None, None, None

        try:
            # Ensure query_features is float32 and on correct device
            query_features = query_features.float().to(self.device)

            # Normalize query features
            query_features = F.normalize(query_features, dim=-1)

            # Cosine similarity
            similarity = (self.image_embeddings @ query_features.T).squeeze()
            similarities = similarity.detach().cpu().numpy()

            # Handle top_percent
            if top_percent is not None:
                k = int(len(similarities) * top_percent)
                if k < 1:
                    k = 1
                threshold = np.partition(similarities, -k)[-k]

            # Filter by threshold
            mask = similarities >= threshold
            filtered_indices = np.where(mask)[0]

            # Get top k
            top_indices = np.argsort(similarities)[-top_k:][::-1]

            return similarities, filtered_indices, top_indices

        except Exception as e:
            print(f"Error during search: {e}")
            return None, None, None
