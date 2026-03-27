import torch
from transformers import AutoImageProcessor, AutoModel
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch.nn.functional as F
from PIL import Image
import os

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
                 ckpt_path="./checkpoints/DINOv2",
                 model_name="facebook/dinov2-large",
                 embedding_path="./embedding_datasets/10percent_dinov2_encoded/all_dinov2_embeddings.parquet",
                 device=None):
        """
        Initialize the DINOv2Model.
        
        Args:
            ckpt_path (str): Path to local checkpoint directory or 'hf' for HuggingFace
            model_name (str): HuggingFace model name (used when ckpt_path='hf')
            embedding_path (str): Path to pre-computed embeddings parquet file
            device (str): Device to use ('cuda', 'cpu', or None for auto-detection)
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
        """Load DINOv2 model and processor from local checkpoint or HuggingFace."""
        print(f"Loading DINOv2 model from {self.ckpt_path}...")
        try:
            if self.ckpt_path == 'hf':
                # Load from HuggingFace
                print(f"Loading from HuggingFace: {self.model_name}")
                self.processor = AutoImageProcessor.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
            elif self.ckpt_path.startswith('ms'):
                # Load from ModelScope
                import modelscope
                if self.ckpt_path.endswith('ai'):
                    print(f"Loading from ModelScope AI: {self.model_name}")
                    # get and print MODELSCOPE_DOMAIN environment variable for debugging
                    modelscope_domain = os.getenv('MODELSCOPE_DOMAIN', 'Not Set')
                    print(f"MODELSCOPE_DOMAIN: {modelscope_domain}")
                    self.model_name = "VoyagerX/dinov2-large"
                self.processor = modelscope.AutoImageProcessor.from_pretrained(self.model_name)
                self.model = modelscope.AutoModel.from_pretrained(self.model_name)
            else:
                self.processor = AutoImageProcessor.from_pretrained(self.ckpt_path)
                self.model = AutoModel.from_pretrained(self.ckpt_path)
            
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
    
    # def normalize_s2(self, input_data):
    #     """
    #     Normalize Sentinel-2 RGB data to true-color values.
        
    #     Converts raw Sentinel-2 reflectance values to normalized true-color values
    #     suitable for the DINOv2 model.
        
    #     Args:
    #         input_data (torch.Tensor or np.ndarray): Raw Sentinel-2 image data
        
    #     Returns:
    #         torch.Tensor or np.ndarray: Normalized true-color image in range [0, 1]
    #     """
    #     return (2.5 * (input_data / 1e4)).clip(0, 1)
    
    def encode_image(self, image, is_sentinel2=False):
        """
        Encode an image into a feature embedding.
        
        Args:
            image (PIL.Image, torch.Tensor, or np.ndarray): Input image
                - PIL.Image: RGB image
                - torch.Tensor: Image tensor with shape [C, H, W] (Sentinel-2) or [H, W, C]
                - np.ndarray: Image array with shape [H, W, C]
            is_sentinel2 (bool): Whether to apply Sentinel-2 normalization
        
        Returns:
            torch.Tensor: Normalized embedding vector with shape [embedding_dim]
        """
        if self.model is None or self.processor is None:
            print("Model not loaded!")
            return None
        
        try:
            # Convert to PIL Image if needed
            if isinstance(image, torch.Tensor):
                if is_sentinel2:
                    # Sentinel-2 data: [C, H, W] -> normalize -> PIL
                    image = self.normalize_s2(image)
                    # Convert to [H, W, C] and then to numpy
                    if image.shape[0] == 3:  # [C, H, W]
                        image = image.permute(1, 2, 0)
                    image_np = (image.cpu().numpy() * 255).astype(np.uint8)
                    image = Image.fromarray(image_np, mode='RGB')
                else:
                    # Regular RGB tensor: [H, W, C] or [C, H, W]
                    if image.shape[0] == 3:  # [C, H, W]
                        image = image.permute(1, 2, 0)
                    image_np = (image.cpu().numpy() * 255).astype(np.uint8)
                    image = Image.fromarray(image_np, mode='RGB')
            elif isinstance(image, np.ndarray):
                if is_sentinel2:
                    image = self.normalize_s2(image)
                # Assume [H, W, C] format
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
                image = Image.fromarray(image, mode='RGB')
            elif isinstance(image, Image.Image):
                image = image.convert("RGB")
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")
            
            # Process image
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                if self.device == "cuda":
                    # with torch.amp.autocast('cuda'):  # disable amp as the official embedding is float32
                    outputs = self.model(pixel_values)
                else:
                    outputs = self.model(pixel_values)

                # Get embeddings: average across sequence dimension
                last_hidden_states = outputs.last_hidden_state
                image_features = last_hidden_states.mean(dim=1)
                
                # # Get embeddings: Use pooler_output (1024-d) to match pre-computed embeddings
                # # If pooler_output is not available, use CLS token (first token)
                # if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                #     image_features = outputs.pooler_output
                # else:
                #     # Use CLS token (first token in sequence)
                #     last_hidden_states = outputs.last_hidden_state
                #     image_features = last_hidden_states[:, 0, :]  # [batch_size, hidden_dim]
                
                # Normalize
                image_features = F.normalize(image_features, dim=-1)
            
            return image_features
            
        except Exception as e:
            print(f"Error encoding image: {e}")
            import traceback
            traceback.print_exc()
            return None
    
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


# Legacy class for backward compatibility
class DINOv2_S2RGB_Embedder(torch.nn.Module):
    """
    Legacy embedding wrapper for DINOv2 and Sentinel-2 data.
    
    This class is kept for backward compatibility with existing code.
    For new projects, please use DINOv2Model instead.
    """

    def __init__(self):
        """Initialize the legacy DINOv2_S2RGB_Embedder."""
        super().__init__()

        # Load the DINOv2 processor and model from Hugging Face
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.model = AutoModel.from_pretrained('facebook/dinov2-base')

        # Define the RGB bands for Sentinel-2 (B04, B03, B02)
        self.bands = ['B04', 'B03', 'B02']

        # Extract the input size from the processor settings
        self.size = self.processor.crop_size['height'], self.processor.crop_size['width']

    def normalize(self, input):
        """
        Normalize Sentinel-2 RGB data to true-color values.
        
        Args:
            input (torch.Tensor): Raw Sentinel-2 image tensor
        
        Returns:
            torch.Tensor: Normalized true-color image
        """
        return (2.5 * (input / 1e4)).clip(0, 1)

    def forward(self, input):
        """
        Forward pass through the model to generate embeddings.
        
        Args:
            input (torch.Tensor): Input Sentinel-2 image tensor with shape [C, H, W]
        
        Returns:
            torch.Tensor: Embedding vector with shape [embedding_dim]
        """
        model_input = self.processor(self.normalize(input), return_tensors="pt")
        outputs = self.model(model_input['pixel_values'].to(self.model.device))
        last_hidden_states = outputs.last_hidden_state
        return last_hidden_states.mean(dim=1).cpu()
