import os
import warnings

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from modelscope.hub.snapshot_download import snapshot_download
from PIL import Image

# Attempt to import get_satclip, but handle potential issues gracefully

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from models.SatCLIP.satclip.load import get_satclip
    print("Successfully imported models.SatCLIP.satclip.load.get_satclip.")

class SatCLIPModel:
    def __init__(self,
                 ckpt_path='ms',
                 embedding_path=None,
                 device=None):

        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        if 'hf' in ckpt_path:
            ckpt_path = hf_hub_download("microsoft/SatCLIP-ViT16-L40", "satclip-vit16-l40.ckpt")
        elif 'ms' in ckpt_path:
            if ckpt_path.endswith('ai'):
                # get env and print
                print("Using modelscope domain:", os.environ.get('MODELSCOPE_DOMAIN', 'not set'))
                # os.environ['MODELSCOPE_DOMAIN'] = "www.modelscope.ai"
                repo_id = "VoyagerX/SatCLIP-ViT16-L40"
            else:
                repo_id = "microsoft/SatCLIP-ViT16-L40"
            cache_dir = snapshot_download(
                repo_id=repo_id,
                allow_file_pattern="satclip-vit16-l40.ckpt"
            )
            ckpt_path = os.path.join(cache_dir, "satclip-vit16-l40.ckpt")

        self.ckpt_path = ckpt_path
        self.embedding_path = embedding_path

        self.model = None
        self.df_embed = None
        self.image_embeddings = None

        self.load_model()
        if self.embedding_path is not None:
            self.load_embeddings()

    def load_model(self):
        if get_satclip is None:
            print("Error: SatCLIP functionality is not available.")
            return

        print(f"Loading SatCLIP model from {self.ckpt_path}...")
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
        # Assuming embeddings are stored similarly to SigLIP
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

    def encode_image(self, image):
        """
        Encode an image into a vector using SatCLIP visual encoder.

        Supports:
        - PIL Image (RGB): adapted to 13 channels (R→B04, G→B03, B→B02, rest zeros).
        - np.ndarray (H, W, 12): 12-band MajorTOM Sentinel-2 data, padded to 13 channels
          by inserting B10=zeros at index 10. Band order assumed:
          [B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12].
        """
        if self.model is None:
            return None

        try:
            if isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[-1] == 12:
                # ---- 12-band Sentinel-2 input ----
                # Normalize: raw uint16 reflectance / 10000
                img = image.astype(np.float32) / 10000.0
                # (H, W, 12) → (12, H, W)
                img = img.transpose(2, 0, 1)
                # Insert B10 (zeros) at index 10 -> (13, H, W)
                _b10 = np.zeros((1, img.shape[1], img.shape[2]), dtype=img.dtype)
                img_13 = np.concatenate([img[:10], _b10, img[10:]], axis=0)

                input_tensor = torch.from_numpy(img_13).unsqueeze(0)  # (1, 13, H, W)
                # Resize to 224x224
                input_tensor = torch.nn.functional.interpolate(
                    input_tensor, size=(224, 224), mode='bicubic', align_corners=False)
                input_tensor = input_tensor.to(self.device)

                with torch.no_grad():
                    img_feature = self.model.encode_image(input_tensor)
                    img_feature = img_feature / img_feature.norm(dim=1, keepdim=True)
                return img_feature

            elif isinstance(image, Image.Image):    # departed
                # ---- RGB PIL Image ----
                image = image.convert("RGB")
                image = image.resize((224, 224))
                img_np = np.array(image).astype(np.float32) / 255.0

                # Construct 13 channels
                # S2 bands: B01, B02(B), B03(G), B04(R), B05, B06, B07, B08, B8A, B09, B10, B11, B12
                input_tensor = np.zeros((13, 224, 224), dtype=np.float32)
                input_tensor[1] = img_np[:, :, 2]  # Blue  → B02
                input_tensor[2] = img_np[:, :, 1]  # Green → B03
                input_tensor[3] = img_np[:, :, 0]  # Red   → B04

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

    def search(self, query_features, top_k=5, top_percent=None, threshold=0.0):
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
