import numpy as np
import geopandas as gpd
import hashlib
from rasterio.io import MemoryFile

from .grid_cell_fragment import *
from .models import *
import cv2

class MajorTOM_Embedder(torch.nn.Module):
    """
    MajorTOM Embedder class that applies a model to geospatial image fragments, 
    computes embeddings, and returns metadata for each fragment.

    This class is designed to work with raster data, where the image is fragmented 
    into smaller tiles, and embeddings are computed for each tile using the provided 
    embedder model. The output is a GeoDataFrame containing spatial metadata and 
    the corresponding embeddings for each tile.

    Attributes:
        embedder: A model that generates embeddings for image fragments.
        frag_params: Dictionary containing fragmentation parameters such as the 
                      target overlap and border shift.
        column_types: Dictionary specifying data types for the output GeoDataFrame columns.
    """
    
    def __init__(self, embedder, target_overlap=0.1, border_shift=True):
        """
        Initializes the MajorTOM Embedder with the given parameters.

        Args:
            embedder (torch.nn.Module): A model that generates embeddings for image fragments.
            target_overlap (float): The target overlap between image fragments. Default is 0.1.
            border_shift (bool): Whether to shift the borders of fragments to avoid edge artifacts. Default is True.
        """
        super().__init__()

        # Model
        self.embedder = embedder

        # Fragmentation Settings
        self.frag_params = params = {
            'fragment_size' : self.embedder.size[0],
            'target_overlap' : target_overlap,
            'border_shift' : border_shift
        }

        # Data types for the output dataframe (commented columns need no conversion)
        self.column_types = {
            #'unique_id' :,
            #'embedding' : ,
            #'timestamp' : ,
            #'product_id' : ,
            #'grid_cell' : ,
            'grid_row_u' : 'int16',
            'grid_col_r' : 'int16',
            'centre_lat' : 'float32',
            'centre_lon' : 'float32',
            #'utm_footprint' : ,
            #'utm_crs' : ,
            #'pixel_bbox' : ,            
        }

    def bands(self):
        """
        Returns the set of input bands in the correct order.

        Returns:
            list: List of input bands used by the embedder.
        """
        return self.embedder.bands

    def size(self):
        """
        Returns the input image size.

        Returns:
            tuple: Tuple representing the image size (height, width).
        """
        return self.embedder.size

    def calculate_checksum(self, geometry, timestamp, product_id, embedding):
        """
        Calculates a checksum for the given geometry, timestamp, product ID, and embedding.

        Args:
            geometry (shapely.geometry): The geometry object representing the fragment's footprint.
            timestamp (str): Timestamp of the data.
            product_id (str): Product identifier.
            embedding (np.ndarray): The embedding of the image fragment.

        Returns:
            str: A SHA256 checksum of the concatenated input parameters.
        """
        combined = f"{geometry}_{timestamp}_{product_id}_{embedding}"
        checksum = hashlib.sha256(combined.encode()).hexdigest()
        return checksum

    def _read_image(self, row):
        """
        Reads and processes the image bands for a given row, performs optional upsampling 
        if the resolution is mismatched, and returns the image data, footprint, and CRS.

        Args:
            row (pandas.Series): The input row containing the image bands.

        Returns:
            torch.Tensor: A tensor containing the stacked image bands.
            shapely.geometry: The footprint of the image.
            rasterio.crs.CRS: The CRS of the image.
        """

        # Read the file
        img = []
        for band in self.embedder.bands:
            with MemoryFile(row[band][0].as_py()) as mem_f:
                with mem_f.open(driver='GTiff') as f:
                    crs = f.crs
                    footprint = box(*f.bounds)
                    img.append(f.read()[0])

        # optional upsampling
        shapes = [layer.shape for layer in img]
        if any([el!=shapes[0] for el in shapes]): # if any resolution mismatch
            h, w = max([el[0] for el in shapes]), max([el[1] for el in shapes]) # maximum size
            for layer_idx, layer in enumerate(img):
                if layer.shape != (h,w):
                    img[layer_idx] = cv2.resize(layer, (h,w), interpolation=cv2.INTER_NEAREST)
        img = torch.from_numpy(np.stack(img,-1).astype(np.float32))

        return img, footprint, crs

    def _encode_images(self, images, timestamps=None):
        """Encode images, forwarding timestamps only to temporal embedders."""
        if getattr(self.embedder, "supports_timestamps", False):
            return self.embedder(images, timestamps=timestamps)
        return self.embedder(images)
        

    def forward(self, row, row_meta, device='cuda'):
        """
        Forward pass of the model: Reads the image, fragments it, computes embeddings 
        for each fragment, and returns a GeoDataFrame with the spatial metadata and 
        embeddings.

        Args:
            row (pandas.Series): The input row containing the image data.
            row_meta (pandas.Series): Metadata associated with the row (e.g., timestamp, product_id).
            device (str): The device to run the model on ('cpu' or 'cuda'). Default is 'cuda'.

        Returns:
            geopandas.GeoDataFrame: A GeoDataFrame containing metadata and embeddings for each fragment.
        """
        # Read file
        img, footprint, crs = self._read_image(row)

        # Fragment the sample
        fragments, xys = fragment_fn(img, **self.frag_params, return_indices=True, verbose=False)

        nrows, ncols, c, h, w = fragments.shape
        # Apply the model
        timestamp = row_meta.timestamp.item()
        timestamps = [timestamp] * (nrows * ncols)
        with torch.no_grad():
            embeddings = self._encode_images(
                fragments.reshape(-1,c,h,w).to(device), timestamps=timestamps
            ).view(nrows, ncols, -1)

        df_rows = []

        # Pack rows for geoparquet
        for r_idx in range(nrows):
            for c_idx in range(ncols):
                embedding = embeddings[r_idx, c_idx].cpu().numpy()
                # spatial features per fragment
                x_offset,y_offset=xys[r_idx,c_idx].int().tolist()
                pixel_bbox = [x_offset, y_offset, x_offset + h,y_offset + w] # in pixels
                utm_footprint = crop_footprint(footprint, *img.shape[:2], pixel_bbox)
                # main footprint is in WGS84 (needs to be consistent across parquet)
                transformer = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)
                geometry = transform(transformer.transform, utm_footprint) # WGS84
                centre_lon, centre_lat = geometry.centroid.coords[0]
                
                row_dict = {
                    'unique_id' : self.calculate_checksum(geometry, row_meta.timestamp.item(), row_meta.product_id.item(), embedding),
                    'embedding' : embedding,
                    'timestamp' : row_meta.timestamp.item(),
                    'product_id' : row_meta.product_id.item(),
                    'grid_cell' : row_meta.grid_cell.item(),
                    'grid_row_u' : row_meta.grid_row_u.item(),
                    'grid_col_r' : row_meta.grid_col_r.item(),
                    'geometry' : geometry,
                    'centre_lat' : centre_lat,
                    'centre_lon' : centre_lon,
                    'utm_footprint' : utm_footprint.wkt,
                    'utm_crs' : crs.to_string(),
                    'pixel_bbox' : pixel_bbox,
                    'parquet_row' : row_meta.parquet_row.item() if 'parquet_row' in row_meta.columns else None,
                    'parquet_url' : row_meta.parquet_url.item() if 'parquet_url' in row_meta.columns else None,
                }
                df_rows.append(row_dict)

        return gpd.GeoDataFrame(df_rows, geometry='geometry', crs='EPSG:4326').astype(self.column_types)

    def forward_batch(self, rows_and_meta, device='cuda'):
        """
        Batch version of forward().

        Args:
            rows_and_meta: Iterable of (row, row_meta) pairs.
            device (str): The device to run the model on ('cpu' or 'cuda').

        Returns:
            geopandas.GeoDataFrame: Metadata and embeddings for every fragment in
            the same order as calling forward() on each input row.
        """
        contexts = []
        fragment_batches = []
        batch_timestamps = []

        for row, row_meta in rows_and_meta:
            img, footprint, crs = self._read_image(row)
            fragments, xys = fragment_fn(img, **self.frag_params, return_indices=True, verbose=False)

            nrows, ncols, c, h, w = fragments.shape
            fragment_batches.append(fragments.reshape(-1, c, h, w))
            batch_timestamps.extend([row_meta.timestamp.item()] * (nrows * ncols))
            contexts.append({
                'img_shape': img.shape,
                'footprint': footprint,
                'crs': crs,
                'row_meta': row_meta,
                'xys': xys,
                'nrows': nrows,
                'ncols': ncols,
                'h': h,
                'w': w,
                'count': nrows * ncols,
            })

        if not fragment_batches:
            return gpd.GeoDataFrame([]).astype(self.column_types)

        fragments_batch = torch.cat(fragment_batches, dim=0).to(device)
        with torch.no_grad():
            embeddings = self._encode_images(fragments_batch, timestamps=batch_timestamps).detach().cpu()

        df_rows = []
        offset = 0

        for context in contexts:
            count = context['count']
            nrows = context['nrows']
            ncols = context['ncols']
            h = context['h']
            w = context['w']
            row_meta = context['row_meta']
            xys = context['xys']
            footprint = context['footprint']
            crs = context['crs']
            img_shape = context['img_shape']
            sample_embeddings = embeddings[offset:offset + count].view(nrows, ncols, -1)
            offset += count

            transformer = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)

            for r_idx in range(nrows):
                for c_idx in range(ncols):
                    embedding = sample_embeddings[r_idx, c_idx].numpy()
                    x_offset, y_offset = xys[r_idx, c_idx].int().tolist()
                    pixel_bbox = [x_offset, y_offset, x_offset + h, y_offset + w]
                    utm_footprint = crop_footprint(footprint, *img_shape[:2], pixel_bbox)
                    geometry = transform(transformer.transform, utm_footprint)
                    centre_lon, centre_lat = geometry.centroid.coords[0]

                    row_dict = {
                        'unique_id': self.calculate_checksum(
                            geometry, row_meta.timestamp.item(), row_meta.product_id.item(), embedding
                        ),
                        'embedding': embedding,
                        'timestamp': row_meta.timestamp.item(),
                        'product_id': row_meta.product_id.item(),
                        'grid_cell': row_meta.grid_cell.item(),
                        'grid_row_u': row_meta.grid_row_u.item(),
                        'grid_col_r': row_meta.grid_col_r.item(),
                        'geometry': geometry,
                        'centre_lat': centre_lat,
                        'centre_lon': centre_lon,
                        'utm_footprint': utm_footprint.wkt,
                        'utm_crs': crs.to_string(),
                        'pixel_bbox': pixel_bbox,
                        'parquet_row': row_meta.parquet_row.item() if 'parquet_row' in row_meta.columns else None,
                        'parquet_url': row_meta.parquet_url.item() if 'parquet_url' in row_meta.columns else None,
                    }
                    df_rows.append(row_dict)

        return gpd.GeoDataFrame(df_rows, geometry='geometry', crs='EPSG:4326').astype(self.column_types)
