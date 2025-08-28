"""Core processing utilities for centroids and data preprocessing."""

import pandas as pd
from pathlib import Path
from typing import Any
from ..config import UnifiedConfig


class CoreProcessor:
    """Handles centroid preprocessing and data transformations"""
    
    def __init__(self, config: UnifiedConfig, utils):
        self.config = config
        self.utils = utils
    
    def preprocess_centroids(self, centroids_tsv: Path) -> pd.DataFrame:
        """
        Preprocess centroids: convert µm to pixels and apply DAPI transformations
        
        Returns preprocessed centroids DataFrame with pixel coordinates
        """
        self.utils.log("Preprocessing centroids for VALIS registration...")
        
        # Load centroids
        df = pd.read_csv(centroids_tsv, sep='\t')
        
        x_col = self.config.get('coordinates.x_column', 'Centroid X µm')
        y_col = self.config.get('coordinates.y_column', 'Centroid Y µm')
        
        if x_col not in df.columns or y_col not in df.columns:
            raise ValueError(f"Required columns {x_col}, {y_col} not found in {centroids_tsv}")
        
        # Convert from microns to pixels
        um_per_pixel = self.config.get('global.microns_per_pixel', 0.5072)
        df['X_px'] = df[x_col] / um_per_pixel
        df['Y_px'] = df[y_col] / um_per_pixel
        
        # Apply DAPI preprocessing transformations to match segmentation
        scale_factor = self.config.get('valis.preprocessing.scale_factor', 0.1022)
        df['X_scaled'] = df['X_px'] * scale_factor
        df['Y_scaled'] = df['Y_px'] * scale_factor
        
        # Apply flip if configured
        flip_axis = self.config.get('global.flip_axis', 'y')
        if flip_axis == 'y':
            # For Y flip, we need the processed DAPI image dimensions
            # This will be updated after DAPI processing
            df['flip_axis'] = 'y'
        elif flip_axis == 'x':
            df['flip_axis'] = 'x'
        
        self.utils.log(f"Preprocessed {len(df)} centroids")
        return df
