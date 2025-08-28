"""Core utilities for unified pipeline."""

import csv
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any
from rich.console import Console


class CoreUtils:
    """Utility functions for core processing and data management"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.console = Console()
        
    def log(self, message: str, level: str = "info"):
        """Log message with appropriate styling"""
        if not self.verbose and level == "debug":
            return
            
        styles = {
            "info": "white",
            "success": "green", 
            "warning": "yellow",
            "error": "red",
            "debug": "dim white"
        }
        
        style = styles.get(level, "white")
        self.console.print(f"[{style}]{message}[/{style}]")
    
    def load_core_grid(self, csv_path: Path) -> List[List[str]]:
        """Load core naming grid from CSV file"""
        try:
            grid = []
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header row with column indices
                for row in reader:
                    # Skip row index (first column) and get core names
                    core_names = row[1:] if len(row) > 1 else []
                    grid.append(core_names)
            self.log(f"Loaded grid from {csv_path.name}: {len(grid)} rows, {len(grid[0]) if grid else 0} cols")
            return grid
        except Exception as e:
            self.log(f"Failed to load {csv_path.name}: {e}", "warning")
            return []
    
    def get_canonical_core_name(self, core_index: int, grid: List[List[str]]) -> str:
        """Get canonical core name from grid based on row-wise index"""
        if not grid:
            return f"core_{core_index+1:02d}"
            
        # Calculate row and column from linear index (left-to-right, top-to-bottom)
        cols = len(grid[0]) if grid else 1
        row = core_index // cols
        col = core_index % cols
        
        if row < len(grid) and col < len(grid[row]):
            return grid[row][col]
        else:
            return f"core_{core_index+1:02d}"  # Fallback
    
    def extract_core_centroids(self, centroids_df: pd.DataFrame, 
                             core_x: int, core_y: int, core_radius: float) -> pd.DataFrame:
        """Extract centroids that fall within a core's spatial bounds"""
        # Prefer original pixel space if available (matches core_x/core_y units)
        if 'X_px' in centroids_df.columns and 'Y_px' in centroids_df.columns:
            x_series = centroids_df['X_px']
            y_series = centroids_df['Y_px']
        elif 'X_scaled' in centroids_df.columns and 'Y_scaled' in centroids_df.columns:
            # Fallback to scaled coordinates if pixel columns are missing
            x_series = centroids_df['X_scaled']
            y_series = centroids_df['Y_scaled']
        else:
            # No spatial columns available
            return centroids_df.copy()
        
        # Calculate distance from core center
        dx = x_series - core_x
        dy = y_series - core_y
        distances = np.sqrt(dx**2 + dy**2)
        
        # Filter centroids within core radius
        core_mask = distances <= core_radius
        return centroids_df[core_mask].copy()
    
    def resolve_crop_path_from_report(self, report: Dict[str, Any], core_name: str) -> Optional[Path]:
        """Resolve the absolute crop image path for a given core name from a detection report.
        Handles both list-style crop_mapping entries and dict-style mappings.
        """
        entries = report.get('crop_mapping', [])
        filtered_dir = report.get('cores_filtered_directory')
        raw_dir = report.get('cores_raw_directory')
        target_fname = f"{core_name}.png"

        # Case 1: mapping dict of filename -> path
        if isinstance(entries, dict):
            path_str = entries.get(target_fname)
            return Path(path_str) if path_str else None

        # Case 2: list of dict entries with 'filename' or 'image_filename'
        if isinstance(entries, list):
            for e in entries:
                fname = e.get('image_filename') or e.get('filename')
                eid = e.get('core_id')
                if fname == target_fname or eid == core_name:
                    # Prefer filtered directory, fall back to raw
                    if filtered_dir and fname:
                        p = Path(filtered_dir) / fname
                        if p.exists():
                            return p
                    if raw_dir and fname:
                        p = Path(raw_dir) / fname
                        if p.exists():
                            return p
                    # If not found on disk, still return the most likely path (filtered then raw)
                    if filtered_dir and fname:
                        return Path(filtered_dir) / fname
                    if raw_dir and fname:
                        return Path(raw_dir) / fname
                    return None
        return None
