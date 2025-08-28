"""VALIS registration engine for per-core alignment."""

import json
import shutil
import pandas as pd
from pathlib import Path
from typing import List, Optional

# Import VALIS components
try:
    from myvalis.valis_alignment_clean import (
        AlignmentConfig, 
        run_alignment_pipeline
    )
except ImportError:
    print("Warning: VALIS alignment module not found. Per-core registration will be skipped.")
    AlignmentConfig = None
    run_alignment_pipeline = None


class ValisEngine:
    """Handles VALIS registration for core pairs"""
    
    def __init__(self, config, utils):
        self.config = config
        self.utils = utils
        self.valis_workdir = config.get_output_dir('valis_workdir')
        
        # Ensure VALIS working directory exists
        self.valis_workdir.mkdir(parents=True, exist_ok=True)
    
    def run_per_core_valis(self, dapi_cores: List, stain_cores: List, 
                          preprocessed_centroids: pd.DataFrame, 
                          dapi_report_path: str, stain_report_path: str,
                          dapi_grid: List, stain_grid: List) -> List[Path]:
        """
        Run VALIS registration for each core pair
        
        Args:
            dapi_cores: List of DAPI core coordinates
            stain_cores: List of stain core coordinates  
            preprocessed_centroids: DataFrame with preprocessed centroids
            dapi_report_path: Path to DAPI detection report
            stain_report_path: Path to stain detection report
            dapi_grid: DAPI core naming grid
            stain_grid: Stain core naming grid
            
        Returns:
            List of paths to per-core transformed centroid files
        """
        if run_alignment_pipeline is None:
            self.utils.log("VALIS module not available, skipping per-core registration", "warning")
            return []
        
        self.utils.log("Starting per-core VALIS registration...")
        
        # Load detection reports to get core crop paths
        with open(dapi_report_path, 'r') as f:
            dapi_report = json.load(f)
        with open(stain_report_path, 'r') as f:
            stain_report = json.load(f)
        
        transformed_files = []
        
        # Match cores between modalities (same order from row-wise sorting)
        min_cores = min(len(dapi_cores), len(stain_cores))
        self.utils.log(f"Processing {min_cores} core pairs (DAPI: {len(dapi_cores)}, Stain: {len(stain_cores)})")
        
        for i in range(min_cores):
            # Get canonical core names from grids
            dapi_core_name = self.utils.get_canonical_core_name(i, dapi_grid)
            stain_core_name = self.utils.get_canonical_core_name(i, stain_grid)
            
            self.utils.log(f"Processing core pair {i+1}/{min_cores}: DAPI={dapi_core_name}, Stain={stain_core_name}")
            
            # Create per-core working directory using DAPI name (primary reference)
            core_workdir = self.valis_workdir / f"core_{dapi_core_name}"
            core_workdir.mkdir(parents=True, exist_ok=True)
            
            # Get core crop image paths using canonical names (robust to report format)
            dapi_crop_path = self.utils.resolve_crop_path_from_report(dapi_report, dapi_core_name)
            stain_crop_path = self.utils.resolve_crop_path_from_report(stain_report, stain_core_name)

            if dapi_crop_path is None or stain_crop_path is None:
                self.utils.log(f"Missing crop images for DAPI={dapi_core_name}, Stain={stain_core_name}, skipping", "warning")
                continue
            
            if not dapi_crop_path.exists() or not stain_crop_path.exists():
                self.utils.log(f"Crop images not found for DAPI={dapi_core_name}, Stain={stain_core_name}, skipping", "warning")
                continue
            
            # Extract per-core centroids using spatial filtering
            dapi_x, dapi_y, dapi_r = dapi_cores[i]
            core_centroids = self.utils.extract_core_centroids(
                preprocessed_centroids, dapi_x, dapi_y, dapi_r * 1.2  # 20% padding
            )
            
            if len(core_centroids) == 0:
                self.utils.log(f"No centroids found for {dapi_core_name}, skipping", "warning")
                continue
            
            # Save per-core centroids
            core_centroids_path = core_workdir / f"{dapi_core_name}_centroids.tsv"
            core_centroids.to_csv(core_centroids_path, sep='\t', index=False)
            
            # Setup VALIS config for this core
            valis_config = AlignmentConfig()
            valis_config.workdir = core_workdir
            valis_config.scale_factor = self.config.get('valis.preprocessing.scale_factor', 0.1022)
            valis_config.flip_axis = self.config.get('global.flip_axis', 'y')
            valis_config.microns_per_pixel = self.config.get('global.microns_per_pixel', 0.5072)
            # Set per-core inputs on config for myvalis API
            valis_config.stain_path = stain_crop_path
            valis_config.dapi_path = dapi_crop_path
            valis_config.centroids_tsv = core_centroids_path
            
            # Configure VALIS to use CPU only (prevents CUDA device mismatches)
            valis_config.superglue_config.update({
                'weights': self.config.get('valis.superglue.weights', 'indoor'),
                'keypoint_threshold': self.config.get('valis.superglue.keypoint_threshold', 0.005),
                'match_threshold': self.config.get('valis.superglue.match_threshold', 0.2),
                'force_cpu': True
            })
            
            # VALIS runs on CPU only for stability
            valis_config.use_gpu = False
            
            valis_config.save_overlays = self.config.get('valis.save_overlays', True)
            valis_config.overlay_alpha = self.config.get('valis.overlay_alpha', 0.5)
            
            # Run VALIS alignment for this core pair
            try:
                output_path = core_workdir / f"{dapi_core_name}_transformed.tsv"
                
                results = run_alignment_pipeline(valis_config)
                # Retrieve output TSV from results and copy/move to desired per-core path
                out_from_valis = None
                try:
                    out_from_valis = results.get('output_tsv_path') if isinstance(results, dict) else None
                except Exception:
                    out_from_valis = None
                if out_from_valis and Path(out_from_valis).exists():
                    try:
                        shutil.copy2(out_from_valis, output_path)
                    except Exception:
                        try:
                            shutil.move(out_from_valis, output_path)
                        except Exception:
                            pass
                
                if output_path.exists():
                    transformed_files.append(output_path)
                    self.utils.log(f"Completed {dapi_core_name}: {len(core_centroids)} centroids transformed", "success")
                else:
                    self.utils.log(f"VALIS alignment failed for {dapi_core_name}", "warning")
                    
            except Exception as e:
                self.utils.log(f"Error processing {dapi_core_name}: {e}", "error")
                continue
        
        self.utils.log(f"VALIS registration completed: {len(transformed_files)} cores processed successfully")
        return transformed_files
    
    def aggregate_results(self, transformed_files: List[Path]) -> Path:
        """
        Aggregate per-core transformed centroids into final output file
        """
        self.utils.log("Aggregating transformed centroids...")
        
        final_output_path = self.config.get_output_dir('final_output')
        
        if not transformed_files:
            self.utils.log("No transformed files to aggregate", "warning")
            # Create empty output file
            pd.DataFrame().to_csv(final_output_path, sep='\t', index=False)
            return final_output_path
        
        # Combine all per-core results
        all_centroids = []
        for file_path in transformed_files:
            try:
                df = pd.read_csv(file_path, sep='\t')
                all_centroids.append(df)
            except Exception as e:
                self.utils.log(f"Failed to read {file_path}: {e}", "error")
        
        if all_centroids:
            combined_df = pd.concat(all_centroids, ignore_index=True)
            combined_df.to_csv(final_output_path, sep='\t', index=False)
            self.utils.log(f"Aggregated {len(combined_df)} centroids to {final_output_path}", "success")
        else:
            self.utils.log("No valid transformed files found", "error")
            pd.DataFrame().to_csv(final_output_path, sep='\t', index=False)
        
        return final_output_path
