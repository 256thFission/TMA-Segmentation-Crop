"""Main unified pipeline orchestrator."""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from .config import UnifiedConfig
from .core import CoreUtils, CoreProcessor
from .segmentation import SegmentationEngine
from .registration import ValisEngine


class UnifiedPipeline:
    """Main unified pipeline class that orchestrates all components"""
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
        self.verbose = config.get('global.verbose', True)
        
        # Initialize paths
        self.outputs_dir = config.get_output_dir('final_output').parent
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.utils = CoreUtils(verbose=self.verbose)
        self.processor = CoreProcessor(config, self.utils)
        self.segmentation = SegmentationEngine(config, self.utils)
        self.registration = ValisEngine(config, self.utils)
        
        # Load canonical core naming grids
        self.dapi_grid = self.utils.load_core_grid(config.get_input_path('pcf_csv'))
        self.stain_grid = self.utils.load_core_grid(config.get_input_path('visium_csv'))
        
    def run(self) -> Dict:
        """
        Execute the complete unified pipeline
        
        Returns results dictionary
        """
        start_time = datetime.now()
        
        self.utils.log("=== Starting Unified Tissue Core Alignment Pipeline ===")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=self.utils.console,
            transient=True
        ) as progress:
            
            # Step 1: Segment DAPI image
            task1 = progress.add_task("Segmenting DAPI cores...", total=100)
            dapi_path = self.config.get_input_path('dapi_image')
            dapi_cores, dapi_report = self.segmentation.run_segmentation(
                'dapi', dapi_path, self.dapi_grid, self.stain_grid
            )
            progress.update(task1, advance=100)
            
            # Step 2: Segment stain image  
            task2 = progress.add_task("Segmenting stain cores...", total=100)
            stain_path = self.config.get_input_path('stain_image')
            stain_cores, stain_report = self.segmentation.run_segmentation(
                'stain', stain_path, self.dapi_grid, self.stain_grid
            )
            progress.update(task2, advance=100)
            
            # Step 3: Preprocess centroids
            task3 = progress.add_task("Preprocessing centroids...", total=100)
            centroids_path = self.config.get_input_path('centroids_tsv')
            preprocessed_centroids = self.processor.preprocess_centroids(centroids_path)
            progress.update(task3, advance=100)
            
            # Step 4: Per-core VALIS registration
            task4 = progress.add_task("Running per-core VALIS registration...", total=100)
            transformed_files = self.registration.run_per_core_valis(
                dapi_cores, stain_cores, preprocessed_centroids, 
                dapi_report, stain_report, self.dapi_grid, self.stain_grid
            )
            progress.update(task4, advance=100)
            
            # Step 5: Aggregate results
            task5 = progress.add_task("Aggregating results...", total=100)
            final_output = self.registration.aggregate_results(transformed_files)
            progress.update(task5, advance=100)
        
        end_time = datetime.now()
        total_time = end_time - start_time
        
        # Generate summary
        results = {
            'processing_time': str(total_time),
            'dapi_cores_detected': len(dapi_cores),
            'stain_cores_detected': len(stain_cores),
            'dapi_report': dapi_report,
            'stain_report': stain_report,
            'final_output': str(final_output),
            'transformed_files': [str(f) for f in transformed_files]
        }
        
        self.utils.log("=== Pipeline Completed Successfully! ===", "success")
        self.utils.log(f"Total processing time: {total_time}")
        self.utils.log(f"DAPI cores: {len(dapi_cores)}, Stain cores: {len(stain_cores)}")
        self.utils.log(f"Final output: {final_output}")
        
        return results
