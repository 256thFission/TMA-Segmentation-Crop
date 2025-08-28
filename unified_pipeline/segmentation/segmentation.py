"""Tissue core segmentation engine."""

import cv2
import json
import tempfile
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

from preprocessor import ImagePreprocessor
from cellpose_detector import CellposeSAMDetector
from visualize import create_simple_overlay, crop_cores_with_masks, crop_cores_raw, sort_cores_rowwise


class SegmentationEngine:
    """Handles tissue core segmentation for both DAPI and stain modalities"""
    
    def __init__(self, config, utils):
        self.config = config
        self.utils = utils
        self.base_dir = config.get_output_dir('base_dir')
        
        # Ensure base directory exists
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def run_segmentation(self, modality: str, image_path: Path, 
                        dapi_grid: List, stain_grid: List) -> Tuple[List, str]:
        """
        Run tissue core segmentation for a given modality (dapi/stain)
        
        Args:
            modality: 'dapi' or 'stain'
            image_path: Path to input image
            dapi_grid: DAPI core naming grid
            stain_grid: Stain core naming grid
            
        Returns:
            tuple: (detected_cores, processing_report_path)
        """
        
        self.utils.log(f"Starting segmentation for {modality} modality...")
        
        # Setup output directories following canonical layout
        img_basename = image_path.stem
        output_dir = self.base_dir / f"{img_basename}_processed"
        
        crops_filtered_dir = output_dir / "cores_filtered"
        crops_raw_dir = output_dir / "cores_raw"
        masks_dir = output_dir / "masks"
        
        for dir_path in [output_dir, crops_filtered_dir, crops_raw_dir, masks_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize segmentation components with modality-specific settings
        preprocessor = ImagePreprocessor(
            downsample_factor=self.config.get(f'segmentation.{modality}.downsample_factor', 64),
            enhance_contrast=self.config.get(f'segmentation.{modality}.preprocessing.enhance_contrast', True),
            flat_field_correction=self.config.get(f'segmentation.{modality}.preprocessing.flat_field_correction', True)
        )
        
        # Cellpose GPU detection - allow GPU usage for better performance
        use_gpu = self.config.get(f'segmentation.{modality}.cellpose.use_gpu', True)
        if use_gpu:
            try:
                import torch
                use_gpu = torch.cuda.is_available()
                if use_gpu:
                    self.utils.log(f"Using GPU for {modality} Cellpose segmentation")
                else:
                    self.utils.log(f"GPU requested but not available for {modality} Cellpose, using CPU")
            except ImportError:
                use_gpu = False
                self.utils.log(f"PyTorch not available for {modality} Cellpose, using CPU")
                
        detector = CellposeSAMDetector(use_gpu=use_gpu)
        
        # Process image
        self.utils.log(f"Processing {modality} image: {image_path.name}")
        original_image, processed_image, scale_factor = preprocessor.process(str(image_path))
        
        if processed_image is None:
            raise RuntimeError(f"Failed to process {modality} image: {image_path}")
        
        # Detect cores
        self.utils.log(f"Detecting tissue cores in {modality} image...")
        detected_cores, detection_mask = detector.detect_cores(
            processed_image,
            flow_threshold=self.config.get(f'segmentation.{modality}.cellpose.flow_threshold', 0.6),
            cellprob_threshold=self.config.get(f'segmentation.{modality}.cellpose.cellprob_threshold', 0.3),
            return_mask=True
        )
        
        # Filter cores by size
        img_width = processed_image.shape[1]
        min_radius = (self.config.get(f'segmentation.{modality}.min_radius_pct', 1.0) / 100.0) * img_width
        max_radius = (self.config.get(f'segmentation.{modality}.max_radius_pct', 30.0) / 100.0) * img_width
        
        filtered_cores = [(x, y, r) for x, y, r in detected_cores 
                         if min_radius <= r <= max_radius]
        
        # Sort cores in row-wise order (from memory - excellent performance)
        sorted_cores = sort_cores_rowwise(filtered_cores)
        
        # Scale coordinates back to original image
        original_cores = []
        for x, y, r in sorted_cores:
            orig_x = int(x * scale_factor)
            orig_y = int(y * scale_factor)
            orig_r = int(r * scale_factor)
            original_cores.append((orig_x, orig_y, orig_r))
        
        # Save detection overlay
        overlay_image = create_simple_overlay(processed_image, sorted_cores, mask=detection_mask)
        overlay_path = output_dir / "detection_overlay.png"
        cv2.imwrite(str(overlay_path), overlay_image)
        
        # Generate canonical core names for this modality
        grid = dapi_grid if modality == 'dapi' else stain_grid
        core_names = [self.utils.get_canonical_core_name(i, grid) for i in range(len(sorted_cores))]
        
        # Generate masks (always needed for downstream processing)
        with tempfile.TemporaryDirectory() as temp_dir:
            mask_info = crop_cores_with_masks(
                original_image, detection_mask, sorted_cores, temp_dir,
                padding_factor=self.config.get(f'segmentation.{modality}.cropping.padding_factor', 1.2),
                scale_factor=scale_factor, masks_dir=str(masks_dir), core_names=core_names
            )
        
        # Generate crops based on masking preference
        skip_masking = self.config.get(f'segmentation.{modality}.cropping.skip_masking', False)
        if skip_masking:
            cropped_info = crop_cores_raw(
                original_image, sorted_cores, str(crops_filtered_dir),
                padding_factor=self.config.get(f'segmentation.{modality}.cropping.padding_factor', 1.2),
                scale_factor=scale_factor, core_names=core_names
            )
        else:
            cropped_info = crop_cores_with_masks(
                original_image, detection_mask, sorted_cores, str(crops_filtered_dir),
                padding_factor=self.config.get(f'segmentation.{modality}.cropping.padding_factor', 1.2),
                scale_factor=scale_factor, masks_dir=None, core_names=core_names
            )
        
        # Always create raw crops for reference
        raw_cropped_info = crop_cores_raw(
            original_image, sorted_cores, str(crops_raw_dir),
            padding_factor=self.config.get(f'segmentation.{modality}.cropping.padding_factor', 1.2),
            scale_factor=scale_factor, core_names=core_names
        )
        
        # Generate processing report following spec format
        expected_cores = self.config.get(f'segmentation.{modality}.expected_cores')
        detection_rate = (len(filtered_cores) / expected_cores * 100) if expected_cores else None
        
        report = {
            'input_path': str(image_path),
            'processing_time': str(datetime.now()),  # Will be updated with actual duration
            'total_detected': len(detected_cores),
            'filtered_cores': len(filtered_cores),
            'expected_cores': expected_cores or 'N/A',
            'detection_rate': f"{detection_rate:.1f}%" if detection_rate else 'N/A',
            'output_directory': str(output_dir),
            'overlay_image': str(overlay_path),
            'cores_filtered_directory': str(crops_filtered_dir),
            'cores_raw_directory': str(crops_raw_dir),
            'masks_directory': str(masks_dir),
            'scale_factor': scale_factor,
            'crop_mapping': cropped_info
        }
        
        report_path = output_dir / 'detection_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.utils.log(f"Segmentation completed for {modality}: {len(filtered_cores)} cores detected", "success")
        
        return original_cores, str(report_path)
