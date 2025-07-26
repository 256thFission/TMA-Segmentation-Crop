#!/usr/bin/env python3
"""
Simplified Cellpose-SAM Tissue Core Segmentation Pipeline
"""

import os
import json
from datetime import datetime
import cv2
import numpy as np
import typer
from typing import Optional

from preprocessor import ImagePreprocessor
from cellpose_detector import CellposeSAMDetector
from visualize import create_simple_overlay, crop_cores_simple, crop_cores_with_masks

# GPU detection
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False


def main(
    input_path: str = typer.Option(..., "--input", "-i", help="Input image path"),
    output: str = typer.Option("simple_output", "--output", "-o", help="Output directory"),
    downsample: int = typer.Option(64, help="Downsample factor"),
    expected_cores: Optional[int] = typer.Option(None, "--expected-cores", help="Optional: Expected number of cores for performance assessment."),
    min_radius_pct: float = typer.Option(1.0, help="Minimum core radius as % of image width"),
    max_radius_pct: float = typer.Option(30.0, help="Maximum core radius as % of image width"),
    flow_threshold: float = typer.Option(0.6, help="Cellpose flow threshold - increase to reduce false positives"),
    cellprob_threshold: float = typer.Option(0.3, help="Cellpose cell probability threshold - increase to reduce noise detections"),
    flat_field: bool = typer.Option(True, help="Enable flat field correction (illumination normalization)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging for debugging.")
):
    """Simplified main execution pipeline"""
    
    # Determine GPU usage
    use_gpu = GPU_AVAILABLE
    gpu_reason = "auto-detected" if GPU_AVAILABLE else "not available"
    
    print("=== Cellpose-SAM Tissue Core Segmentation ===")

    if verbose:
        print("--- Configuration ---")
        print(f"Input: {input_path}")
        print(f"Output: {output}")
        print(f"Downsample factor: {downsample}")
        if expected_cores:
            print(f"Expected cores: {expected_cores}")
        print(f"Radius filter: {min_radius_pct}% - {max_radius_pct}% of image width")
        print(f"Cellpose params: flow_threshold={flow_threshold}, cellprob_threshold={cellprob_threshold}")
        print(f"Flat field correction: {'on' if flat_field else 'off'}")
        print(f"GPU available: {GPU_AVAILABLE}")
        print(f"Using GPU: {use_gpu} ({gpu_reason})")
        print("---------------------")
    
    # Check input
    if not os.path.exists(input_path):
        print(f"CRITICAL: Input file not found: {input_path}")
        raise typer.Exit(code=1)
    
    # Create output directory
    os.makedirs(output, exist_ok=True)
    img_basename = os.path.splitext(os.path.basename(input_path))[0]
    img_dir = os.path.join(output, img_basename + "_processed")
    os.makedirs(img_dir, exist_ok=True)
    crops_dir = os.path.join(img_dir, "individual_cores")
    os.makedirs(crops_dir, exist_ok=True)
    
    start_time = datetime.now()
    print(f"[1/5] Initializing...", end="")
    
    try:
        preprocessor = ImagePreprocessor(
            downsample_factor=downsample,
            enhance_contrast=True,
            flat_field_correction=flat_field,
            flat_field_after_downsample=True
        )
        detector = CellposeSAMDetector(use_gpu=use_gpu)
        print("Done.")

        # Process image
        print(f"[2/5] Loading and preprocessing image...", end="")
        original_image, processed_image, scale_factor = preprocessor.process(input_path)
        if processed_image is None:
            print(f"CRITICAL: Failed to load or process image at {input_path}")
            raise typer.Exit(code=1)
        print("Done.")
        if verbose:
            print(f"      Original shape: {original_image.shape}, Processed shape: {processed_image.shape}")
        
        # Detect cores
        print(f"[3/5] Detecting tissue cores...", end="")
        detected_cores, detection_mask = detector.detect_cores(
            processed_image,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            return_mask=True
        )
        print("Done.")
        
        # Calculate radius thresholds based on image dimensions
        img_width = processed_image.shape[1]
        min_radius = (min_radius_pct / 100.0) * img_width
        max_radius = (max_radius_pct / 100.0) * img_width
        
        print(f"[4/5] Filtering and scaling cores...", end="")
        filtered_cores = []
        for x, y, r in detected_cores:
            if min_radius <= r <= max_radius:
                filtered_cores.append((x, y, r))
        
        if verbose:
            print(f"\n      Initial detections: {len(detected_cores)}")
            print(f"      Radius thresholds (pixels): {min_radius:.1f} - {max_radius:.1f}")
            print(f"      Cores after filtering: {len(filtered_cores)}")
        
        # Scale coordinates back to original image
        original_cores = []
        for x, y, r in filtered_cores:
            orig_x = int(x * scale_factor)
            orig_y = int(y * scale_factor)
            orig_r = int(r * scale_factor)
            original_cores.append((orig_x, orig_y, orig_r))
        
        print("Done.")

        # Save artifacts
        print(f"[5/5] Saving results...", end="")
        overlay_image = create_simple_overlay(processed_image, filtered_cores, mask=detection_mask)
        overlay_path = os.path.join(output, "detection_overlay.png")
        cv2.imwrite(overlay_path, overlay_image)

        cropped_info = crop_cores_with_masks(
            original_image, detection_mask, filtered_cores, crops_dir,
            padding_factor=1.2, scale_factor=scale_factor
        )
        print("Done.")

        # Generate report
        end_time = datetime.now()
        total_time = end_time - start_time
        
        detection_rate = (len(filtered_cores) / expected_cores * 100) if expected_cores else None
        
        report = {
            'input_path': input_path,
            'processing_time': str(total_time),
            'total_detected': len(detected_cores),
            'filtered_cores': len(filtered_cores),
            'expected_cores': expected_cores if expected_cores is not None else 'N/A',
            'detection_rate': f"{detection_rate:.1f}%" if detection_rate is not None else 'N/A',
            'individual_cores_cropped': len(cropped_info),
            'output_directory': output,
            'overlay_image': overlay_path,
            'cores_directory': crops_dir
        }
        
        report_path = os.path.join(output, 'detection_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Summary
        print("\n=== PROCESSING COMPLETE ===")
        print(f"Total time: {total_time}")
        print(f"Final core count: {len(filtered_cores)}")
        if expected_cores is not None and detection_rate is not None:
            print(f"Expected cores: {expected_cores}")
            print(f"Detection rate: {detection_rate:.1f}%" if detection_rate is not None else "N/A")
        print(f"Results saved to: {output}")

        # Performance assessment only if expected_cores is provided
        if detection_rate is not None:
            if detection_rate >= 90:
                print("\nPerformance:  EXCELLENT (≥ 90%)")
            elif detection_rate >= 75:
                print("\nPerformance:  GOOD (≥ 75%)")
            elif detection_rate >= 60:
                print("\nPerformance:  FAIR (≥ 60%)")
            else:
                print("\nPerformance:  POOR (< 60%)")
        
        return 0
        
    except Exception as e:
        print(f"\nCRITICAL: An error occurred during processing: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        raise typer.Exit(code=1)




if __name__ == "__main__":
    typer.run(main)