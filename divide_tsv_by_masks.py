#!/usr/bin/env python3
"""
Script to divide a TSV file containing cell centroids by mask region (core).
"""

import os
import sys
import csv
import json
import cv2
import numpy as np
import argparse
from pathlib import Path
from collections import defaultdict


def convert_um_to_pixels(coord_um, conversion_factor=0.5072):
    return round(coord_um / conversion_factor)


def load_crop_mapping_and_masks(pipeline_output_dir):
    report_path = os.path.join(pipeline_output_dir, '..', 'detection_report.json')
    
    if not os.path.exists(report_path):
        print(f"Error: Detection report not found: {report_path}")
        return None
    
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    if 'crop_mapping' not in report:
        print("Error: No crop mapping data found in detection report")
        return None
    
    crop_mapping = {}
    print("Loading mask images...")
    
    for crop_info in report['crop_mapping']:
        core_id = crop_info['core_id']
        x1, y1, x2, y2 = crop_info['crop_bounds']
        # Some runs (e.g., when --skip-masking is used) won't have mask files.
        # In that case, fall back to bounds-only assignment by leaving mask=None.
        mask = None
        if 'mask_filename' in crop_info:
            mask_path = os.path.join(pipeline_output_dir, 'masks', crop_info['mask_filename'])
            # Load mask image once
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    print(f"  Loaded mask for core {core_id:02d}: {mask.shape}")
                else:
                    print(f"  Warning: Could not load mask: {mask_path}")
            else:
                print(f"  Warning: Mask file not found: {mask_path}")
        else:
            print(f"  Note: No 'mask_filename' in report for core {core_id:02d}; using bounds-only assignment.")
        
        crop_mapping[core_id] = {
            'bounds': (x1, y1, x2, y2),
            'mask': mask  # Store mask in memory
        }
    
    print(f"Loaded crop mapping and masks for {len(crop_mapping)} cores")
    return crop_mapping


def find_core_for_centroid(x_pixel, y_pixel, crop_mapping):
    """Find which core a centroid belongs to based on crop bounds and mask regions."""
    for core_id, crop_info in crop_mapping.items():
        x1, y1, x2, y2 = crop_info['bounds']
        
        # Check if coordinates are within crop bounds
        if x1 <= x_pixel < x2 and y1 <= y_pixel < y2:
            # Use pre-loaded mask for this core
            mask = crop_info['mask']
            if mask is not None:
                # Convert coordinates to crop-relative coordinates
                crop_x = x_pixel - x1
                crop_y = y_pixel - y1
                
                # Check if within mask bounds and in mask region
                if (0 <= crop_y < mask.shape[0] and 
                    0 <= crop_x < mask.shape[1]):
                    if mask[crop_y, crop_x] == 255:  # White pixels in mask
                        return core_id
            else:
                # No mask available (e.g., pipeline run with --skip-masking). Use bounds-only.
                return core_id
    
    return None  # Centroid not found in any mask


def parse_tsv_file(tsv_path):
    """Parse the input TSV file and return rows with converted coordinates."""
    rows_with_pixels = []
    
    with open(tsv_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter='\t')
        
        for row in reader:
            # Convert coordinates from µm to pixels
            x_um = float(row['Centroid X µm'])
            y_um = float(row['Centroid Y µm'])
            
            x_pixel = convert_um_to_pixels(x_um)
            y_pixel = convert_um_to_pixels(y_um)
            
            row_data = {
                'Image': row['Image'],
                'Object ID': row['Object ID'],
                'Centroid X µm': row['Centroid X µm'],
                'Centroid Y µm': row['Centroid Y µm'],
                'x_pixel': x_pixel,
                'y_pixel': y_pixel
            }
            
            rows_with_pixels.append(row_data)
    
    return rows_with_pixels


def write_core_tsv_files(core_data, output_dir):
    """Write separate TSV files for each core."""
    os.makedirs(output_dir, exist_ok=True)
    
    for core_id, rows in core_data.items():
        output_file = os.path.join(output_dir, f"core{core_id:02d}_cellseg.tsv")
        
        with open(output_file, 'w', newline='', encoding='utf-8') as file:
            # Write header
            fieldnames = ['Image', 'Object ID', 'Centroid X µm', 'Centroid Y µm']
            writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()
            
            # Write data rows
            for row in rows:
                writer.writerow({
                    'Image': row['Image'],
                    'Object ID': row['Object ID'],
                    'Centroid X µm': row['Centroid X µm'],
                    'Centroid Y µm': row['Centroid Y µm']
                })
        
        print(f"Created {output_file} with {len(rows)} centroids")


def main():
    parser = argparse.ArgumentParser(description='Divide TSV file by mask regions')
    parser.add_argument('--tsv', required=True, help='Input TSV file path')
    parser.add_argument('--pipeline-output', required=True, help='Pipeline output directory')
    parser.add_argument('--output', help='Output directory for core TSV files (default: sibling of pipeline output)')
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.tsv):
        print(f"Error: TSV file not found: {args.tsv}")
        return 1
    
    if not os.path.exists(args.pipeline_output):
        print(f"Error: Pipeline output directory not found: {args.pipeline_output}")
        return 1
    
    # Load crop mapping data and masks from pipeline
    crop_mapping = load_crop_mapping_and_masks(args.pipeline_output)
    if not crop_mapping:
        return 1
    
    # Set output directory (sibling to pipeline output by default)
    if args.output:
        output_dir = args.output
    else:
        pipeline_parent = os.path.dirname(args.pipeline_output)
        pipeline_name = os.path.basename(args.pipeline_output)
        output_dir = os.path.join(pipeline_parent, f"{pipeline_name}_cellseg_by_core")
    
    print(f"=== TSV Division by Mask Regions ===")
    print(f"Input TSV: {args.tsv}")
    print(f"Pipeline output: {args.pipeline_output}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Crop mapping and masks already loaded above
    print()
    
    # Parse TSV file
    print("Parsing TSV file...")
    rows_with_pixels = parse_tsv_file(args.tsv)
    print(f"Parsed {len(rows_with_pixels)} centroids")
    print()
    
    # Map centroids to cores
    print("Mapping centroids to cores...")
    core_data = defaultdict(list)
    unmapped_count = 0
    
    for row in rows_with_pixels:
        core_id = find_core_for_centroid(row['x_pixel'], row['y_pixel'], crop_mapping)
        
        if core_id is not None:
            core_data[core_id].append(row)
        else:
            unmapped_count += 1
    
    print(f"Mapped centroids to {len(core_data)} cores")
    print(f"Unmapped centroids: {unmapped_count}")
    
    # Display mapping summary
    for core_id in sorted(core_data.keys()):
        print(f"  Core {core_id:02d}: {len(core_data[core_id])} centroids")
    print()
    
    # Write output files
    print("Writing core TSV files...")
    write_core_tsv_files(core_data, output_dir)
    
    print(f"\nCompleted! Output files written to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
