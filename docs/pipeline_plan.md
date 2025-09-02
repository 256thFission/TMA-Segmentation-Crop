# Tissue Core Alignment Pipeline — Issues and Implementation Plan

This document summarizes the current gaps and a staged plan to complete the pipeline that aligns tissue cores between stain and DAPI images, and maps coordinates across modalities.

## Goal

- Input: two WSIs (stain, DAPI), 2 CSV grids, global DAPI centroids TSV.
- Output: per-core transformed centroids in stain coordinates and robust stain↔DAPI WSI coordinate mapping.


Canonical directory layout (example)

```
inputs/
  tissue_dapi_fullres.tif
  tissue_hires_image.png
  rko4_pcf.csv
  rko4_visium.csv
  rkoX_cellseg.tsv   # global DAPI centroids
  config.yaml

fullres2/
  tissue_dapi_fullres_processed/ # dapi slide
    cores_filtered/
    cores_raw/
    masks/
    detection_overlay.png
    detection_report.json
  tissue_hires_image_processed/ #stain sldie            
    cores_filtered/
    cores_raw/
    masks/
    detection_overlay.png
    detection_report.json

valis_work/
  core_01/
    src/
    results/
      overlays/
    centroids_transformed.tsv
  core_02/
    ...

# Final aggregated output
outputs/
  rkoX_cellseg_transformed.tsv
```

Pipeline flow (high level)

inputs -> fullres2/<modality>_processed (segment + masks + report) -> µm→px conversion + DAPI preprocess (scale+Y-flip) -> VALIS registration -> per-core transformed TSVs in valis_work/core_<ID>/ -> aggregate -> outputs/rkoX_cellseg_transformed.tsv

## Defaults / Preferences

- **Microns-per-pixel default**: 0.5072 µm/px unless overridden.
- **Flip axis**: vertical flip (flip along Y axis).
- **Paths as variables**: use provided variable paths for WSIs:
  - DAPI: `inputs/tissue_dapi_fullres.tif`
  - Stain: `inputs/tissue_hires_image.png`
- **Canonical core IDs**: enforce IDs from the CSV grids.
- **Cores per row**: read from config, default to 4.
- **Per-core outputs**: disabled by default; only final transforms unless a flag enables per-core artifacts.

## Artifacts specification

- **Segmentation report**: `fullres2/<basename>_processed/detection_report.json`
  - Global keys: `input_path`, `processing_time`, `total_detected`, `filtered_cores`, `expected_cores`, `detection_rate`, `output_directory`, `overlay_image`, `cores_filtered_directory`, `cores_raw_directory`, `masks_directory`, `scale_factor`.
  - Per-core list `crop_mapping` entries include: `core_id`, `image_filename`, `mask_filename`, `center_original`, `radius_original`, `crop_bounds` (x1,y1,x2,y2), `crop_size`, `mask_id`.
- **Segmentation artifacts**: under `fullres2/<basename>_processed/`
  - `cores_filtered/`, `cores_raw/`, `masks/`, `detection_overlay.png`, `detection_report.json`.
- **VALIS per-core outputs**: under `valis_work/core_<ID>/`
  - `src/`, `results/overlays/`, `centroids_transformed.tsv`.
-
  - **Final aggregated output**: `outputs/rkoX_cellseg_transformed.tsv` (collected from per-core results).
