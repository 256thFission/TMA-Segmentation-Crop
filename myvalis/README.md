r# VALIS Tissue Core Alignment Pipeline

This module provides a complete pipeline for aligning tissue cores using VALIS (Virtual Alignment of pathoLogy Image Series) with SuperPoint/SuperGlue feature matching, and transforming cell centroid coordinates between image spaces.

## Overview

The VALIS alignment pipeline is designed to register DAPI (nuclear channel) images to stained tissue images (H&E, IHC) and transform cell centroid coordinates from the original DAPI space to the aligned stain space. This is particularly useful for correlating cell segmentation data between different imaging modalities.

## Files

- **`valis_alignment_clean.py`** - Core pipeline implementation with all alignment functions
- **`valis_cli.py`** - Command-line interface for easy usage
- **`README.md`** - This documentation file

## Features

- **Image Preprocessing**: Automatic DAPI image downscaling and horizontal flipping
- **Advanced Registration**: Uses SuperPoint/SuperGlue deep learning features for robust alignment
- **Coordinate Transformation**: Transforms cell centroids from DAPI to stain coordinate systems
- **CLI Interface**: User-friendly command-line interface with validation
- **Error Handling**: Comprehensive error handling and cleanup
- **Progress Tracking**: Rich console output with progress indicators

## Installation

Ensure you have the required environment set up. From the project root:

```bash
# Create and activate the unified environment
conda env create -f environment_unified.yml
conda activate cellSAM_valis_env
```

## Usage

### Command Line Interface (Recommended)

The CLI provides the easiest way to use the alignment pipeline:

#### Basic Usage

```bash
# Navigate to the myvalis directory
cd myvalis/

# Basic alignment command
python valis_cli.py align \
    /path/to/stain_image.png \
    /path/to/dapi_image.png \
    /path/to/centroids.tsv
```

#### Advanced Usage with Options

```bash
python valis_cli.py align \
    --workdir ./my_alignment_work \
    --output ./results/transformed_centroids.tsv \
    --scale-factor 0.15 \
    --no-flip \
    --x-col "X_coordinate" \
    --y-col "Y_coordinate" \
    --weights outdoor \
    --keypoint-threshold 0.01 \
    --match-threshold 0.3 \
    --verbose \
    /path/to/stain_image.png \
    /path/to/dapi_image.png \
    /path/to/centroids.tsv
```

#### CLI Commands

1. **`align`** - Main alignment command
2. **`info`** - Display information about a TSV file
3. **`validate`** - Validate input files before processing

#### Key Parameters

- **`--scale-factor`** - DAPI downscaling factor (default: 0.1022)
- **`--flip/--no-flip`** - Whether to flip DAPI horizontally (default: True)
- **`--x-col`** - X coordinate column name (default: "Centroid X µm")
- **`--y-col`** - Y coordinate column name (default: "Centroid Y µm")
- **`--weights`** - SuperGlue weights: "indoor" or "outdoor" (default: "indoor")
- **`--workdir`** - Working directory for temporary files (default: "./valis_work")
- **`--verbose`** - Enable detailed output
- **`--dry-run`** - Show configuration without processing

#### Examples

```bash
# Check TSV file structure
python valis_cli.py info centroids.tsv

# Validate inputs before processing
python valis_cli.py validate stain.png dapi.png centroids.tsv

# Run with outdoor SuperGlue weights for better outdoor/macro images
python valis_cli.py align --weights outdoor stain.png dapi.png centroids.tsv

# Process with different scaling and column names
python valis_cli.py align \
    --scale-factor 0.2 \
    --x-col "X_position" \
    --y-col "Y_position" \
    stain.png dapi.png centroids.tsv
```

### Python API Usage

For programmatic usage or custom workflows:

```python
from valis_alignment_clean import AlignmentConfig, run_alignment_pipeline

# Create configuration
config = AlignmentConfig()
config.stain_path = Path("stain_image.png")
config.dapi_path = Path("dapi_image.png") 
config.centroids_tsv = Path("centroids.tsv")
config.workdir = Path("./alignment_work")

# Customize parameters
config.scale_factor = 0.15
config.flip_horizontal = False
config.superglue_config['weights'] = 'outdoor'

# Run pipeline
results = run_alignment_pipeline(config)

# Access results
transformed_df = results['transformed_centroids']
registration_error = results['registration_error']
```

## Input Requirements

### Images

- **Stain Image**: RGB tissue image (H&E, IHC, etc.) serving as reference
- **DAPI Image**: Nuclear channel image to be aligned to stain image
- **Supported Formats**: PNG, TIFF, JPEG

### Centroids TSV File

Required format: Tab-separated values with headers including:
- **X coordinate column**: Default "Centroid X µm" (customizable)
- **Y coordinate column**: Default "Centroid Y µm" (customizable)
- **Additional columns**: Preserved in output

Example TSV structure:
```
Cell_ID	Centroid X µm	Centroid Y µm	Area	Class
1	1234.5	2345.6	150.2	Epithelial
2	1456.7	2567.8	120.8	Stromal
...
```

## Output

### Transformed Centroids TSV

The output TSV contains all original columns plus:
- **`Stain_X_px`** - X coordinates in stain image space (integer pixels)
- **`Stain_Y_px`** - Y coordinates in stain image space (integer pixels)

### Working Directory Structure

```
valis_work/
├── src/                          # Source images for registration
│   ├── stain_image.png
│   └── dapi_processed.png
├── results/                      # VALIS registration outputs
│   ├── aligned/
│   └── registration_stats.csv
├── dapi_processed.png           # Preprocessed DAPI image
└── centroids_transformed.tsv   # Final output
```

## Configuration Parameters

### Image Preprocessing
- **`scale_factor`**: Downscaling factor for DAPI (default: 0.1022)
- **`flip_horizontal`**: Horizontal flip of DAPI (default: True)

### SuperGlue Registration
- **`weights`**: Model weights - "indoor" (default) or "outdoor"
- **`keypoint_threshold`**: Keypoint detection sensitivity (default: 0.005)
- **`match_threshold`**: Feature matching threshold (default: 0.2)
- **`force_cpu`**: Disable CUDA acceleration (default: False)

### Column Mapping
- **`centroid_x_col`**: X coordinate column name
- **`centroid_y_col`**: Y coordinate column name

## Performance Notes

- **GPU Acceleration**: Automatically uses CUDA if available
- **Memory Usage**: Large images are processed efficiently with downscaling
- **Processing Time**: Typical core alignment takes 30-60 seconds
- **Registration Quality**: SuperGlue provides robust feature matching even with different staining

## Error Handling

The pipeline includes comprehensive error handling:

- **File validation**: Checks for missing or unreadable files
- **Column validation**: Verifies required TSV columns exist
- **JVM cleanup**: Automatic VALIS Java Virtual Machine cleanup
- **Graceful failure**: Clear error messages with suggested fixes

## Troubleshooting

### Common Issues

1. **Import Error**: Ensure VALIS environment is activated
   ```bash
   conda activate cellSAM_valis_env
   ```

2. **CUDA Out of Memory**: Use `--force-cpu` flag
   ```bash
   python valis_cli.py align --force-cpu ...
   ```

3. **Registration Failure**: Try different SuperGlue weights
   ```bash
   python valis_cli.py align --weights outdoor ...
   ```

4. **Missing Columns**: Check TSV structure with info command
   ```bash
   python valis_cli.py info centroids.tsv
   ```

### Performance Tuning

- **Large Images**: Adjust `--scale-factor` to balance speed vs accuracy
- **Poor Matches**: Try `--weights outdoor` for macro/overview images
- **Low Features**: Lower `--keypoint-threshold` for more features
- **Strict Matching**: Increase `--match-threshold` for stricter matches

## Integration with TataLab Pipeline

This VALIS module integrates with the broader tissue analysis pipeline:

1. **Input**: Uses tissue cores from `simple_main.py` segmentation
2. **Processing**: Aligns DAPI to stain images with coordinate transformation
3. **Output**: Provides aligned centroids for downstream spatial analysis

## Dependencies

Key packages (installed via `environment_unified.yml`):
- `valis-wsi` - Core VALIS package
- `torch` - PyTorch for SuperPoint/SuperGlue
- `opencv` - Image processing
- `pandas` - Data manipulation
- `typer` - CLI framework
- `rich` - Enhanced console output

## Version Compatibility

- **Python**: 3.9 (required by VALIS)
- **VALIS**: Latest version from PyPI
- **PyTorch**: 2.2.0 (CPU version for compatibility)
- **OpenCV**: 4.5.0+

## Citation

If using this pipeline in research, please cite:
- VALIS: [VALIS paper/repository]
- SuperGlue: Sarlin et al. "SuperGlue: Learning Feature Matching with Graph Neural Networks"
- SuperPoint: DeTone et al. "SuperPoint: Self-Supervised Interest Point Detection and Description"
