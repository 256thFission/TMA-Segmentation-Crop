# TataLab Tissue Analysis Pipeline - Complete Instruction Manual

This manual provides comprehensive instructions for using the TataLab tissue analysis pipeline, which includes a unified pipeline and individual CLI tools for automated tissue processing and alignment.

## 📋 Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Environment Setup](#environment-setup)
3. [Unified Pipeline (Recommended)](#unified-pipeline-recommended)
4. [Individual CLI Tools](#individual-cli-tools)
   - [CLI Tool 1: Tissue Core Segmentation](#cli-tool-1-tissue-core-segmentation)
   - [CLI Tool 2: VALIS Image Alignment](#cli-tool-2-valis-image-alignment)
5. [Complete Workflow Examples](#complete-workflow-examples)
6. [Troubleshooting](#troubleshooting)
7. [Performance Tips](#performance-tips)

---

## 🔬 Pipeline Overview

The TataLab pipeline provides both unified and individual processing approaches:

### **Unified Pipeline (Recommended)**
- **`unified_pipeline.py`** - Complete end-to-end processing with YAML configuration
- Integrates segmentation + VALIS registration according to pipeline specification
- Processes both DAPI and stain modalities with per-core alignment
- Canonical directory structure with final aggregated results

### **Individual CLI Tools**
- **`simple_main.py`** - Automated tissue core detection and segmentation using Cellpose-SAM
- **`myvalis/valis_cli.py`** - VALIS-based image alignment with coordinate transformation

### Typical Workflows

**Unified Approach:**
```
YAML Config → Unified Pipeline → Segmentation (DAPI + Stain) → Per-Core VALIS → Aggregated Results
```

**Individual Tools Approach:**
```
Raw Tissue Image → Core Segmentation → Individual Cores → VALIS Alignment → Aligned Coordinates
```

---

## 🛠️ Environment Setup

### Option 1: Unified Environment (Recommended)

This setup supports both the unified pipeline and individual CLI tools in a single environment.

```bash
# 1. Create the unified environment
conda env create -f environment_unified.yml

# 2. Activate environment  
conda activate cellSAM_valis_env

# 3. Install additional unified pipeline dependencies
pip install pyyaml rich pandas valis

# 4. Verify installation
python unified_pipeline.py --help
python simple_main.py --help
python myvalis/valis_cli.py --help
```

**Key Dependencies Included:**
- Python 3.9 (required by VALIS)
- Cellpose 4.0+ for segmentation
- VALIS-WSI for alignment
- PyYAML for configuration management
- Rich for enhanced CLI output
- Pandas for data manipulation
- OpenJDK 11 + Bioformats for slide reading
- PyTorch 2.2.0 (CPU version for stability)
- Scientific computing stack (numpy, scipy, opencv, etc.)

### Option 2: Legacy Separate Environment

For segmentation-only workflows:

```bash
# 1. Create basic environment
conda create -n cellpose_env python=3.9 -y
conda activate cellpose_env

# 2. Install conda packages
conda install -c conda-forge numpy scipy scikit-image opencv tifffile typer -y

# 3. Install pip packages  
pip install -r requirements.txt
```

### Option 3: Container-Based Setup

For portable, reproducible analysis:

```bash
# Build container (one-time setup)
apptainer build unified_pipeline.sif unified_pipeline.def

# Run with data binding
apptainer exec --bind $PWD:/workspace unified_pipeline.sif python simple_main.py [options]
```

---

## 🚀 Unified Pipeline (Recommended)

### Purpose
Complete end-to-end tissue core alignment pipeline that integrates segmentation and VALIS registration according to the pipeline specification in `docs/pipeline_plan.md`.

### Key Features
- **YAML Configuration**: Single config file for all parameters
- **Dual Modality Processing**: Segments both DAPI and stain images
- **Canonical Directory Structure**: Follows specification layout
- **Per-Core VALIS Registration**: Individual core alignment in `valis_work/core_<ID>/`
- **Aggregated Results**: Final transformed centroids in `outputs/`

### Basic Usage

```bash
python unified_pipeline.py --config inputs/config.yaml
```

### Configuration File Structure

The unified pipeline uses a YAML configuration file (`inputs/config.yaml`) with the following sections:

```yaml
# Input files and paths
inputs:
  dapi_image: "inputs/tissue_dapi_fullres.tif"
  stain_image: "inputs/tissue_hires_image.png"
  centroids_tsv: "inputs/rkoX_cellseg.tsv"

# Global settings
global:
  microns_per_pixel: 0.5072
  cores_per_row: 4
  expected_cores: 20
  flip_axis: "y"
  verbose: true

# Segmentation parameters (both modalities)
segmentation:
  downsample_factor: 64
  min_radius_pct: 1.0
  max_radius_pct: 30.0
  cellpose:
    flow_threshold: 0.6
    cellprob_threshold: 0.3
    use_gpu: true

# VALIS registration parameters
valis:
  preprocessing:
    scale_factor: 0.1022
  superglue:
    weights: "indoor"
    keypoint_threshold: 0.005
    match_threshold: 0.2
  save_overlays: true
```

### Output Directory Structure

The unified pipeline creates the canonical directory structure specified in the pipeline plan:

```
inputs/
  tissue_dapi_fullres.tif
  tissue_hires_image.png
  rkoX_cellseg.tsv
  config.yaml

fullres2/
  tissue_dapi_fullres_processed/    # DAPI segmentation results
    cores_filtered/
    cores_raw/
    masks/
    detection_overlay.png
    detection_report.json
  tissue_hires_image_processed/     # Stain segmentation results
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

outputs/
  rkoX_cellseg_transformed.tsv      # Final aggregated results
```

### Command Line Options

| Parameter | Flag | Type | Default | Description |
|-----------|------|------|---------|-------------|
| **Config File** | `--config`, `-c` | string | `inputs/config.yaml` | Path to YAML configuration file |
| **Verbose Mode** | `--verbose`, `-v` | flag | From config | Override config verbose setting |

### Example Commands

```bash
# Basic usage with default configuration
python unified_pipeline.py

# Custom configuration file
python unified_pipeline.py --config my_config.yaml

# Override verbose setting
python unified_pipeline.py --config inputs/config.yaml --verbose

# Check configuration without running
python unified_pipeline.py --config inputs/config.yaml --verbose | head -20
```

### Pipeline Stages

The unified pipeline executes the following stages:

1. **Load Configuration**: Parse YAML config and validate paths
2. **DAPI Segmentation**: Detect and crop tissue cores from DAPI image
3. **Stain Segmentation**: Detect and crop tissue cores from stain image  
4. **Centroid Preprocessing**: Convert coordinates from µm to pixels
5. **Per-Core VALIS**: Register each core pair individually
6. **Result Aggregation**: Combine transformed centroids into final output

### Performance Assessment

The unified pipeline provides comprehensive performance reporting:

- **Detection rates** for both modalities
- **Processing times** for each stage
- **Registration quality** metrics per core
- **Final summary** with total centroids transformed

---

## 🔧 Individual CLI Tools

For advanced users who need granular control, the individual CLI tools remain available:

## 🔍 CLI Tool 1: Tissue Core Segmentation

### Purpose
Automated detection and segmentation of individual tissue cores from tissue microarray (TMA) images.

### Basic Usage

```bash
python simple_main.py --input your_image.tif --output results
```

### Complete Parameter Reference

| Parameter | Flag | Type | Default | Description | Recommended Value |
|-----------|------|------|---------|-------------|------------------|
| **Input Image** | `--input`, `-i` | string | Required | Path to input tissue image (TIFF, PNG, JPG) | `inputs/tissue_dapi_fullres.tif` |
| **Output Directory** | `--output`, `-o` | string | `simple_output` | Directory for all output files and crops | `results` |
| **Downsample Factor** | `--downsample` | int | 64 | Factor to reduce image size for processing | `8` (thumbnail), `64` (full-res) |
| **Expected Cores** | `--expected-cores` | int | 20 | Expected number of cores (for performance assessment) | `20` |
| **Min Radius %** | `--min-radius-pct` | float | 1.0 | Minimum core radius as % of image width | `1.0` |
| **Max Radius %** | `--max-radius-pct` | float | 30.0 | Maximum core radius as % of image width | `30.0` |
| **Flow Threshold** | `--flow-threshold` | float | 0.6 | Cellpose flow threshold (higher = fewer false positives) | `0.6` |
| **Cell Prob Threshold** | `--cellprob-threshold` | float | 0.3 | Cell probability threshold (higher = less noise) | `0.3` |
| **Flat Field Correction** | `--flat-field` | flag | True | Enable illumination normalization | Default ON |
| **Skip Masking** | `--skip-masking` | flag | False | Export raw crops without masks | - |
| **Verbose Mode** | `--verbose`, `-v` | flag | False | Enable detailed logging | - |

### Input Requirements

- **Image Formats**: TIFF, PNG, JPEG
- **Image Types**: RGB or grayscale tissue microarray images
- **Recommended Resolution**: 30k+ pixels for optimal detection

### Output Structure

```
results/
├── processed_image.tif          # Preprocessed image
├── overlay_simple.png           # Detection overlay
├── core_info.json              # Detection metadata
├── masks/                      # Individual core masks
│   ├── core_001_mask.tif
│   └── ...
└── cropped_cores/              # Individual core images
    ├── core_001.tif
    └── ...
```

### Example Commands

```bash
# Basic segmentation with performance assessment
python simple_main.py --input tissue_sample.tif --output results --expected-cores 20

# High-resolution processing with verbose output
python simple_main.py \
    --input high_res_image.tif \
    --output detailed_results \
    --downsample 8 \
    --flow-threshold 0.4 \
    --cellprob-threshold 0.2 \
    --verbose

# Quick processing for thumbnails
python simple_main.py \
    --input thumbnail.png \
    --output quick_results \
    --downsample 2 \
    --expected-cores 12
```

---

## 🎯 CLI Tool 2: VALIS Image Alignment

### Purpose
Align tissue cores between different imaging modalities (e.g., DAPI to H&E) and transform cell centroid coordinates.

### Basic Usage

```bash
cd myvalis/
python valis_cli.py align stain_image.png dapi_image.png centroids.tsv
```

### CLI Commands

1. **`align`** - Main alignment command
2. **`info`** - Display TSV file structure  
3. **`validate`** - Validate input files

### Complete Parameter Reference

| Parameter | Flag | Type | Default | Description |
|-----------|------|------|---------|-------------|
| **Stain Image** | positional | string | Required | Reference tissue image (H&E, IHC) |
| **DAPI Image** | positional | string | Required | Nuclear channel image to align |
| **Centroids TSV** | positional | string | Required | Cell centroid coordinates file |
| **Working Directory** | `--workdir` | string | `./valis_work` | Temporary processing directory |
| **Output File** | `--output` | string | `centroids_transformed.tsv` | Output transformed coordinates |
| **Scale Factor** | `--scale-factor` | float | 0.1022 | DAPI downscaling factor |
| **Horizontal Flip** | `--flip/--no-flip` | flag | True | Flip DAPI horizontally |
| **X Column** | `--x-col` | string | `"Centroid X µm"` | X coordinate column name |
| **Y Column** | `--y-col` | string | `"Centroid Y µm"` | Y coordinate column name |
| **SuperGlue Weights** | `--weights` | string | `indoor` | Model weights: "indoor" or "outdoor" |
| **Keypoint Threshold** | `--keypoint-threshold` | float | 0.005 | Keypoint detection sensitivity |
| **Match Threshold** | `--match-threshold` | float | 0.2 | Feature matching threshold |
| **Force CPU** | `--force-cpu` | flag | False | Disable CUDA acceleration |
| **Verbose Mode** | `--verbose` | flag | False | Enable detailed output |
| **Dry Run** | `--dry-run` | flag | False | Show configuration without processing |

### Input Requirements

#### Images
- **Stain Image**: RGB tissue image (H&E, IHC, etc.) - serves as reference
- **DAPI Image**: Nuclear channel image to be aligned
- **Formats**: PNG, TIFF, JPEG

#### Centroids TSV File
Required tab-separated format with headers:

```tsv
Cell_ID	Centroid X µm	Centroid Y µm	Area	Class
1	1234.5	2345.6	150.2	Epithelial
2	1456.7	2567.8	120.8	Stromal
```

- **X Column**: Default "Centroid X µm" (customizable with `--x-col`)
- **Y Column**: Default "Centroid Y µm" (customizable with `--y-col`)
- **Additional Columns**: Preserved in output

### Output Structure

#### Transformed Centroids TSV
Original columns plus new aligned coordinates:
- **`Stain_X_px`** - X coordinates in stain image space (integer pixels)
- **`Stain_Y_px`** - Y coordinates in stain image space (integer pixels)

#### Working Directory
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

### Example Commands

```bash
# Check TSV file structure first
python valis_cli.py info centroids.tsv

# Validate all inputs before processing
python valis_cli.py validate stain.png dapi.png centroids.tsv

# Basic alignment
python valis_cli.py align stain.png dapi.png centroids.tsv

# Advanced alignment with custom parameters
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
    stain.png dapi.png centroids.tsv

# For macro/overview images
python valis_cli.py align --weights outdoor stain.png dapi.png centroids.tsv

# CPU-only processing (if GPU memory issues)
python valis_cli.py align --force-cpu stain.png dapi.png centroids.tsv
```

---

## 🔄 Complete Workflow Examples

### Example 1: Basic TMA Processing (Unified Pipeline - Recommended)

```bash
# Step 1: Prepare configuration
# Edit inputs/config.yaml with your file paths

# Step 2: Run complete pipeline
python unified_pipeline.py --config inputs/config.yaml

# Results automatically generated in canonical structure:
# - fullres2/ contains segmentation results for both modalities
# - valis_work/ contains per-core registration results
# - outputs/ contains final aggregated transformed centroids
```

### Example 2: Custom Configuration Example

```bash
# Create custom config for specific experiment
cat > my_experiment_config.yaml << EOF
inputs:
  dapi_image: "data/experiment_1/dapi_fullres.tif"
  stain_image: "data/experiment_1/he_stain.png"
  centroids_tsv: "data/experiment_1/cell_centroids.tsv"

outputs:
  base_dir: "results/experiment_1/fullres2"
  valis_workdir: "results/experiment_1/valis_work"
  final_output: "results/experiment_1/outputs/transformed_centroids.tsv"

global:
  expected_cores: 16
  cores_per_row: 4
  verbose: true

segmentation:
  downsample_factor: 32
  cellpose:
    flow_threshold: 0.7
    cellprob_threshold: 0.4
EOF

# Run with custom configuration
python unified_pipeline.py --config my_experiment_config.yaml
```

### Example 3: Individual Tools Workflow (Legacy)

For users who need granular control over each step:

```bash
# Step 1: Segment DAPI tissue cores
python simple_main.py \
    --input inputs/tissue_dapi_fullres.tif \
    --output fullres2/dapi_results \
    --expected-cores 20

# Step 2: Segment stain tissue cores  
python simple_main.py \
    --input inputs/tissue_hires_image.png \
    --output fullres2/stain_results \
    --expected-cores 20

# Step 3: Process each core pair with VALIS
cd myvalis/
python valis_cli.py align \
    ../fullres2/stain_results/cropped_cores/core_001.tif \
    ../fullres2/dapi_results/cropped_cores/core_001.tif \
    ../inputs/core_001_centroids.tsv \
    --workdir ../valis_work/core_01
```

### Example 4: Container-Based Processing

```bash
# Build container with unified pipeline
apptainer build unified_pipeline.sif unified_pipeline.def

# Run unified pipeline in container
apptainer exec --bind $PWD:/workspace unified_pipeline.sif \
    python unified_pipeline.py \
    --config /workspace/inputs/config.yaml

# Alternative: Run individual tools in container
apptainer exec --bind $PWD:/workspace unified_pipeline.sif \
    python simple_main.py \
    --input /workspace/inputs/tissue_sample.tif \
    --output /workspace/results
```

---

## 🔧 Troubleshooting

### Environment Issues

**Memory Error: "Conda SSL certificate issues"**
```bash
# Use the unified environment instead of pip
conda activate cellSAM_valis_env
# Environment includes pre-resolved dependencies
```

**ImportError: "Could not import valis_alignment module"**
```bash
# Ensure you're in the unified environment
conda activate cellSAM_valis_env
# Check VALIS installation
python -c "import valis; print('VALIS OK')"
```

### Segmentation Issues

**Poor Detection Rate (<90%)**
- Lower `--flow-threshold` (try 0.4-0.5)
- Adjust `--downsample` factor
- Check `--min-radius-pct` and `--max-radius-pct`

**Processing Hangs/Takes Forever**
- Increase `--downsample` factor (try 64 or higher)
- Reduce image resolution before processing
- Enable `--verbose` to monitor progress







