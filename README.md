# Cellpose-SAM Tissue Core Segmentation

Automated tissue core detection and segmentation for tissue microarray (TMA) extraction


It  automatically segments and crops individual tissue cores from microscopy images. 

## Setup Instructions


### 1. Create a Conda Env


```bash
# Create a new conda environment with Python 3.9
conda create -n cellpose_env python=3.9

# Activate the environment
conda activate cellpose_env
```

### 2. Install Dependencies

Install the required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```


## Basic Usage

```bash
python simple_main.py --input your_image.tif --output results


!!! Note, you'll most likely have to adjust the downsampling factor. ~8 works well for the thumbnail stain images, and the default works bettetr for the ~ 30k pixel image.

```

## Example with Performance Assessment

```bash
python simple_main.py --input tissue_sample.tif --output results --expected-cores 20
```

## CLI Parameters

| Parameter | Flag | Type | Default | Description | Recommended Value |
|-----------|------|------|---------|-------------|------------------|
| **Input Image** | `--input`, `-i` | string | Required | Path to input tissue image (supports TIFF, PNG, JPG) | `work/inputs/tissue_dapi_fullres.tif` |
| **Output Directory** | `--output`, `-o` | string | `simple_output` | Directory for all output files and crops | `results` |
| **Downsample Factor** | `--downsample` | int | 8 | Factor to reduce image size for processing (higher = faster, lower = more detail) | `64` |
| **Expected Cores** | `--expected-cores` | int | 20 | Number of tissue cores expected in the image (for performance assessment) | `20` |
| **Minimum Radius %** | `--min-radius-pct` | float | 1.0 | Minimum core radius as percentage of image width (filters small detections) | 1.0 |
| **Maximum Radius %** | `--max-radius-pct` | float | 30.0 | Maximum core radius as percentage of image width (filters large detections) | 30.0 |
| **Flow Threshold** | `--flow-threshold` | float | 0.4 | Cellpose flow threshold - higher values reduce false positives | `0.6` |
| **Cell Probability Threshold** | `--cellprob-threshold` | float | 0.0 | Cellpose cell probability threshold - higher values reduce noise detections | `0.3` |
| **Flat Field Correction** | `--flat-field` | flag | True | Enable flat field correction (illumination normalization).  | Default ON |


## Processing Pipeline

Raw Image → Format Conversion → Flat Field Correction → Downsampling → 
Contrast Enhancement → Noise Reduction → Processed Image

