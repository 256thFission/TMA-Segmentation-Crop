# Unified Tissue Analysis Pipeline Installation

This pipeline combines cell segmentation (cellSAM/cellpose) and whole slide image registration (Valis) in a single portable container.

## Quick Start (Researchers)

### Prerequisites
- Linux system with Apptainer/Singularity installed
- No other dependencies needed!

### 1. Download Container
```bash
# Download the pre-built container (when available)
wget https://your-server.com/unified_pipeline.sif

# OR build from definition file
apptainer build unified_pipeline.sif unified_pipeline.def
```

### 2. Run Your Analysis
```bash
# For interactive Jupyter Lab (recommended for exploration)
apptainer run --bind /path/to/your/data:/workspace unified_pipeline.sif

# For command-line analysis
apptainer exec --bind /path/to/your/data:/workspace unified_pipeline.sif python your_script.py

# For cell segmentation pipeline
apptainer exec --bind /path/to/your/data:/workspace unified_pipeline.sif python simple_main.py
```

### 3. Access Jupyter Lab
After running the container, open your browser to: `http://localhost:8888`

## What's Included

### ✅ Cell Segmentation Stack
- **cellpose** 4.0+ - Advanced cell segmentation
- **numpy, scipy, scikit-image** - Core scientific computing
- **opencv** - Computer vision processing
- **tifffile** - TIFF image handling

### ✅ Valis Registration Stack  
- **valis-wsi** - Whole slide image registration
- **libvips 8.15.2** - High-performance image processing
- **OpenSlide** - Slide format support
- **Java JDK 11 + Bioformats** - Multi-format slide reading
- **PyTorch 2.2.0** - Deep learning backend

### ✅ Analysis Environment
- **Jupyter Lab** - Interactive notebooks
- **matplotlib, seaborn, pandas** - Data visualization and analysis

## Directory Structure
```
/workspace/     # Mount your data here
/data/          # Additional data directory  
/results/       # Output directory
```

## Example Workflows

### Cell Segmentation
```bash
apptainer exec --bind $PWD:/workspace unified_pipeline.sif \
  python simple_main.py --input /workspace/images --output /workspace/results
```

### Valis Registration
```bash
apptainer exec --bind $PWD:/workspace unified_pipeline.sif \
  python valis_script.py --src_dir /workspace/slides --dst_dir /workspace/registered
```

### Interactive Analysis
```bash
# Start Jupyter Lab
apptainer run --bind $PWD:/workspace unified_pipeline.sif

# Then open notebook and use both tools:
import cellpose
from valis import registration
# ... your analysis code
```

## Troubleshooting

### Memory Issues
```bash
# Increase memory limit for large images
apptainer run --memory=20g --bind $PWD:/workspace unified_pipeline.sif
```

### File Permissions
```bash
# If permission issues occur
chmod -R 755 /path/to/your/data
```

### Java/Bioformats Issues
The container includes pre-configured Java and Bioformats. If issues occur:
```bash
# Check Java setup
apptainer exec unified_pipeline.sif echo $JAVA_HOME
apptainer exec unified_pipeline.sif java -version
```

## Building from Source

```bash
# Clone repository
git clone https://github.com/your-lab/tissue-analysis-pipeline
cd tissue-analysis-pipeline

# Build container (requires ~30 minutes)
apptainer build unified_pipeline.sif unified_pipeline.def

# Test installation
./test_container.sh
```

## Support
- 📧 Contact: [your-email@university.edu]
- 📖 Documentation: [link-to-detailed-docs]
- 🐛 Issues: [github-issues-link]
