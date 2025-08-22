#!/bin/bash
# Build script for unified tissue analysis pipeline container

set -e  # Exit on any error

echo "=== Building Unified Tissue Analysis Pipeline Container ==="
echo "This will take approximately 30-45 minutes..."
echo ""

# Check if apptainer/singularity is available
if ! command -v apptainer &> /dev/null && ! command -v singularity &> /dev/null; then
    echo "❌ Error: Neither apptainer nor singularity found."
    echo "Please install Apptainer: https://apptainer.org/docs/admin/main/installation.html"
    exit 1
fi

# Use apptainer if available, fallback to singularity
CONTAINER_CMD="apptainer"
if ! command -v apptainer &> /dev/null; then
    CONTAINER_CMD="singularity"
fi

echo "Using container runtime: $CONTAINER_CMD"
echo ""

# Build the container
echo "Building container from definition file..."
$CONTAINER_CMD build --fakeroot unified_pipeline.sif unified_pipeline.def

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Container built successfully: unified_pipeline.sif"
    echo ""
    
    # Show container info
    echo "Container information:"
    $CONTAINER_CMD inspect unified_pipeline.sif
    echo ""
    
    # Test the container
    echo "Testing container..."
    $CONTAINER_CMD exec unified_pipeline.sif python3 -c "
import numpy as np
import cellpose
from valis import registration
import cv2
print('✅ All major packages imported successfully')
print(f'NumPy version: {np.__version__}')
print(f'OpenCV version: {cv2.__version__}')
print('Container is ready for use!')
"
    
    echo ""
    echo "🚀 Container is ready! Usage:"
    echo "  Interactive: $CONTAINER_CMD run --bind \$PWD:/workspace unified_pipeline.sif"
    echo "  Scripts: $CONTAINER_CMD exec --bind \$PWD:/workspace unified_pipeline.sif python your_script.py"
    
else
    echo "❌ Container build failed. Check the error messages above."
    exit 1
fi
