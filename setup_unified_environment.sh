#!/bin/bash
# Unified Environment Setup Script
# This script creates a unified conda environment for both cellSAM and Valis workflows

set -e  # Exit on any error

echo "=== Setting up Unified cellSAM + Valis Environment ==="

# Step 1: Create the unified conda environment
echo "[1/4] Creating unified conda environment..."
conda env create -f environment_unified.yml

# Step 2: Activate the environment
echo "[2/4] Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate cellSAM_valis_env

# Step 3: Set up Java environment variables
echo "[3/4] Setting up Java environment..."
export JAVA_HOME=$CONDA_PREFIX
echo "export JAVA_HOME=$CONDA_PREFIX" >> ~/.bashrc

# Step 4: Download Bioformats jar (required for Valis)
echo "[4/4] Downloading Bioformats jar..."
mkdir -p ~/bioformats
wget -O ~/bioformats/bioformats_package.jar https://downloads.openmicroscopy.org/bio-formats/7.0.0/artifacts/bioformats_package.jar

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "To activate the unified environment, run:"
echo "  conda activate cellSAM_valis_env"
echo ""
echo "Bioformats jar is located at: ~/bioformats/bioformats_package.jar"
echo "Add this path to your Valis configuration if needed."
