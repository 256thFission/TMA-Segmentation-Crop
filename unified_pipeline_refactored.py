#!/usr/bin/env python3
"""
Unified Tissue Core Alignment Pipeline - Refactored Entry Point

This is the new main entry point that uses the modular unified_pipeline package.
Run this instead of the old unified_pipeline.py file.

Usage:
    python unified_pipeline_refactored.py --config inputs/config.yaml
"""

import typer
from unified_pipeline.cli import main

if __name__ == "__main__":
    typer.run(main)
