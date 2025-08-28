# Unified Pipeline - Modular Architecture

This document describes the refactored modular architecture of the unified tissue core alignment pipeline.

## Directory Structure

```
unified_pipeline/
├── __init__.py                 # Package initialization  
├── config/
│   ├── __init__.py
│   └── config.py              # Configuration management (UnifiedConfig)
├── segmentation/
│   ├── __init__.py  
│   └── segmentation.py        # Tissue core segmentation (SegmentationEngine)
├── registration/
│   ├── __init__.py
│   └── valis.py              # VALIS registration logic (ValisEngine) 
├── core/
│   ├── __init__.py
│   ├── processing.py         # Core processing utilities (CoreProcessor)
│   └── utils.py             # General utilities (CoreUtils)
├── pipeline.py              # Main orchestration (UnifiedPipeline)
└── cli.py                   # Command-line interface
```

## Key Improvements

### **Separation of Concerns**
- **Configuration**: `config/config.py` - YAML loading, validation, path resolution
- **Segmentation**: `segmentation/segmentation.py` - Tissue core detection and cropping
- **Registration**: `registration/valis.py` - VALIS alignment and result aggregation
- **Core Processing**: `core/processing.py` - Centroid preprocessing and transformations
- **Utilities**: `core/utils.py` - Logging, core naming, spatial filtering
- **Orchestration**: `pipeline.py` - Main pipeline coordination
- **CLI**: `cli.py` - Command-line interface

### **Maintainability Benefits**
- **Modular**: Each component has single responsibility
- **Testable**: Components can be tested independently  
- **Readable**: ~150 lines per module vs 692 lines monolithic
- **Extensible**: Easy to add new modalities or registration methods
- **Reusable**: Components can be used independently

## Usage

### **Option 1: Container Execution (Recommended)**
```bash
# Build container with unified pipeline
apptainer build unified_pipeline.sif unified_pipeline.def

# Run refactored pipeline in container
apptainer exec --bind $PWD:/workspace unified_pipeline.sif \
    python unified_pipeline_refactored.py \
    --config /workspace/inputs/config.yaml
```

### **Internal API Changes**
If extending the pipeline, use the new modular components:
```python
from unified_pipeline.config import UnifiedConfig
from unified_pipeline.segmentation import SegmentationEngine
from unified_pipeline.registration import ValisEngine
```

## File Overview

| Component | Responsibility | Key Classes |
|-----------|---------------|-------------|
| `config/config.py` | Configuration management | `UnifiedConfig` |
| `segmentation/segmentation.py` | Core detection & cropping | `SegmentationEngine` |
| `registration/valis.py` | VALIS alignment | `ValisEngine` |  
| `core/processing.py` | Centroid preprocessing | `CoreProcessor` |
| `core/utils.py` | Utilities & logging | `CoreUtils` |
| `pipeline.py` | Main orchestration | `UnifiedPipeline` |
| `cli.py` | Command-line interface | `main()` |

The refactored architecture maintains all existing functionality while dramatically improving code organization, maintainability, and extensibility.
