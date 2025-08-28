"""Configuration management for unified pipeline."""

import yaml
from pathlib import Path
from typing import Any, Dict


class UnifiedConfig:
    """Unified configuration loaded from YAML"""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.data = self._load_config()
        self._validate_config()
        
    def _load_config(self) -> dict:
        """Load YAML configuration file"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"Failed to load config from {self.config_path}: {e}")
    
    def _validate_config(self):
        """Validate required configuration sections"""
        required_sections = ['inputs', 'outputs', 'global', 'segmentation', 'valis', 'coordinates']
        for section in required_sections:
            if section not in self.data:
                raise ValueError(f"Missing required config section: {section}")
    
    def get(self, key_path: str, default=None) -> Any:
        """Get configuration value using dot notation (e.g., 'segmentation.cellpose.flow_threshold')"""
        keys = key_path.split('.')
        value = self.data
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def get_input_path(self, key: str) -> Path:
        """Get input file path, resolved relative to config directory"""
        path_str = self.get(f'inputs.{key}')
        if not path_str:
            raise ValueError(f"Input path '{key}' not found in config")
        
        path = Path(path_str)
        if not path.is_absolute():
            path = self.config_path.parent / path
        return path
    
    def get_output_dir(self, key: str) -> Path:
        """Get output directory path, resolved relative to config directory"""
        path_str = self.get(f'outputs.{key}')
        if not path_str:
            raise ValueError(f"Output path '{key}' not found in config")
        
        path = Path(path_str)
        if not path.is_absolute():
            path = self.config_path.parent / path
        return path
