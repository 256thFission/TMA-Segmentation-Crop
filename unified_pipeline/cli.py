"""Command-line interface for unified pipeline."""

import json
import typer
from pathlib import Path
from rich.console import Console

from .config import UnifiedConfig
from .pipeline import UnifiedPipeline

console = Console()


def main(
    config_path: str = typer.Option(
        "inputs/config.yaml",
        "--config", "-c",
        help="Path to YAML configuration file"
    ),
    verbose: bool = typer.Option(
        None,
        "--verbose", "-v", 
        help="Enable verbose output (overrides config)"
    )
):
    """
    Unified Tissue Core Alignment Pipeline
    
    Integrates segmentation and VALIS registration according to pipeline_plan.md specification.
    """
    
    try:
        # Load configuration
        config = UnifiedConfig(config_path)
        
        # Override verbose setting if provided
        if verbose is not None:
            config.data['global']['verbose'] = verbose
        
        # Initialize and run pipeline
        pipeline = UnifiedPipeline(config)
        results = pipeline.run()
        
        # Save run summary
        summary_path = Path(config_path).parent / "pipeline_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        console.print(f"[green]Pipeline summary saved to: {summary_path}[/green]")
        
    except Exception as e:
        console.print(f"[red]Pipeline failed: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)
