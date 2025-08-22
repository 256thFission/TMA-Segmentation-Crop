#!/usr/bin/env python3
"""
VALIS Tissue Core Alignment CLI

Command-line interface for aligning tissue cores using VALIS with SuperPoint/SuperGlue
and transforming cell centroid coordinates between image spaces.

Example usage:
    python valis_cli.py align data/stain.png data/dapi.png data/centroids.tsv
    python valis_cli.py align --scale-factor 0.15 --no-flip data/stain.png data/dapi.png data/centroids.tsv
"""

import sys
from pathlib import Path
from typing import Optional, Tuple
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich import print as rprint
import pandas as pd

# Import the main pipeline functions
# Assuming the main pipeline code is in valis_alignment.py
try:
    from valis_alignment import (
        AlignmentConfig, 
        run_alignment_pipeline,
        preprocess_dapi_image,
        registration
    )
except ImportError:
    rprint("[red]Error: Could not import valis_alignment module.[/red]")
    rprint("[yellow]Make sure the main pipeline code is saved as 'valis_alignment.py'[/yellow]")
    sys.exit(1)

# Initialize typer app and rich console
app = typer.Typer(
    name="valis-align",
    help="VALIS tissue core alignment with coordinate transformation",
    add_completion=False
)
console = Console()

class CLIConfig(AlignmentConfig):
    """Extended configuration class for CLI usage"""
    
    def __init__(self):
        super().__init__()
        self.verbose = False
        self.output_tsv = None
        
    def update_from_cli(self, **kwargs):
        """Update configuration from CLI arguments"""
        for key, value in kwargs.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)

def validate_input_files(*file_paths: Path) -> bool:
    """Validate that input files exist and are readable"""
    for file_path in file_paths:
        if not file_path.exists():
            console.print(f"[red]Error: File not found: {file_path}[/red]")
            return False
        if not file_path.is_file():
            console.print(f"[red]Error: Not a file: {file_path}[/red]")
            return False
    return True

def validate_tsv_columns(tsv_path: Path, x_col: str, y_col: str) -> bool:
    """Validate that TSV file has required columns"""
    try:
        df = pd.read_csv(tsv_path, sep='\t', nrows=1)
        if x_col not in df.columns:
            console.print(f"[red]Error: Column '{x_col}' not found in {tsv_path}[/red]")
            return False
        if y_col not in df.columns:
            console.print(f"[red]Error: Column '{y_col}' not found in {tsv_path}[/red]")
            return False
        return True
    except Exception as e:
        console.print(f"[red]Error reading TSV file {tsv_path}: {e}[/red]")
        return False

@app.command()
def align(
    stain_image: Path = typer.Argument(
        ..., 
        help="Path to stained tissue image (reference)",
        exists=True,
        file_okay=True,
        readable=True
    ),
    dapi_image: Path = typer.Argument(
        ..., 
        help="Path to DAPI/nuclear channel image",
        exists=True,
        file_okay=True,
        readable=True
    ),
    centroids_tsv: Path = typer.Argument(
        ..., 
        help="Path to TSV file with cell centroids",
        exists=True,
        file_okay=True,
        readable=True
    ),
    # Output options
    workdir: Optional[Path] = typer.Option(
        "./valis_work",
        "--workdir", "-w",
        help="Working directory for temporary files and outputs"
    ),
    output_tsv: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output path for transformed centroids TSV (default: workdir/centroids_transformed.tsv)"
    ),
    # Image preprocessing options
    scale_factor: float = typer.Option(
        0.1022,
        "--scale-factor", "-s",
        min=0.01,
        max=1.0,
        help="Scaling factor for DAPI downsampling"
    ),
    flip_horizontal: bool = typer.Option(
        True,
        "--flip/--no-flip",
        help="Whether to flip DAPI image horizontally"
    ),
    # TSV column options
    x_column: str = typer.Option(
        "Centroid X µm",
        "--x-col",
        help="Name of X coordinate column in TSV"
    ),
    y_column: str = typer.Option(
        "Centroid Y µm", 
        "--y-col",
        help="Name of Y coordinate column in TSV"
    ),
    # SuperGlue options
    weights: str = typer.Option(
        "indoor",
        "--weights",
        help="SuperGlue weights (indoor/outdoor)"
    ),
    keypoint_threshold: float = typer.Option(
        0.005,
        "--keypoint-threshold",
        min=0.001,
        max=0.1,
        help="SuperPoint keypoint detection threshold"
    ),
    match_threshold: float = typer.Option(
        0.2,
        "--match-threshold", 
        min=0.0,
        max=1.0,
        help="SuperGlue matching threshold"
    ),
    force_cpu: bool = typer.Option(
        False,
        "--force-cpu",
        help="Force CPU usage (disable CUDA)"
    ),
    # CLI options
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose output"
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show configuration and exit without processing"
    )
):
    """
    Align tissue cores using VALIS and transform cell centroids.
    
    This command registers a DAPI image to a stained reference image and transforms
    cell centroid coordinates from the original DAPI space to the aligned stain space.
    """
    
    # Validate inputs
    if not validate_input_files(stain_image, dapi_image, centroids_tsv):
        raise typer.Exit(1)
    
    if not validate_tsv_columns(centroids_tsv, x_column, y_column):
        raise typer.Exit(1)
    
    # Create and configure alignment config
    config = CLIConfig()
    config.stain_path = stain_image
    config.dapi_path = dapi_image
    config.centroids_tsv = centroids_tsv
    config.workdir = workdir
    config.scale_factor = scale_factor
    config.flip_horizontal = flip_horizontal
    config.centroid_x_col = x_column
    config.centroid_y_col = y_column
    config.verbose = verbose
    
    # Update SuperGlue configuration
    config.superglue_config.update({
        'weights': weights,
        'keypoint_threshold': keypoint_threshold,
        'match_threshold': match_threshold,
        'force_cpu': force_cpu
    })
    
    # Set output path
    if output_tsv:
        config.output_tsv = output_tsv
    else:
        config.output_tsv = config.workdir / 'centroids_transformed.tsv'
    
    # Display configuration
    display_config(config, dry_run)
    
    if dry_run:
        console.print("[yellow]Dry run completed. Exiting without processing.[/yellow]")
        raise typer.Exit(0)
    
    # Run alignment pipeline
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            
            task = progress.add_task("Running VALIS alignment pipeline...", total=None)
            
            if verbose:
                console.print("\n[blue]Starting alignment pipeline...[/blue]")
            
            results = run_alignment_pipeline(config)
            
            progress.update(task, description="Pipeline completed successfully!")
        
        # Copy results to specified output path if different
        if config.output_tsv != results['output_tsv_path']:
            import shutil
            shutil.copy2(results['output_tsv_path'], config.output_tsv)
            console.print(f"[green]Results copied to: {config.output_tsv}[/green]")
        
        # Display results summary
        display_results(results, verbose)
        
    except Exception as e:
        console.print(f"[red]Pipeline failed: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)

def display_config(config: CLIConfig, dry_run: bool = False):
    """Display current configuration in a nice table"""
    
    table = Table(title="VALIS Alignment Configuration" + (" (DRY RUN)" if dry_run else ""))
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="yellow")
    
    # Input files
    table.add_row("Stain Image", str(config.stain_path))
    table.add_row("DAPI Image", str(config.dapi_path))
    table.add_row("Centroids TSV", str(config.centroids_tsv))
    table.add_row("Working Directory", str(config.workdir))
    table.add_row("Output TSV", str(config.output_tsv))
    
    # Preprocessing
    table.add_row("Scale Factor", f"{config.scale_factor:.4f}")
    table.add_row("Flip Horizontal", str(config.flip_horizontal))
    table.add_row("X Column", config.centroid_x_col)
    table.add_row("Y Column", config.centroid_y_col)
    
    # SuperGlue config
    table.add_row("SuperGlue Weights", config.superglue_config['weights'])
    table.add_row("Keypoint Threshold", f"{config.superglue_config['keypoint_threshold']:.3f}")
    table.add_row("Match Threshold", f"{config.superglue_config['match_threshold']:.2f}")
    table.add_row("Force CPU", str(config.superglue_config['force_cpu']))
    
    console.print(table)
    console.print()

def display_results(results: dict, verbose: bool = False):
    """Display pipeline results summary"""
    
    console.print("\n[green bold]✓ Pipeline Completed Successfully![/green bold]\n")
    
    # Results table
    table = Table(title="Results Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="yellow")
    
    # Basic stats
    n_centroids = len(results['transformed_centroids'])
    table.add_row("Input Centroids", f"{n_centroids:,}")
    table.add_row("Output File", str(results['output_tsv_path']))
    
    # Image dimensions
    orig_h, orig_w = results['original_dapi_shape']
    proc_h, proc_w = results['processed_dapi_shape']
    table.add_row("Original DAPI Size", f"{orig_w} × {orig_h}")
    table.add_row("Processed DAPI Size", f"{proc_w} × {proc_h}")
    
    # Registration error
    if 'registration_error' in results and len(results['registration_error']) > 0:
        err_df = results['registration_error']
        if 'rigid_D' in err_df.columns:
            mean_error = err_df['rigid_D'].mean()
            table.add_row("Mean Registration Error", f"{mean_error:.3f} pixels")
    
    console.print(table)
    
    if verbose and 'registration_error' in results:
        console.print("\n[blue]Detailed Registration Error:[/blue]")
        console.print(results['registration_error'].to_string())

@app.command()
def info(
    tsv_file: Path = typer.Argument(
        ...,
        help="Path to TSV file to analyze",
        exists=True,
        file_okay=True,
        readable=True
    )
):
    """Display information about a centroids TSV file."""
    
    try:
        df = pd.read_csv(tsv_file, sep='\t')
        
        table = Table(title=f"TSV File Information: {tsv_file.name}")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="yellow")
        
        table.add_row("File Path", str(tsv_file))
        table.add_row("Number of Rows", f"{len(df):,}")
        table.add_row("Number of Columns", str(len(df.columns)))
        
        # Display columns
        console.print(table)
        
        console.print(f"\n[blue]Columns ({len(df.columns)}):[/blue]")
        for i, col in enumerate(df.columns, 1):
            console.print(f"  {i:2d}. {col}")
        
        # Look for coordinate-like columns
        coord_cols = [col for col in df.columns if any(word in col.lower() 
                     for word in ['x', 'y', 'centroid', 'coord', 'position'])]
        
        if coord_cols:
            console.print(f"\n[green]Potential coordinate columns:[/green]")
            for col in coord_cols:
                console.print(f"  • {col}")
        
        # Show first few rows
        console.print(f"\n[blue]First 3 rows:[/blue]")
        console.print(df.head(3).to_string())
        
    except Exception as e:
        console.print(f"[red]Error reading TSV file: {e}[/red]")
        raise typer.Exit(1)

@app.command()
def validate(
    stain_image: Path = typer.Argument(..., help="Path to stain image"),
    dapi_image: Path = typer.Argument(..., help="Path to DAPI image"),
    centroids_tsv: Path = typer.Argument(..., help="Path to centroids TSV"),
    x_column: str = typer.Option("Centroid X µm", "--x-col", help="X coordinate column name"),
    y_column: str = typer.Option("Centroid Y µm", "--y-col", help="Y coordinate column name")
):
    """Validate input files and configuration before running alignment."""
    
    console.print("[blue]Validating input files and configuration...[/blue]\n")
    
    all_valid = True
    
    # Check file existence
    files_to_check = [
        ("Stain image", stain_image),
        ("DAPI image", dapi_image), 
        ("Centroids TSV", centroids_tsv)
    ]
    
    for name, file_path in files_to_check:
        if file_path.exists():
            console.print(f"[green]✓[/green] {name}: {file_path}")
        else:
            console.print(f"[red]✗[/red] {name}: {file_path} (not found)")
            all_valid = False
    
    # Check TSV columns
    if centroids_tsv.exists():
        if validate_tsv_columns(centroids_tsv, x_column, y_column):
            console.print(f"[green]✓[/green] TSV columns: '{x_column}', '{y_column}' found")
        else:
            all_valid = False
    
    # Check dependencies
    try:
        import cv2
        import torch
        from valis import registration
        console.print("[green]✓[/green] Dependencies: All required packages available")
        console.print(f"   • OpenCV version: {cv2.__version__}")
        console.print(f"   • PyTorch version: {torch.__version__}")
        console.print(f"   • CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        console.print(f"[red]✗[/red] Dependencies: Missing package - {e}")
        all_valid = False
    
    console.print()
    if all_valid:
        console.print("[green bold]✓ All validations passed! Ready to run alignment.[/green bold]")
    else:
        console.print("[red bold]✗ Validation failed. Please fix the issues above.[/red bold]")
        raise typer.Exit(1)

@app.callback()
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Show version and exit")
):
    """
    VALIS Tissue Core Alignment CLI
    
    Align tissue cores using VALIS with SuperPoint/SuperGlue and transform 
    cell centroid coordinates between image spaces.
    """
    if version:
        console.print("VALIS Alignment CLI v1.0.0")
        raise typer.Exit()

if __name__ == "__main__":
    app()