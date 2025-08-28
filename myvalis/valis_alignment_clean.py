# VALIS Tissue Core Alignment Pipeline
# Two-channel tissue core alignment with VALIS + SuperPoint/SuperGlue
# Registers a stained core to its nuclear (DAPI) counterpart and re-projects 
# cell centroid tables to integer pixel coordinates.

import os
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import torch

from valis import registration, feature_detectors, feature_matcher
from skimage import transform
from valis.feature_matcher import SuperGlueMatcher
from valis.feature_detectors import SuperPointFD
from valis.superglue_models.superpoint import SuperPoint as _VSuperPoint
print(f"CUDA available: {torch.cuda.is_available()}")

# =============================================================================
# CONFIGURATION
# =============================================================================

class AlignmentConfig:
    """Configuration parameters for tissue core alignment"""
    
    def __init__(self):
        # Input paths
        self.stain_path = Path('./inputs/hires/core_03_raw.png')           # Reference (H&E/IHC)
        self.dapi_path = Path('./inputs/fullres/tissue_dapi_fullres_processed/cores_filtered/core_01_raw.png')  # Nuclear channel
        self.centroids_tsv = Path('./inputs/tsvs/core01_cellseg.tsv')      # 4-column TSV with centroids
        self.workdir = Path('./valis_work')                               # Scratch/output folder
        
        # DAPI preprocessing parameters
        self.scale_factor = 0.1022  # Downscaling factor for DAPI image
        self.flip_horizontal = True  # Whether to flip DAPI horizontally
        self.microns_per_pixel = 0.5072  # Microns per pixel for coordinate conversion
        self.flip_axis = 'y'  # Flip axis: 'none', 'x' (horizontal), or 'y' (vertical)
        self.processed_dapi_shape = None  # Will be set during preprocessing
        
        # SuperGlue matching parameters
        self.superglue_config = {
            'weights': 'indoor',           # or 'outdoor'
            'keypoint_threshold': 0.005,
            'nms_radius': 4,
            'sinkhorn_iterations': 100,
            'match_threshold': 0.2,
            'force_cpu': False
        }
        
        # Expected TSV column names for centroids
        self.centroid_x_col = 'Centroid X µm'
        self.centroid_y_col = 'Centroid Y µm'

        # Overlay export options
        self.save_overlays = True
        self.overlay_alpha = 0.5  # alpha for blending overlays

# =============================================================================
# PREPROCESSING FUNCTIONS
# =============================================================================

def preprocess_dapi_image(dapi_path, scale_factor, flip_axis='y', output_path=None):
    """
    Load, downscale, and optionally flip the DAPI image
    
    Parameters:
    -----------
    dapi_path : Path
        Path to original DAPI image
    scale_factor : float
        Scaling factor for downsampling
    flip_axis : str
        Flip axis: 'none', 'x' (horizontal), or 'y' (vertical)
    output_path : Path, optional
        Where to save processed image
        
    Returns:
    --------
    tuple: (processed_image, output_path, original_shape, processed_shape)
    """
    
    # Load the original DAPI image
    dapi_img = cv2.imread(str(dapi_path))
    if dapi_img is None:
        raise FileNotFoundError(f"Could not load image from path: {dapi_path}")
    
    original_shape = dapi_img.shape[:2]  # (height, width)
    
    # Calculate new dimensions
    new_width = int(dapi_img.shape[1] * scale_factor)
    new_height = int(dapi_img.shape[0] * scale_factor)
    new_dim = (new_width, new_height)
    
    # Downscale the image
    downscaled_img = cv2.resize(dapi_img, new_dim, interpolation=cv2.INTER_AREA)
    
    # Apply flip based on axis
    if flip_axis == 'x':
        processed_img = cv2.flip(downscaled_img, 1)  # Horizontal flip
    elif flip_axis == 'y':
        processed_img = cv2.flip(downscaled_img, 0)  # Vertical flip
    else:
        processed_img = downscaled_img
    
    processed_shape = processed_img.shape[:2]  # (height, width)
    
    # Save if output path provided
    if output_path:
        cv2.imwrite(str(output_path), processed_img)
        print(f"Processed DAPI image saved to: {output_path}")
    
    print(f"Original DAPI dimensions: {original_shape}")
    print(f"Processed DAPI dimensions: {processed_shape}")
    
    return processed_img, output_path, original_shape, processed_shape

# =============================================================================
# VALIS REGISTRATION
# =============================================================================

def setup_valis_registration(src_dir, dst_dir, reference_name, superglue_config):
    """
    Initialize VALIS registration with SuperGlue matcher
    
    Parameters:
    -----------
    src_dir : str
        Source directory containing images
    dst_dir : str
        Destination directory for results
    reference_name : str
        Name of reference image file
    superglue_config : dict
        SuperGlue configuration parameters
        
    Returns:
    --------
    registration.Valis: Configured VALIS registration object
    """
    
    # Initialize SuperGlue matcher
    superglue_matcher = feature_matcher.SuperGlueMatcher(**superglue_config)
    print("SuperGlue matcher initialized")
    
    # Create VALIS registration object (no need for custom detector since we patched the class)
    reg = registration.Valis(
        src_dir=str(src_dir),
        dst_dir=str(dst_dir),
        reference_img_f=reference_name,
        align_to_reference=True,
        crop="reference",
        create_masks=False,
        feature_detector_cls=None,  # Let VALIS choose, our patch will handle it
        matcher=superglue_matcher
    )
    
    return reg

def run_registration(reg):
    """
    Execute VALIS registration
    
    Parameters:
    -----------
    reg : registration.Valis
        VALIS registration object
        
    Returns:
    --------
    tuple: (rigid_reg, nonrigid_reg, error_df)
    """
    
    try:
        print("Starting VALIS registration...")
        rigid_reg, nonrigid_reg, err_df = reg.register()
        print("Registration completed successfully!")
        print(f"\nRegistration error summary:")
        print(err_df)
        return rigid_reg, nonrigid_reg, err_df
    except Exception as e:
        print(f"Registration failed: {e}")
        registration.kill_jvm()
        raise

# =============================================================================
# COORDINATE TRANSFORMATION
# =============================================================================

def create_coordinate_transformers(reg, stain_name, dapi_name):
    """
    Create coordinate transformation functions between stain and DAPI spaces
    
    Parameters:
    -----------
    reg : registration.Valis
        VALIS registration object
    stain_name : str
        Name of stain image file
    dapi_name : str
        Name of DAPI image file
        
    Returns:
    --------
    tuple: (stain_to_dapi_func, dapi_to_stain_func)
    """
    
    stain_slide = reg.get_slide(stain_name)
    dapi_slide = reg.get_slide(dapi_name)
    
    def stain_xy_to_dapi_xy(xy_int, level_src=0, level_dst=0):
        """Transform integer coordinates from stain to DAPI space"""
        xy_float = stain_slide.warp_xy_from_to(
            xy_int.astype(float),
            dapi_slide,
            src_pt_level=level_src,
            dst_slide_level=level_dst,
            non_rigid=True
        )
        return np.rint(xy_float).astype(np.int32)
    
    def dapi_xy_to_stain_xy(xy_int, level_src=0, level_dst=0):
        """Transform integer coordinates from DAPI to stain space"""
        xy_float = dapi_slide.warp_xy_from_to(
            xy_int.astype(float),
            stain_slide,
            src_pt_level=level_src,
            dst_slide_level=level_dst,
            non_rigid=True
        )
        return np.rint(xy_float).astype(np.int32)
    
    return stain_xy_to_dapi_xy, dapi_xy_to_stain_xy

def _blend_and_save(img_a, img_b, out_path, alpha=0.5):
    """Alpha-blend two images and save to out_path. Resizes img_b to img_a size."""
    try:
        if img_a is None or img_b is None:
            return False
        # Ensure 3 channels
        if len(img_a.shape) == 2:
            img_a = cv2.cvtColor(img_a, cv2.COLOR_GRAY2BGR)
        if len(img_b.shape) == 2:
            img_b = cv2.cvtColor(img_b, cv2.COLOR_GRAY2BGR)
        # Resize B to A
        h, w = img_a.shape[:2]
        img_b_rs = cv2.resize(img_b, (w, h), interpolation=cv2.INTER_AREA)
        overlay = cv2.addWeighted(img_a, 1 - alpha, img_b_rs, alpha, 0)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), overlay)
        return True
    except Exception:
        return False

def _safe_imread(path_like):
    try:
        return cv2.imread(str(path_like))
    except Exception:
        return None

# =============================================================================
# FIX VALIS SUPERPOINT DEVICE MISMATCH 
# =============================================================================

# Monkey patch SuperPointFD to handle optional mask and defer device handling
original_detect_and_compute = SuperPointFD._detect_and_compute

def patched_detect_and_compute(self, img, mask=None):
    """
    Wrapper for SuperPointFD._detect_and_compute that supports mask and delegates
    device consistency to a patched SuperPoint.forward. No manual device forcing here.
    """
    # Delegate to underlying implementations
    if self.kp_detector is None and self.kp_descriptor is None:
        kp_pos_xy, desc = self.detect_and_compute_sg(img)
    else:
        kp_pos_xy = self.detect(img)
        desc = self.compute(img, kp_pos_xy)
    return kp_pos_xy, desc

# Also patch the detect_and_compute_sg method to handle tensor device placement
original_detect_and_compute_sg = SuperPointFD.detect_and_compute_sg

def patched_detect_and_compute_sg(self, img):
    """
    Patched wrapper that delegates to the original method; device co-location
    is handled inside SuperPoint.forward.
    """
    try:
        return original_detect_and_compute_sg(self, img)
    except Exception as e:
        print(f"Warning: SuperPoint processing failed: {e}")
        return original_detect_and_compute_sg(self, img)

# Apply the monkey patches for SuperPointFD
SuperPointFD._detect_and_compute = patched_detect_and_compute
SuperPointFD.detect_and_compute_sg = patched_detect_and_compute_sg

# Monkey-patch SuperPoint.forward to co-locate input tensor with model device
_ORIG_SP_FORWARD = _VSuperPoint.forward
def _PATCHED_SP_FORWARD(self, data):
    """
    Ensure data['image'] tensor is on the same device/dtype as model params.
    Does not force a specific device; follows wherever the model lives (CPU or CUDA).
    """
    try:
        dev = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        if isinstance(data, dict) and 'image' in data:
            img = data['image']
            if not torch.is_tensor(img):
                img = torch.as_tensor(img)
            # Shallow copy to avoid mutating caller reference
            data = dict(data)
            data['image'] = img.to(dev, dtype=dtype, non_blocking=False)
        else:
            # Fallback if upstream API changes
            try:
                data = data.to(dev)  # type: ignore[attr-defined]
            except Exception:
                pass
    except Exception as e:
        print(f"Warning: SuperPoint.forward device colocation failed: {e}")
    return _ORIG_SP_FORWARD(self, data)
_VSuperPoint.forward = _PATCHED_SP_FORWARD

print("Applied SuperPointFD patches and SuperPoint.forward device co-location patch")

# =============================================================================
# CONFIGURATION
# =============================================================================


def _find_slide_thumb(slide, keywords):
    """Best-effort to find a thumbnail path on a Slide object matching keywords."""
    try:
        for attr in dir(slide):
            name = attr.lower()
            if 'thumb' in name and all(k in name for k in keywords):
                val = getattr(slide, attr, None)
                if isinstance(val, str) and os.path.exists(val):
                    return val
        # Common guesses as fallback
        guesses = [
            'processed_img_thumb_f',
            'rigid_reg_thumb_f',
            'non_rigid_reg_thumb_f',
        ]
        for g in guesses:
            val = getattr(slide, g, None)
            if isinstance(val, str) and os.path.exists(val):
                # Filter by keywords
                if all(k in g.lower() for k in keywords):
                    return val
    except Exception:
        pass
    return None

def save_overlays_safe(reg, config, src_dir, stain_name, dapi_processed_path, dst_dir):
    """Save pre- and post-registration overlays into results/overlays/."""
    try:
        # Check if save_overlays is enabled in config (use different variable name)
        should_save_overlays = getattr(config, 'save_overlays', True)
        if not should_save_overlays:
            return
        overlays_dir = Path(dst_dir) / 'overlays'

        # Pre-registration overlay (stain vs processed DAPI)
        stain_img = _safe_imread(Path(src_dir) / stain_name)
        dapi_proc_img = _safe_imread(Path(src_dir) / Path(dapi_processed_path).name)
        alpha_value = getattr(config, 'overlay_alpha', 0.5)
        _blend_and_save(stain_img, dapi_proc_img, overlays_dir / 'overlay_before.png', alpha=alpha_value)

        # Post-registration overlays using thumbnails if available
        stain_slide = reg.get_slide(stain_name)
        dapi_slide = reg.get_slide(Path(dapi_processed_path).name)

        # Rigid
        stain_rigid = _find_slide_thumb(stain_slide, keywords=['rigid'])
        dapi_rigid = _find_slide_thumb(dapi_slide, keywords=['rigid'])
        if stain_rigid and dapi_rigid:
            _blend_and_save(_safe_imread(stain_rigid), _safe_imread(dapi_rigid), overlays_dir / 'overlay_after_rigid.png', alpha=alpha_value)

        # Non-rigid
        stain_nonrigid = _find_slide_thumb(stain_slide, keywords=['non', 'rigid'])
        dapi_nonrigid = _find_slide_thumb(dapi_slide, keywords=['non', 'rigid'])
        if stain_nonrigid and dapi_nonrigid:
            _blend_and_save(_safe_imread(stain_nonrigid), _safe_imread(dapi_nonrigid), overlays_dir / 'overlay_after_nonrigid.png', alpha=alpha_value)
    except Exception as e:
        print(f"Warning: Could not save overlays: {e}")
        pass

# Keep old function name for compatibility
save_overlays = save_overlays_safe

def transform_centroids(centroids_tsv, transform_func, config, output_path):
    """
    Transform cell centroids from original DAPI space to aligned stain space
    
    Parameters:
    -----------
    centroids_tsv : Path
        Path to input TSV file with centroids
    transform_func : callable
        Function to transform coordinates from DAPI to stain space
    config : AlignmentConfig
        Configuration object with transformation parameters
    output_path : Path
        Path to save transformed centroids
        
    Returns:
    --------
    pd.DataFrame: DataFrame with transformed coordinates
    """
    
    # Load centroid data
    df = pd.read_csv(centroids_tsv, sep='\t')
    print(f"Loaded {len(df)} centroids from {centroids_tsv}")
    
    # Validate required columns exist
    if config.centroid_x_col not in df.columns or config.centroid_y_col not in df.columns:
        raise ValueError(f"Required columns {config.centroid_x_col}, {config.centroid_y_col} not found in TSV")
    
    # Get original coordinates (in microns)
    xy_original = df[[config.centroid_x_col, config.centroid_y_col]].values
    
    # Step 1: Convert from microns to pixels
    xy_pixels = xy_original / config.microns_per_pixel
    
    # Step 2: Scale coordinates from original DAPI space to downscaled space
    xy_scaled = xy_pixels * config.scale_factor
    
    # Step 3: Account for flip if applied
    if config.flip_axis == 'x' and config.processed_dapi_shape is not None:
        processed_width = config.processed_dapi_shape[1]  # width
        xy_scaled[:, 0] = (processed_width - 1) - xy_scaled[:, 0]
    elif config.flip_axis == 'y' and config.processed_dapi_shape is not None:
        processed_height = config.processed_dapi_shape[0]  # height
        xy_scaled[:, 1] = (processed_height - 1) - xy_scaled[:, 1]
    
    # Step 4: Transform to stain space using VALIS registration
    xy_stain = transform_func(xy_scaled.astype(np.int32))
    
    # Add transformed coordinates to dataframe
    df['Stain_X_px'] = xy_stain[:, 0]
    df['Stain_Y_px'] = xy_stain[:, 1]
    
    # Save results
    df.to_csv(output_path, sep='\t', index=False)
    print(f'Saved transformed centroids to: {output_path}')
    
    return df
# MAIN PIPELINE
# =============================================================================

def run_alignment_pipeline(config=None, **kwargs):
    """
    Run the complete tissue core alignment pipeline
    
    Parameters:
    -----------
    config : AlignmentConfig or None
        Configuration object with all parameters. If None, legacy keyword
        arguments can be provided for backward compatibility.
    
    Returns:
    --------
    dict: Results dictionary with transformed coordinates and registration info
    """
    
    # Backward-compatibility layer: allow calling with legacy keyword args
    # such as stain_image_path, dapi_image_path, centroids_tsv_path, etc.
    if config is None:
        cfg = AlignmentConfig()
        try:
            stain = kwargs.pop('stain_image_path', None) or kwargs.pop('stain_path', None)
            dapi = kwargs.pop('dapi_image_path', None) or kwargs.pop('dapi_path', None)
            tsv = kwargs.pop('centroids_tsv_path', None) or kwargs.pop('centroids_tsv', None)
            workdir = kwargs.pop('workdir', None)
            scale_factor = kwargs.pop('scale_factor', None)
            flip_axis = kwargs.pop('flip_axis', None)
            um_per_pixel = kwargs.pop('microns_per_pixel', None) or kwargs.pop('um_per_pixel', None)
            x_col = kwargs.pop('x_column', None)
            y_col = kwargs.pop('y_column', None)
            weights = kwargs.pop('weights', None)
            keypoint_threshold = kwargs.pop('keypoint_threshold', None)
            match_threshold = kwargs.pop('match_threshold', None)
            force_cpu = kwargs.pop('force_cpu', None)
            save_overlays_flag = kwargs.pop('save_overlays', None)
            overlay_alpha = kwargs.pop('overlay_alpha', None)

            if stain: cfg.stain_path = Path(stain)
            if dapi: cfg.dapi_path = Path(dapi)
            if tsv: cfg.centroids_tsv = Path(tsv)
            if workdir: cfg.workdir = Path(workdir)
            if scale_factor is not None: cfg.scale_factor = float(scale_factor)
            if flip_axis is not None: cfg.flip_axis = str(flip_axis)
            if um_per_pixel is not None: cfg.microns_per_pixel = float(um_per_pixel)
            if x_col: cfg.centroid_x_col = x_col
            if y_col: cfg.centroid_y_col = y_col
            if weights is not None: cfg.superglue_config['weights'] = weights
            if keypoint_threshold is not None: cfg.superglue_config['keypoint_threshold'] = float(keypoint_threshold)
            if match_threshold is not None: cfg.superglue_config['match_threshold'] = float(match_threshold)
            if force_cpu is not None: cfg.superglue_config['force_cpu'] = bool(force_cpu)
            if save_overlays_flag is not None: cfg.save_overlays = bool(save_overlays_flag)
            if overlay_alpha is not None: cfg.overlay_alpha = float(overlay_alpha)

        except Exception:
            # If anything goes wrong while parsing kwargs, fall back silently
            pass

        config = cfg
    elif kwargs:
        # Ignore unexpected kwargs but keep compatibility if callers still send them
        try:
            print(f"Warning: Ignoring unexpected kwargs in run_alignment_pipeline: {list(kwargs.keys())}")
        except Exception:
            pass
    
    # Create working directories
    config.workdir.mkdir(exist_ok=True)
    src_dir = config.workdir / 'src'
    dst_dir = config.workdir / 'results'
    src_dir.mkdir(exist_ok=True)
    dst_dir.mkdir(exist_ok=True)
    
    # Step 1: Preprocess DAPI image
    print("=" * 60)
    print("STEP 1: Preprocessing DAPI image")
    print("=" * 60)
    
    dapi_processed_path = config.workdir / 'dapi_processed.png'
    processed_img, _, original_shape, processed_shape = preprocess_dapi_image(
        config.dapi_path, 
        config.scale_factor, 
        config.flip_axis, 
        dapi_processed_path
    )
    
    # Store processed shape in config for coordinate transforms
    config.processed_dapi_shape = processed_shape
    
    # Force CPU execution for SuperGlue to avoid device mismatches
    config.superglue_config['force_cpu'] = True

    # Step 2: Copy images to source directory
    print("\n" + "=" * 60)
    print("STEP 2: Preparing images for registration")
    print("=" * 60)
    
    # Copy reference (stain) image
    shutil.copy2(config.stain_path, src_dir / config.stain_path.name)
    print(f"Copied stain image: {config.stain_path.name}")
    
    # Copy processed DAPI image
    shutil.copy2(dapi_processed_path, src_dir / dapi_processed_path.name)
    print(f"Copied processed DAPI image: {dapi_processed_path.name}")
    
    # Step 3: Run VALIS registration
    print("\n" + "=" * 60)
    print("STEP 3: Running VALIS registration")
    print("=" * 60)
    
    reg = setup_valis_registration(
        src_dir, 
        dst_dir, 
        config.stain_path.name, 
        config.superglue_config
    )
    
    rigid_reg, nonrigid_reg, err_df = run_registration(reg)
    
    # Save overlays (best-effort) - wrap in try/except to prevent crashes
    try:
        # Call the explicitly named function to avoid any accidental shadowing
        save_overlays_safe(reg, config, src_dir, config.stain_path.name, dapi_processed_path, dst_dir)
    except Exception as e:
        print(f"Warning: Failed to save overlays: {e}")
    
    # Step 4: Transform centroids
    print("\n" + "=" * 60)
    print("STEP 4: Transforming cell centroids")
    print("=" * 60)
    
    # Create coordinate transformation functions
    stain_to_dapi, dapi_to_stain = create_coordinate_transformers(
        reg, 
        config.stain_path.name, 
        dapi_processed_path.name
    )
    
    # Transform centroids
    output_tsv = config.workdir / 'centroids_transformed.tsv'
    transformed_df = transform_centroids(
        config.centroids_tsv, 
        dapi_to_stain, 
        config, 
        output_tsv
    )
    
    # Step 5: Cleanup
    print("\n" + "=" * 60)
    print("STEP 5: Cleanup")
    print("=" * 60)
    
    registration.kill_jvm()
    print("JVM cleaned up successfully")
    
    # Return results
    results = {
        'registration_error': err_df,
        'transformed_centroids': transformed_df,
        'output_tsv_path': output_tsv,
        'original_dapi_shape': original_shape,
        'processed_dapi_shape': processed_shape,
        'transform_functions': (stain_to_dapi, dapi_to_stain)
    }
    
    print(f"\nPipeline completed successfully!")
    print(f"Transformed centroids saved to: {output_tsv}")
    
    return results

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Initialize configuration
    config = AlignmentConfig()
    
    # Run the pipeline
    try:
        results = run_alignment_pipeline(config)
        
        # Display summary
        print("\n" + "=" * 60)
        print("PIPELINE SUMMARY")
        print("=" * 60)
        print(f"Input centroids: {len(results['transformed_centroids'])}")
        print(f"Output file: {results['output_tsv_path']}")
        print(f"Registration error (mean): {results['registration_error']['rigid_D'].mean():.3f} pixels")
        
    except Exception as e:
        print(f"Pipeline failed with error: {e}")
        registration.kill_jvm()  # Ensure cleanup on failure
        raise