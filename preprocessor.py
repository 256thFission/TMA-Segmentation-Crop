"""
Image preprocessing utilities for tissue core segmentation
Handles loading, downsampling, and enhancement
"""

import cv2
import numpy as np
import tifffile
from skimage import exposure, filters, morphology, segmentation
from scipy import ndimage
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')


class ImagePreprocessor:
    """Image preprocessing for tissue core detection"""
    
    def __init__(self, downsample_factor=128, enhance_contrast=True, median_filter_size=3, flat_field_correction=False):
        """
        Initialize preprocessor with simplified parameters
        
        Args:
            downsample_factor: Downsampling factor to apply
            enhance_contrast: Whether to apply contrast enhancement
            median_filter_size: Size of median filter for noise reduction
            flat_field_correction: Whether to apply flat field correction. Correction is always applied after downsampling.
        """
        self.downsample_factor = downsample_factor
        self.enhance_contrast = enhance_contrast
        self.median_filter_size = median_filter_size
        self.flat_field_correction = flat_field_correction
        self.actual_downsample_factor = downsample_factor
    
    def process(self, image_path):
        """
        Main preprocessing pipeline 
        
        Returns:
            original_image: Full resolution image
            processed_image: Downsampled and enhanced image
            scale_factor: Factor to scale coordinates back to original
        """
        # Load image
        original_image = self.load_image(image_path)
        if original_image is None:
            return None, None, None
        
        # Convert to grayscale if not already
        if len(original_image.shape) > 2:
            if original_image.shape[2] > 1:
                original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
            else:
                original_image = original_image[:, :, 0]
        
        self.actual_downsample_factor = self.downsample_factor
        print(f"Using downsample factor: {self.actual_downsample_factor}")
        
        # Downsample for processing
        processed_image = self.downsample_image(original_image)
        
        # Apply flat field correction 
        if self.flat_field_correction:
            print("Applying flat field correction...")
            processed_image = self.apply_flat_field_correction(processed_image)
        
        # Enhance image
        if self.enhance_contrast:
            processed_image = self.enhance_contrast_adaptive(processed_image)
        
        # Apply noise reduction
        if self.median_filter_size > 0:
            processed_image = self.apply_median_filter(processed_image)
        
        return original_image, processed_image, self.actual_downsample_factor
    
    def load_image(self, image_path):
        """Load image from various formats"""
        try:
            # Try TIFF first
            if image_path.lower().endswith(('.tif', '.tiff')):
                img = tifffile.imread(image_path)
                print(f"Loaded TIFF image: {img.shape}, dtype: {img.dtype}")
                
                # Handle multi-dimensional TIFF
                if len(img.shape) > 2:
                    if len(img.shape) == 4:  # (T, Z, Y, X) or (T, C, Y, X)
                        img = img[0, 0]  # Take first time and z/channel
                    elif len(img.shape) == 3:
                        if img.shape[0] < img.shape[2]:  # Likely (C, Y, X)
                            img = img[0]  # Take first channel
                        else:  # Likely (Y, X, C)
                            img = img[:, :, 0]  # Take first channel
                
                return img
            
            # Try OpenCV for other formats
            else:
                img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    print(f"Loaded image: {img.shape}, dtype: {img.dtype}")
                    return img
                
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
        
        return None
    
    def downsample_image(self, image):
        """Downsample image efficiently using calculated factor"""
        height, width = image.shape
        new_height = int(height / self.actual_downsample_factor)
        new_width = int(width / self.actual_downsample_factor)
        
        print(f"Downsampling from {width}x{height} to {new_width}x{new_height} (factor: {self.actual_downsample_factor})")
        
        # Use area interpolation for downsampling (good for reducing aliasing)
        downsampled = cv2.resize(
            image, 
            (new_width, new_height), 
            interpolation=cv2.INTER_AREA
        )
        
        return downsampled
    
    def enhance_contrast_adaptive(self, image):
        """Apply adaptive histogram equalization"""
        print("Applying adaptive contrast enhancement...")
        
        # Normalize to 0-1 range for CLAHE
        if image.dtype != np.uint8:
            image_norm = exposure.rescale_intensity(image, out_range=(0, 255)).astype(np.uint8)
        else:
            image_norm = image
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image_norm)
        
        return enhanced
    
    def apply_median_filter(self, image):
        """Apply median filter for noise reduction"""
        if self.median_filter_size > 1:
            print(f"Applying median filter (size: {self.median_filter_size})...")
            return cv2.medianBlur(image, self.median_filter_size)
        return image
    
    def normalize_image(self, image, percentile_low=1, percentile_high=99):
        """Normalize image intensities based on percentiles"""
        p_low = np.percentile(image, percentile_low)
        p_high = np.percentile(image, percentile_high)
        
        image_norm = np.clip((image - p_low) / (p_high - p_low), 0, 1)
        return (image_norm * 255).astype(np.uint8)
    
    def create_tissue_mask(self, image, threshold_method='otsu'):
        """Create a binary mask to identify tissue regions"""
        if threshold_method == 'otsu':
            threshold, binary_mask = cv2.threshold(
                image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        elif threshold_method == 'adaptive':
            binary_mask = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
        else:
            # Simple threshold
            threshold = np.mean(image) + np.std(image)
            _, binary_mask = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        
        # Clean up mask with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        return binary_mask
    
    def apply_flat_field_correction(self, image):
        """
        Apply flat field correction to compensate for uneven illumination
        
        Args:
            image: Input grayscale image
            
        Returns:
            Corrected image with rescaled intensity
        """
        # This is a compute-intensive step, especially on large images
        
        # Create illumination model using multiple approaches
        illumination_gaussian = self._create_gaussian_illumination_model(image)
        illumination_poly = self._create_polynomial_illumination_model(image)
        
        # Combine models (weighted average)
        illumination = 0.7 * illumination_gaussian + 0.3 * illumination_poly
        
        # Ensure illumination values are not too small to avoid division issues
        illumination = np.maximum(illumination, np.percentile(illumination, 5))
        
        # Apply correction: divide by illumination
        corrected = image.astype(np.float64) / illumination
        
        # Rescale to maintain original intensity range using skimage.exposure.rescale_intensity
        corrected_rescaled = exposure.rescale_intensity(
            corrected, 
            in_range='image',  # Use image's min/max as input range
            out_range=(image.min(), image.max())  # Maintain original range
        )
        
        # Convert back to original dtype
        corrected_final = corrected_rescaled.astype(image.dtype)
        
        print(f"Flat field correction applied. Original range: [{image.min()}, {image.max()}], "
              f"Corrected range: [{corrected_final.min()}, {corrected_final.max()}]")
        
        return corrected_final
    
    def _create_gaussian_illumination_model(self, image, sigma_factor=0.1):
        """
        Create illumination model using heavy Gaussian blur
        
        Args:
            image: Input image
            sigma_factor: Gaussian sigma as fraction of image width
            
        Returns:
            Illumination model
        """
        # Calculate sigma based on image size for scale-invariant blur
        sigma = sigma_factor * image.shape[1]  # Use image width
        
        # Apply heavy Gaussian blur to create illumination model
        illumination = ndimage.gaussian_filter(image.astype(np.float64), sigma=sigma)
        
        return illumination
    
    def _create_polynomial_illumination_model(self, image, poly_order=2):
        """
        Create illumination model by fitting polynomial surface to background regions
        
        Args:
            image: Input image
            poly_order: Order of polynomial surface (default: 2 for quadratic)
            
        Returns:
            Illumination model
        """
        # Identify background regions (areas between tissue cores)
        background_mask = self._identify_background_regions(image)
        
        # Get coordinates for all pixels
        h, w = image.shape
        y_indices, x_indices = np.ogrid[:h, :w]
        
        # Create coordinate meshgrids that match image shape
        x_coords = np.broadcast_to(x_indices, (h, w)).astype(np.float64) / w  # Normalize to [0, 1]
        y_coords = np.broadcast_to(y_indices, (h, w)).astype(np.float64) / h  # Normalize to [0, 1]
        
        # Extract background pixel coordinates and intensities
        bg_x = x_coords[background_mask]
        bg_y = y_coords[background_mask]
        bg_intensities = image[background_mask].astype(np.float64)
        
        if len(bg_intensities) < 100:  # Fallback if too few background pixels
            print("Warning: Few background pixels found, using downsampled approach")
            return self._create_gaussian_illumination_model(image, sigma_factor=0.05)
        
        # Create polynomial surface function
        def polynomial_2d(coords, *params):
            x, y = coords
            if poly_order == 1:
                a, b, c = params
                return a + b*x + c*y
            elif poly_order == 2:
                a, b, c, d, e, f = params
                return a + b*x + c*y + d*x*x + e*y*y + f*x*y
            else:
                raise ValueError("Only poly_order 1 or 2 supported")
        
        # Fit polynomial to background intensities
        try:
            if poly_order == 1:
                initial_params = [np.mean(bg_intensities), 0, 0]
            else:  # poly_order == 2
                initial_params = [np.mean(bg_intensities), 0, 0, 0, 0, 0]
            
            popt, _ = curve_fit(
                polynomial_2d, 
                (bg_x, bg_y), 
                bg_intensities,
                p0=initial_params,
                maxfev=2000
            )
            
            # Create illumination surface for entire image
            illumination = polynomial_2d((x_coords, y_coords), *popt)
            
        except Exception as e:
            print(f"Warning: Polynomial fitting failed ({e}), using Gaussian fallback")
            illumination = self._create_gaussian_illumination_model(image, sigma_factor=0.05)
        
        return illumination
    
    def _identify_background_regions(self, image, erosion_size=5):
        """
        Identify background regions (areas between tissue cores) using morphological operations
        
        Args:
            image: Input image
            erosion_size: Size of erosion kernel for background identification
            
        Returns:
            Binary mask where True indicates background regions
        """
        # Create tissue mask using Otsu thresholding
        tissue_mask = self.create_tissue_mask(image)
        
        background_mask = ~tissue_mask
        
        # Erode background mask to ensure we're sampling pure background
        kernel = morphology.disk(erosion_size)
        background_mask = morphology.binary_erosion(background_mask, kernel)
        
        background_mask = morphology.remove_small_objects(
            background_mask, 
            min_size=100,
            connectivity=2
        )
        
        print(f"Background regions identified: {np.sum(background_mask)} pixels "
              f"({100.0 * np.sum(background_mask) / background_mask.size:.1f}% of image)")
        
        return background_mask