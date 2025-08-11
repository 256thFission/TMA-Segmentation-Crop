"""
Cellpose-SAM detector for circular tissue core segmentation
"""

import cv2
from typing import List, Optional, Tuple

import numpy as np
import gc
import warnings
from typing import List, Tuple, Optional, Union

warnings.filterwarnings('ignore')

try:
    from cellpose import models
    CELLPOSE_AVAILABLE = True
    print("Cellpose is available")
except ImportError:
    CELLPOSE_AVAILABLE = False
    print("Warning: cellpose not available")


class CellposeSAMDetector:
    """Cellpose-SAM detector optimized for circular tissue cores"""
    
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu
        self.model = None
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize Cellpose-SAM model"""        
        try:
            print("Initializing Cellpose-SAM model...")            
            self.model = models.CellposeModel(gpu=self.use_gpu)
            
            print(f"Cellpose-SAM initialized successfully (GPU: {self.use_gpu})")
            
        except Exception as e:
            print(f"Failed to initialize Cellpose: {e}")
            self.model = None
    
    def detect_cores(self, image: np.ndarray, diameter: Optional[int] = None, 
                    flow_threshold: float = 0.6, cellprob_threshold: float = 0.3, 
                    return_mask: bool = False) -> Union[List[Tuple[int, int, int]], Tuple[List[Tuple[int, int, int]], np.ndarray]]:
        """
        Detect circular tissue cores using Cellpose-SAM
        
        Args:
            image: Input image (grayscale)
            diameter: Expected diameter (None for auto-detect)
            flow_threshold: Flow threshold (default 0.4)
            cellprob_threshold: Cell probability threshold (default 0.0)
            return_mask: If True, return both cores and segmentation mask
        
        Returns:
            List of (x, y, radius) tuples, or tuple of (cores_list, mask_array) if return_mask=True
        """
        try:
            print(f"Running Cellpose-SAM detection...")
            print(f"Image shape: {image.shape}")
            print(f"Parameters: diameter={diameter}, flow_threshold={flow_threshold}, cellprob_threshold={cellprob_threshold}")

            if len(image.shape) == 2:
                # Convert grayscale to 3-channel
                image_input = np.stack([image, image, image], axis=-1)
            else:
                image_input = image
            
            print(f"Processing full image ({image.shape[0]}x{image.shape[1]})")
            cores, filtered_image, original_mask = self._process_full_image(image_input, diameter, flow_threshold, cellprob_threshold)
            if return_mask:
                return cores, original_mask
            else:
                return cores
        except Exception as e:
            print(f"Error: Cellpose detection failed: {e}")
            import traceback
            traceback.print_exc()
            print("Returning empty list.")
            if return_mask:
                return [], np.array([])
            else:
                return []
    
    def _process_full_image(self, image: np.ndarray, diameter: Optional[int], flow_threshold: float, cellprob_threshold: float) -> Tuple[List[Tuple[int, int, int]], np.ndarray, np.ndarray]:
        """Process the full image at once
        
        Args:
            image: Input image as a numpy array
            diameter: Expected diameter of cores (optional)
            flow_threshold: Flow threshold for Cellpose
            cellprob_threshold: Cell probability threshold for Cellpose
            
        Returns:
            Tuple containing:
            - cores_list: List of (x, y, radius) tuples  
            - filtered_image: Image with only mask pixels preserved, background removed
            - original_mask: Original Cellpose segmentation mask with object IDs
        """
        try:
            print("Processing full image with Cellpose-SAM...")
            
            # Run Cellpose-SAM
            masks, flows, styles = self.model.eval(
                [image], 
                diameter=diameter,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold
            )
            
            mask = masks[0] if isinstance(masks, list) else masks
            
            # Convert masks to core coordinates
            cores = self._extract_cores_from_mask(mask)
            
            filtered_image = self._apply_mask_filtering(image, mask)
            
            print(f"Cellpose-SAM detected {len(cores)} cores")
            return cores, filtered_image, mask
            
        except Exception as e:
            print(f"Full image processing failed: {e}")
            raise
    

    
    def _apply_mask_filtering(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply mask filtering to preserve only mask pixels, remove background
        
        Args:
            image: Original input image (H, W, C) or (H, W)
            mask: Cellpose segmentation mask (H, W) with object IDs
            
        Returns:
            Filtered image with only mask pixels preserved, background set to zero
        """
        binary_mask = mask > 0
        
        if len(image.shape) == 3:  # RGB image (H, W, C)
            binary_mask_expanded = np.expand_dims(binary_mask, axis=2)
            binary_mask_expanded = np.repeat(binary_mask_expanded, image.shape[2], axis=2)
            filtered_image = image * binary_mask_expanded
        else:  # Grayscale image (H, W)
            filtered_image = image * binary_mask
        
        print(f"Mask filtering applied: {np.sum(binary_mask)} pixels preserved out of {binary_mask.size} total")
        return filtered_image
    
    def _extract_cores_from_mask(self, mask: np.ndarray, offset_x: int = 0, offset_y: int = 0) -> List[Tuple[int, int, int]]:
        """Extract circular core coordinates from Cellpose mask"""
        cores = []
        
        unique_ids = np.unique(mask)[1:]
        
        for mask_id in unique_ids:
            binary_mask = (mask == mask_id).astype(np.uint8)
            
            # Find contours
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                continue
            
            # Get the largest contour
            contour = max(contours, key=cv2.contourArea)
            
            # Filter by size (minimum area to avoid noise)
            if cv2.contourArea(contour) >= 100:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"]) + offset_x
                    cy = int(M["m01"] / M["m00"]) + offset_y
                    
                    area = cv2.contourArea(contour)
                    radius = int(np.sqrt(area / np.pi))
                    
                    cores.append((cx, cy, radius))
        
        return cores
    

    
    def is_available(self) -> bool:
        return self.model is not None
    
    def get_model_info(self) -> dict:
        if self.model is None:
            return {'available': False}
        
        return {
            'available': True,
            'gpu_enabled': self.use_gpu,
            'model_type': 'Cellpose-SAM'
        }