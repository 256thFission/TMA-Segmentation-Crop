"""
Visualization functions for tissue core segmentation
"""

import os
import cv2
import numpy as np


def create_simple_overlay(image, cores, mask=None):
    """Create simple detection overlay using binary mask regions
    
    Args:
        image: Input image (grayscale or BGR)
        cores: List of (x, y, radius) tuples
        mask: Segmentation mask from Cellpose (optional)
    """
    # Convert to color for visualization
    if len(image.shape) == 2:
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        overlay = image.copy()
    
    if mask is not None:
        # Create composite colored mask overlay
        # Generate different colors for each core
        unique_ids = np.unique(mask)[1:]  # Exclude background (0)
        
        # Create a single composite mask overlay to avoid progressive darkening
        composite_mask = np.zeros_like(overlay)
        
        for i, mask_id in enumerate(unique_ids):
            # Create binary mask for this core
            core_mask = (mask == mask_id).astype(np.uint8)
            
            # Generate a unique color for this core (use more vibrant colors)
            hue = int((i * 360) / len(unique_ids))  # Full hue range
            color_hsv = np.array([[[hue % 180, 255, 255]]], dtype=np.uint8)  # Full saturation and value
            color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0, 0]
            color = (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2]))
            
            # Add colored mask to composite (don't apply to overlay yet)
            composite_mask[core_mask > 0] = color
            
            # Draw contour outline on composite
            contours, _ = cv2.findContours(core_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(composite_mask, contours, -1, color, 2)
        
        # Apply composite mask to overlay ONCE to avoid progressive darkening
        overlay = cv2.addWeighted(overlay, 0.8, composite_mask, 0.2, 0)
        
        # Add labels and center points
        for i, mask_id in enumerate(unique_ids):
            core_mask = (mask == mask_id).astype(np.uint8)
            
            # Find center for label placement
            M = cv2.moments(core_mask)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Draw center point
                cv2.circle(overlay, (cx, cy), 3, (255, 255, 255), -1)
                
                # Put label with black outline for better visibility
                label = f"{i + 1}"
                # Black outline
                cv2.putText(overlay, label, (cx - 10, cy + 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
                # White text
                cv2.putText(overlay, label, (cx - 10, cy + 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    else:
        # Fallback: draw circles if no mask provided
        for i, (x, y, radius) in enumerate(cores):
            cv2.circle(overlay, (int(x), int(y)), int(radius), (0, 255, 0), 2)
            cv2.circle(overlay, (int(x), int(y)), 3, (0, 0, 255), -1)
            label = f"{i + 1}"
            cv2.putText(overlay, label, (int(x) - 10, int(y) + 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    return overlay


def crop_cores_simple(image, cores, output_dir, padding_factor=1.2):
    """Simple core cropping function with circular mask"""
    cropped_info = []
    
    for i, (x, y, r) in enumerate(cores):
        try:
            # Define a square crop region
            crop_r = int(r * padding_factor)
            x1, y1 = x - crop_r, y - crop_r
            x2, y2 = x + crop_r, y + crop_r

            # Ensure crop boundaries are within image dimensions
            x1_safe, y1_safe = max(0, x1), max(0, y1)
            x2_safe, y2_safe = min(image.shape[1], x2), min(image.shape[0], y2)
            
            crop = image[y1_safe:y2_safe, x1_safe:x2_safe]

            # Create a circular mask that exactly matches the crop dimensions
            mask = np.zeros(crop.shape[:2], dtype=bool)  # Create mask with exact crop dimensions
            
            # Calculate center relative to the actual crop
            center_y, center_x = y - y1_safe, x - x1_safe
            
            # Create the circular mask
            for mask_y_idx in range(crop.shape[0]):
                for mask_x_idx in range(crop.shape[1]):
                    dist = np.sqrt((mask_x_idx - center_x)**2 + (mask_y_idx - center_y)**2)
                    if dist <= r:
                        mask[mask_y_idx, mask_x_idx] = True

            # Apply mask (guaranteed to have matching shapes)
            if len(crop.shape) == 3:
                masked_crop = crop * mask[..., np.newaxis]
            else:
                masked_crop = crop * mask

            # Save crop
            crop_filename = f"core_{i+1:02d}.png"
            crop_path = os.path.join(output_dir, crop_filename)
            
            # Convert to uint8 for saving
            if masked_crop.dtype != np.uint8:
                masked_crop = cv2.normalize(masked_crop, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            cv2.imwrite(crop_path, masked_crop)
            
            crop_info = {
                'filename': crop_filename,
                'center': (x, y),
                'radius': r,
                'crop_size': crop.shape
            }
            
            cropped_info.append(crop_info)
            
        except Exception as e:
            print(f"Failed to crop core {i+1}: {e}")
            continue
    
    return cropped_info


def crop_cores_with_masks(image, mask, cores, output_dir, padding_factor=1.2, scale_factor=1.0, masks_dir=None):
    """Crop cores and save both image and binary mask for each core
    
    Args:
        image: Original image (grayscale or color)
        mask: Segmentation mask from Cellpose (same scale as processed image)
        cores: List of (x, y, radius) tuples (in processed image coordinates)
        output_dir: Directory to save crops
        padding_factor: Padding around each core (default 1.2)
        scale_factor: Factor to scale coordinates from processed to original image
        masks_dir: Directory to save masks (if None, saves to output_dir)
        
    Returns:
        List of crop information dictionaries
    """
    cropped_info = []
    
    # Scale mask to original image size if needed
    if scale_factor != 1.0:
        orig_height, orig_width = int(mask.shape[0] * scale_factor), int(mask.shape[1] * scale_factor)
        scaled_mask = cv2.resize(mask.astype(np.uint8), (orig_width, orig_height), interpolation=cv2.INTER_NEAREST)
    else:
        scaled_mask = mask.astype(np.uint8)
    
    for i, (x, y, r) in enumerate(cores):
        try:
            # Scale coordinates to original image if needed
            orig_x = int(x * scale_factor)
            orig_y = int(y * scale_factor) 
            orig_r = int(r * scale_factor)
            
            # Calculate square crop boundaries with padding
            crop_r = int(orig_r * padding_factor)
            x1 = max(0, orig_x - crop_r)
            y1 = max(0, orig_y - crop_r)
            x2 = min(image.shape[1], orig_x + crop_r)
            y2 = min(image.shape[0], orig_y + crop_r)
            
            # Extract image crop using safe boundaries
            image_crop = image[y1:y2, x1:x2]
            
            # Extract mask crop using the SAME safe boundaries to ensure matching dimensions
            # Get the mask value at the core center
            mask_y = min(orig_y, scaled_mask.shape[0] - 1)
            mask_x = min(orig_x, scaled_mask.shape[1] - 1)
            core_mask_id = scaled_mask[mask_y, mask_x]
            
            # Create binary mask for this specific core
            core_binary_mask = (scaled_mask == core_mask_id).astype(np.uint8)
            # Use the same safe boundaries for mask crop
            mask_crop = core_binary_mask[y1:y2, x1:x2]
            
            # Ensure mask_crop exactly matches image_crop dimensions (robust fix for broadcasting errors)
            if mask_crop.shape != image_crop.shape[:2]:
                mask_crop = cv2.resize(mask_crop, (image_crop.shape[1], image_crop.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # Apply mask filtering to the image crop
            # Only preserve pixels within the detected core boundary
            if len(image_crop.shape) == 3:  # Color image
                # Expand mask to match image channels
                mask_3d = np.expand_dims(mask_crop, axis=2)
                mask_3d = np.repeat(mask_3d, image_crop.shape[2], axis=2)
                filtered_image_crop = image_crop * mask_3d
            else:  # Grayscale image
                filtered_image_crop = image_crop * mask_crop
            
            # Save filtered image crop (with mask applied)
            image_filename = f"core_{i+1:02d}_image.png"
            image_path = os.path.join(output_dir, image_filename)
            
            # Ensure proper data type for saving
            if filtered_image_crop.dtype != np.uint8:
                image_normalized = cv2.normalize(filtered_image_crop, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            else:
                image_normalized = filtered_image_crop
                
            cv2.imwrite(image_path, image_normalized)
            
            # Save binary mask crop (convert to 255 for visibility)
            mask_filename = f"core_{i+1:02d}_mask.png"
            # Use masks_dir if provided, otherwise fall back to output_dir
            mask_save_dir = masks_dir if masks_dir is not None else output_dir
            mask_path = os.path.join(mask_save_dir, mask_filename)
            mask_crop_255 = mask_crop * 255  # Convert 0/1 to 0/255 for saving
            cv2.imwrite(mask_path, mask_crop_255)
            
            crop_info = {
                'core_id': i + 1,
                'image_filename': image_filename,
                'mask_filename': mask_filename,
                'center_original': (orig_x, orig_y),
                'radius_original': orig_r,
                'crop_bounds': (x1, y1, x2, y2),
                'crop_size': image_crop.shape,
                'mask_id': int(core_mask_id)
            }
            
            cropped_info.append(crop_info)
            
        except Exception as e:
            print(f"Failed to crop core {i+1}: {e}")
            continue
    
    return cropped_info