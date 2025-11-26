"""
Additional preprocessing utilities for thermal data
"""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


def normalize_thermal(thermal_img, method='minmax'):
    """
    Normalize thermal image to [0, 1]
    
    Args:
        thermal_img: Thermal image (numpy array)
        method: 'minmax' or 'percentile'
    
    Returns:
        normalized: Normalized thermal image
    """
    if method == 'minmax':
        min_val = thermal_img.min()
        max_val = thermal_img.max()
        if max_val > min_val:
            normalized = (thermal_img - min_val) / (max_val - min_val)
        else:
            normalized = thermal_img
    elif method == 'percentile':
        p2, p98 = np.percentile(thermal_img, [2, 98])
        normalized = np.clip((thermal_img - p2) / (p98 - p2), 0, 1)
    else:
        normalized = thermal_img / 255.0
    
    return normalized.astype(np.float32)


def smooth_thermal(thermal_img, method='bilateral', **kwargs):
    """
    Smooth thermal image to reduce noise
    
    Args:
        thermal_img: Thermal image
        method: 'bilateral', 'gaussian', or 'median'
        **kwargs: Additional parameters for smoothing
    
    Returns:
        smoothed: Smoothed thermal image
    """
    if method == 'bilateral':
        d = kwargs.get('d', 9)
        sigma_color = kwargs.get('sigma_color', 75)
        sigma_space = kwargs.get('sigma_space', 75)
        smoothed = cv2.bilateralFilter(thermal_img, d, sigma_color, sigma_space)
    elif method == 'gaussian':
        sigma = kwargs.get('sigma', 1.0)
        smoothed = gaussian_filter(thermal_img, sigma=sigma)
    elif method == 'median':
        ksize = kwargs.get('ksize', 5)
        smoothed = cv2.medianBlur(thermal_img, ksize)
    else:
        smoothed = thermal_img
    
    return smoothed


def enhance_thermal_contrast(thermal_img, alpha=1.5, beta=0):
    """
    Enhance thermal image contrast
    
    Args:
        thermal_img: Thermal image
        alpha: Contrast control (1.0-3.0)
        beta: Brightness control (0-100)
    
    Returns:
        enhanced: Enhanced thermal image
    """
    enhanced = cv2.convertScaleAbs(thermal_img, alpha=alpha, beta=beta)
    return enhanced


def create_thermal_mask(thermal_img, threshold_percentile=90):
    """
    Create mask for valid thermal regions
    
    Args:
        thermal_img: Thermal image
        threshold_percentile: Percentile threshold for valid regions
    
    Returns:
        mask: Binary mask
    """
    threshold = np.percentile(thermal_img, threshold_percentile)
    mask = (thermal_img > threshold).astype(np.uint8) * 255
    return mask

