"""
Align RGB and Thermal images from KAIST dataset
Handles calibration and synchronization
"""

import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm


def align_images(rgb_img, thermal_img, calibration_matrix=None):
    """
    Align RGB and thermal images using calibration matrix or feature matching
    
    Args:
        rgb_img: RGB image (numpy array)
        thermal_img: Thermal image (numpy array)
        calibration_matrix: Optional homography matrix
    
    Returns:
        aligned_thermal: Aligned thermal image
    """
    if calibration_matrix is not None:
        # Use provided calibration matrix
        h, w = rgb_img.shape[:2]
        aligned_thermal = cv2.warpPerspective(thermal_img, calibration_matrix, (w, h))
        return aligned_thermal
    else:
        # Feature-based alignment (ORB)
        orb = cv2.ORB_create()
        
        # Convert thermal to 3-channel for feature detection
        if len(thermal_img.shape) == 2:
            thermal_3ch = cv2.cvtColor(thermal_img, cv2.COLOR_GRAY2BGR)
        else:
            thermal_3ch = thermal_img
        
        # Find keypoints and descriptors
        kp1, des1 = orb.detectAndCompute(rgb_img, None)
        kp2, des2 = orb.detectAndCompute(thermal_3ch, None)
        
        if des1 is None or des2 is None:
            # Fallback: simple resize
            h, w = rgb_img.shape[:2]
            return cv2.resize(thermal_img, (w, h))
        
        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        if len(matches) < 4:
            # Not enough matches, use resize
            h, w = rgb_img.shape[:2]
            return cv2.resize(thermal_img, (w, h))
        
        # Extract matched points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
        
        # Find homography
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        
        if M is None:
            h, w = rgb_img.shape[:2]
            return cv2.resize(thermal_img, (w, h))
        
        # Warp thermal image
        h, w = rgb_img.shape[:2]
        aligned_thermal = cv2.warpPerspective(thermal_img, M, (w, h))
        
        return aligned_thermal


def process_kaist_dataset(raw_rgb_dir, raw_thermal_dir, output_rgb_dir, output_thermal_dir, 
                          target_size=(256, 256), use_calibration=True):
    """
    Process KAIST dataset: align and resize RGB-Thermal pairs
    
    Args:
        raw_rgb_dir: Directory with raw RGB images
        raw_thermal_dir: Directory with raw thermal images
        output_rgb_dir: Output directory for processed RGB
        output_thermal_dir: Output directory for processed thermal
        target_size: Target image size (width, height)
        use_calibration: Whether to use calibration-based alignment
    """
    os.makedirs(output_rgb_dir, exist_ok=True)
    os.makedirs(output_thermal_dir, exist_ok=True)
    
    # Get RGB files
    rgb_files = sorted([f for f in os.listdir(raw_rgb_dir) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"Found {len(rgb_files)} RGB images")
    
    calibration_matrix = None
    if use_calibration:
        # Try to load calibration matrix if available
        calib_path = os.path.join(raw_rgb_dir, 'calibration_matrix.npy')
        if os.path.exists(calib_path):
            calibration_matrix = np.load(calib_path)
            print("Loaded calibration matrix")
    
    processed = 0
    for rgb_file in tqdm(rgb_files, desc="Processing pairs"):
        rgb_path = os.path.join(raw_rgb_dir, rgb_file)
        
        # Find corresponding thermal file (same name or pattern)
        thermal_file = rgb_file  # Adjust naming pattern if needed
        thermal_path = os.path.join(raw_thermal_dir, thermal_file)
        
        if not os.path.exists(thermal_path):
            # Try alternative naming
            base_name = os.path.splitext(rgb_file)[0]
            for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                alt_path = os.path.join(raw_thermal_dir, base_name + ext)
                if os.path.exists(alt_path):
                    thermal_path = alt_path
                    break
            else:
                continue  # Skip if no thermal match found
        
        # Load images
        rgb_img = cv2.imread(rgb_path)
        thermal_img = cv2.imread(thermal_path, cv2.IMREAD_GRAYSCALE)
        
        if rgb_img is None or thermal_img is None:
            continue
        
        # Align thermal to RGB
        aligned_thermal = align_images(rgb_img, thermal_img, calibration_matrix)
        
        # Resize both to target size
        rgb_resized = cv2.resize(rgb_img, target_size)
        thermal_resized = cv2.resize(aligned_thermal, target_size)
        
        # Normalize thermal to [0, 255]
        if thermal_resized.max() > 255:
            thermal_resized = (thermal_resized / thermal_resized.max() * 255).astype(np.uint8)
        
        # Save processed images
        output_rgb_path = os.path.join(output_rgb_dir, rgb_file)
        output_thermal_path = os.path.join(output_thermal_dir, rgb_file)
        
        cv2.imwrite(output_rgb_path, rgb_resized)
        cv2.imwrite(output_thermal_path, thermal_resized)
        
        processed += 1
    
    print(f"Processed {processed} RGB-Thermal pairs")
    return processed


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Align and preprocess KAIST dataset')
    parser.add_argument('--raw_rgb', type=str, required=True, help='Raw RGB images directory')
    parser.add_argument('--raw_thermal', type=str, required=True, help='Raw thermal images directory')
    parser.add_argument('--output_rgb', type=str, default='../data/processed/rgb', help='Output RGB directory')
    parser.add_argument('--output_thermal', type=str, default='../data/processed/thermal', help='Output thermal directory')
    parser.add_argument('--size', type=int, default=256, help='Target image size')
    parser.add_argument('--no_calibration', action='store_true', help='Disable calibration-based alignment')
    
    args = parser.parse_args()
    
    process_kaist_dataset(
        args.raw_rgb,
        args.raw_thermal,
        args.output_rgb,
        args.output_thermal,
        target_size=(args.size, args.size),
        use_calibration=not args.no_calibration
    )

