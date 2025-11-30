"""
Process FLIR ADAS dataset
FLIR dataset structure:
  - thermal_8_bit/  (thermal images)
  - RGB/            (RGB images)
  - Annotations/    (JSON files)
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import shutil


def load_flir_mapping(flir_dir):
    """Load RGB to Thermal mapping from JSON file"""
    mapping_file = os.path.join(flir_dir, 'rgb_to_thermal_vid_map.json')
    if os.path.exists(mapping_file):
        with open(mapping_file, 'r') as f:
            return json.load(f)
    return None


def match_flir_pairs(flir_dir, output_rgb_dir, output_thermal_dir, max_pairs=8000):
    """
    Match FLIR RGB and Thermal image pairs
    FLIR ADAS v2 structure:
    - images_rgb_train/data/
    - images_thermal_train/data/
    - images_rgb_val/data/
    - images_thermal_val/data/
    """
    # Try FLIR ADAS v2 structure first
    rgb_train_dir = os.path.join(flir_dir, 'images_rgb_train', 'data')
    thermal_train_dir = os.path.join(flir_dir, 'images_thermal_train', 'data')
    rgb_val_dir = os.path.join(flir_dir, 'images_rgb_val', 'data')
    thermal_val_dir = os.path.join(flir_dir, 'images_thermal_val', 'data')
    
    # Fallback to old structure
    if not os.path.exists(rgb_train_dir):
        rgb_train_dir = os.path.join(flir_dir, 'RGB')
        thermal_train_dir = os.path.join(flir_dir, 'thermal_8_bit')
        rgb_val_dir = None
        thermal_val_dir = None
    
    if not os.path.exists(rgb_train_dir):
        raise FileNotFoundError(f"RGB directory not found: {rgb_train_dir}")
    if not os.path.exists(thermal_train_dir):
        raise FileNotFoundError(f"Thermal directory not found: {thermal_train_dir}")
    
    # Load mapping if available (for video frames)
    mapping = load_flir_mapping(flir_dir)
    
    # Get all RGB images from train and val
    rgb_files = []
    rgb_to_thermal_map = {}
    
    if os.path.exists(rgb_train_dir):
        train_files = [f for f in os.listdir(rgb_train_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        rgb_files.extend(train_files)
        print(f"Found {len(train_files)} RGB training images")
    
    if rgb_val_dir and os.path.exists(rgb_val_dir):
        val_files = [f for f in os.listdir(rgb_val_dir) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        rgb_files.extend(val_files)
        print(f"Found {len(val_files)} RGB validation images")
    
    # Build mapping: for train/val images, filenames should match
    # For video frames, use the mapping file
    if mapping:
        print(f"Loaded mapping file with {len(mapping)} entries")
        for rgb_file in rgb_files:
            if rgb_file in mapping:
                rgb_to_thermal_map[rgb_file] = mapping[rgb_file]
    
    rgb_files = sorted(rgb_files)
    print(f"Total: {len(rgb_files)} RGB images")
    
    # Create output directories
    os.makedirs(output_rgb_dir, exist_ok=True)
    os.makedirs(output_thermal_dir, exist_ok=True)
    
    matched_pairs = 0
    skipped = 0
    
    for rgb_file in tqdm(rgb_files[:max_pairs], desc="Matching pairs"):
        # Try train directory first
        rgb_path = os.path.join(rgb_train_dir, rgb_file)
        is_train = os.path.exists(rgb_path)
        
        # If not in train, try val
        if not is_train and rgb_val_dir:
            rgb_path = os.path.join(rgb_val_dir, rgb_file)
            if not os.path.exists(rgb_path):
                skipped += 1
                continue
        
        # Find corresponding thermal file
        # First check if we have a mapping
        thermal_file = rgb_file
        if rgb_file in rgb_to_thermal_map:
            thermal_file = rgb_to_thermal_map[rgb_file]
        
        # Try same filename first (for train/val images)
        if is_train:
            thermal_path = os.path.join(thermal_train_dir, thermal_file)
        else:
            thermal_path = os.path.join(thermal_val_dir, thermal_file) if thermal_val_dir else None
        
        # If not found, try same name in both directories
        if not thermal_path or not os.path.exists(thermal_path):
            # Try train directory
            thermal_path = os.path.join(thermal_train_dir, thermal_file)
            if not os.path.exists(thermal_path):
                # Try val directory
                if thermal_val_dir:
                    thermal_path = os.path.join(thermal_val_dir, thermal_file)
                if not os.path.exists(thermal_path):
                    # Try with different extensions
                    base_name = os.path.splitext(thermal_file)[0]
                    found = False
                    for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                        alt_path = os.path.join(thermal_train_dir, base_name + ext)
                        if os.path.exists(alt_path):
                            thermal_path = alt_path
                            thermal_file = os.path.basename(alt_path)
                            found = True
                            break
                        if thermal_val_dir:
                            alt_path = os.path.join(thermal_val_dir, base_name + ext)
                            if os.path.exists(alt_path):
                                thermal_path = alt_path
                                thermal_file = os.path.basename(alt_path)
                                found = True
                                break
                    if not found:
                        skipped += 1
                        continue
        
        # Copy files to output directories
        output_rgb = os.path.join(output_rgb_dir, rgb_file)
        output_thermal = os.path.join(output_thermal_dir, rgb_file)
        
        # Use symlinks to save space, fallback to copy
        try:
            if not os.path.exists(output_rgb):
                os.symlink(os.path.abspath(rgb_path), output_rgb)
            if not os.path.exists(output_thermal):
                os.symlink(os.path.abspath(thermal_path), output_thermal)
        except:
            # Fallback to copying
            shutil.copy2(rgb_path, output_rgb)
            shutil.copy2(thermal_path, output_thermal)
        
        matched_pairs += 1
    
    print(f"\n✅ Matched {matched_pairs} RGB-Thermal pairs")
    if skipped > 0:
        print(f"⚠️  Skipped {skipped} RGB images (no thermal match)")
    
    return matched_pairs


def resize_flir_images(rgb_dir, thermal_dir, target_size=(256, 256)):
    """Resize FLIR images to target size"""
    print(f"\n📐 Resizing images to {target_size}...")
    
    rgb_files = sorted([f for f in os.listdir(rgb_dir) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    for rgb_file in tqdm(rgb_files, desc="Resizing"):
        rgb_path = os.path.join(rgb_dir, rgb_file)
        thermal_path = os.path.join(thermal_dir, rgb_file)
        
        if not os.path.exists(thermal_path):
            continue
        
        # Load and resize RGB
        rgb_img = cv2.imread(rgb_path)
        if rgb_img is None:
            continue
        rgb_resized = cv2.resize(rgb_img, target_size)
        cv2.imwrite(rgb_path, rgb_resized)
        
        # Load and resize thermal
        thermal_img = cv2.imread(thermal_path, cv2.IMREAD_GRAYSCALE)
        if thermal_img is None:
            continue
        thermal_resized = cv2.resize(thermal_img, target_size)
        cv2.imwrite(thermal_path, thermal_resized)
    
    print("✅ Resizing complete!")


def main():
    parser = argparse.ArgumentParser(description='Process FLIR ADAS Dataset')
    parser.add_argument('--input', type=str, required=True,
                       help='FLIR dataset root directory (contains RGB/ and thermal_8_bit/)')
    parser.add_argument('--output_rgb', type=str, default='../data/processed/rgb',
                       help='Output directory for RGB images')
    parser.add_argument('--output_thermal', type=str, default='../data/processed/thermal',
                       help='Output directory for thermal images')
    parser.add_argument('--max_pairs', type=int, default=8000,
                       help='Maximum number of pairs to process')
    parser.add_argument('--resize', type=int, default=256,
                       help='Target image size (256, 512, etc.)')
    parser.add_argument('--skip_resize', action='store_true',
                       help='Skip resizing step')
    
    args = parser.parse_args()
    
    print("🔥 FLIR ADAS Dataset Processor")
    print("=" * 50)
    
    # Match pairs
    pairs = match_flir_pairs(
        args.input,
        args.output_rgb,
        args.output_thermal,
        max_pairs=args.max_pairs
    )
    
    if pairs == 0:
        print("❌ No pairs matched. Check your FLIR dataset structure.")
        return
    
    # Resize if requested
    if not args.skip_resize:
        resize_flir_images(
            args.output_rgb,
            args.output_thermal,
            target_size=(args.resize, args.resize)
        )
    
    print(f"\n✅ FLIR dataset processed!")
    print(f"   RGB images: {args.output_rgb} ({pairs} images)")
    print(f"   Thermal images: {args.output_thermal} ({pairs} images)")
    print(f"\n🚀 Ready for training!")
    print(f"   python model/train.py --rgb_dir {args.output_rgb} --thermal_dir {args.output_thermal}")


if __name__ == "__main__":
    main()

