"""
Process FLIR ADAS dataset using COCO JSON files to match RGB-Thermal pairs
This is more reliable than filename matching
"""

import os
import json
import cv2
import shutil
from tqdm import tqdm
import argparse


def load_coco_images(coco_file):
    """Load image list from COCO JSON file"""
    with open(coco_file, 'r') as f:
        data = json.load(f)
    return {img['id']: img['file_name'] for img in data.get('images', [])}


def match_from_coco(flir_dir, output_rgb_dir, output_thermal_dir, max_pairs=8000):
    """
    Match RGB-Thermal pairs using COCO JSON files
    FLIR uses same image IDs in both RGB and thermal COCO files
    """
    # Paths
    rgb_train_coco = os.path.join(flir_dir, 'images_rgb_train', 'coco.json')
    thermal_train_coco = os.path.join(flir_dir, 'images_thermal_train', 'coco.json')
    rgb_val_coco = os.path.join(flir_dir, 'images_rgb_val', 'coco.json')
    thermal_val_coco = os.path.join(flir_dir, 'images_thermal_val', 'coco.json')
    
    rgb_train_dir = os.path.join(flir_dir, 'images_rgb_train', 'data')
    thermal_train_dir = os.path.join(flir_dir, 'images_thermal_train', 'data')
    rgb_val_dir = os.path.join(flir_dir, 'images_rgb_val', 'data')
    thermal_val_dir = os.path.join(flir_dir, 'images_thermal_val', 'data')
    
    # Load COCO files
    rgb_train_images = load_coco_images(rgb_train_coco) if os.path.exists(rgb_train_coco) else {}
    thermal_train_images = load_coco_images(thermal_train_coco) if os.path.exists(thermal_train_coco) else {}
    rgb_val_images = load_coco_images(rgb_val_coco) if os.path.exists(rgb_val_coco) else {}
    thermal_val_images = load_coco_images(thermal_val_coco) if os.path.exists(thermal_val_coco) else {}
    
    print(f"Loaded {len(rgb_train_images)} RGB train images from COCO")
    print(f"Loaded {len(thermal_train_images)} Thermal train images from COCO")
    print(f"Loaded {len(rgb_val_images)} RGB val images from COCO")
    print(f"Loaded {len(thermal_val_images)} Thermal val images from COCO")
    
    # Find matching IDs
    train_matches = set(rgb_train_images.keys()) & set(thermal_train_images.keys())
    val_matches = set(rgb_val_images.keys()) & set(thermal_val_images.keys())
    all_matches = list(train_matches) + list(val_matches)
    
    print(f"\nFound {len(train_matches)} matching train pairs")
    print(f"Found {len(val_matches)} matching val pairs")
    print(f"Total: {len(all_matches)} matching pairs")
    
    # Create output directories
    os.makedirs(output_rgb_dir, exist_ok=True)
    os.makedirs(output_thermal_dir, exist_ok=True)
    
    matched = 0
    skipped = 0
    
    # Process train matches
    for img_id in tqdm(list(train_matches)[:max_pairs], desc="Processing pairs"):
        rgb_file = rgb_train_images[img_id]
        thermal_file = thermal_train_images[img_id]
        
        # Handle file_name that may include "data/" prefix
        rgb_file_clean = rgb_file.replace('data/', '') if 'data/' in rgb_file else rgb_file
        thermal_file_clean = thermal_file.replace('data/', '') if 'data/' in thermal_file else thermal_file
        
        rgb_path = os.path.join(rgb_train_dir, rgb_file_clean)
        thermal_path = os.path.join(thermal_train_dir, thermal_file_clean)
        
        if not os.path.exists(rgb_path) or not os.path.exists(thermal_path):
            skipped += 1
            continue
        
        # Use image ID as output filename to ensure uniqueness
        output_rgb = os.path.join(output_rgb_dir, f"{img_id}.jpg")
        output_thermal = os.path.join(output_thermal_dir, f"{img_id}.jpg")
        
        # Copy files
        try:
            shutil.copy2(rgb_path, output_rgb)
            shutil.copy2(thermal_path, output_thermal)
            matched += 1
        except Exception as e:
            print(f"Error copying {img_id}: {e}")
            skipped += 1
    
    # Process val matches if we need more
    if matched < max_pairs and val_matches:
        remaining = max_pairs - matched
        for img_id in tqdm(list(val_matches)[:remaining], desc="Processing val pairs"):
            rgb_file = rgb_val_images[img_id]
            thermal_file = thermal_val_images[img_id]
            
            # Handle file_name that may include "data/" prefix
            rgb_file_clean = rgb_file.replace('data/', '') if 'data/' in rgb_file else rgb_file
            thermal_file_clean = thermal_file.replace('data/', '') if 'data/' in thermal_file else thermal_file
            
            rgb_path = os.path.join(rgb_val_dir, rgb_file_clean)
            thermal_path = os.path.join(thermal_val_dir, thermal_file_clean)
            
            if not os.path.exists(rgb_path) or not os.path.exists(thermal_path):
                skipped += 1
                continue
            
            output_rgb = os.path.join(output_rgb_dir, f"{img_id}.jpg")
            output_thermal = os.path.join(output_thermal_dir, f"{img_id}.jpg")
            
            try:
                shutil.copy2(rgb_path, output_rgb)
                shutil.copy2(thermal_path, output_thermal)
                matched += 1
            except Exception as e:
                skipped += 1
    
    print(f"\n✅ Matched {matched} RGB-Thermal pairs")
    if skipped > 0:
        print(f"⚠️  Skipped {skipped} pairs")
    
    return matched


def resize_images(rgb_dir, thermal_dir, target_size=(256, 256)):
    """Resize images to target size"""
    print(f"\n📐 Resizing images to {target_size}...")
    
    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.jpg')])
    
    for rgb_file in tqdm(rgb_files, desc="Resizing"):
        rgb_path = os.path.join(rgb_dir, rgb_file)
        thermal_path = os.path.join(thermal_dir, rgb_file)
        
        if not os.path.exists(thermal_path):
            continue
        
        # Resize RGB
        rgb_img = cv2.imread(rgb_path)
        if rgb_img is not None:
            rgb_resized = cv2.resize(rgb_img, target_size)
            cv2.imwrite(rgb_path, rgb_resized)
        
        # Resize thermal
        thermal_img = cv2.imread(thermal_path, cv2.IMREAD_GRAYSCALE)
        if thermal_img is not None:
            thermal_resized = cv2.resize(thermal_img, target_size)
            cv2.imwrite(thermal_path, thermal_resized)
    
    print("✅ Resizing complete!")


def main():
    parser = argparse.ArgumentParser(description='Process FLIR ADAS Dataset using COCO JSON')
    parser.add_argument('--input', type=str, required=True,
                       help='FLIR dataset root directory')
    parser.add_argument('--output_rgb', type=str, default='../data/processed/rgb',
                       help='Output directory for RGB images')
    parser.add_argument('--output_thermal', type=str, default='../data/processed/thermal',
                       help='Output directory for thermal images')
    parser.add_argument('--max_pairs', type=int, default=8000,
                       help='Maximum number of pairs to process')
    parser.add_argument('--resize', type=int, default=256,
                       help='Target image size')
    parser.add_argument('--skip_resize', action='store_true',
                       help='Skip resizing step')
    
    args = parser.parse_args()
    
    print("🔥 FLIR ADAS Dataset Processor (COCO-based)")
    print("=" * 50)
    
    pairs = match_from_coco(
        args.input,
        args.output_rgb,
        args.output_thermal,
        max_pairs=args.max_pairs
    )
    
    if pairs == 0:
        print("❌ No pairs matched.")
        return
    
    if not args.skip_resize:
        resize_images(
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

