"""
Alternative: Download FLIR ADAS Dataset (Easier to get, smaller)
FLIR ADAS is a smaller, publicly available thermal dataset
"""

import os
import requests
import json
from pathlib import Path


def download_flir_info():
    """Get FLIR ADAS dataset information"""
    print("🔥 FLIR ADAS Dataset (Alternative to KAIST)")
    print("=" * 50)
    print("\nFLIR ADAS is easier to download and smaller (~2GB)")
    print("\nDownload steps:")
    print("1. Visit: https://www.flir.com/oem/adas/adas-dataset-form/")
    print("2. Fill out the form (free, no approval needed)")
    print("3. Download the dataset")
    print("4. Extract to data/raw/flir/")
    print("\nThe dataset contains:")
    print("- RGB images")
    print("- Thermal images")
    print("- JSON annotations")
    print("- ~14,000 image pairs")
    print("\nAfter download, run:")
    print("  python utils/process_flir.py --input data/raw/flir --output data/processed")


def check_flir_structure(flir_dir):
    """Check if FLIR dataset is properly structured"""
    required_dirs = ['thermal_8_bit', 'RGB']
    
    if not os.path.exists(flir_dir):
        return False, f"Directory not found: {flir_dir}"
    
    missing = []
    for dir_name in required_dirs:
        if not os.path.exists(os.path.join(flir_dir, dir_name)):
            missing.append(dir_name)
    
    if missing:
        return False, f"Missing directories: {', '.join(missing)}"
    
    return True, "FLIR structure looks good!"


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='FLIR ADAS Dataset Helper')
    parser.add_argument('--check', type=str, default='data/raw/flir',
                       help='Check FLIR dataset structure')
    
    args = parser.parse_args()
    
    if os.path.exists(args.check):
        valid, message = check_flir_structure(args.check)
        if valid:
            print(f"✅ {message}")
        else:
            print(f"❌ {message}")
    else:
        download_flir_info()

