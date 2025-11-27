"""
Download and setup KAIST Multispectral Dataset
KAIST dataset contains RGB and thermal (LWIR) image pairs

Options:
1. Hugging Face (easiest, no registration)
2. Official KAIST website (requires registration)
"""

import os
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm
import argparse

try:
    from huggingface_hub import hf_hub_download, snapshot_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("⚠️  huggingface-hub not installed. Install with: pip install huggingface-hub")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


def download_file(url, destination, description="Downloading"):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    with open(destination, 'wb') as f, tqdm(
        desc=description,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))
    
    return destination


def extract_archive(archive_path, extract_to):
    """Extract zip or tar archive"""
    print(f"Extracting {archive_path} to {extract_to}...")
    os.makedirs(extract_to, exist_ok=True)
    
    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_to)
    elif archive_path.endswith('.tar'):
        with tarfile.open(archive_path, 'r') as tar_ref:
            tar_ref.extractall(extract_to)
    
    print("Extraction complete!")


def setup_kaist_structure(raw_dir, output_rgb_dir, output_thermal_dir):
    """
    Organize KAIST dataset into RGB and Thermal folders
    KAIST structure: images/set00/V000/I00000.jpg (RGB)
                     images/set00/V000/I00000.jpg (Thermal - same structure)
    """
    print("Organizing KAIST dataset structure...")
    
    # Find all RGB and thermal images
    rgb_images = []
    thermal_images = []
    
    # Common KAIST dataset patterns
    for root, dirs, files in os.walk(raw_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(root, file)
                # Check if it's thermal or RGB based on path or filename
                if 'thermal' in root.lower() or 'lwir' in root.lower() or 'I' in file:
                    thermal_images.append(filepath)
                elif 'visible' in root.lower() or 'rgb' in root.lower() or 'V' in file:
                    rgb_images.append(filepath)
    
    print(f"Found {len(rgb_images)} RGB images")
    print(f"Found {len(thermal_images)} thermal images")
    
    # Create output directories
    os.makedirs(output_rgb_dir, exist_ok=True)
    os.makedirs(output_thermal_dir, exist_ok=True)
    
    # Copy/link images (we'll use symlinks to save space)
    copied_rgb = 0
    copied_thermal = 0
    
    for rgb_path in tqdm(rgb_images[:8000], desc="Organizing RGB images"):
        filename = os.path.basename(rgb_path)
        dest = os.path.join(output_rgb_dir, filename)
        if not os.path.exists(dest):
            try:
                os.symlink(os.path.abspath(rgb_path), dest)
            except:
                import shutil
                shutil.copy2(rgb_path, dest)
            copied_rgb += 1
    
    for thermal_path in tqdm(thermal_images[:8000], desc="Organizing thermal images"):
        filename = os.path.basename(thermal_path)
        dest = os.path.join(output_thermal_dir, filename)
        if not os.path.exists(dest):
            try:
                os.symlink(os.path.abspath(thermal_path), dest)
            except:
                import shutil
                shutil.copy2(thermal_path, dest)
            copied_thermal += 1
    
    print(f"\n✅ Organized {copied_rgb} RGB images")
    print(f"✅ Organized {copied_thermal} thermal images")
    
    return copied_rgb, copied_thermal


def download_from_huggingface(repo_id, download_dir):
    """Download KAIST dataset from Hugging Face"""
    if not HF_AVAILABLE:
        print("❌ huggingface-hub not installed. Install with: pip install huggingface-hub")
        return False
    
    print(f"📥 Downloading from Hugging Face: {repo_id}")
    print("This may take a while (dataset is large)...")
    
    try:
        # Try to download the dataset
        snapshot_download(
            repo_id=repo_id,
            local_dir=download_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print("✅ Download complete!")
        return True
    except Exception as e:
        print(f"❌ Error downloading from Hugging Face: {e}")
        print("\nTrying alternative methods...")
        return False


def main():
    parser = argparse.ArgumentParser(description='Download KAIST Multispectral Dataset')
    parser.add_argument('--download_dir', type=str, default='data/raw', 
                       help='Directory to download dataset')
    parser.add_argument('--output_rgb', type=str, default='data/processed/rgb',
                       help='Output directory for RGB images')
    parser.add_argument('--output_thermal', type=str, default='data/processed/thermal',
                       help='Output directory for thermal images')
    parser.add_argument('--skip_download', action='store_true',
                       help='Skip download, only organize existing files')
    parser.add_argument('--use_huggingface', action='store_true',
                       help='Try downloading from Hugging Face')
    parser.add_argument('--hf_repo', type=str, 
                       default='kaist-cv/kaist-multispectral-pedestrian-dataset',
                       help='Hugging Face repository ID')
    
    args = parser.parse_args()
    
    print("🔥 KAIST Multispectral Dataset Downloader")
    print("=" * 50)
    
    if not args.skip_download:
        # Try Hugging Face first if requested
        if args.use_huggingface and HF_AVAILABLE:
            print("\n📥 Attempting download from Hugging Face...")
            success = download_from_huggingface(args.hf_repo, args.download_dir)
            if success:
                args.skip_download = True  # Dataset downloaded, now organize
            else:
                print("\n⚠️  Hugging Face download failed. Trying manual methods...")
        
        if not args.skip_download:
            print("\n📥 Download Instructions:")
            print("\nOption 1: Hugging Face (Easiest)")
            print("  python download_kaist.py --use_huggingface")
            print("\nOption 2: Official KAIST Website")
            print("  1. Visit: http://multispectral.kaist.ac.kr/pedestrian/")
            print("  2. Register and request access")
            print("  3. Download the dataset")
            print("  4. Extract to:", args.download_dir)
            print("\nOption 3: Alternative - FLIR ADAS (Easier, smaller)")
            print("  Visit: https://www.flir.com/oem/adas/adas-dataset-form/")
            print("  Download FLIR ADAS dataset (~2GB, no approval needed)")
            print("\nAfter downloading, run:")
            print("  python download_kaist.py --skip_download")
            
            response = input("\nDo you have the dataset downloaded? (y/n): ")
            if response.lower() != 'y':
                print("\n💡 Quick start options:")
                print("  1. Try Hugging Face: python download_kaist.py --use_huggingface")
                print("  2. Download FLIR ADAS (easier): python download_flir_alternative.py")
                print("  3. Manual download from KAIST website")
                return
    
    # Organize existing dataset
    if os.path.exists(args.download_dir):
        print(f"\n📁 Organizing dataset from {args.download_dir}...")
        rgb_count, thermal_count = setup_kaist_structure(
            args.download_dir,
            args.output_rgb,
            args.output_thermal
        )
        
        print(f"\n✅ Dataset ready!")
        print(f"   RGB images: {args.output_rgb} ({rgb_count} images)")
        print(f"   Thermal images: {args.output_thermal} ({thermal_count} images)")
        print(f"\nNext step: Run preprocessing")
        print(f"   python utils/align_kaist.py --raw_rgb {args.output_rgb} --raw_thermal {args.output_thermal}")
    else:
        print(f"\n❌ Dataset directory not found: {args.download_dir}")
        print("Please download the dataset first or specify the correct path.")


if __name__ == "__main__":
    main()

