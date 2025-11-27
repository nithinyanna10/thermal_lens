# 📥 KAIST Multispectral Dataset Download Guide

## Overview

The KAIST Multispectral Dataset contains:
- **RGB images** (visible spectrum)
- **Thermal images** (LWIR - Long Wave Infrared)
- **Synchronized pairs** for training
- **~50GB** total size
- **~95,000 image pairs** (we'll use 6-8k for training)

## Download Methods

### Method 1: Official KAIST Website (Recommended)

1. **Visit the official website:**
   - URL: http://multispectral.kaist.ac.kr/pedestrian/
   - Or search: "KAIST multispectral dataset"

2. **Register for access:**
   - Fill out the registration form
   - Wait for approval (usually 1-2 days)
   - You'll receive download links via email

3. **Download the dataset:**
   - Download all sets (set00-set24)
   - Or download specific sets for faster download
   - Extract to `data/raw/`

### Method 2: Academic Torrents (If Available)

Some academic datasets are available via:
- Academic Torrents
- University repositories
- Research paper supplementary materials

### Method 3: Alternative Datasets

If KAIST is not available, consider:
- **FLIR ADAS Dataset** (smaller, easier to get)
- **CVC-14 Dataset** (pedestrian detection)
- **Custom dataset** (collect your own RGB-Thermal pairs)

## Quick Setup Script

After downloading, use our script:

```bash
# Organize the downloaded dataset
python download_kaist.py --skip_download

# Or specify custom paths
python download_kaist.py \
    --download_dir /path/to/kaist \
    --output_rgb data/processed/rgb \
    --output_thermal data/processed/thermal \
    --skip_download
```

## Dataset Structure

KAIST dataset typically has this structure:

```
kaist_dataset/
├── images/
│   ├── set00/
│   │   ├── V000/          # Visible (RGB)
│   │   │   ├── I00000.jpg
│   │   │   ├── I00001.jpg
│   │   │   └── ...
│   │   └── I00000/        # Infrared (Thermal)
│   │       ├── I00000.jpg
│   │       ├── I00001.jpg
│   │       └── ...
│   ├── set01/
│   └── ...
```

## Processing Steps

1. **Download** → Extract to `data/raw/`
2. **Organize** → Run `download_kaist.py --skip_download`
3. **Align** → Run `utils/align_kaist.py` to align RGB-Thermal pairs
4. **Train** → Use aligned pairs for training

## Expected Results

After processing:
- **6,000-8,000 aligned pairs** for training
- **1,000-2,000 pairs** for validation
- **256×256 resolution** (resized from original)
- **Perfect alignment** between RGB and thermal

## Troubleshooting

### Dataset not found?
- Check the path in `--download_dir`
- Ensure files are extracted
- Check file permissions

### Not enough pairs?
- Download more sets from KAIST
- Use all available sets (set00-set24)
- Combine with other thermal datasets

### Alignment issues?
- Run `utils/align_kaist.py` with calibration
- Check if RGB and thermal filenames match
- Verify image formats

## File Size Estimates

- **Full dataset**: ~50GB
- **6-8k pairs**: ~2-3GB (after processing)
- **Processed 256×256**: ~500MB

## Next Steps After Download

1. ✅ Download dataset
2. ✅ Organize with `download_kaist.py`
3. ✅ Align with `utils/align_kaist.py`
4. ✅ Train with `model/train.py`
5. ✅ Test with webcam!

---

**Note**: KAIST dataset requires registration. If you need immediate access, consider using FLIR ADAS or creating a smaller custom dataset for testing.

