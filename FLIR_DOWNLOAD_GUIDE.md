# 📥 FLIR ADAS Dataset - What You Need to Download

## Quick Answer

**You need to download ALL 12 parts (00-11)** to get the complete dataset.

However, for **training the thermal prediction model**, you only need:
- ✅ `images_rgb_train/` - RGB training images (input)
- ✅ `images_thermal_train/` - Thermal training images (ground truth)
- ✅ `images_rgb_val/` - RGB validation images
- ✅ `images_thermal_val/` - Thermal validation images

**You DON'T need:**
- ❌ `video_rgb_test/` - Video data (optional, for later)
- ❌ `video_thermal_test/` - Video data (optional, for later)

## Download Steps

### Option 1: Download All Parts (Recommended)

The dataset is split into 12 parts (00-11) totaling ~2-3GB.

**On Mac/Linux:**
```bash
# Create download directory
mkdir -p ~/Downloads/FLIR_ADAS
cd ~/Downloads/FLIR_ADAS

# Download all 12 parts
for i in {00..11}; do 
    curl --output FLIR_ADAS_v2.zip.${i} \
    https://adas-dataset-v2.flirconservator.com/dataset/parts/FLIR_ADAS_v2.zip.${i}
done

# Combine all parts into one zip file
cat FLIR_ADAS_v2.zip.* > FLIR_ADAS_v2.zip

# Extract
unzip -q FLIR_ADAS_v2.zip

# Move to our project
mv FLIR_ADAS_v2/* /Users/nithinyanna/Downloads/thermal-lens/data/raw/flir/
```

**On Windows:**
1. Download all 12 parts manually from the website
2. Use 7-Zip to extract (it will auto-combine parts)
3. Extract to: `thermal-lens/data/raw/flir/`

### Option 2: Manual Download

1. Go to: https://www.flir.com/oem/adas/adas-dataset-form/
2. Fill out the form
3. Download all 12 parts (FLIR_ADAS_v2.zip.00 through .11)
4. Combine and extract (see instructions above)

## What You'll Get

After extraction, you'll have:
```
FLIR_ADAS_v2/
├── images_rgb_train/        ← Need this (9,233 images)
│   └── data/                ← RGB training images
├── images_thermal_train/     ← Need this (9,711 images)
│   └── data/                ← Thermal training images
├── images_rgb_val/          ← Need this (validation)
│   └── data/
├── images_thermal_val/       ← Need this (validation)
│   └── data/
├── video_rgb_test/          ← Optional (for video inference)
├── video_thermal_test/       ← Optional (for video inference)
└── rgb_to_thermal_vid_map.json
```

## Dataset Size

- **Total size**: ~2-3GB (compressed)
- **After extraction**: ~5-6GB
- **Training images**: 9,233 RGB + 9,711 Thermal pairs
- **Validation images**: ~1,000 pairs (10% split)

## After Download

Once downloaded and extracted:

```bash
cd /Users/nithinyanna/Downloads/thermal-lens

# Process the FLIR dataset
python utils/process_flir.py \
    --input data/raw/flir/FLIR_ADAS_v2 \
    --output_rgb data/processed/rgb \
    --output_thermal data/processed/thermal \
    --max_pairs 8000 \
    --resize 256
```

This will:
1. Match RGB-Thermal pairs
2. Resize to 256×256
3. Organize for training

## Why All 12 Parts?

The dataset is split into 12 parts because:
- Large file size (~2-3GB total)
- More reliable downloads (can resume if one part fails)
- Better for slower connections

**You need all parts** because the training images are distributed across all parts. You can't train with just one part.

## Quick Start Script

I'll create a download script for you:

```bash
# Run this after you download the parts
./download_flir.sh
```

## Summary

✅ **Download**: All 12 parts (00-11)  
✅ **Extract**: Combine and unzip  
✅ **Process**: Run `process_flir.py`  
✅ **Train**: Use processed images  

**Total time**: ~30-60 minutes (depending on internet speed)

---

**Next step**: Download the 12 parts, then we'll process them!

