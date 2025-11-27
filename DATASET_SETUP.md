# 📥 Dataset Setup Guide - KAIST Multispectral

## Quick Start Options

### Option 1: FLIR ADAS (Easiest - Recommended for Quick Start)

**Why FLIR ADAS?**
- ✅ No registration/approval needed
- ✅ Smaller (~2GB vs 50GB)
- ✅ Faster download
- ✅ Good quality RGB-Thermal pairs
- ✅ ~14,000 image pairs

**Steps:**
1. Visit: https://www.flir.com/oem/adas/adas-dataset-form/
2. Fill out the form (free, instant access)
3. Download the dataset
4. Extract to `data/raw/flir/`
5. Run: `python utils/process_flir.py --input data/raw/flir`

### Option 2: KAIST Multispectral (Best Quality)

**Why KAIST?**
- ✅ Largest dataset (~95,000 pairs)
- ✅ Best quality
- ✅ Research standard
- ❌ Requires registration (1-2 day wait)
- ❌ Large download (~50GB)

**Steps:**
1. Visit: http://multispectral.kaist.ac.kr/pedestrian/
2. Register and request access
3. Wait for approval email
4. Download dataset
5. Extract to `data/raw/kaist/`
6. Run: `python download_kaist.py --skip_download`

### Option 3: Use Existing Dummy Data (For Testing)

We already have 200 dummy pairs generated. For better results:
```bash
# Generate more dummy data
python utils/generate_dummy_data.py --num_images 1000
```

## Recommended: Start with FLIR ADAS

FLIR ADAS is the best balance of:
- Ease of access
- Dataset size
- Quality
- Speed

**Download FLIR ADAS now:**
1. Go to: https://www.flir.com/oem/adas/adas-dataset-form/
2. Fill form → Download → Extract
3. We'll process it next!

## After Download

Once you have either dataset:

```bash
# For FLIR
python utils/process_flir.py --input data/raw/flir --output data/processed

# For KAIST  
python download_kaist.py --skip_download
python utils/align_kaist.py --raw_rgb data/processed/rgb --raw_thermal data/processed/thermal
```

## What You'll Get

After processing:
- **6,000-8,000 training pairs** (RGB + Thermal)
- **1,000-2,000 validation pairs**
- **256×256 aligned images**
- **Ready for training!**

---

**💡 Recommendation: Start with FLIR ADAS for fastest results!**

