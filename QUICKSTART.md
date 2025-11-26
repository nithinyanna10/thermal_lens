# 🚀 Thermal Lens v0.1 - Quick Start Guide

## Prerequisites

- Python 3.8+
- pip
- (Optional) CUDA for GPU training
- (Optional) MPS for Apple Silicon Macs

## Step 1: Setup Environment

```bash
# Make setup script executable
chmod +x setup.sh

# Run setup
./setup.sh
```

Or manually:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Test Model Architecture

Before training, verify the model works:

```bash
python test_model.py
```

This will:
- ✅ Test model forward pass
- ✅ Verify output shapes
- ✅ Check device compatibility (CPU/CUDA/MPS)

## Step 3: Prepare Dataset

### Option A: Using KAIST Dataset

1. Download KAIST multispectral dataset
2. Extract to `data/raw/`:
   ```
   data/raw/
   ├── rgb/        # RGB images
   └── thermal/    # Thermal images
   ```

3. Align and preprocess:
```bash
python utils/align_kaist.py \
    --raw_rgb data/raw/rgb \
    --raw_thermal data/raw/thermal \
    --output_rgb data/processed/rgb \
    --output_thermal data/processed/thermal \
    --size 256
```

### Option B: Using Your Own Dataset

Place RGB-Thermal pairs in:
- `data/processed/rgb/` - RGB images
- `data/processed/thermal/` - Thermal images (grayscale)

Images should be:
- Same filename (e.g., `image001.png` in both folders)
- Aligned (same scene, same perspective)
- Any size (will be resized to 256×256 during training)

## Step 4: Train Model

```bash
cd model
python train.py \
    --rgb_dir ../data/processed/rgb \
    --thermal_dir ../data/processed/thermal \
    --batch_size 16 \
    --epochs 20 \
    --lr 3e-4 \
    --device cuda  # or 'mps' for Mac, 'cpu' for CPU
```

Training will:
- Split data 80/20 (train/val)
- Save best model to `checkpoints/best_model.pth`
- Save checkpoints every 5 epochs
- Display training/validation loss

**Expected training time:**
- CPU: ~2-4 hours for 20 epochs
- GPU (CUDA): ~30-60 minutes
- MPS (Apple Silicon): ~1-2 hours

## Step 5: Export to ONNX

After training, export for fast inference:

```bash
python model/export_onnx.py \
    --checkpoint checkpoints/best_model.pth \
    --output model/thermal_unet.onnx
```

## Step 6: Test Inference

### Single Image

```bash
python model/infer.py \
    --model checkpoints/best_model.pth \
    --image path/to/image.jpg \
    --output result.png \
    --colormap inferno
```

### Real-time Webcam

```bash
python demos/webcam_infer.py \
    --model model/thermal_unet.onnx \
    --camera 0 \
    --colormap inferno \
    --blend 0.6
```

Controls:
- `q` - Quit
- `s` - Save current frame

## Step 7: Web App (Optional)

### Start Backend

```bash
cd app/backend
python main.py
```

Backend runs on `http://localhost:8000`

### Test API

```bash
curl -X POST "http://localhost:8000/predict" \
    -F "file=@path/to/image.jpg" \
    --output thermal_result.png
```

## Troubleshooting

### Import Errors
```bash
# Make sure you're in the project root
cd /path/to/thermal-lens

# Activate virtual environment
source venv/bin/activate
```

### CUDA Out of Memory
- Reduce batch size: `--batch_size 8`
- Use smaller image size: `--image_size 128`

### MPS Issues (Mac)
- If MPS fails, use CPU: `--device cpu`
- Some operations may not support MPS yet

### Dataset Issues
- Ensure RGB and thermal images have matching filenames
- Check image formats (PNG, JPG supported)
- Verify images are readable

## Next Steps

After v0.1 works:
- Experiment with different architectures
- Try different loss functions
- Add data augmentation
- Optimize for edge devices
- Build mobile app

## Getting Help

- Check `README.md` for full documentation
- Review error messages carefully
- Test with `test_model.py` first
- Verify dataset format

---

**Happy training! 🔥**

