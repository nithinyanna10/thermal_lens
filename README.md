# 🔥 Thermal Lens v0.1

**Real-time thermal vision hallucination from RGB using deep learning**

Train a tiny UNet to predict thermal heatmaps from regular RGB webcam input. This is the v0.1 MVP - proving the thermal prediction pipeline works end-to-end.

## 🎯 v0.1 Goals

- ✅ Train a tiny UNet on RGB→Thermal pairs
- ✅ Use KAIST multispectral dataset (RGB + LWIR thermal)
- ✅ Produce thermal heatmap (grayscale or inferno colormap)
- ✅ Real-time inference at 15–30 FPS on Mac
- ✅ Export ONNX
- ✅ Build minimal UI overlay

## 📁 Project Structure

```
thermal-lens/
├── data/
│   ├── raw/               # KAIST raw dataset
│   ├── processed/         # aligned & resized pairs
├── model/
│   ├── unet.py            # UNet architecture
│   ├── dataset.py         # PyTorch dataset loader
│   ├── train.py           # Training loop
│   ├── infer.py           # Inference + ONNX pipeline
│   ├── export_onnx.py     # ONNX export
├── app/
│   ├── backend/           # FastAPI
│   │   ├── main.py
│   ├── frontend/          # React app
│   │   ├── src/
│   │   ├── index.css
├── utils/
│   ├── align_kaist.py     # Align RGB & Thermal frames
│   ├── preprocess.py
├── demos/
│   ├── webcam_infer.py    # Live webcam thermal mode
└── requirements.txt
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

Download KAIST multispectral dataset and place in `data/raw/`:

```bash
python utils/align_kaist.py \
    --raw_rgb data/raw/rgb \
    --raw_thermal data/raw/thermal \
    --output_rgb data/processed/rgb \
    --output_thermal data/processed/thermal \
    --size 256
```

### 3. Train Model

```bash
cd model
python train.py \
    --rgb_dir ../data/processed/rgb \
    --thermal_dir ../data/processed/thermal \
    --batch_size 16 \
    --epochs 20 \
    --lr 3e-4
```

### 4. Export to ONNX

```bash
python export_onnx.py \
    --checkpoint ../checkpoints/best_model.pth \
    --output ../model/thermal_unet.onnx
```

### 5. Run Webcam Demo

```bash
python demos/webcam_infer.py \
    --model model/thermal_unet.onnx \
    --camera 0 \
    --colormap inferno
```

## 🧠 Model Architecture

**Tiny UNet:**
- Input: RGB image (3 channels, 256×256)
- Output: Thermal heatmap (1 channel, 256×256)
- 4 down blocks + 4 up blocks with skip connections
- ~10M parameters
- Optimized for 15-30 FPS inference

**Loss Function:**
- 80% L1 Loss (image reconstruction)
- 20% SSIM Loss (edge preservation)

## 📊 Training

- **Epochs:** 20
- **Batch Size:** 16
- **Learning Rate:** 3e-4
- **Optimizer:** AdamW
- **Augmentations:** Flip, rotate, brightness shifts

After training, the model learns:
- Human skin = hotter
- Bright metal = cooler
- Stove/finger/laptop = visible
- Hot vs cold mug = distinguishable
- AC vents = cold

## 🎥 Real-time Inference

The webcam demo (`demos/webcam_infer.py`) provides:
- Live thermal hallucination from webcam
- Inferno colormap overlay
- Real-time FPS display
- Save frames with 's' key

## 🌐 Web App

Start the FastAPI backend:

```bash
cd app/backend
python main.py
```

Then open the frontend (React app) to upload images and get thermal predictions.

## 📝 Notes

- v0.1 focuses on proving the pipeline works
- Model is intentionally small for real-time performance
- ONNX export enables fast inference across platforms
- KAIST dataset provides high-quality RGB-Thermal pairs

## 🔮 Future Enhancements

- Multi-scale UNet for better detail
- Temporal consistency for video
- Edge device optimization (TensorRT, CoreML)
- Mobile app integration
- Advanced colormaps and visualization

## 📄 License

MIT

---

**"I trained a neural network to hallucinate thermal vision from RGB."** 🔥

