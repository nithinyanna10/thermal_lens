# 🎥 Live Webcam Thermal Inference Guide

## Quick Start

### Option 1: Using the helper script
```bash
./run_webcam.sh
```

### Option 2: Direct command
```bash
source venv/bin/activate
python demos/webcam_infer.py \
    --model model/thermal_unet.onnx \
    --camera 0 \
    --colormap inferno \
    --blend 0.6
```

## Controls

- **'q'** - Quit the application
- **'s'** - Save current frame as `thermal_frame_XXX.jpg`

## Command Options

```bash
python demos/webcam_infer.py \
    --model model/thermal_unet.onnx \    # ONNX model path
    --camera 0 \                         # Camera ID (0, 1, 2, etc.)
    --size 256 \                         # Input image size
    --colormap inferno \                 # Colormap: inferno, hot, grayscale
    --blend 0.6 \                        # Blending alpha (0-1)
    --no-fps                             # Hide FPS display
```

## Troubleshooting

### Camera not found?
```bash
# Test which camera works
python test_camera.py 0
python test_camera.py 1
python test_camera.py 2
```

### Low FPS?
- The model runs on CPU by default
- For better performance, ensure you're using the ONNX model (not PyTorch)
- Reduce `--size` to 128 for faster inference (lower quality)

### Window not showing?
- Make sure you're running in a terminal (not background)
- Check if OpenCV can access your display
- On Mac, you may need to grant camera permissions

## Expected Performance

- **FPS**: 15-30 on Mac with CPU
- **Latency**: ~30-60ms per frame
- **Resolution**: Input resized to 256×256, output scaled to webcam resolution

## Tips

1. **Lighting**: Better lighting = better thermal predictions
2. **Distance**: Keep objects 1-3 feet away for best results
3. **Movement**: Model works best with relatively static scenes
4. **Hot objects**: Try holding a warm cup, laptop, or your hand near the camera

## What to Expect

The model will predict thermal patterns based on RGB input:
- **Warm objects** (hands, laptops) appear brighter/hotter
- **Cool objects** (metal, shadows) appear darker/cooler
- **Gradients** show temperature transitions

Remember: This is a **prediction** from RGB, not actual thermal imaging!

---

**Enjoy your thermal vision! 🔥**

