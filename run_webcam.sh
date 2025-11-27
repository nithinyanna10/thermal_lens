#!/bin/bash
# Quick script to run webcam thermal inference

cd "$(dirname "$0")"
source venv/bin/activate

echo "🔥 Starting Thermal Lens Webcam Inference..."
echo ""
echo "Controls:"
echo "  - Press 'q' to quit"
echo "  - Press 's' to save current frame"
echo ""
echo "Starting in 3 seconds..."
sleep 3

python demos/webcam_infer.py \
    --model model/thermal_unet.onnx \
    --camera 0 \
    --colormap inferno \
    --blend 0.6

