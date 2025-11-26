"""
Inference script for RGB → Thermal prediction
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
import argparse
import os

from unet import TinyUNet


def load_model(checkpoint_path, device='cpu'):
    """Load trained model from checkpoint"""
    model = TinyUNet(n_channels=3, n_classes=1)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path, image_size=256):
    """Preprocess image for inference"""
    if isinstance(image_path, str):
        img = Image.open(image_path).convert('RGB')
    else:
        img = image_path
    
    # Resize
    img = img.resize((image_size, image_size))
    
    # Convert to tensor
    img_array = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)  # Add batch dimension
    
    return img_tensor, img


def predict_thermal(model, rgb_tensor, device='cpu'):
    """Predict thermal image from RGB"""
    rgb_tensor = rgb_tensor.to(device)
    
    with torch.no_grad():
        thermal_pred = model(rgb_tensor)
        thermal_pred = thermal_pred.squeeze().cpu().numpy()
    
    return thermal_pred


def apply_inferno_colormap(thermal_map):
    """Apply inferno colormap to thermal prediction"""
    import matplotlib.cm as cm
    try:
        # Try new API first (matplotlib 3.7+)
        inferno = cm.colormaps['inferno']
    except (AttributeError, KeyError):
        # Fallback to old API
        inferno = cm.get_cmap('inferno')
    thermal_colored = inferno(thermal_map)[:, :, :3]  # Remove alpha channel
    return (thermal_colored * 255).astype(np.uint8)


def blend_images(rgb_img, thermal_colored, alpha=0.6):
    """Blend RGB and thermal images"""
    rgb_array = np.array(rgb_img)
    if rgb_array.shape[:2] != thermal_colored.shape[:2]:
        thermal_colored = cv2.resize(thermal_colored, (rgb_array.shape[1], rgb_array.shape[0]))
    
    blended = cv2.addWeighted(rgb_array, 1-alpha, thermal_colored, alpha, 0)
    return blended


def infer_image(model_path, image_path, output_path=None, device='cpu', colormap='inferno', blend=True):
    """Run inference on a single image"""
    # Load model
    model = load_model(model_path, device)
    
    # Preprocess
    rgb_tensor, rgb_img = preprocess_image(image_path)
    
    # Predict
    thermal_pred = predict_thermal(model, rgb_tensor, device)
    
    # Apply colormap
    if colormap == 'inferno':
        thermal_colored = apply_inferno_colormap(thermal_pred)
    else:
        thermal_colored = (thermal_pred * 255).astype(np.uint8)
        thermal_colored = cv2.applyColorMap(thermal_colored, cv2.COLORMAP_HOT)
    
    # Blend if requested
    if blend:
        result = blend_images(rgb_img, thermal_colored)
    else:
        result = thermal_colored
    
    # Save or return
    if output_path:
        result_img = Image.fromarray(result)
        result_img.save(output_path)
        print(f"Saved result to {output_path}")
    
    return result, thermal_pred


def main():
    parser = argparse.ArgumentParser(description='Infer thermal from RGB image')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True, help='Path to input RGB image')
    parser.add_argument('--output', type=str, default=None, help='Path to save output')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu', help='Device')
    parser.add_argument('--colormap', type=str, default='inferno', choices=['inferno', 'hot', 'grayscale'], help='Colormap')
    parser.add_argument('--blend', action='store_true', default=True, help='Blend with original RGB')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.image))[0]
        args.output = f"{base_name}_thermal.png"
    
    result, thermal = infer_image(
        args.model,
        args.image,
        args.output,
        device=str(device),
        colormap=args.colormap,
        blend=args.blend
    )
    
    print("Inference complete!")


if __name__ == "__main__":
    main()

