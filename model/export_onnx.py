"""
Export UNet model to ONNX format for fast inference
"""

import torch
import torch.onnx
import argparse
import os

from unet import TinyUNet


def export_to_onnx(checkpoint_path, output_path, image_size=256, device='cpu'):
    """Export PyTorch model to ONNX"""
    
    # Load model
    model = TinyUNet(n_channels=3, n_classes=1)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, image_size, image_size).to(device)
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=18,  # Updated to 18 for better compatibility
        do_constant_folding=True,
        input_names=['rgb_input'],
        output_names=['thermal_output'],
        dynamic_axes={
            'rgb_input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'thermal_output': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    )
    
    print(f"Model exported to {output_path}")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {model(dummy_input).shape}")


def main():
    parser = argparse.ArgumentParser(description='Export UNet to ONNX')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='../model/thermal_unet.onnx', help='Output ONNX path')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu', help='Device')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    export_to_onnx(args.checkpoint, args.output, args.image_size, str(device))
    
    print("Export complete!")


if __name__ == "__main__":
    main()

