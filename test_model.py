"""
Quick test script to verify model architecture works
Run this before training to ensure everything is set up correctly
"""

import torch
import sys
import os

# Add model directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'model'))

from unet import TinyUNet


def test_model():
    """Test UNet model forward pass"""
    print("🧪 Testing TinyUNet Architecture")
    print("=" * 50)
    
    # Create model
    model = TinyUNet(n_channels=3, n_classes=1)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Model created successfully")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    print("\n📊 Testing forward pass...")
    batch_size = 2
    image_size = 256
    
    # Create dummy RGB input
    dummy_input = torch.randn(batch_size, 3, image_size, image_size)
    print(f"  Input shape: {dummy_input.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Verify output is in [0, 1] range (sigmoid)
    assert output.min() >= 0 and output.max() <= 1, "Output should be in [0, 1] range"
    assert output.shape == (batch_size, 1, image_size, image_size), "Output shape mismatch"
    
    print("\n✅ All tests passed!")
    print("\n🎯 Model is ready for training!")
    
    return model


def test_device_compatibility():
    """Test model on available devices"""
    print("\n🔧 Testing device compatibility...")
    
    model = TinyUNet(n_channels=3, n_classes=1)
    dummy_input = torch.randn(1, 3, 256, 256)
    
    # Test CPU
    print("  Testing CPU...")
    model_cpu = model.cpu()
    output_cpu = model_cpu(dummy_input)
    print(f"    ✓ CPU works: {output_cpu.shape}")
    
    # Test CUDA if available
    if torch.cuda.is_available():
        print("  Testing CUDA...")
        model_cuda = model.cuda()
        output_cuda = model_cuda(dummy_input.cuda())
        print(f"    ✓ CUDA works: {output_cuda.shape}")
    
    # Test MPS (Apple Silicon) if available
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("  Testing MPS (Apple Silicon)...")
        model_mps = model.to('mps')
        output_mps = model_mps(dummy_input.to('mps'))
        print(f"    ✓ MPS works: {output_mps.shape}")
    
    print("\n✅ Device compatibility test complete!")


if __name__ == "__main__":
    try:
        model = test_model()
        test_device_compatibility()
        print("\n" + "=" * 50)
        print("🚀 Ready to start training!")
        print("Run: python model/train.py")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

