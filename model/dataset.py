"""
PyTorch Dataset for KAIST RGB-Thermal pairs
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms


class ThermalDataset(Dataset):
    """
    Dataset for RGB → Thermal image pairs
    Expects aligned pairs in processed/ directory
    """
    
    def __init__(self, rgb_dir, thermal_dir, transform=None, mode='train'):
        self.rgb_dir = rgb_dir
        self.thermal_dir = thermal_dir
        self.transform = transform
        self.mode = mode
        
        # Get list of RGB images
        self.rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.thermal_files = sorted([f for f in os.listdir(thermal_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        # Match RGB and thermal files
        self.pairs = []
        for rgb_file in self.rgb_files:
            thermal_file = rgb_file  # Assuming same filename
            if thermal_file in self.thermal_files:
                self.pairs.append((rgb_file, thermal_file))
        
        print(f"Found {len(self.pairs)} RGB-Thermal pairs")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        rgb_file, thermal_file = self.pairs[idx]
        
        # Load RGB image
        rgb_path = os.path.join(self.rgb_dir, rgb_file)
        rgb_img = Image.open(rgb_path).convert('RGB')
        
        # Load thermal image
        thermal_path = os.path.join(self.thermal_dir, thermal_file)
        thermal_img = Image.open(thermal_path).convert('L')  # Grayscale
        
        # Convert to numpy arrays
        rgb_array = np.array(rgb_img).astype(np.float32) / 255.0
        thermal_array = np.array(thermal_img).astype(np.float32) / 255.0
        
        # Convert to tensors
        rgb_tensor = torch.from_numpy(rgb_array).permute(2, 0, 1)  # HWC -> CHW
        thermal_tensor = torch.from_numpy(thermal_array).unsqueeze(0)  # Add channel dimension
        
        # Apply transforms if provided
        if self.transform:
            # Stack for joint augmentation
            stacked = torch.cat([rgb_tensor, thermal_tensor], dim=0)
            stacked = self.transform(stacked)
            rgb_tensor = stacked[:3]
            thermal_tensor = stacked[3:4]
        
        return rgb_tensor, thermal_tensor


def get_transforms(mode='train', image_size=256):
    """Get data augmentation transforms"""
    
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            # Brightness/contrast can be added here
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
        ])


if __name__ == "__main__":
    # Test dataset
    rgb_dir = "../data/processed/rgb"
    thermal_dir = "../data/processed/thermal"
    
    if os.path.exists(rgb_dir) and os.path.exists(thermal_dir):
        dataset = ThermalDataset(rgb_dir, thermal_dir, transform=None)
        if len(dataset) > 0:
            rgb, thermal = dataset[0]
            print(f"RGB shape: {rgb.shape}")
            print(f"Thermal shape: {thermal.shape}")
        else:
            print("No pairs found. Run preprocessing first.")
    else:
        print("Data directories not found. Run preprocessing first.")

