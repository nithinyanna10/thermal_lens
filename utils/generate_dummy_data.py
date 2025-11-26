"""
Generate dummy RGB-Thermal pairs for testing
Useful when you don't have KAIST dataset yet
"""

import numpy as np
import cv2
import os
from tqdm import tqdm


def generate_dummy_thermal_from_rgb(rgb_image):
    """
    Generate a fake thermal image from RGB
    This simulates thermal patterns for testing
    """
    # Convert to grayscale
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    
    # Simulate thermal patterns:
    # - Bright areas (skin, warm objects) = hot
    # - Dark areas (shadows, cool objects) = cold
    # - Add some smooth gradients
    
    # Base thermal from brightness (inverted - bright = hot)
    thermal = 255 - gray
    
    # Add smooth gradients
    h, w = thermal.shape
    y, x = np.ogrid[:h, :w]
    center_x, center_y = w // 2, h // 2
    
    # Radial gradient (center is warmer)
    radial = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    radial = radial / radial.max()
    thermal = thermal * 0.7 + (1 - radial) * 255 * 0.3
    
    # Add some noise
    noise = np.random.normal(0, 10, thermal.shape)
    thermal = np.clip(thermal + noise, 0, 255).astype(np.uint8)
    
    # Smooth with bilateral filter
    thermal = cv2.bilateralFilter(thermal, 9, 75, 75)
    
    return thermal


def generate_dummy_dataset(output_rgb_dir, output_thermal_dir, num_images=100, image_size=256):
    """
    Generate dummy RGB-Thermal pairs for testing
    
    Args:
        output_rgb_dir: Output directory for RGB images
        output_thermal_dir: Output directory for thermal images
        num_images: Number of image pairs to generate
        image_size: Size of generated images
    """
    os.makedirs(output_rgb_dir, exist_ok=True)
    os.makedirs(output_thermal_dir, exist_ok=True)
    
    print(f"Generating {num_images} dummy RGB-Thermal pairs...")
    print(f"Image size: {image_size}x{image_size}")
    
    for i in tqdm(range(num_images), desc="Generating images"):
        # Generate random RGB image
        rgb = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
        
        # Add some structure (gradients, shapes)
        y, x = np.ogrid[:image_size, :image_size]
        
        # Add circular gradient
        center = image_size // 2
        mask = (x - center)**2 + (y - center)**2
        mask = mask / mask.max()
        rgb = (rgb * (1 - mask * 0.3)).astype(np.uint8)
        
        # Add some colored rectangles
        for _ in range(3):
            x1 = np.random.randint(0, image_size // 2)
            y1 = np.random.randint(0, image_size // 2)
            x2 = x1 + np.random.randint(image_size // 4, image_size // 2)
            y2 = y1 + np.random.randint(image_size // 4, image_size // 2)
            color = np.random.randint(0, 255, 3)
            cv2.rectangle(rgb, (x1, y1), (x2, y2), color.tolist(), -1)
        
        # Generate corresponding thermal
        thermal = generate_dummy_thermal_from_rgb(rgb)
        
        # Save images
        rgb_filename = f"dummy_{i:04d}.png"
        thermal_filename = f"dummy_{i:04d}.png"
        
        cv2.imwrite(os.path.join(output_rgb_dir, rgb_filename), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_thermal_dir, thermal_filename), thermal)
    
    print(f"\n✅ Generated {num_images} image pairs")
    print(f"RGB images: {output_rgb_dir}")
    print(f"Thermal images: {output_thermal_dir}")
    print("\nYou can now test training with:")
    print(f"python model/train.py --rgb_dir {output_rgb_dir} --thermal_dir {output_thermal_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate dummy RGB-Thermal pairs for testing')
    parser.add_argument('--output_rgb', type=str, default='data/processed/rgb', help='Output RGB directory')
    parser.add_argument('--output_thermal', type=str, default='data/processed/thermal', help='Output thermal directory')
    parser.add_argument('--num_images', type=int, default=100, help='Number of image pairs to generate')
    parser.add_argument('--size', type=int, default=256, help='Image size')
    
    args = parser.parse_args()
    
    generate_dummy_dataset(
        args.output_rgb,
        args.output_thermal,
        num_images=args.num_images,
        image_size=args.size
    )

