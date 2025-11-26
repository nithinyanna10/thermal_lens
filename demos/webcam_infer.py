"""
Real-time webcam thermal inference
The WOW moment - live thermal hallucination from RGB webcam
"""

import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.cm as cm
import argparse
import time
import onnxruntime as ort  # For ONNX inference


def load_onnx_model(onnx_path):
    """Load ONNX model for fast inference"""
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    return session


def preprocess_frame(frame, image_size=256):
    """Preprocess webcam frame for inference"""
    # Resize
    frame_resized = cv2.resize(frame, (image_size, image_size))
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    frame_normalized = frame_rgb.astype(np.float32) / 255.0
    
    # Convert to CHW format
    frame_chw = np.transpose(frame_normalized, (2, 0, 1))
    
    # Add batch dimension
    frame_batch = np.expand_dims(frame_chw, axis=0)
    
    return frame_batch, frame


def apply_inferno_colormap(thermal_map):
    """Apply inferno colormap to thermal prediction"""
    try:
        # Try new API first (matplotlib 3.7+)
        inferno = cm.colormaps['inferno']
    except (AttributeError, KeyError):
        # Fallback to old API
        inferno = cm.get_cmap('inferno')
    thermal_colored = inferno(thermal_map)[:, :, :3]  # Remove alpha
    return (thermal_colored * 255).astype(np.uint8)


def blend_thermal_overlay(rgb_frame, thermal_colored, alpha=0.6):
    """Blend thermal overlay with RGB frame"""
    # Resize thermal to match RGB frame size
    h, w = rgb_frame.shape[:2]
    thermal_resized = cv2.resize(thermal_colored, (w, h))
    
    # Convert RGB frame to RGB if needed
    if len(rgb_frame.shape) == 3 and rgb_frame.shape[2] == 3:
        rgb_frame_rgb = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
    else:
        rgb_frame_rgb = rgb_frame
    
    # Blend
    blended = cv2.addWeighted(rgb_frame_rgb, 1-alpha, thermal_resized, alpha, 0)
    
    # Convert back to BGR for display
    blended_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
    
    return blended_bgr


def run_webcam_inference(onnx_path, camera_id=0, image_size=256, colormap='inferno', blend_alpha=0.6, show_fps=True):
    """
    Run real-time webcam thermal inference
    
    Args:
        onnx_path: Path to ONNX model
        camera_id: Webcam ID (usually 0)
        image_size: Input image size
        colormap: Colormap to use ('inferno', 'hot', 'grayscale')
        blend_alpha: Blending alpha (0-1)
        show_fps: Whether to display FPS
    """
    # Load ONNX model
    print(f"Loading ONNX model from {onnx_path}...")
    session = load_onnx_model(onnx_path)
    input_name = session.get_inputs()[0].name
    print("Model loaded!")
    
    # Open webcam
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return
    
    print("Starting webcam inference...")
    print("Press 'q' to quit, 's' to save frame")
    
    fps_history = []
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Preprocess
            start_time = time.time()
            frame_batch, original_frame = preprocess_frame(frame, image_size)
            
            # Inference
            output = session.run(None, {input_name: frame_batch})
            thermal_pred = output[0][0, 0]  # Remove batch and channel dims
            
            # Apply colormap
            if colormap == 'inferno':
                thermal_colored = apply_inferno_colormap(thermal_pred)
            elif colormap == 'hot':
                thermal_gray = (thermal_pred * 255).astype(np.uint8)
                thermal_colored = cv2.applyColorMap(thermal_gray, cv2.COLORMAP_HOT)
            else:
                thermal_gray = (thermal_pred * 255).astype(np.uint8)
                thermal_colored = cv2.cvtColor(thermal_gray, cv2.COLOR_GRAY2BGR)
            
            # Blend with original
            result = blend_thermal_overlay(frame, thermal_colored, alpha=blend_alpha)
            
            # Calculate FPS
            inference_time = time.time() - start_time
            fps = 1.0 / inference_time if inference_time > 0 else 0
            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = np.mean(fps_history)
            
            # Display FPS
            if show_fps:
                cv2.putText(result, f"FPS: {avg_fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show result
            cv2.imshow('Thermal Lens - Live Inference', result)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save frame
                filename = f"thermal_frame_{frame_count}.jpg"
                cv2.imwrite(filename, result)
                print(f"Saved frame to {filename}")
            
            frame_count += 1
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nProcessed {frame_count} frames")
        if fps_history:
            print(f"Average FPS: {np.mean(fps_history):.1f}")


def main():
    parser = argparse.ArgumentParser(description='Real-time webcam thermal inference')
    parser.add_argument('--model', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID')
    parser.add_argument('--size', type=int, default=256, help='Input image size')
    parser.add_argument('--colormap', type=str, default='inferno', choices=['inferno', 'hot', 'grayscale'], help='Colormap')
    parser.add_argument('--blend', type=float, default=0.6, help='Blending alpha (0-1)')
    parser.add_argument('--no-fps', action='store_true', help='Hide FPS display')
    
    args = parser.parse_args()
    
    run_webcam_inference(
        args.model,
        camera_id=args.camera,
        image_size=args.size,
        colormap=args.colormap,
        blend_alpha=args.blend,
        show_fps=not args.no_fps
    )


if __name__ == "__main__":
    main()

