"""
Debug script to test webcam inference and see what the model is outputting
"""

import cv2
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt

# Load model
onnx_path = "model/thermal_unet.onnx"
session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("Press SPACE to capture and analyze a frame, 'q' to quit")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Show original
    cv2.imshow('Original', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):  # Spacebar to capture
        # Preprocess
        frame_resized = cv2.resize(frame, (256, 256))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_normalized = frame_rgb.astype(np.float32) / 255.0
        frame_chw = np.transpose(frame_normalized, (2, 0, 1))
        frame_batch = np.expand_dims(frame_chw, axis=0)
        
        # Inference
        output = session.run(None, {input_name: frame_batch})
        thermal_pred = output[0][0, 0]
        
        print(f"\n=== Frame {frame_count} Analysis ===")
        print(f"Thermal pred shape: {thermal_pred.shape}")
        print(f"Thermal pred min: {thermal_pred.min():.4f}")
        print(f"Thermal pred max: {thermal_pred.max():.4f}")
        print(f"Thermal pred mean: {thermal_pred.mean():.4f}")
        print(f"Thermal pred std: {thermal_pred.std():.4f}")
        
        # Show thermal prediction
        thermal_normalized = (thermal_pred - thermal_pred.min()) / (thermal_pred.max() - thermal_pred.min() + 1e-8)
        thermal_display = (thermal_normalized * 255).astype(np.uint8)
        cv2.imshow('Thermal Prediction', thermal_display)
        
        # Save for inspection
        cv2.imwrite(f'debug_thermal_{frame_count}.png', thermal_display)
        print(f"Saved debug_thermal_{frame_count}.png")
        
        frame_count += 1

cap.release()
cv2.destroyAllWindows()

