"""
Quick test to check if camera is available
"""

import cv2
import sys

def test_camera(camera_id=0):
    """Test if camera is accessible"""
    print(f"Testing camera {camera_id}...")
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"❌ Camera {camera_id} is not available")
        print("Try a different camera ID (--camera 1, 2, etc.)")
        return False
    
    # Try to read a frame
    ret, frame = cap.read()
    if not ret:
        print(f"❌ Camera {camera_id} opened but cannot read frames")
        cap.release()
        return False
    
    print(f"✅ Camera {camera_id} is working!")
    print(f"   Resolution: {frame.shape[1]}x{frame.shape[0]}")
    cap.release()
    return True

if __name__ == "__main__":
    camera_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if test_camera(camera_id):
        print("\n🚀 Camera is ready! You can now run:")
        print("   python demos/webcam_infer.py --model model/thermal_unet.onnx --camera", camera_id)
    else:
        print("\n💡 Try:")
        print("   python test_camera.py 0")
        print("   python test_camera.py 1")
        print("   python test_camera.py 2")

