"""
FastAPI backend for Thermal Lens web app
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import io
import matplotlib.cm as cm
import uvicorn

app = FastAPI(title="Thermal Lens API", version="0.1.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model session
model_session = None
input_name = None


def load_model(onnx_path):
    """Load ONNX model"""
    global model_session, input_name
    model_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = model_session.get_inputs()[0].name
    print(f"Model loaded from {onnx_path}")


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    # Update path as needed
    model_path = "../../model/thermal_unet.onnx"
    try:
        load_model(model_path)
    except Exception as e:
        print(f"Warning: Could not load model: {e}")


def preprocess_image(image_bytes, image_size=256):
    """Preprocess uploaded image"""
    # Load image
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    original_size = img.size
    
    # Resize
    img_resized = img.resize((image_size, image_size))
    
    # Convert to numpy
    img_array = np.array(img_resized).astype(np.float32) / 255.0
    
    # Convert to CHW
    img_chw = np.transpose(img_array, (2, 0, 1))
    img_batch = np.expand_dims(img_chw, axis=0)
    
    return img_batch, original_size


def apply_inferno_colormap(thermal_map):
    """Apply inferno colormap"""
    inferno = cm.get_cmap('inferno')
    thermal_colored = inferno(thermal_map)[:, :, :3]
    return (thermal_colored * 255).astype(np.uint8)


@app.post("/predict")
async def predict_thermal(file: UploadFile = File(...)):
    """Predict thermal image from uploaded RGB image"""
    if model_session is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Read image
        image_bytes = await file.read()
        img_batch, original_size = preprocess_image(image_bytes)
        
        # Inference
        output = model_session.run(None, {input_name: img_batch})
        thermal_pred = output[0][0, 0]
        
        # Apply colormap
        thermal_colored = apply_inferno_colormap(thermal_pred)
        
        # Resize to original size
        thermal_resized = cv2.resize(thermal_colored, original_size)
        
        # Convert to bytes
        thermal_pil = Image.fromarray(thermal_resized)
        img_byte_arr = io.BytesIO()
        thermal_pil.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return StreamingResponse(io.BytesIO(img_byte_arr.read()), media_type="image/png")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse({
        "status": "healthy",
        "model_loaded": model_session is not None
    })


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

