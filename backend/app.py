import torch
import torchvision.transforms as T
from ultralytics import YOLO
from PIL import Image
import numpy as np
import google.generativeai as genai
import pyttsx3
import asyncio
from transformers import BlipProcessor, BlipForConditionalGeneration
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, WebSocket
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from io import BytesIO

# WebSocket server settings
connected_clients = set()

# Google Gemini API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyAGRLFM6FV5p8SJGpmW1MVf4r1D_ozST8U"  # Replace with your Gemini API key
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# 🧠 Load YOLOv8 model
yolo_model = YOLO("yolov8n.pt")

# 🧠 Load MiDaS model
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)

# 🔁 MiDaS Transform
midas_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(256),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# 🎯 Estimate Depth
def estimate_depth(image):
    input_tensor = midas_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = midas(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False
        ).squeeze()
    return prediction.cpu().numpy()

# 🧠 Object Detection with YOLO
def detect_objects(image):
    results = yolo_model(image)
    return results[0]

# 📌 Estimate object positions using depth
def get_object_positions(image, results, depth_map):
    positions = []
    close_objects = []
    for box, cls_id in zip(results.boxes.xyxy, results.boxes.cls):
        x1, y1, x2, y2 = map(int, box.tolist())
        object_name = yolo_model.names[int(cls_id)]
        obj_depth = depth_map[y1:y2, x1:x2]
        mean_depth = np.mean(obj_depth)
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        horizontal = "left" if center_x < image.width * 0.33 else "right" if center_x > image.width * 0.66 else "center"
        vertical = "top" if center_y < image.height * 0.33 else "bottom" if center_y > image.height * 0.66 else "middle"
        distance = "near" if mean_depth < 0.4 else "far"

        description = f"{object_name} is in the {horizontal}-{vertical} and appears {distance}"
        positions.append(description)

        if mean_depth < 1000.0:  # If object is too close
            close_objects.append(object_name)

    return positions, close_objects

# 🔊 Speak description (if needed on backend)
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# 🎯 WebSocket Server to notify frontend
async def send_alerts(alert_message):
    if connected_clients:
        # Send message to all connected clients
        for websocket in connected_clients:
            await websocket.send_text(alert_message)

# 🌐 FastAPI App
app = FastAPI()

# Enable CORS if frontend is separate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket Connection Handler using FastAPI's WebSocket
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle data from frontend if necessary
            print("Received from frontend:", data)
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        connected_clients.remove(websocket)

# Main Route for scene description
@app.post("/describe-scene")
async def describe_scene(file: UploadFile = File(...)):
    # Read and process image
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = image.resize((640, 480))

    # Run heavy operations in background threads
    results = await asyncio.to_thread(detect_objects, image)
    object_names = [yolo_model.names[int(cls)] for cls in results.boxes.cls]

    depth_map = await asyncio.to_thread(estimate_depth, image)
    positions, close_objects = await asyncio.to_thread(get_object_positions, image, results, depth_map)
    # print(depth_map)
    # Compose Gemini prompt
    prompt = (
        "Describe this scene for a visually impaired person using simple and non-fancy language "
        "so they can understand their surroundings and navigate safely. Start your description with the sentence\n"
        "So now currently, and then describe the image.\n"
        f"• Detected objects: {', '.join(object_names)}\n"
        f"• Object positions: {', '.join(positions)}"
    )

    # Generate description using Gemini API
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content([prompt, image], generation_config={"temperature": 0.2})
    scene_description = response.text

    # If close objects detected, send alert through WebSocket
    if close_objects:
        alert_message = f"Alert: The following object(s) are too close: {', '.join(close_objects)}"
        await send_alerts(alert_message)

    # Return description to frontend
    return JSONResponse(content={"description": scene_description})
