import os
import cv2
import numpy as np
import requests
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, Form, status
from fastapi.responses import JSONResponse
from bson import ObjectId
from db import users_collection  # Your MongoDB setup
from ultralytics import YOLO  # YOLOv8
from jose import jwt
from datetime import datetime, timedelta
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from utils import load_file, create_vectorstore, get_qa_chain
import shutil
import tempfile
from fastapi.middleware.cors import CORSMiddleware
import uuid


SECRET_KEY = os.getenv("JWT_SECRET")  # Use a strong secret key in production
print("SECRET_KEY: ",SECRET_KEY)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 3
REFRESH_TOKEN_EXPIRE_DAYS = 30


sessions = {}
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:5173"] for stricter config
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Load YOLOv8n model (face-detection fine-tuned is better, but default works)
model = YOLO("yolov8n.pt")
mtcnn = MTCNN(image_size=160, margin=0, keep_all=False)
resnet = InceptionResnetV1(pretrained='vggface2').eval()


def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def create_refresh_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_facenet_embedding(image: np.ndarray):
    # Convert BGR to RGB
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Detect and align
    aligned_face = mtcnn(rgb_img)
    if aligned_face is None:
        return None
    with torch.no_grad():
        embedding = resnet(aligned_face.unsqueeze(0)).squeeze().cpu().numpy()
    return embedding

def detect_faces_yolo(image: np.ndarray):
    results = model.predict(image)
    detections = results[0].boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
    return detections

def crop_face(image: np.ndarray, box, margin=0.2):
    h, w, _ = image.shape
    x1, y1, x2, y2 = box
    # Add margin and clip
    dx = int((x2 - x1) * margin)
    dy = int((y2 - y1) * margin)
    x1 = max(0, int(x1 - dx))
    y1 = max(0, int(y1 - dy))
    x2 = min(w, int(x2 + dx))
    y2 = min(h, int(y2 + dy))
    return image[y1:y2, x1:x2]

@app.get("/")
async def health_check():
    return JSONResponse(content={"status": "ok"})

@app.post("/python/login", status_code=status.HTTP_200_OK)
async def log_in_face_recognition(
    user_id: str = Form(...), 
    file: UploadFile = File(...),
):
    contents = await file.read()
    np_image = np.frombuffer(contents, np.uint8)
    bgr_image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    if bgr_image is None:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"statusCode": 400, "message": "Invalid image file."}
        )

    # YOLO face detection
    detections = detect_faces_yolo(bgr_image)
    if len(detections) == 0:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"statusCode": 400, "message": "No face detected with YOLO."}
        )

    # For simplicity, use the first face
    face_crop = crop_face(bgr_image, detections[0])
    rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

    unknown_embedding = get_facenet_embedding(face_crop)
    if unknown_embedding is None:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"statusCode": 400, "message": "No face found in cropped image."}
        )


    # Fetch user by ID from MongoDB
    try:
        user_record = await users_collection.find_one({"_id": ObjectId(user_id)})
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"statusCode": 400, "message": f"Invalid user ID. {e}"}
        )

    if not user_record:
        return JSONResponse(
            status_code=404,
            content={"statusCode": 404, "message": "User not found."}
        )

    user_image_url = user_record.get("faceImage")
    if not user_image_url:
        return JSONResponse(
            status_code=404,
            content={"statusCode": 404, "message": "No face image found for this user."}
        )

    try:
        response = requests.get(user_image_url)
        response.raise_for_status()
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"statusCode": 500, "message": f"Failed to download face image. {e}"}
        )

    known_image_bgr = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    known_detections = detect_faces_yolo(known_image_bgr)
    if len(known_detections) == 0:
        return JSONResponse(
            status_code=400,
            content={"statusCode": 400, "message": "No face detected in stored image."}
        )

    known_crop = crop_face(known_image_bgr, known_detections[0])
    known_rgb = cv2.cvtColor(known_crop, cv2.COLOR_BGR2RGB)
    known_embedding = get_facenet_embedding(known_crop)
    if known_embedding is None:
        return JSONResponse(
            status_code=400,
            content={"statusCode": 400, "message": "Stored face image is invalid."}
        )

    distance = np.linalg.norm(unknown_embedding - known_embedding)
    match = distance < 0.8  # You can tune this threshold


    if match:
        access_token = create_access_token({"_id": user_id})
        refresh_token = create_refresh_token({"_id": user_id})
        return JSONResponse(
            status_code=200,
            content={"statusCode": 200, "message": "Login successful!", "_id": user_id,"accessToken": access_token,
            "refreshToken": refresh_token}
        )

    return JSONResponse(
        status_code=401,
        content={"statusCode": 401, "message": "Face does not match."}
    )

@app.post("/python/ask")
async def ask_question(file: UploadFile = File(...), query: str = Form(...)):
    try:
        suffix = os.path.splitext(file.filename)[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            file_path = tmp.name

        docs = load_file(file_path)
        vectorstore = create_vectorstore(docs)
        qa_chain = get_qa_chain(vectorstore)

        answer = qa_chain.run(query)
        return JSONResponse(content={"answer": answer})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/python/start_chat")
async def start_chat(file: UploadFile = File(...), query: str = Form(...)):
    try:
        suffix = os.path.splitext(file.filename)[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            file_path = tmp.name

        docs = load_file(file_path)
        vectorstore = create_vectorstore(docs)
        qa_chain = get_qa_chain(vectorstore)

        answer = qa_chain.run(query)

        # Generate session ID and store qa_chain or vectorstore
        session_id = str(uuid.uuid4())
        sessions[session_id] = qa_chain  # or store vectorstore if you want

        return JSONResponse(content={"session_id": session_id, "answer": answer})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/python/continue_chat")
async def continue_chat(session_id: str = Form(...), query: str = Form(...)):
    try:
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        qa_chain = sessions[session_id]
        answer = qa_chain.run(query)

        return JSONResponse(content={"answer": answer})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
# Ensure Vercel sees this app
handler = app