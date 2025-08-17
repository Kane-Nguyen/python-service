import os
import cv2
import numpy as np
import requests
from fastapi import FastAPI, UploadFile, File, Form, status
from fastapi.responses import JSONResponse
from bson import ObjectId
from db import users_collection  # Your MongoDB setup
from jose import jwt
from datetime import datetime, timedelta
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from utils import load_file, create_vectorstore, get_qa_chain
import shutil
import tempfile
from fastapi.middleware.cors import CORSMiddleware
import uuid

# JWT Config
SECRET_KEY = os.getenv("JWT_SECRET")  
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 3
REFRESH_TOKEN_EXPIRE_DAYS = 30

# FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Face recognition setup
device = torch.device("cpu")
mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

sessions = {}

# JWT helpers
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

# Face embedding
def get_facenet_embedding(image: np.ndarray):
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    aligned_face = mtcnn(rgb_img)
    if aligned_face is None:
        return None
    with torch.no_grad():
        embedding = resnet(aligned_face.unsqueeze(0).to(device)).squeeze().cpu().numpy()
    return embedding

# Detect faces using MTCNN
def detect_faces_mtcnn(image: np.ndarray):
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(rgb_img)
    if boxes is None:
        return []
    return boxes  # [x1, y1, x2, y2]

# Crop face
def crop_face(image: np.ndarray, box, margin=0.2):
    h, w, _ = image.shape
    x1, y1, x2, y2 = box
    dx = int((x2 - x1) * margin)
    dy = int((y2 - y1) * margin)
    x1 = max(0, int(x1 - dx))
    y1 = max(0, int(y1 - dy))
    x2 = min(w, int(x2 + dx))
    y2 = min(h, int(y2 + dy))
    return image[y1:y2, x1:x2]

# Health check
@app.get("/")
async def health_check():
    return JSONResponse(content={"status": "ok"})

# Login with face recognition
@app.post("/python/login", status_code=status.HTTP_200_OK)
async def log_in_face_recognition(
    user_id: str = Form(...),
    file: UploadFile = File(...),
):
    contents = await file.read()
    np_image = np.frombuffer(contents, np.uint8)
    bgr_image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    if bgr_image is None:
        return JSONResponse(status_code=400, content={"message": "Invalid image file."})

    # Detect face
    detections = detect_faces_mtcnn(bgr_image)
    if len(detections) == 0:
        return JSONResponse(status_code=400, content={"message": "No face detected."})

    # Use first detected face
    face_crop = crop_face(bgr_image, detections[0])
    unknown_embedding = get_facenet_embedding(face_crop)
    if unknown_embedding is None:
        return JSONResponse(status_code=400, content={"message": "Face could not be encoded."})

    # Fetch user
    try:
        user_record = await users_collection.find_one({"_id": ObjectId(user_id)})
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": f"Invalid user ID. {e}"})

    if not user_record:
        return JSONResponse(status_code=404, content={"message": "User not found."})

    user_image_url = user_record.get("faceImage")
    if not user_image_url:
        return JSONResponse(status_code=404, content={"message": "No face image for this user."})

    try:
        response = requests.get(user_image_url)
        response.raise_for_status()
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Failed to download face image. {e}"})

    known_image_bgr = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    known_detections = detect_faces_mtcnn(known_image_bgr)
    if len(known_detections) == 0:
        return JSONResponse(status_code=400, content={"message": "No face detected in stored image."})

    known_crop = crop_face(known_image_bgr, known_detections[0])
    known_embedding = get_facenet_embedding(known_crop)
    if known_embedding is None:
        return JSONResponse(status_code=400, content={"message": "Stored face image is invalid."})

    # Compare embeddings
    distance = np.linalg.norm(unknown_embedding - known_embedding)
    match = distance < 0.8

    if match:
        access_token = create_access_token({"_id": user_id})
        refresh_token = create_refresh_token({"_id": user_id})
        return JSONResponse(
            status_code=200,
            content={
                "message": "Login successful!",
                "_id": user_id,
                "accessToken": access_token,
                "refreshToken": refresh_token
            }
        )

    return JSONResponse(status_code=401, content={"message": "Face does not match."})

# Ask question endpoint
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

# Start chat
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

        session_id = str(uuid.uuid4())
        sessions[session_id] = qa_chain
        return JSONResponse(content={"session_id": session_id, "answer": answer})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Continue chat
@app.post("/python/continue_chat")
async def continue_chat(session_id: str = Form(...), query: str = Form(...)):
    try:
        if session_id not in sessions:
            return JSONResponse(status_code=404, content={"message": "Session not found"})

        qa_chain = sessions[session_id]
        answer = qa_chain.run(query)
        return JSONResponse(content={"answer": answer})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
