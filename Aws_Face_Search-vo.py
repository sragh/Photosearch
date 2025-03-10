import os
import boto3
import json
import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from facenet_pytorch import InceptionResnetV1, MTCNN
import torch
from PIL import Image, ImageDraw
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

#from fastapi import FastAPI

app = FastAPI()
print("API Initialized.")  # Debug print
# Define a route for "/"
@app.get("/")
def read_root():
    print("Root endpoint hit!")  # Debug print
    return {"message": "Welcome to Ref Image Search API!"}

# AWS Configuration
S3_BUCKET = "first-face-search"
EMBEDDINGS_FILE = "face_embeddings.json"
s3 = boto3.client("s3")

# Load FaceNet Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
mtcnn = MTCNN(keep_all=True, device=device)  # For face detection

# Extract embedding for a given image path
def extract_embedding(image_path):
    image = preprocess_image(image_path)
    with torch.no_grad():
        embedding = model(image).cpu().numpy()
    return embedding.flatten()

# Load embeddings from S3
def load_embeddings_from_s3():
    try:
        response = s3.get_object(Bucket=S3_BUCKET, Key=EMBEDDINGS_FILE)
        embeddings = json.loads(response["Body"].read().decode("utf-8"))
        return embeddings
    except Exception as e:
        print("Error loading embeddings:", e)
        return {}

# Preprocess image function
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    faces, _ = mtcnn.detect(image)
    if faces is not None and len(faces) > 0:
        x1, y1, x2, y2 = map(int, faces[0])
        image = image.crop((x1, y1, x2, y2))
    image = image.resize((160, 160))
    image = np.array(image) / 255.0
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    return image

# Draw bounding boxes on images
def draw_bounding_box(image_path):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    faces, _ = mtcnn.detect(image)
    if faces is not None:
        for box in faces:
            x1, y1, x2, y2 = map(int, box)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=20)
    image.save(image_path)  # Save with bounding box

# Upload image to S3
def upload_to_s3(image_path):
    s3.upload_file(image_path, S3_BUCKET, image_path)
    return f"https://{S3_BUCKET}.s3.amazonaws.com/{image_path}"

# Match reference image API
@app.post("/match")
async def match_reference_image(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid image format")

    file_location = f"temp_{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())

    reference_embedding = extract_embedding(file_location)
    stored_embeddings = load_embeddings_from_s3()
    matches = []

    for image_name, embedding in stored_embeddings.items():
        similarity = cosine_similarity([reference_embedding], [embedding])[0][0]
        if similarity > 0.6:
            draw_bounding_box(image_name)  # Draw bounding box on match
            url_with_box = upload_to_s3(image_name)  # Upload to S3
            matches.append({"image_name": image_name, "similarity": similarity, "url": url_with_box})

    matches.sort(key=lambda x: x["similarity"], reverse=True)
    os.remove(file_location)  # Clean up

    if not matches:
        return JSONResponse(content={"message": "No matches found."})
    return JSONResponse(content={"matches": matches})

# Run FastAPI
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
