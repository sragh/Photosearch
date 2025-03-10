import os
import pickle
import time
from datetime import datetime

import streamlit as st
from PIL import Image, ImageDraw
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import boto3

# -------------------------------
# Configuration for Local Refresh
# -------------------------------
EMBEDDING_DIR = "SavedEmbeddings"
LOCAL_EMBEDDING_FILE = os.path.join(EMBEDDING_DIR, "PhotoAlbumLocalEmbeddings.pkl")
CACHE_BUCKET = "face-search-embedding-cache"  # S3 bucket for cache
CACHE_KEY = "embedding_cache.pkl"              # S3 key for cache

# -------------------------------
# Initialize S3 Client
# -------------------------------
s3 = boto3.client('s3')

# -------------------------------
# Load Face Detection and Recognition Models
# -------------------------------
@st.cache_resource
def load_face_models():
    mtcnn = MTCNN(image_size=160, margin=10, keep_all=True, post_process=True, min_face_size=20)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    return mtcnn, resnet

mtcnn, resnet = load_face_models()

def get_face_embeddings(image, mtcnn, resnet):
    """Detect faces and compute embeddings from an image."""
    face_crops = mtcnn(image)
    boxes, _ = mtcnn.detect(image)
    if face_crops is None or boxes is None:
        return None, None
    if face_crops.ndim == 3:
        face_crops = face_crops.unsqueeze(0)
    embeddings = resnet(face_crops)
    return embeddings, boxes

def save_album_data(album_data):
    """Save the embedding cache locally."""
    if not os.path.exists(EMBEDDING_DIR):
        os.makedirs(EMBEDDING_DIR)
    with open(LOCAL_EMBEDDING_FILE, "wb") as f:
        pickle.dump(album_data, f)

def upload_cache_to_s3():
    """Upload the local embedding cache to S3."""
    with open(LOCAL_EMBEDDING_FILE, "rb") as f:
        s3.upload_fileobj(f, CACHE_BUCKET, CACHE_KEY)
    st.success("Embedding cache uploaded to S3.")

# -------------------------------
# Streamlit UI for Local Refresh
# -------------------------------
st.title("Local Refresh of Album Embeddings")
st.write("Upload your album images to compute face embeddings, bounding boxes, and metadata.")

album_data = []  # List to hold dictionaries for each image

new_album_files = st.file_uploader("Upload Album Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if st.button("Refresh Album Embeddings"):
    if new_album_files:
        st.write(f"Processing {len(new_album_files)} images...")
        progress = st.progress(0)
        total = len(new_album_files)
        for idx, file in enumerate(new_album_files):
            try:
                image = Image.open(file).convert("RGB")
            except Exception as e:
                st.error(f"Error loading {file.name}: {e}")
                continue

            embeddings, boxes = get_face_embeddings(image, mtcnn, resnet)
            if embeddings is None:
                st.warning(f"No face detected in {file.name}; skipping.")
            else:
                # Save the embedding, bounding boxes, and additional metadata
                album_data.append({
                    "embedding": embeddings, 
                    "filename": file.name,
                    "bounding_boxes": boxes,
                    "timestamp": datetime.utcnow().isoformat()
                })
                # Annotate image preview with bounding boxes
                annotated_image = image.copy()
                draw = ImageDraw.Draw(annotated_image)
                for box in boxes:
                    draw.rectangle([tuple(box[:2]), tuple(box[2:])], outline="red", width=20)
                st.image(annotated_image, caption=f"Annotated: {file.name}", width=224)
            progress.progress((idx+1)/total)
            time.sleep(0.05)
        save_album_data(album_data)
        st.success(f"Processed and saved embeddings for {len(album_data)} images.")
        upload_cache_to_s3()
    else:
        st.warning("Please upload album images.")
