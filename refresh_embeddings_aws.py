import io
import pickle
import time
import gc
from datetime import datetime

import streamlit as st
from PIL import Image, ImageDraw
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import boto3

# -------------------------------
# Configuration for AWS Refresh
# -------------------------------
ALBUM_BUCKET = "first-face-search"              # S3 bucket with album images
CACHE_BUCKET = "face-search-embedding-cache"      # S3 bucket for embedding cache
CACHE_KEY = "embedding_cache.pkl"                 # S3 key for cache
CHUNK_SIZE = 1  # Process one image at a time

# -------------------------------
# Initialize S3 Client
# -------------------------------
s3 = boto3.client('s3')

# -------------------------------
# Load Models
# -------------------------------
def load_models():
    mtcnn = MTCNN(image_size=160, margin=10, keep_all=True, post_process=True, min_face_size=20)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    return mtcnn, resnet

mtcnn, resnet = load_models()

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

def compute_album_embeddings():
    """Process album images from the S3 bucket and build the embedding cache."""
    album_data = []
    response = s3.list_objects_v2(Bucket=ALBUM_BUCKET)
    objects = response.get("Contents", [])
    valid_extensions = (".jpg", ".jpeg", ".png")
    keys = [obj["Key"] for obj in objects if obj["Key"].lower().endswith(valid_extensions)]
    total = len(keys)
    st.write(f"Found {total} images in {ALBUM_BUCKET}.")
    progress = st.progress(0)
    
    for idx, key in enumerate(keys):
        try:
            s3_object = s3.get_object(Bucket=ALBUM_BUCKET, Key=key)
            image_bytes = s3_object["Body"].read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            st.error(f"Error loading image {key}: {e}")
            continue
        
        embeddings, boxes = get_face_embeddings(image, mtcnn, resnet)
        if embeddings is None:
            st.warning(f"No face detected in {key}; skipping.")
        else:
            album_data.append({
                "embedding": embeddings,
                "filename": key,
                "bounding_boxes": boxes,
                "timestamp": datetime.utcnow().isoformat()
            })
            # Optional: Show annotated preview for the first few images
            if idx < 3:
                annotated_image = image.copy()
                draw = ImageDraw.Draw(annotated_image)
                for box in boxes:
                    draw.rectangle([tuple(box[:2]), tuple(box[2:])], outline="red", width=20)
                st.image(annotated_image, caption=f"Annotated: {key}", width=224)
        
        progress.progress((idx+1)/total)
        del image, image_bytes, embeddings, boxes
        gc.collect()
        time.sleep(0.05)
        
    st.success(f"Processed embeddings for {len(album_data)} images.")
    return album_data

def upload_cache_to_s3(album_data):
    """Upload the embedding cache to S3."""
    cache_buffer = io.BytesIO()
    pickle.dump(album_data, cache_buffer)
    cache_buffer.seek(0)
    s3.put_object(Bucket=CACHE_BUCKET, Key=CACHE_KEY, Body=cache_buffer)
    st.success(f"Uploaded embedding cache with {len(album_data)} entries to S3.")

# -------------------------------
# Streamlit UI for AWS Refresh
# -------------------------------
st.title("AWS Refresh of Album Embeddings")
if st.button("Refresh Album Embeddings from S3 Album Bucket"):
    album_data = compute_album_embeddings()
    upload_cache_to_s3(album_data)
