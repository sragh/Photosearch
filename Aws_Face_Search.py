import os
import io
import pickle
import time
import gc
import streamlit as st
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw
import boto3

# -------------------------------
# S3 Bucket Configurations
# -------------------------------
ALBUM_BUCKET = "first-face-search"  # Your album images bucket
CACHE_BUCKET = "face-search-embedding-cache"  # Bucket to store embedding cache
CACHE_KEY = "embedding_cache.pkl"  # Key name for cached embeddings
CHUNK_SIZE = 1  # Process one image at a time

# -------------------------------
# Model Initialization & Caching
# -------------------------------
@st.cache_resource
def load_face_models():
    mtcnn = MTCNN(image_size=160, margin=10, keep_all=True, post_process=True, min_face_size=20)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    return mtcnn, resnet

mtcnn, resnet = load_face_models()

def get_face_embeddings(image, mtcnn, resnet):
    """
    Detect faces in the image and return:
      - face crops (tensor),
      - embeddings (tensor),
      - bounding boxes (np.ndarray of shape (n,4)).
    Returns (None, None, None) if no face is detected.
    """
    face_crops = mtcnn(image)
    boxes, _ = mtcnn.detect(image)
    if face_crops is None:
        return None, None, None
    if face_crops.ndim == 3:
        face_crops = face_crops.unsqueeze(0)
    embeddings = resnet(face_crops)
    return face_crops, embeddings, boxes

# -------------------------------
# S3 Embedding Cache Functions
# -------------------------------
def load_album_data_from_cache():
    """
    Attempts to load the album embeddings cache from the CACHE_BUCKET in S3.
    Returns the album data if available, else None.
    """
    s3 = boto3.client('s3')
    try:
        response = s3.get_object(Bucket=CACHE_BUCKET, Key=CACHE_KEY)
        cache_bytes = response['Body'].read()
        album_data = pickle.loads(cache_bytes)
        st.info(f"Loaded album embeddings from S3 cache with {len(album_data)} entries.")
        return album_data
    except Exception as e:
        st.info("No album cache found in S3. Please refresh album embeddings.")
        return None

def save_album_data_to_cache(album_data):
    """
    Saves the album data (embedding, metadata) as a pickle file to the CACHE_BUCKET in S3.
    """
    s3 = boto3.client('s3')
    cache_buffer = io.BytesIO()
    pickle.dump(album_data, cache_buffer)
    cache_buffer.seek(0)
    s3.put_object(Bucket=CACHE_BUCKET, Key=CACHE_KEY, Body=cache_buffer)
    st.success(f"Saved album embeddings to S3 cache with {len(album_data)} entries.")

# -------------------------------
# Compute Album Embeddings (Chunk-Based, Memory Optimized)
# -------------------------------
def compute_album_embeddings_chunked():
    """
    Lists image objects in the ALBUM_BUCKET, processes them one by one,
    computes face embeddings, and returns a list of tuples:
      (embedding (tensor), filename (S3 key), bounding boxes).
    Only minimal data is stored in memory.
    """
    s3 = boto3.client('s3')
    album_data = []
    
    # List objects in the album bucket
    response = s3.list_objects_v2(Bucket=ALBUM_BUCKET)
    objects = response.get('Contents', [])
    valid_extensions = ('.jpg', '.jpeg', '.png')
    keys = [obj['Key'] for obj in objects if obj['Key'].lower().endswith(valid_extensions)]
    
    total_files = len(keys)
    if total_files == 0:
        st.warning("No valid images found in the album S3 bucket.")
        return album_data

    progress_bar = st.progress(0)
    
    for i, key in enumerate(keys):
        try:
            s3_object = s3.get_object(Bucket=ALBUM_BUCKET, Key=key)
            image_bytes = s3_object['Body'].read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            st.error(f"Error loading image {key}: {e}")
            continue

        face_crops, embeddings, boxes = get_face_embeddings(image, mtcnn, resnet)
        # Do not store the full image; only store embedding, key, and boxes
        if embeddings is None or embeddings.shape[0] == 0 or boxes is None:
            st.warning(f"No face detected in image {key}; skipping.")
        else:
            album_data.append((embeddings, key, boxes))
        
        progress_bar.progress((i + 1) / total_files)
        # Clean up memory explicitly
        del image, face_crops, embeddings, boxes, image_bytes
        gc.collect()
        time.sleep(0.05)
    
    st.success(f"Processed {len(album_data)} images with detected faces out of {total_files} files.")
    return album_data

# -------------------------------
# Utility: Load Image from S3 on Demand
# -------------------------------
def load_image_from_s3(key):
    s3 = boto3.client('s3')
    try:
        s3_object = s3.get_object(Bucket=ALBUM_BUCKET, Key=key)
        image_bytes = s3_object['Body'].read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return image
    except Exception as e:
        st.error(f"Error reloading image {key}: {e}")
        return None

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Face-based Reference Photo Search (AWS) - Memory Optimized")
st.write("This app processes album images from S3 in small chunks and stores only essential data (embeddings, filename, bounding boxes) to minimize memory usage.")

# Option to refresh album embeddings
if st.button("Refresh Album Embeddings"):
    st.info("Refreshing album embeddings with memory-optimized processing (chunk size: 1)...")
    album_data = compute_album_embeddings_chunked()
    if album_data:
        save_album_data_to_cache(album_data)
    else:
        st.error("No album embeddings computed. Please check the images in the album bucket.")
else:
    album_data = load_album_data_from_cache()

# --- Reference Image Processing ---
ref_embedding = None
ref_file = st.file_uploader("Upload Reference Image", type=["jpg", "jpeg", "png"], key="ref")
if ref_file:
    st.write("Processing reference image...")
    ref_progress = st.progress(0)
    try:
        ref_image = Image.open(ref_file).convert("RGB")
    except Exception as e:
        st.error(f"Error loading reference image: {e}")
    else:
        st.image(ref_image, caption="Reference Image", width=224)
        face_crops, ref_embeddings, _ = get_face_embeddings(ref_image, mtcnn, resnet)
        ref_progress.progress(0.5)
        time.sleep(0.2)
        if ref_embeddings is None or ref_embeddings.shape[0] == 0:
            st.error("No face detected in the reference image. Please try another image.")
        else:
            st.write("Detected face(s) in the reference image:")
            for face in face_crops:
                face_img = Image.fromarray((face.permute(1, 2, 0).mul(255).byte().numpy()))
                st.image(face_img, width=160)
            ref_embedding = ref_embeddings[0]
            ref_progress.progress(1.0)
            st.success("Reference image processed.")
else:
    st.info("Please upload a reference image.")

# --- Run Similarity Search ---
if album_data and ref_file and st.button("Run Search"):
    if not album_data:
        st.error("No album embeddings available.")
    elif ref_embedding is None:
        st.error("No face detected in the reference image.")
    else:
        st.write("Running similarity search...")
        search_progress = st.progress(0)
        results = []
        total_album = len(album_data)
        for idx, (embeddings, fname, boxes) in enumerate(album_data):
            # Compare each stored embedding with the reference embedding
            sims = F.cosine_similarity(embeddings, ref_embedding.unsqueeze(0))
            max_sim, max_idx = sims.max(0)
            max_sim = max_sim.item()
            if max_sim >= 0.7:
                # For display, reload the image from S3
                orig_img = load_image_from_s3(fname)
                if orig_img is None:
                    continue
                # Optionally, re-detect bounding boxes for annotation
                re_boxes, _ = mtcnn.detect(orig_img)
                if re_boxes is not None and len(re_boxes) > max_idx:
                    best_box = re_boxes[int(max_idx)]
                    best_box = [int(round(coord)) for coord in best_box]
                else:
                    best_box = None
                results.append((orig_img, max_sim, fname, best_box))
            search_progress.progress((idx + 1) / total_album)
            time.sleep(0.05)
        
        if results:
            st.write("### Matching Images (Similarity Score ≥ 0.7):")
            results.sort(key=lambda x: x[1], reverse=True)
            for img, score, fname, best_box in results:
                annotated_img = img.copy()
                draw = ImageDraw.Draw(annotated_img)
                if best_box is not None:
                    draw.rectangle(best_box, outline="red", width=20)
                    st.write(f"Filename: {fname} | Similarity Score: {score:.4f} | Bounding Box: {best_box}")
                else:
                    st.write(f"Filename: {fname} | Similarity Score: {score:.4f} | No bounding box available")
                st.image(annotated_img, width=224)
        else:
            st.write("No matching images found with similarity score above 0.7.")
