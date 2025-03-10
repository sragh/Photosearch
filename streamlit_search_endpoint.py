import os
import io
import pickle
import time
import tempfile
import zipfile

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
import boto3
from facenet_pytorch import MTCNN, InceptionResnetV1

# -------------------------------
# S3 Configuration
# -------------------------------
ALBUM_BUCKET = "first-face-search"               # S3 bucket with album images
CACHE_BUCKET = "face-search-embedding-cache"       # S3 bucket for embedding cache
CACHE_KEY = "embedding_cache.pkl"                  # S3 key for cache

# -------------------------------
# Initialize S3 Client
# -------------------------------
s3 = boto3.client("s3")

# -------------------------------
# Load Models
# -------------------------------
@st.cache_resource
def load_models():
    mtcnn = MTCNN(image_size=160, margin=10, keep_all=True, post_process=True, min_face_size=20)
    resnet = InceptionResnetV1(pretrained="vggface2").eval()
    return mtcnn, resnet

mtcnn, resnet = load_models()

def get_face_embedding(image):
    """Compute the face embedding for a given image."""
    face_crops = mtcnn(image)
    if face_crops is None:
        return None
    if face_crops.ndim == 3:
        face_crops = face_crops.unsqueeze(0)
    embedding = resnet(face_crops)[0]
    return embedding

# -------------------------------
# Load Embedding Cache from S3
# -------------------------------
@st.cache_resource(show_spinner=False)
def load_embedding_cache():
    try:
        response = s3.get_object(Bucket=CACHE_BUCKET, Key=CACHE_KEY)
        album_data = pickle.loads(response["Body"].read())
        st.info(f"Loaded cache with {len(album_data)} embeddings.")
        return album_data
    except Exception as e:
        st.error("Error loading embedding cache from S3: " + str(e))
        return []

album_data = load_embedding_cache()

# -------------------------------
# Utility: Download Image from S3
# -------------------------------
def load_image_from_s3(s3_key):
    try:
        response = s3.get_object(Bucket=ALBUM_BUCKET, Key=s3_key)
        image_bytes = response["Body"].read()
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        st.error(f"Error loading image {s3_key}: {e}")
        return None

# -------------------------------
# Utility: Create ZIP of Images
# -------------------------------
def create_zip(image_keys):
    temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    with zipfile.ZipFile(temp_zip, "w") as zf:
        for key in image_keys:
            try:
                response = s3.get_object(Bucket=ALBUM_BUCKET, Key=key)
                data = response["Body"].read()
                zf.writestr(os.path.basename(key), data)
            except Exception as e:
                st.error(f"Error retrieving {key}: {e}")
    temp_zip.close()
    return temp_zip.name

# -------------------------------
# Streamlit UI for Public Search
# -------------------------------
st.title("Public Face-based Image Search")
st.write("Upload a reference image to search for similar faces in the album.")

ref_file = st.file_uploader("Upload Reference Image", type=["jpg", "jpeg", "png"])
if ref_file:
    try:
        ref_image = Image.open(ref_file).convert("RGB")
        st.image(ref_image, caption="Reference Image", width=224)
    except Exception as e:
        st.error("Error processing reference image: " + str(e))
    
    st.write("Computing embedding for the reference image...")
    query_embedding = get_face_embedding(ref_image)
    if query_embedding is None:
        st.error("No face detected in the reference image.")
    else:
        st.success("Reference image processed.")
        st.write("Running similarity search...")
        results = []
        # Loop over each cached album entry (each with embedding, filename, bounding_boxes, and metadata)
        for entry in album_data:
            emb = entry["embedding"]
            sim = F.cosine_similarity(emb, query_embedding.unsqueeze(0)).item()
            if sim >= 0.7:
                results.append({
                    "s3_key": entry["filename"],
                    "score": sim,
                    "bounding_boxes": entry["bounding_boxes"],
                    "metadata": {"timestamp": entry.get("timestamp", "")}
                })
        if results:
            results.sort(key=lambda x: x["score"], reverse=True)
            st.write(f"Found {len(results)} matching images:")
            download_keys = []
            for res in results:
                s3_url = f"https://{ALBUM_BUCKET}.s3.amazonaws.com/{res['s3_key']}"
                st.write(f"**Filename:** {res['s3_key']}")
                st.write(f"**Similarity:** {res['score']:.4f}")
                st.write(f"**Bounding Boxes:** {res['bounding_boxes']}")
                st.write(f"**Metadata:** {res['metadata']}")
                st.markdown(f"[Download Image]({s3_url})")
                # Load image and annotate with bounding boxes
                img = load_image_from_s3(res["s3_key"])
                if img:
                    annotated_img = img.copy()
                    draw = ImageDraw.Draw(annotated_img)
                    if res["bounding_boxes"]:
                        for box in res["bounding_boxes"]:
                            #draw.rectangle(box, outline="red", width=2)
                            draw.rectangle([tuple(box[:2]), tuple(box[2:])], outline="red", width=20)
                    st.image(annotated_img, width=224)
                    download_keys.append(res["s3_key"])
            # Option to download all matches as ZIP
            if download_keys:
                zip_file = create_zip(download_keys)
                with open(zip_file, "rb") as f:
                    st.download_button("Download All as ZIP", f, file_name="matches.zip", mime="application/zip")
        else:
            st.write("No matching images found with similarity ≥ 0.7.")
else:
    st.info("Please upload a reference image.")
