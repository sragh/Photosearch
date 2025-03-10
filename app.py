import os
import pickle
import time
import streamlit as st
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw

# -------------------------------
# Configuration for Local Embeddings
# -------------------------------
EMBEDDING_DIR = "SavedEmbeddings"
EMBEDDING_FILE = os.path.join(EMBEDDING_DIR, "PhotoAlbumLocalEmbeddings.pkl")

# -------------------------------
# Summary of What This Code Does:
# - Model Initialization & Caching:
#   • Loads pre-trained face detection (MTCNN) and face recognition (InceptionResnetV1) models using facenet-pytorch.
#   • Caches the loaded models for efficient reuse.
#  
# - Album Images Upload, Processing & Local Storage:
#   • Checks for locally stored embeddings (in EMBEDDING_FILE) and loads them with a progress bar.
#   • Allows uploading new album images via a file uploader.
#   • Provides a "Load New Album Images" button that, when clicked, deletes the old embeddings and processes new images.
#   • Processes each album image to detect faces, obtain bounding boxes, and compute face embeddings.
#   • Stores additional metadata (filename, original image, bounding boxes) along with embeddings.
#   • Saves the processed album data locally in the EMBEDDING_FILE.
#  
# - Reference Image Upload & Processing:
#   • Enables uploading a reference image via the file uploader.
#   • Processes the reference image to detect faces and compute face embeddings.
#   • Displays detected face(s) from the reference image for user verification.
#   • Uses a progress bar to show the reference image processing progress.
#  
# - Similarity Search & Results Display:
#   • Provides a "Run Search" button that computes the cosine similarity between the reference face embedding and each album image's face embeddings.
#   • Filters and displays all album images with a similarity score of 0.7 or higher.
#   • For each matching image, identifies the face (bounding box) that best matches the reference.
#   • Annotates the album image to indicate where the matching face is.
#   • Displays the filename, similarity score, and annotated image.
# -------------------------------

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
    """
    Detect faces in the image and return:
      - face crops (tensor),
      - embeddings (tensor),
      - bounding boxes (np.ndarray of shape (n,4)).
    If no face is detected, returns (None, None, None).
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
# Helper Functions for Local Storage
# -------------------------------
def save_album_data(album_data):
    if not os.path.exists(EMBEDDING_DIR):
        os.makedirs(EMBEDDING_DIR)
    with open(EMBEDDING_FILE, "wb") as f:
        pickle.dump(album_data, f)

def load_album_data():
    if os.path.exists(EMBEDDING_FILE):
        with open(EMBEDDING_FILE, "rb") as f:
            return pickle.load(f)
    return None

# -------------------------------
# Streamlit App Interface
# -------------------------------
st.title("Face-based Reference Photo Search")
st.write("This app uses locally stored embeddings for efficient search. Upload a reference image to find matching faces in the photo album.")

# --- Album Images Processing & Local Embeddings ---
album_data = None

# Option to load new album images
new_album_files = st.file_uploader("Upload New Album Images (to create new local embeddings)", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="new_album")

if st.button("Load New Album Images"):
    if new_album_files:
        st.write(f"Processing {len(new_album_files)} new album images...")
        album_progress = st.progress(0)
        total_album = len(new_album_files)
        temp_album_data = []
        for idx, file in enumerate(new_album_files):
            try:
                image = Image.open(file).convert("RGB")
            except Exception as e:
                st.error(f"Error loading album image {file.name}: {e}")
                continue
            face_crops, embeddings, boxes = get_face_embeddings(image, mtcnn, resnet)
            if embeddings is None or embeddings.shape[0] == 0 or boxes is None:
                st.warning(f"No face detected in image {file.name}; skipping.")
            else:
                temp_album_data.append((image, embeddings, file.name, boxes))
            album_progress.progress((idx + 1) / total_album)
            time.sleep(0.05)
        album_data = temp_album_data
        save_album_data(album_data)
        st.success(f"Processed and saved {len(album_data)} album images with detected faces.")
    else:
        st.warning("Please upload new album images to load.")

# If no new album images were loaded, attempt to load local embeddings.
if album_data is None:
    loaded_data = load_album_data()
    if loaded_data is not None:
        # Show a quick progress bar for local loading.
        load_progress = st.progress(0)
        time.sleep(0.2)
        load_progress.progress(1.0)
        album_data = loaded_data
        st.info(f"Loaded local embeddings for {len(album_data)} album images.")
    else:
        st.info("No local album embeddings found. Please load new album images.")

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
            st.write("Detected face(s) in reference image:")
            for face in face_crops:
                face_img = Image.fromarray((face.permute(1, 2, 0).mul(255).byte().numpy()))
                st.image(face_img, width=160)
            ref_embedding = ref_embeddings[0]
            ref_progress.progress(1.0)
            st.success("Reference image processed.")
else:
    st.info("Please upload a reference image.")

# --- Run Search Using Local Embeddings ---
if album_data and ref_file and st.button("Run Search"):
    if not album_data:
        st.error("No album images with detected faces available.")
    elif ref_embedding is None:
        st.error("No face detected in the reference image.")
    else:
        st.write("Running search using local embeddings...")
        search_progress = st.progress(0)
        results = []
        total_album = len(album_data)
        for idx, (orig_img, embeddings, fname, boxes) in enumerate(album_data):
            sims = F.cosine_similarity(embeddings, ref_embedding.unsqueeze(0))
            max_sim, max_idx = sims.max(0)
            max_sim = max_sim.item()
            if max_sim >= 0.7:
                max_idx = int(max_idx.item())
                re_boxes, _ = mtcnn.detect(orig_img)
                if re_boxes is not None and len(re_boxes) > max_idx:
                    best_box = re_boxes[max_idx]
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
