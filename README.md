# 📸 Reference Image Search

## 🛤️ Project Overview
**Reference Image Search** is a solution designed to help you find yourself in a sea of photos effortlessly. Imagine you've returned from a road trip with friends or a reunion with your undergraduate classmates. You have a massive photo album and want to quickly locate all pictures that include you. This project is here to solve that problem — efficiently and cost-effectively!

---

## 🎯 Key Features
- **Cost-Effective Face Recognition:** Leverages PyTorch for face recognition and Streamlit for a simple UI.
- **AWS Integration:** Uses AWS S3 for storage and SageMaker for model development and search functionality.
- **Efficient and On-Demand:** 
   - Not real-time (processing can take 10+ seconds if needed).
   - Optimized for sparse queries (50-70 searches over a few days).
- **Versatile Search:** 
   - Identifies **all similar images** based on embeddings, not just the top-N matches.
   - Handles reference images of different resolutions seamlessly.

---

## 🏗️ Architecture Assumptions
- **Cloud Provider:** AWS
- **Model Development:** SageMaker for training and inference.
- **Local Development:** VS Code for coding and testing.

---

## 📂 AWS Setup
1. **S3 Buckets:**
   - `AlbumBucket` for storing all original images.
   - `SavedEmbeddings` for storing image embeddings.
   - `SageMakerArtifacts` for storing model artifacts.

2. **IAM Roles and Policies:**
   - Define roles for each S3 bucket with minimal and secure access policies.

---

## 🛠️ Functionality
### 1. Create Embeddings Locally
- Extract embeddings from all images using PyTorch.
- Upload embeddings to the `SavedEmbeddings` S3 bucket.

### 2. Create Embeddings on AWS
- Run a similar embedding process directly on AWS.
- Store results in the `SavedEmbeddings` bucket.

### 3. Perform Image Search on SageMaker
- Use SageMaker to search embeddings efficiently.
- Provide a public endpoint using Streamlit for user-friendly access.
- Accelerate search using GPU on SageMaker.

---

## 🚀 Getting Started
- Clone the repository:
  ```bash
  git clone https://github.com/sragh/Photosearch.git

Install dependencies:
pip install -r requirements.txt

Run the Streamlit app:
streamlit run app.py

Future enhancements could include integrating additional cloud cost optimizations, expanding to real-time search capabilities, and enhancing the UI with more search filters. Find your moments in a snap! 📸✨


```
