from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import torch
import pandas as pd
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel

# === Flask App Setup ===
app = Flask(__name__)
CORS(app)

# === Load Model and Captions Once ===
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
captions_df = pd.read_csv("model/social_media_captions_400.csv")
candidate_captions = captions_df['Caption'].tolist()

# === Your Helper Functions ===
def load_and_preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    return inputs

def generate_image_embeddings(inputs):
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
    return image_features

def match_captions(image_features):
    text_inputs = processor(
        text=candidate_captions,
        return_tensors="pt",
        padding=True,
        truncation=True  # Fixes long caption error
    )
    with torch.no_grad():
        text_features = clip_model.get_text_features(**text_inputs)

    image_features = image_features.detach().cpu().numpy()
    text_features = text_features.detach().cpu().numpy()

    similarities = cosine_similarity(image_features, text_features)
    best_indices = similarities.argsort(axis=1)[0][::-1]

    best_captions = [candidate_captions[i] for i in best_indices]
    best_similarities = similarities[0][best_indices].tolist()

    return best_captions[:5], best_similarities[:5]

#  Paste the API Route BELOW This Line ===
@app.route("/api/match", methods=["POST"])
def api_match():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image = request.files['image']
    image_path = os.path.join("static/uploads", image.filename)
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    image.save(image_path)

    inputs = load_and_preprocess_image(image_path)
    features = generate_image_embeddings(inputs)
    best_captions, similarities = match_captions(features)

    return jsonify({
        "results": list(zip(best_captions, similarities))
    })

@app.route("/")
def home():
    return " PicLingo Flask backend is running!"

# === Run the Flask App ===
if __name__ == "__main__":
    app.run(debug=True)
