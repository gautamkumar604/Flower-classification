"""
Flask Backend for Flower Classification App
This API provides an endpoint to predict flower types from uploaded images.
It also serves the built React/Vite frontend from frontend/dist.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import numpy as np
import io
import json
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend development

# Configuration
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
IMG_SIZE = (224, 224)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'flower_model.h5')
CLASS_NAMES_PATH = os.path.join(BASE_DIR, 'model', 'class_names.json')
FRONTEND_DIST_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'dist'))
FRONTEND_SRC_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
FRONTEND_DIR = FRONTEND_DIST_DIR if os.path.isdir(FRONTEND_DIST_DIR) else FRONTEND_SRC_DIR

# Global model and class names
model = None
class_names = None


def load_flower_model():
    global model, class_names

    try:
        print("🔄 Loading model...")

        model_path = MODEL_PATH
        class_path = CLASS_NAMES_PATH

        print(f"Model path: {model_path}")
        print(f"Class names path: {class_path}")

        model = load_model(model_path)

        with open(class_path, 'r') as f:
            class_names = json.load(f)

        print("✅ Model loaded successfully")
        print("✅ Class names loaded successfully")

    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        model = None
        class_names = None

def allowed_file(filename):
    """Validate allowed image file extensions."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_bytes):
    """Preprocess the uploaded image for model prediction."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize(IMG_SIZE, Image.Resampling.LANCZOS)

        image_array = np.array(image, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)
        return preprocess_input(image_array)
    except Exception as exc:
        raise ValueError(f"Error processing image: {exc}")


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'class_names_loaded': class_names is not None,
        'frontend_dir': FRONTEND_DIR
    }), 200


@app.route('/', methods=['GET'])
def serve_frontend():
    if not os.path.isdir(FRONTEND_DIR):
        return jsonify({'error': 'Frontend build not found. Run npm run build in the frontend folder.'}), 500
    return send_from_directory(FRONTEND_DIR, 'index.html')


@app.route('/<path:path>', methods=['GET'])
def serve_frontend_static(path):
    if not os.path.isdir(FRONTEND_DIR):
        return jsonify({'error': 'Frontend build not found. Run npm run build in the frontend folder.'}), 500

    if '..' in path or os.path.isabs(path):
        return send_from_directory(FRONTEND_DIR, 'index.html')

    full_path = os.path.join(FRONTEND_DIR, path)
    if os.path.exists(full_path) and os.path.isfile(full_path):
        return send_from_directory(FRONTEND_DIR, path)

    return send_from_directory(FRONTEND_DIR, 'index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None or class_names is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 503

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only JPG and PNG are allowed.'}), 400

    file_bytes = file.read()
    if len(file_bytes) > MAX_FILE_SIZE:
        return jsonify({'error': 'File size exceeds 5MB limit'}), 413

    try:
        image_data = preprocess_image(file_bytes)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    try:
        predictions = model.predict(image_data, verbose=0)
        predicted_index = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_index]) * 100
        predicted_class = class_names[predicted_index]

        return jsonify({
            'class': predicted_class,
            'confidence': round(confidence, 2)
        }), 200
    except Exception as exc:
        print(f"❌ Prediction error: {exc}")
        return jsonify({'error': 'Failed to generate prediction'}), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File size too large. Maximum size is 5MB.'}), 413


@app.errorhandler(404)
def handle_404(error):
    if not os.path.isdir(FRONTEND_DIR):
        return jsonify({'error': 'Frontend build not found. Run npm run build in the frontend folder.'}), 500
    return send_from_directory(FRONTEND_DIR, 'index.html')


# app = Flask(__name__)

load_flower_model()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)