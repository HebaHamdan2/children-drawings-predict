import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from werkzeug.utils import secure_filename
from PIL import Image
import logging

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for the Flask app
CORS(app, resources={r"/*": {"origins": "*"}})

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)

# Load the YOLO model - use an absolute path or environment variable if needed
MODEL_PATH = os.getenv("MODEL_PATH", "../best.pt")  # Ensure the path is correct for deployment
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Ensure the path is correct.")

try:
    model = YOLO(MODEL_PATH)
    logging.info("YOLO model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Error loading YOLO model: {e}")

# Define allowed file extensions for uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure the 'uploads' directory exists for temporary file storage
UPLOADS_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to resize the uploaded image
def resize_image(filepath, target_size=(224, 224)):
    try:
        with Image.open(filepath) as img:
            img = img.resize(target_size)
            img.save(filepath)
            logging.info(f"Image resized to: {target_size}")
    except Exception as e:
        raise RuntimeError(f"Error resizing image: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image file is included in the request
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    
    # Validate file format
    if not (file and allowed_file(file.filename)):
        return jsonify({'error': 'Invalid file format. Allowed formats: png, jpg, jpeg'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOADS_DIR, filename)

    try:
        # Save the uploaded file to the server
        file.save(filepath)
        logging.info(f"Image saved at: {filepath}")

        # Open and log the image properties
        with Image.open(filepath) as img:
            logging.info(f"Original image size: {img.size}")

        # Resize the image
        resize_image(filepath)

        # Run inference using the YOLO model
        results = model.predict(source=filepath, show=False)

        # Cleanup the temporary file after prediction
        os.remove(filepath)

        # Extract and format results
        if isinstance(results, list) and hasattr(results[0], 'probs') and results[0].probs is not None:
            probs = results[0].probs.data.numpy()
            label_names = {0: 'Anger and aggression', 1: 'Anxiety', 2: 'Happy', 3: 'Sad'}
            predictions = {label_names.get(i, f'Class {i}'): f"{prob * 100:.2f}%" for i, prob in enumerate(probs)}
            return jsonify({'predictions': predictions}), 200
        else:
            return jsonify({'error': 'Unable to process the image'}), 500

    except Exception as e:
        # Log the error and cleanup
        logging.error(f"Error during prediction: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': f"Error during prediction: {str(e)}"}), 500

# Main entry point for Render deployment
if __name__ == '__main__':
    # Ensure the app uses the correct port on the cloud platform
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
