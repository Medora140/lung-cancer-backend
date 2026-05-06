import os
import io
import sys

# DEBUG & HARD FAIL CHECK
print("PYTHON VERSION:", sys.version)
assert "3.11" in sys.version, f"WRONG PYTHON VERSION ON RENDER: {sys.version}. Expected 3.11.x"

import base64
import numpy as np
import tensorflow as tf
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

# Removed old print statement from below


app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# Constants
IMG_SIZE = 224
CLASS_NAMES = ["Adenocarcinoma", "Large Cell Carcinoma", "Normal", "Squamous Cell Carcinoma"]
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "efficientnet_lung_model.h5"
)

# Global variable for model
model = None

def load_ai_model():
    global model
    if model is None:
        print("Current directory:", os.getcwd())
        print("Files in directory:", os.listdir())
        print("MODEL PATH:", MODEL_PATH)
        print("MODEL EXISTS:", os.path.exists(MODEL_PATH))
        if os.path.exists(MODEL_PATH):
            try:
                # Load model without compiling to save memory
                model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                print("Model loaded successfully.")
            except Exception as e:
                print(f"Error loading model: {e}")
        else:
            print(f"Error: Model not found at {MODEL_PATH}")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    try:
        grad_model = tf.keras.models.Model(
            [model.inputs],
            [model.get_layer(last_conv_layer_name).output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
            
        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        return heatmap.numpy()
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return None

def preprocess_image(image):
    # Convert PIL Image to OpenCV
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Preprocessing pipeline
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    resized = cv2.resize(enhanced, (IMG_SIZE, IMG_SIZE))
    rgb_image = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    
    # Prepare for model
    img_array = np.expand_dims(rgb_image, axis=0)
    return img_array.astype('float32') / 255.0, image_cv

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Ensure model is loaded (Singleton pattern)
    if model is None:
        load_ai_model()
        if model is None:
            return jsonify({'error': 'Model file missing on server. Ensure efficientnet_lung_model.h5 is in the backend folder.'}), 500

    try:
        # Load and preprocess
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_array, original_cv2_img = preprocess_image(image)

        # Predict
        preds = model.predict(img_array)[0]
        idx = np.argmax(preds)
        label = CLASS_NAMES[idx]
        confidence = float(preds[idx])

        # Generate Grad-CAM Heatmap
        last_conv_layer_name = "top_conv" 
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
        
        response_data = {
            'class': label,
            'confidence': round(confidence * 100, 2),
            'all_predictions': {CLASS_NAMES[i]: round(float(preds[i]) * 100, 2) for i in range(len(CLASS_NAMES))}
        }

        # Superimpose heatmap if successful
        if heatmap is not None:
            heatmap_resized = cv2.resize(heatmap, (original_cv2_img.shape[1], original_cv2_img.shape[0]))
            heatmap_resized = np.uint8(255 * heatmap_resized)
            heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
            superimposed_img = cv2.addWeighted(original_cv2_img, 0.6, heatmap_color, 0.4, 0)
            
            _, buffer_heat = cv2.imencode('.jpg', superimposed_img)
            heat_base64 = base64.b64encode(buffer_heat).decode('utf-8')
            response_data['heatmap_image'] = f"data:image/jpeg;base64,{heat_base64}"

        # Original image for reference
        _, buffer_orig = cv2.imencode('.jpg', original_cv2_img)
        orig_base64 = base64.b64encode(buffer_orig).decode('utf-8')
        response_data['original_image'] = f"data:image/jpeg;base64,{orig_base64}"

        return jsonify(response_data)

    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

# Eager loading for production stability
load_ai_model()

if __name__ == '__main__':
    # Local dev
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
