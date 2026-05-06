import os
import gc
import random
import io
import base64
import threading
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import cv2

# 1. DETERMINISTIC TENSORFLOW SETUP (STRICT CPU)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["PYTHONHASHSEED"] = "42"

# Global Seeds
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# CPU Isolation
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# 2. OFFICIAL PREPROCESSING
from tensorflow.keras.applications.efficientnet import preprocess_input

app = Flask(__name__)
CORS(app)

IMG_SIZE = 224
CLASS_NAMES = ["Adenocarcinoma", "Large Cell Carcinoma", "Normal", "Squamous Cell Carcinoma"]
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "efficientnet_lung_model.h5")

# 3. GLOBAL MODEL & INFERENCE LOCK
model = None
model_lock = threading.Lock()

def load_ai_model():
    global model
    if model is None:
        import keras
        try:
            # Load with compile=False to avoid optimizer overhead/RAM usage
            model = keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
        except Exception as e:
            print(f"Critical Load Error: {e}")
    return model

# 4. STABLE PREPROCESSING
def preprocess_image(image_pil):
    img = image_pil.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
    img_array = np.array(img).astype("float32")
    img_array = np.expand_dims(img_array, axis=0)
    # Official EfficientNet normalization
    img_array = preprocess_input(img_array)
    return img_array

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    global model
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    
    if model is None:
        load_ai_model()

    try:
        file = request.files['file']
        img_bytes = file.read()
        
        # Load and Preprocess
        image_pil = Image.open(io.BytesIO(img_bytes))
        img_array = preprocess_image(image_pil)
        
        # Prepare original image for response
        original_cv2 = cv2.cvtColor(np.array(image_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
        _, buffer_orig = cv2.imencode('.jpg', original_cv2)
        base64_image = f"data:image/jpeg;base64,{base64.b64encode(buffer_orig).decode('utf-8')}"

        # 5. SAFE LOCKED INFERENCE (FIXES INSTABILITY)
        with model_lock:
            # 6. FIX INPUT SIGNATURE WARNING: Wrap input in a list
            # This satisfies the [['input_layer']] expectation of Keras Functional API
            preds_tensor = model([img_array], training=False)
            preds = preds_tensor.numpy()[0]
        
        idx = np.argmax(preds)
        label = CLASS_NAMES[idx]
        confidence = float(preds[idx])

        result = {
            "class": label,
            "confidence": round(confidence * 100, 2),
            "original_image": base64_image
        }

        # 7. MEMORY STABILIZATION
        del preds
        del img_array
        del original_cv2
        gc.collect()
        # tf.keras.backend.clear_session() is NOT called to keep model loaded

        return jsonify(result)

    except Exception as e:
        print(f"Inference Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
