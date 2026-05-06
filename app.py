import os
# 1. DETERMINISTIC SETTINGS (MUST BE AT TOP)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["PYTHONHASHSEED"] = "42"

import tensorflow as tf
import numpy as np
import random
import io
import base64
import cv2
import threading
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

# 2. OFFICIAL PREPROCESSING IMPORT
from tensorflow.keras.applications.efficientnet import preprocess_input

# Global Seeds
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# CPU Thread Limiting
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

app = Flask(__name__)
CORS(app)

IMG_SIZE = 224
CLASS_NAMES = ["Adenocarcinoma", "Large Cell Carcinoma", "Normal", "Squamous Cell Carcinoma"]
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "efficientnet_lung_model.h5")

model = None
heatmap_store = {}

def load_ai_model():
    global model
    if model is None:
        import keras
        try:
            model = keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
        except Exception as e:
            print(f"Load Error: {e}")
    return model

# 3. SAFE PREPROCESSING FUNCTION
def preprocess_image(image_pil):
    """
    Ensures RGB consistency and official EfficientNet scaling.
    """
    # Force RGB to avoid transparency/greyscale issues
    img = image_pil.convert("RGB")
    # High-quality resize
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
    # Convert to array
    img_array = np.array(img).astype("float32")
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Official EfficientNet preprocessing
    img_array = preprocess_input(img_array)
    return img_array

def generate_heatmap_async(job_id, img_array_copy, original_cv2_img):
    global model
    try:
        # Create GradCAM model
        grad_model = tf.keras.models.Model(
            [model.inputs],
            [model.get_layer("top_conv").output, model.output]
        )
        
        with tf.GradientTape() as tape:
            # Thread-safe inference for GradCAM
            conv_outputs, predictions = grad_model(img_array_copy, training=False)
            pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
            
        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        
        heatmap_np = heatmap.numpy()
        heatmap_resized = cv2.resize(heatmap_np, (original_cv2_img.shape[1], original_cv2_img.shape[0]))
        heatmap_resized = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(original_cv2_img, 0.6, heatmap_color, 0.4, 0)
        
        _, buffer = cv2.imencode('.jpg', superimposed_img)
        b64_heatmap = base64.b64encode(buffer).decode('utf-8')
        heatmap_store[job_id] = f"data:image/jpeg;base64,{b64_heatmap}"
    except Exception as e:
        print(f"GradCAM Error: {e}")
        heatmap_store[job_id] = None

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
        
        # 4. COLOR PIPELINE FIX
        # Load once with PIL for RGB preprocessing
        image_pil = Image.open(io.BytesIO(img_bytes))
        
        # Preprocess for model
        img_array = preprocess_image(image_pil)
        
        # Create BGR copy for OpenCV GradCAM
        original_cv2_img = cv2.cvtColor(np.array(image_pil.convert("RGB")), cv2.COLOR_RGB2BGR)

        # 5. FIX INPUT STRUCTURE & THREAD SAFETY
        # Use np.copy() to prevent reference mutation in background thread
        inference_input = np.copy(img_array)
        
        # Use direct call with training=False for deterministic inference
        # Wrap in list to satisfy [['input_layer']] structure if needed
        preds_tensor = model(inference_input, training=False)
        preds = preds_tensor.numpy()[0]
        
        idx = np.argmax(preds)
        label = CLASS_NAMES[idx]
        confidence = float(preds[idx])

        job_id = str(uuid.uuid4())
        heatmap_store[job_id] = "processing"
        
        # 6. THREAD-SAFE GRADCAM CALL
        threading.Thread(
            target=generate_heatmap_async, 
            args=(job_id, np.copy(img_array), original_cv2_img)
        ).start()

        _, buffer_orig = cv2.imencode('.jpg', original_cv2_img)
        base64_image = f"data:image/jpeg;base64,{base64.b64encode(buffer_orig).decode('utf-8')}"

        result = {
            "class": label,
            "confidence": round(confidence * 100, 2),
            "job_id": job_id,
            "original_image": base64_image
        }

        # Explicit memory cleanup
        del preds
        del img_array
        del inference_input

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/heatmap/<job_id>', methods=['GET'])
def get_heatmap(job_id):
    status = heatmap_store.get(job_id)
    if status == "processing":
        return jsonify({"status": "processing"})
    elif status is None:
        return jsonify({"status": "failed"})
    else:
        return jsonify({"status": "completed", "heatmap_image": status})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
