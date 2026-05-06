import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

import io
import base64
import numpy as np
import cv2
import threading
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

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
            print(f"Error: {e}")
    return model

def generate_heatmap_async(job_id, img_array, original_cv2_img):
    global model
    try:
        # GradCAM logic
        grad_model = tf.keras.models.Model(
            [model.inputs],
            [model.get_layer("top_conv").output, model.output]
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
        
        # Processing heatmap
        heatmap_np = heatmap.numpy()
        heatmap_resized = cv2.resize(heatmap_np, (original_cv2_img.shape[1], original_cv2_img.shape[0]))
        heatmap_resized = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(original_cv2_img, 0.6, heatmap_color, 0.4, 0)
        
        _, buffer = cv2.imencode('.jpg', superimposed_img)
        b64_heatmap = base64.b64encode(buffer).decode('utf-8')
        heatmap_store[job_id] = f"data:image/jpeg;base64,{b64_heatmap}"
    except Exception as e:
        print(f"Async GradCAM Error: {e}")
        heatmap_store[job_id] = None

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
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        original_cv2_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Optimized preprocessing
        img_array = np.array(image.resize((IMG_SIZE, IMG_SIZE))).astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array, verbose=0)[0]
        idx = np.argmax(preds)
        label = CLASS_NAMES[idx]
        confidence = float(preds[idx])

        job_id = str(uuid.uuid4())
        heatmap_store[job_id] = "processing"
        
        # Start background task
        threading.Thread(target=generate_heatmap_async, args=(job_id, img_array, original_cv2_img)).start()

        _, buffer_orig = cv2.imencode('.jpg', original_cv2_img)
        base64_image = f"data:image/jpeg;base64,{base64.b64encode(buffer_orig).decode('utf-8')}"

        result = {
            "class": label,
            "confidence": round(confidence * 100, 2),
            "job_id": job_id,
            "original_image": base64_image
        }

        # Memory Cleanup
        del preds
        del img_array

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
