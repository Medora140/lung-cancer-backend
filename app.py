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
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

IMG_SIZE = 224
CLASS_NAMES = ["Adenocarcinoma", "Large Cell Carcinoma", "Normal", "Squamous Cell Carcinoma"]
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "efficientnet_lung_model.h5"
)

model = None

def load_ai_model():
    global model
    if model is None:
        import keras
        print("Loading AI model into RAM...")
        try:
            model = keras.models.load_model(
                MODEL_PATH,
                compile=False,
                safe_mode=False
            )
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Critical Model Load Error: {e}")
    return model

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
    except:
        return None

def preprocess_image(image):
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    rgb_image = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    img_array = np.expand_dims(rgb_image, axis=0)
    return img_array.astype('float32') / 255.0, image_cv

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    global model
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    
    file = request.files['file']
    
    if model is None:
        load_ai_model()
        if model is None:
            return jsonify({'error': 'Model unavailable'}), 500

    try:
        print("Running prediction...")
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_array, original_cv2_img = preprocess_image(image)

        preds = model.predict(img_array, verbose=0)[0]
        idx = np.argmax(preds)
        label = CLASS_NAMES[idx]
        confidence = float(preds[idx])

        heatmap = make_gradcam_heatmap(img_array, model, "top_conv")
        
        response_data = {
            'class': label,
            'confidence': round(confidence * 100, 2),
            'all_predictions': {CLASS_NAMES[i]: round(float(preds[i]) * 100, 2) for i in range(len(CLASS_NAMES))}
        }

        if heatmap is not None:
            heatmap_resized = cv2.resize(heatmap, (original_cv2_img.shape[1], original_cv2_img.shape[0]))
            heatmap_resized = np.uint8(255 * heatmap_resized)
            heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
            superimposed_img = cv2.addWeighted(original_cv2_img, 0.6, heatmap_color, 0.4, 0)
            _, buffer_heat = cv2.imencode('.jpg', superimposed_img)
            response_data['heatmap_image'] = f"data:image/jpeg;base64,{base64.b64encode(buffer_heat).decode('utf-8')}"

        _, buffer_orig = cv2.imencode('.jpg', original_cv2_img)
        response_data['original_image'] = f"data:image/jpeg;base64,{base64.b64encode(buffer_orig).decode('utf-8')}"

        return jsonify(response_data)

    except Exception as e:
        print("Prediction error:", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
