import os
import numpy as np
from PIL import Image
from scipy.ndimage import convolve
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
import io
import base64

app = Flask(__name__)

# ── REGISTER CUSTOM ACTIVATION ───────────────────────────────
@tf.keras.utils.register_keras_serializable()
def tlu(x):
    """Truncated Linear Unit — clips activations to [-3, 3]"""
    return tf.clip_by_value(x, -3.0, 3.0)

# ── CONFIG ───────────────────────────────────────────────────
IMG_SIZE = 256
MODEL_PATH = os.environ.get("MODEL_PATH", "final_model.keras")

# ── SRM FILTER BANK ──────────────────────────────────────────
SRM_KERNELS = [
    np.array([[-1,-1,-1],[-1, 8,-1],[-1,-1,-1]], np.float32) / 8.0,
    np.array([[ 0, 0, 0],[-1, 2,-1],[ 0, 0, 0]], np.float32) / 2.0,
    np.array([[ 0,-1, 0],[ 0, 2, 0],[ 0,-1, 0]], np.float32) / 2.0,
    np.array([[-1, 0, 0],[ 0, 2, 0],[ 0, 0,-1]], np.float32) / 2.0,
    np.array([[ 0, 0,-1],[ 0, 2, 0],[-1, 0, 0]], np.float32) / 2.0,
    np.array([[ 0,-1, 0],[-1, 4,-1],[ 0,-1, 0]], np.float32) / 4.0,
]

def apply_srm(arr):
    out = np.mean([
        np.stack([convolve(arr[:,:,c], k) for c in range(3)], axis=-1)
        for k in SRM_KERNELS
    ], axis=0)
    out -= out.min()
    mx = out.max()
    if mx > 0:
        out /= mx
    return out.astype(np.float32)

def preprocess_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    original_size = img.size
    if img.size != (IMG_SIZE, IMG_SIZE):
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    filtered = apply_srm(arr)
    return np.expand_dims(filtered, axis=0), original_size

# ── LOAD MODEL ───────────────────────────────────────────────
print(f"Loading model from: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"tlu": tlu})
print("Model loaded successfully!")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    allowed = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        return jsonify({"error": f"Unsupported format: {ext}"}), 400

    img_bytes = file.read()

    img_for_info = Image.open(io.BytesIO(img_bytes))
    width, height = img_for_info.size
    file_size_kb = len(img_bytes) / 1024

    tensor, _ = preprocess_image(img_bytes)
    prob = float(model.predict(tensor, verbose=0).flatten()[0])
    label = "Stego" if prob >= 0.5 else "Clean"
    confidence = prob if label == "Stego" else (1.0 - prob)

    preview = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    preview.thumbnail((400, 400), Image.LANCZOS)
    buf = io.BytesIO()
    preview.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return jsonify({
        "label": label,
        "probability": round(prob, 4),
        "confidence": round(confidence * 100, 1),
        "preview": img_b64,
        "filename": file.filename,
        "width": width,
        "height": height,
        "file_size_kb": round(file_size_kb, 1),
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
