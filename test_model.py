import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from scipy.ndimage import convolve

# =========================
# 🔥 TLU (same as training)
# =========================
def tlu(x):
    return tf.clip_by_value(x, -3.0, 3.0)

# =========================
# 🔥 SRM FILTER (MUST MATCH TRAINING)
# =========================
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

# =========================
# LOAD MODEL
# =========================
model = keras.models.load_model(
    "final_model.keras",
    custom_objects={"tlu": tlu}
)

IMG_SIZE = 256

# =========================
# PREPROCESS (MATCH TRAINING)
# =========================
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img = image.img_to_array(img) / 255.0  # normalize
    img = apply_srm(img)                  # 🔥 CRITICAL
    return np.expand_dims(img, axis=0)

# =========================
# TEST FOLDER
# =========================
test_folder = "output/"

print("\n===== STEGANOGRAPHY MODEL TEST =====\n")

for img_name in os.listdir(test_folder):
    path = os.path.join(test_folder, img_name)

    img = preprocess_image(path)
    pred = model.predict(img, verbose=0)[0][0]

    label = "Stego (1)" if pred > 0.5 else "Clean (0)"

    print(f"{img_name} -> {label} | score: {pred:.4f}")