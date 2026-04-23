# 5_predict.py

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

model = load_model("steganalysis_model.h5")

def predict_image(image_path):
    img = Image.open(image_path).resize((32, 32))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)[0][0]
    
    if prediction > 0.5:
        print(f"Result: 🔴 STEGO IMAGE (confidence: {prediction*100:.1f}%)")
    else:
        print(f"Result: 🟢 NORMAL IMAGE (confidence: {(1-prediction)*100:.1f}%)")

# Test it
predict_image("test_image.png")