import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# ── CONFIG ─────────────────────────────────────────────────
COVER_DIR     = "./BOSSbase_1.01"
STEGO_DIR     = "./BOSSbase_1.01_stego"
MAX_PER_CLASS = 500    # safe limit — increase later if RAM allows
IMG_SIZE      = 128    # 128x128 is enough for classification
BATCH_SIZE    = 50     # process 50 at a time

# ── Step 1: Generate stego if missing ──────────────────────
if not os.path.exists(STEGO_DIR) or len(os.listdir(STEGO_DIR)) == 0:
    os.makedirs(STEGO_DIR, exist_ok=True)
    files = [f for f in sorted(os.listdir(COVER_DIR)) if f.endswith(".pgm")]
    print(f"Generating {min(len(files), MAX_PER_CLASS)} stego images...")
    for fname in files[:MAX_PER_CLASS]:
        img = cv2.imread(os.path.join(COVER_DIR, fname), cv2.IMREAD_GRAYSCALE)
        secret = np.random.randint(0, 2, img.shape, dtype=np.uint8)
        stego = (img & 0b11111110) | secret
        cv2.imwrite(os.path.join(STEGO_DIR, fname), stego)
    print("Stego images generated!")

# ── Step 2: Build path list only (no images in RAM yet) ────
records = []
for fname in sorted(os.listdir(COVER_DIR))[:MAX_PER_CLASS]:
    if fname.endswith(".pgm"):
        records.append({"filepath": os.path.join(COVER_DIR, fname), "label": 0})
for fname in sorted(os.listdir(STEGO_DIR))[:MAX_PER_CLASS]:
    if fname.endswith(".pgm"):
        records.append({"filepath": os.path.join(STEGO_DIR, fname), "label": 1})

df = pd.DataFrame(records)
print(f"Files found — cover: {(df.label==0).sum()}, stego: {(df.label==1).sum()}")

# ── Step 3: Load + extract in small batches ─────────────── 
def extract(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img.flatten().astype("float32") / 255.0  # normalize here

print("Loading in batches...")
all_features = []
for i in range(0, len(df), BATCH_SIZE):
    batch = df["filepath"].iloc[i:i+BATCH_SIZE]
    all_features.append(np.array([extract(fp) for fp in batch]))
    print(f"  {min(i+BATCH_SIZE, len(df))}/{len(df)} images loaded", end="\r")

X = np.vstack(all_features)
y = df["label"].values
print(f"\nDone. X: {X.shape}, y: {y.shape}")

# ── Step 4: Save so you never reload again ─────────────────
np.save("features.npy", X)
np.save("labels.npy",   y)
print("Saved features.npy and labels.npy")

# ── Step 5: Split ──────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape} | Test: {X_test.shape}")
print("Ready for training!")