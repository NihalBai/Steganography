import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# ── CONFIG — change these to control memory usage ──────────
MAX_PER_CLASS = 500   # 500 cover + 500 stego = 1000 total
IMG_SIZE      = 128   # 128x128 instead of 256x256 (4x less RAM)
BATCH_SIZE    = 100   # process 100 images at a time
COVER_DIR     = "./BOSSbase_1.01"
STEGO_DIR     = "./BOSSbase_1.01_stego"

# ── Step 1: Build DataFrame (paths only, no images) ────────
def build_dataframe(cover_dir, stego_dir, max_per_class):
    records = []
    for fname in sorted(os.listdir(cover_dir))[:max_per_class]:
        if fname.endswith(".pgm"):
            records.append({"filepath": os.path.join(cover_dir, fname),
                            "filename": fname, "label": 0, "class": "cover"})
    for fname in sorted(os.listdir(stego_dir))[:max_per_class]:
        if fname.endswith(".pgm"):
            records.append({"filepath": os.path.join(stego_dir, fname),
                            "filename": fname, "label": 1, "class": "stego"})
    return pd.DataFrame(records)

df = build_dataframe(COVER_DIR, STEGO_DIR, MAX_PER_CLASS)
print(f"Total images: {len(df)}")
print(df["class"].value_counts())

# ── Step 2: Clean — check files without loading all ────────
def is_valid(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    return img is not None

print("\nChecking for corrupted images...")
df["valid"] = df["filepath"].apply(is_valid)
print(f"Corrupted/missing: {df['valid'].eq(False).sum()}")
df = df[df["valid"]].drop(columns="valid").reset_index(drop=True)
print(f"Clean images: {len(df)}")

# ── Step 3: Extract features in batches (safe RAM) ─────────
def extract_features(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img.flatten().astype("float32")

print("\nLoading in batches...")
all_features = []

for i in range(0, len(df), BATCH_SIZE):
    batch_paths = df["filepath"].iloc[i:i+BATCH_SIZE]
    batch_feats = np.array([extract_features(fp) for fp in batch_paths])
    all_features.append(batch_feats)
    print(f"  Batch {i//BATCH_SIZE + 1} done — {min(i+BATCH_SIZE, len(df))}/{len(df)} images", end="\r")
    
features = np.vstack(all_features)
print(f"\nFeatures shape: {features.shape}")

# ── Step 4: Normalize ──────────────────────────────────────
print("Normalizing...")
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# ── Step 5: Lightweight DataFrame (no pixel columns) ───────
df_meta = df[["filename", "label", "class"]].copy()
df_meta["mean_pixel"]  = features_scaled.mean(axis=1).round(4)
df_meta["std_pixel"]   = features_scaled.std(axis=1).round(4)
df_meta["min_pixel"]   = features_scaled.min(axis=1).round(4)
df_meta["max_pixel"]   = features_scaled.max(axis=1).round(4)

print(f"\nMissing values: {df_meta.isnull().sum().sum()}")
print(f"Duplicates:     {df_meta.duplicated().sum()}")
print(df_meta.head())

# ── Step 6: Save ───────────────────────────────────────────
df_meta.to_csv("dataset_clean.csv", index=False)
np.save("features.npy",  features_scaled)   # save array separately
np.save("labels.npy",    df["label"].values)
print("\nSaved: dataset_clean.csv / features.npy / labels.npy")

# ── Step 7: Split ──────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    features_scaled, df["label"].values,
    test_size=0.2, random_state=42, stratify=df["label"].values
)

print(f"\nTrain: {X_train.shape} | Test: {X_test.shape}")
print("Done — ready for training!")