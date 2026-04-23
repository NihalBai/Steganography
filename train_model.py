import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten,
                                     Dense, Dropout, BatchNormalization)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ─────────────────────────────────────────────
# CONFIG  (matched to clean.py output)
# ─────────────────────────────────────────────
IMG_SIZE    = 256          # clean.py saves at 256x256
BATCH_SIZE  = 32
EPOCHS      = 50
NORMAL_DIR  = "dataset/clean"   # ← clean.py output
STEGO_DIR   = "dataset/stego"   # ← clean.py output
MODEL_PATH  = "steganalysis_model.h5"

# ─────────────────────────────────────────────
# STEP 1 — LOAD IMAGES  (NO RESIZE!)
# ─────────────────────────────────────────────
print("Loading images...")

images = []
labels = []

def load_images_from_folder(folder, label):
    count = 0
    for fname in os.listdir(folder):
        if not fname.lower().endswith(".png"):
            continue
        path = os.path.join(folder, fname)
        try:
            img = Image.open(path).convert("RGB")
            # Only resize if image is not already correct size
            if img.size != (IMG_SIZE, IMG_SIZE):
                img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
            arr = np.array(img, dtype=np.float32) / 255.0
            images.append(arr)
            labels.append(label)
            count += 1
        except Exception as e:
            print(f"  Skipped {fname}: {e}")
    print(f"  Loaded {count} images from '{folder}' → label={label}")

load_images_from_folder(NORMAL_DIR, label=0)
load_images_from_folder(STEGO_DIR,  label=1)

X = np.array(images)
y = np.array(labels)

print(f"\nTotal images : {len(X)}")
print(f"Normal (0)   : {np.sum(y == 0)}")
print(f"Stego  (1)   : {np.sum(y == 1)}")
print(f"Image shape  : {X[0].shape}")

# ─────────────────────────────────────────────
# STEP 2 — SPLIT DATA
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y          # keep balance between classes
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train,
    test_size=0.1,
    random_state=42,
    stratify=y_train
)

print(f"\nTrain : {len(X_train)}")
print(f"Val   : {len(X_val)}")
print(f"Test  : {len(X_test)}")

# ─────────────────────────────────────────────
# STEP 3 — DATA AUGMENTATION (only on train)
# ─────────────────────────────────────────────
# NOTE: Only horizontal/vertical flips — NO zoom, NO rotation
# Rotation/zoom would destroy LSB pixel patterns
datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True
)
datagen.fit(X_train)

# ─────────────────────────────────────────────
# STEP 4 — BUILD CNN
# ─────────────────────────────────────────────
print("\nBuilding model...")

model = Sequential([

    # ── Block 1 ──────────────────────────────
    Conv2D(32, (3,3), activation='relu', padding='same',
           input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    BatchNormalization(),
    Conv2D(32, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),

    # ── Block 2 ──────────────────────────────
    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),

    # ── Block 3 ──────────────────────────────
    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.3),

    # ── Block 4 ──────────────────────────────
    Conv2D(256, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.3),

    # ── Classifier ───────────────────────────
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(1, activation='sigmoid')   # 0=Normal  1=Stego
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ─────────────────────────────────────────────
# STEP 5 — CALLBACKS
# ─────────────────────────────────────────────
callbacks = [
    # Stop early if no improvement
    EarlyStopping(
        monitor='val_accuracy',
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),
    # Reduce learning rate when stuck
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1
    )
]

# ─────────────────────────────────────────────
# STEP 6 — TRAIN
# ─────────────────────────────────────────────
print("\nTraining started...")

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

# ─────────────────────────────────────────────
# STEP 7 — EVALUATE
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("EVALUATION ON TEST SET")
print("="*50)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy : {acc*100:.2f}%")
print(f"Test Loss     : {loss:.4f}")

y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=["Normal", "Stego"]
))

# ─────────────────────────────────────────────
# STEP 8 — SAVE MODEL
# ─────────────────────────────────────────────
model.save(MODEL_PATH)
print(f"\nModel saved → {MODEL_PATH}")

# ─────────────────────────────────────────────
# STEP 9 — PLOTS
# ─────────────────────────────────────────────

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Normal","Stego"],
            yticklabels=["Normal","Stego"])
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
print("Saved → confusion_matrix.png")

# Training History
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['accuracy'],     label='Train Accuracy')
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
axes[0].set_title('Accuracy over Epochs')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history.history['loss'],     label='Train Loss')
axes[1].plot(history.history['val_loss'], label='Val Loss')
axes[1].set_title('Loss over Epochs')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("training_history.png")
plt.show()
print("Saved → training_history.png")

print("\nDone! ✅")