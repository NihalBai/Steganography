# 4_evaluate.py

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

model = load_model("steganalysis_model.h5")

# Make predictions
y_pred = (model.predict(X_test) > 0.5).astype(int)

# ── Classification Report ─────────────────
print("Classification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=["Normal", "Stego"]
))

# ── Confusion Matrix ──────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=["Normal","Stego"],
            yticklabels=["Normal","Stego"])
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.savefig("confusion_matrix.png")
plt.show()

# ── Training History ──────────────────────
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss')
plt.legend()

plt.savefig("training_history.png")
plt.show()