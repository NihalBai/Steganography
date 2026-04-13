import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# ── Step 1: Load data ──────────────────────────────────────
X = np.load("features.npy")
y = np.load("labels.npy")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Reshape: (N, 16384) → (N, 1, 128, 128) for CNN
X_train = X_train.reshape(-1, 1, 128, 128)
X_test  = X_test.reshape(-1, 1, 128, 128)

# Convert to tensors
train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long))
test_ds  = TensorDataset(torch.tensor(X_test),  torch.tensor(y_test,  dtype=torch.long))

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=32)

print(f"Train: {X_train.shape} | Test: {X_test.shape}")

# ── Step 2: CNN Model ──────────────────────────────────────
class StegCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.MaxPool2d(2),                             # 64x64

            # Block 2
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.MaxPool2d(2),                             # 32x32

            # Block 3
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.MaxPool2d(2),                             # 16x16
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 64),            nn.ReLU(),
            nn.Linear(64, 2)               # cover or stego
        )

    def forward(self, x):
        return self.classifier(self.features(x))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model     = StegCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# ── Step 3: Train ──────────────────────────────────────────
EPOCHS = 10
train_losses, train_accs = [], []

print("\nTraining CNN...")
for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out  = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct    += (out.argmax(1) == yb).sum().item()
        total      += len(yb)

    scheduler.step()
    acc = correct / total
    train_losses.append(total_loss / len(train_loader))
    train_accs.append(acc)
    print(f"Epoch {epoch+1:02d}/{EPOCHS} — Loss: {train_losses[-1]:.4f} | Acc: {acc:.3f}")

# ── Step 4: Evaluate ───────────────────────────────────────
model.eval()
all_preds, all_labels, all_probs = [], [], []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        out   = model(xb)
        probs = torch.softmax(out, dim=1)[:, 1]
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_labels.extend(yb.numpy())
        all_probs.extend(probs.cpu().numpy())

print("\n── Classification Report ─────────────────────────")
print(classification_report(all_labels, all_preds,
      target_names=["Cover", "Stego"]))

# ── Step 5: Plots ──────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Loss curve
axes[0].plot(train_losses, marker="o")
axes[0].set_title("Training Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")

# Accuracy curve
axes[1].plot(train_accs, marker="o", color="green")
axes[1].set_title("Training Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[2],
            xticklabels=["Cover", "Stego"],
            yticklabels=["Cover", "Stego"])
axes[2].set_title("Confusion Matrix")
axes[2].set_ylabel("Actual")
axes[2].set_xlabel("Predicted")

plt.tight_layout()
plt.savefig("cnn_results.png")
print("Saved: cnn_results.png")

# ROC Curve
fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Stego CNN")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png")
print("Saved: roc_curve.png")

# Save model
torch.save(model.state_dict(), "steg_cnn.pth")
print(f"\nModel saved: steg_cnn.pth")
print(f"AUC Score  : {roc_auc:.3f}")