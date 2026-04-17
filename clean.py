import os
import json
import hashlib
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

# ─── Configuration ────────────────────────────────────────────
CLEAN_FOLDER  = "input"     # source clean images
STEGO_FOLDER  = "out_steg"     # stego output
OUTPUT_CLEAN  = "dataset_clean/clean"
OUTPUT_STEGO  = "dataset_clean/stego"
REPORT_FILE   = "cleaning_report.csv"
IMG_SIZE      = (256, 256)
MIN_FILE_SIZE = 1
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

os.makedirs(OUTPUT_CLEAN, exist_ok=True)
os.makedirs(OUTPUT_STEGO, exist_ok=True)

# ─── Step 1: Scan ─────────────────────────────────────────────
def scan_images(folder, label):
    records = []
    for filename in os.listdir(folder):
        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_EXT:
            continue
        filepath = os.path.join(folder, filename)
        records.append({
            "filename":     filename,
            "filepath":     filepath,
            "label":        label,
            "ext":          ext,
            "file_size_kb": round(os.path.getsize(filepath) / 1024, 2)
        })
    return records

print("📂 Scanning images...")
records = scan_images(CLEAN_FOLDER, label=0) + scan_images(STEGO_FOLDER, label=1)
df = pd.DataFrame(records)
print(f"   Total found: {len(df)} images\n")

# ─── Step 2: Remove unreadable ────────────────────────────────
print("🔍 Checking for corrupted files...")

def try_open(filepath):
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True
    except Exception:
        return False

df["readable"] = df["filepath"].apply(try_open)
print(f"   Unreadable: {(~df['readable']).sum()}")
df = df[df["readable"]].copy()

# ─── Step 3: Remove too-small files ───────────────────────────
print("\n🗑️  Removing empty files...")
before = len(df)
df = df[df["file_size_kb"] >= MIN_FILE_SIZE]
print(f"   Removed: {before - len(df)}")

# ─── Step 4: Load image properties ───────────────────────────
print("\n📐 Loading image properties...")

def get_image_props(filepath):
    try:
        with Image.open(filepath) as img:
            w, h     = img.size
            mode     = img.mode
            channels = len(img.getbands())
            arr      = np.array(img.convert("RGB")).tobytes()
            md5      = hashlib.md5(arr).hexdigest()
            return w, h, mode, channels, md5
    except Exception:
        return None, None, None, None, None

props = df["filepath"].apply(lambda p: pd.Series(
    get_image_props(p),
    index=["width", "height", "mode", "channels", "md5"]
))
df = pd.concat([df, props], axis=1)

# ─── Step 5: Drop nulls ───────────────────────────────────────
print("\n🧹 Dropping null rows...")
before = len(df)
df = df.dropna()
print(f"   Removed: {before - len(df)} | Remaining: {len(df)}")

# ─── Step 6: Remove duplicates ────────────────────────────────
print("\n👯 Removing duplicates...")
before = len(df)
df = df.drop_duplicates(subset="md5", keep="first")
print(f"   Removed: {before - len(df)} duplicates")

# ─── Step 7: Flag wrong sizes ─────────────────────────────────
print("\n📏 Checking sizes...")
df["correct_size"] = (df["width"] == IMG_SIZE[0]) & (df["height"] == IMG_SIZE[1])
print(f"   Need resize: {(~df['correct_size']).sum()}")

# ─── Step 8: Flag wrong color modes ───────────────────────────
print("\n🎨 Checking color modes...")
print(df["mode"].value_counts().to_string())
df["needs_conversion"] = df["mode"] != "RGB"
print(f"   Need RGB conversion: {df['needs_conversion'].sum()}")

# ─── Step 9: Class balance ────────────────────────────────────
print("\n⚖️  Class balance...")
balance = df["label"].value_counts()
print(f"   Clean (0): {balance.get(0, 0)}")
print(f"   Stego (1): {balance.get(1, 0)}")
if abs(balance.get(0,0) - balance.get(1,0)) > 10:
    print("   ⚠️  Class imbalance detected!")
else:
    print("   ✓ Balanced")

# ─── Step 10: Normalize & save ────────────────────────────────
print("\n💾 Normalizing and saving...")

def normalize_and_save(row):
    try:
        img      = Image.open(row["filepath"]).convert("RGB")
        img      = img.resize(IMG_SIZE, Image.LANCZOS)
        arr      = np.array(img, dtype=np.float32) / 255.0
        arr_u8   = (arr * 255).astype(np.uint8)
        img_out  = Image.fromarray(arr_u8)
        folder   = OUTPUT_CLEAN if row["label"] == 0 else OUTPUT_STEGO
        out_path = os.path.join(folder, Path(row["filename"]).stem + ".png")
        img_out.save(out_path, "PNG")
        return "success"
    except Exception as e:
        return f"failed: {e}"

df["save_status"] = df.apply(normalize_and_save, axis=1)
print(f"   Saved: {(df['save_status'] == 'success').sum()} / {len(df)}")

# ─── Step 11: Pixel statistics ────────────────────────────────
print("\n📊 Computing pixel statistics...")

def pixel_stats(filepath):
    try:
        arr = np.array(Image.open(filepath).convert("L"), dtype=np.float32)
        return round(float(arr.mean()), 4), round(float(arr.std()), 4)
    except Exception:
        return None, None

stats = df["filepath"].apply(lambda p: pd.Series(
    pixel_stats(p), index=["pixel_mean", "pixel_std"]
))
df = pd.concat([df, stats], axis=1)
print(df.groupby("label")[["pixel_mean","pixel_std"]].mean().round(4).to_string())

# ─── Step 12: Save CSV report ─────────────────────────────────
report_cols = ["filename","label","ext","file_size_kb",
               "width","height","mode","channels",
               "correct_size","needs_conversion",
               "pixel_mean","pixel_std","save_status"]
df[report_cols].to_csv(REPORT_FILE, index=False)
print(f"\n📝 Report saved → {REPORT_FILE}")

# ─── Step 13: Visual summary ──────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Dataset Cleaning Report", fontsize=14)

axes[0,0].bar(["Clean (0)", "Stego (1)"],
              [balance.get(0,0), balance.get(1,0)],
              color=["steelblue","coral"])
axes[0,0].set_title("Class Distribution")

df.groupby("label")["file_size_kb"].plot(
    kind="hist", bins=20, alpha=0.6, ax=axes[0,1], legend=True)
axes[0,1].set_title("File Size Distribution (KB)")
axes[0,1].legend(["Clean","Stego"])

df.boxplot(column="pixel_mean", by="label", ax=axes[1,0])
axes[1,0].set_title("Pixel Mean per Class")
axes[1,0].set_xlabel("Label")

df.boxplot(column="pixel_std", by="label", ax=axes[1,1])
axes[1,1].set_title("Pixel Std per Class")
axes[1,1].set_xlabel("Label")

plt.tight_layout()
plt.savefig("cleaning_summary.png", dpi=150)
plt.show()

# ─── Final summary ────────────────────────────────────────────
print("\n" + "="*45)
print("✅ CLEANING COMPLETE")
print("="*45)
print(f"  Total processed : {len(df)}")
print(f"  Clean images    : {balance.get(0,0)}")
print(f"  Stego images    : {balance.get(1,0)}")
print(f"  Output folder   : dataset_clean/")
print(f"  Report          : {REPORT_FILE}")
print(f"  Plot            : cleaning_summary.png")
print("="*45)