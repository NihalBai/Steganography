import os
import json
import random
import string
import logging
import numpy as np
from PIL import Image
from pathlib import Path
from stegano import lsb

# ─── Configuration ────────────────────────────────────────────
INPUT_FOLDER  = "input"     # source clean images
OUTPUT_FOLDER = "out_steg"     # stego output
LOG_FILE      = "stego_log.json"
IMG_SIZE      = (256, 256)
MSG_LENGTH    = 200
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ─── Logging setup ────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("stego_process.log"),
        logging.StreamHandler()
    ]
)

# ─── Helper functions ─────────────────────────────────────────
def random_text(length=MSG_LENGTH):
    """Generate random alphanumeric secret message."""
    return ''.join(random.choices(
        string.ascii_letters + string.digits, k=length
    ))

def preprocess_image(path):
    """Resize and convert image to RGB before embedding."""
    img = Image.open(path).convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    return img

def verify_stego(stego_path, expected_message):
    """Decode stego image and verify hidden message survived."""
    try:
        decoded = lsb.reveal(stego_path)
        return decoded == expected_message
    except Exception:
        return False

def get_file_size(path):
    """Return file size in KB."""
    return round(os.path.getsize(path) / 1024, 2)

def pixel_diff(original_path, stego_path):
    """Compute mean absolute pixel difference between original and stego."""
    try:
        orig  = np.array(Image.open(original_path).convert("RGB"), dtype=np.float32)
        stego = np.array(Image.open(stego_path).convert("RGB"),    dtype=np.float32)
        return round(float(np.mean(np.abs(orig - stego))), 6)
    except Exception:
        return None

# ─── Main processing ──────────────────────────────────────────
records = []
success = 0
failed  = 0

image_files = [
    f for f in os.listdir(INPUT_FOLDER)
    if Path(f).suffix.lower() in SUPPORTED_EXT
]

if not image_files:
    logging.warning("No supported images found in input folder.")
else:
    logging.info(f"Found {len(image_files)} images to process.")

for filename in image_files:
    input_path  = os.path.join(INPUT_FOLDER, filename)
    output_name = Path(filename).stem + ".png"   # always PNG
    output_path = os.path.join(OUTPUT_FOLDER, output_name)
    temp_path   = os.path.join(OUTPUT_FOLDER, "_temp.png")

    try:
        # 1. Preprocess image
        img = preprocess_image(input_path)
        img.save(temp_path)

        # 2. Embed secret message
        secret_message = random_text()
        stego_img = lsb.hide(temp_path, secret_message)
        stego_img.save(output_path)
        os.remove(temp_path)

        # 3. Verify embedding
        verified = verify_stego(output_path, secret_message)

        # 4. Compute pixel difference
        diff = pixel_diff(input_path, output_path)

        # 5. Log result
        record = {
            "filename":      filename,
            "output":        output_name,
            "message":       secret_message,
            "verified":      verified,
            "original_kb":   get_file_size(input_path),
            "stego_kb":      get_file_size(output_path),
            "pixel_diff":    diff,
            "status":        "success" if verified else "verify_failed"
        }
        records.append(record)

        if verified:
            success += 1
            logging.info(f"✓ {filename} → {output_name} | diff={diff} | verified")
        else:
            logging.warning(f"⚠ {filename} → embedding not verified")

    except Exception as e:
        failed += 1
        logging.error(f"✗ {filename} → {e}")
        records.append({
            "filename": filename,
            "status":   "failed",
            "error":    str(e)
        })

# ─── Save JSON log ────────────────────────────────────────────
with open(LOG_FILE, "w") as f:
    json.dump(records, f, indent=2)

# ─── Final summary ────────────────────────────────────────────
total = len(image_files)
print("\n" + "="*45)
print("✅ STEGANO COMPLETE")
print("="*45)
print(f"  Total processed : {total}")
print(f"  Success         : {success}")
print(f"  Failed          : {failed}")
print(f"  Output folder   : {OUTPUT_FOLDER}")
print(f"  JSON log        : {LOG_FILE}")
print(f"  Process log     : stego_process.log")
print("="*45)