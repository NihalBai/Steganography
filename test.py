"""
Phase 1 — LSB Steganography Baseline
Team: Person A (encoder/decoder) + Person B (metrics/testing)
"""

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


# ─────────────────────────────────────────────
# PERSON A — Encoder & Decoder
# ─────────────────────────────────────────────

def encode(image_path: str, secret_message: str, output_path: str) -> np.ndarray:
    """Hide a secret message inside a cover image using LSB."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Convert message to binary + add delimiter to mark end
    binary_message = ''.join(format(ord(c), '08b') for c in secret_message)
    binary_message += '1111111111111110'  # 16-bit end delimiter

    flat = image.flatten()

    if len(binary_message) > len(flat):
        raise ValueError(
            f"Message too long! Max {len(flat) // 8} chars for this image, "
            f"got {len(secret_message)} chars."
        )

    stego = flat.copy()
    for i, bit in enumerate(binary_message):
        stego[i] = (stego[i] & 0b11111110) | int(bit)  # replace LSB

    stego_image = stego.reshape(image.shape)
    cv2.imwrite(output_path, stego_image)
    print(f"[Encoder] Message hidden → saved to {output_path}")
    return stego_image


def decode(stego_path: str) -> str:
    """Extract the hidden message from a stego image."""
    image = cv2.imread(stego_path)
    if image is None:
        raise FileNotFoundError(f"Stego image not found: {stego_path}")

    flat = image.flatten()
    bits = [str(pixel & 1) for pixel in flat]

    chars = []
    for i in range(0, len(bits) - 16, 8):
        byte = ''.join(bits[i:i+8])
        next_byte = ''.join(bits[i+8:i+16])
        # Check for end delimiter
        if byte + next_byte == '1111111111111110':
            break
        chars.append(chr(int(byte, 2)))

    message = ''.join(chars)
    print(f"[Decoder] Recovered message: {message}")
    return message


# ─────────────────────────────────────────────
# PERSON B — Metrics & Quality Evaluation
# ─────────────────────────────────────────────

def evaluate(original_path: str, stego_path: str) -> dict:
    """Compute PSNR and SSIM between original and stego image."""
    original = cv2.imread(original_path)
    stego    = cv2.imread(stego_path)

    if original is None or stego is None:
        raise FileNotFoundError("Could not load one of the images.")
    if original.shape != stego.shape:
        raise ValueError("Images must have the same dimensions.")

    psnr_val = psnr(original, stego, data_range=255)

    # SSIM needs channel_axis for color images
    ssim_val = ssim(original, stego, channel_axis=2, data_range=255)

    capacity = (original.shape[0] * original.shape[1] * original.shape[2]) // 8

    print("\n── Quality Metrics ──────────────────────")
    print(f"  PSNR : {psnr_val:.2f} dB  (good if > 38 dB)")
    print(f"  SSIM : {ssim_val:.4f}     (good if > 0.95)")
    print(f"  Max capacity : {capacity} chars for this image")
    print("─────────────────────────────────────────\n")

    return {"psnr": psnr_val, "ssim": ssim_val, "capacity": capacity}


def visual_diff(original_path: str, stego_path: str, output_path: str = "diff.png"):
    """Save an amplified difference image to visualize hidden data."""
    original = cv2.imread(original_path).astype(np.int16)
    stego    = cv2.imread(stego_path).astype(np.int16)

    diff = np.abs(original - stego).astype(np.uint8)
    diff_amplified = cv2.convertScaleAbs(diff, alpha=50)  # amplify x50

    cv2.imwrite(output_path, diff_amplified)
    print(f"[Visual diff] Saved amplified diff → {output_path}")
    return diff_amplified


# ─────────────────────────────────────────────
# MAIN — Run the full pipeline
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import urllib.request, os

    # Download a free test image if you don't have one
    TEST_IMAGE = "cover.png"
    STEGO_IMAGE = "stego.png"
    SECRET = "Hello from steganography project! Phase 1 complete."

    if not os.path.exists(TEST_IMAGE):
    # Generate a random 512x512 color image
        img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        cv2.imwrite(TEST_IMAGE, img)
        print("Generated random cover.png")

    # Step 1 — Hide message
    stego = encode(TEST_IMAGE, SECRET, STEGO_IMAGE)

    # Step 2 — Recover message
    recovered = decode(STEGO_IMAGE)

    # Step 3 — Check accuracy
    assert recovered == SECRET, "ERROR: Recovered message doesn't match!"
    print("[Test] Encode → Decode: PASSED ✓")

    # Step 4 — Measure quality (Person B)
    metrics = evaluate(TEST_IMAGE, STEGO_IMAGE)

    # Step 5 — Visual diff
    visual_diff(TEST_IMAGE, STEGO_IMAGE, "diff_amplified.png")

    print("Phase 1 complete. Check stego.png and diff_amplified.png.")