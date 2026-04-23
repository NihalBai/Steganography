import sys
from stegano import lsb

if len(sys.argv) != 2:
    print("Usage: python detect.py <image_file>")
    exit()

image_path = sys.argv[1]

try:
    hidden = lsb.reveal(image_path)

    if hidden:
        print("🟥 STEGO FOUND")
        print("Hidden message:", hidden)
    else:
        print("🟩 CLEAN (no hidden data found)")

except Exception as e:
    print("🟩 CLEAN (or not readable stego format)")