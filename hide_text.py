import os
import random
import string
from stegano import lsb

input_folder = "images"
output_folder = "output"

os.makedirs(output_folder, exist_ok=True)

def random_text(length=20):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

for filename in os.listdir(input_folder):
    if filename.endswith(".png"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        secret_message = random_text()

        secret_img = lsb.hide(input_path, secret_message)
        secret_img.save(output_path)

        print(f"{filename} -> hidden text: {secret_message}")