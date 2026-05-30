# StegoScan — Local Setup Guide

## Project Structure
stego_detector/
├── app.py
├── requirements.txt
├── final_model.keras       ← PUT YOUR MODEL HERE
└── templates/
    └── index.html

## Steps

### 1. Install Python 3.10+
Make sure Python is installed: https://www.python.org/downloads/

### 2. Create a virtual environment
```bash
python -m venv venv
```

### 3. Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Place your model file
Copy your `final_model.keras` file into the `stego_detector/` folder.

### 6. Run the app
```bash
python app.py
```

### 7. Open in browser
Go to: http://localhost:5000

## Notes
- If your model file has a different name, set the env variable:
  MODEL_PATH=my_model.keras python app.py
- Supported image formats: PNG, JPG, JPEG, BMP, WEBP
