# StegoScan — PVD Steganography Detector

> A deep learning system for detecting **Pixel Value Differencing (PVD) steganography** in images, built with a custom Residual CNN and SRM filter bank preprocessing.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Training (Kaggle)](#training-kaggle)
- [Dataset](#dataset)
- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)

---

## Overview

StegoScan is a steganalysis tool trained to classify images as **clean** or **stego** (containing a hidden message embedded via PVD). It was developed as a Master's capstone project in AI & Cybersecurity.

Key highlights:
- **~95.78%** validation accuracy | **AUC 0.9915**
- **~83.08%** test accuracy | **AUC 0.9579** on 4,000 held-out images
- Custom preprocessing using **SRM (Spatial Rich Model)** filter bank to expose noise residuals
- Flask web interface for real-time inference via drag-and-drop upload

---

## Architecture

```
Input Image (256×256 RGB)
        │
        ▼
  SRM Filter Bank          ← 6 high-pass kernels; removes content, exposes noise
        │
        ▼
  Residual CNN              ← Custom from-scratch (not pretrained)
  + TLU Activation          ← Truncated Linear Unit: clips to [-3, 3]
        │
        ▼
  Sigmoid Output            ← 0 = Clean, 1 = Stego
```

> **Why SRM?** Raw RGB input achieved only ~57% accuracy. SRM filters strip color/content information, making subtle PVD artifacts detectable.

> **Why a custom model?** Transfer learning (MobileNetV2 pretrained on ImageNet) degraded performance — steganalysis requires features invisible to natural image classifiers.

---

## Results

| Split      | Accuracy | AUC    |
|------------|----------|--------|
| Validation | 95.78%   | 0.9915 |
| Test       | 83.08%   | 0.9579 |

**Test classification report:**

| Class | Precision | Recall | F1   |
|-------|-----------|--------|------|
| Clean | 1.00      | 0.66   | 0.80 |
| Stego | 0.75      | 1.00   | 0.86 |

The model is highly sensitive (recall 1.00 on stego), with some false positives on clean images.

---

## Repository Structure

```
StegoScan/
│
├── webapp/
│   └── stego_detector/        # Flask web application
│       ├── app.py             # Routes & inference logic
│       ├── final_model.keras  # Trained model (not in repo — see below)
│       ├── templates/         # HTML UI (drag-and-drop upload)
│       └── static/            # CSS / JS assets
│
├── notebooks/
│   ├── bestver95.ipynb        # Full training pipeline (Kaggle, P100 GPU)
│   └── testnotebook.ipynb     # Evaluation on held-out test set
│
├── samples/                   # 4 sample images to try with the web app
│   ├── clean.png              # Clean image example
│   ├── clean2.png             # Clean image example
│   ├── seg.png                # Stego image example
│   └── seg2.png               # Stego image example
│
├── .gitignore
└── README.md
```

> **Note:** `final_model.keras` is not included due to file size. Place it inside `webapp/stego_detector/` after training or downloading it (see [Training](#training-kaggle)).

---

## Getting Started

### Prerequisites

- Python 3.9+
- TensorFlow 2.x
- Flask

### Installation

```bash
git clone https://github.com/NihalBai/Steganography.git
cd Steganography
pip install tensorflow flask pillow scipy numpy
```

### Running the Web App

```bash
cd webapp/stego_detector
python app.py
```

Open `http://localhost:5000` in your browser. Drag and drop any image — the app returns a **Clean / Stego** prediction with a confidence score. You can use the images in `samples/` to try it out.

> Make sure `final_model.keras` is in the same folder as `app.py`.

---

## Training (Kaggle)

The model was trained on Kaggle using a **P100 GPU**. To reproduce:

1. Upload `notebooks/bestver95.ipynb` to Kaggle
2. Attach the dataset: [`petrdufek/stego-pvd-dataset`](https://www.kaggle.com/datasets/petrdufek/stego-pvd-dataset)
3. Enable GPU accelerator (P100)
4. Run all cells — the trained model will be saved as `final_model.keras`

Key decisions documented in the notebook:
- `tf.data` lazy pipeline with `prefetch(AUTOTUNE)` to avoid RAM crashes
- Custom `@tf.keras.utils.register_keras_serializable()` on `tlu` for model serialization
- 3-split strategy: train / val / test (val for early stopping, test for unbiased evaluation)

---

## Dataset

**[Stego-PVD-Dataset](https://www.kaggle.com/datasets/petrdufek/stego-pvd-dataset)** by petrdufek

- ~16,000 images split across train / val / test
- Balanced classes: clean images vs. PVD-encoded stego images

---

## How It Works

**Pixel Value Differencing (PVD)** hides secret data by modifying the difference between adjacent pixel values within a range imperceptible to the human eye. These modifications leave subtle statistical traces in the image noise residual.

**SRM (Spatial Rich Model)** extracts those noise residuals by applying high-pass convolutional kernels that suppress image content. The residual map is then fed to the CNN, which learns to distinguish natural noise from PVD artifacts.

---

## Tech Stack

| Component       | Tools                                   |
|-----------------|-----------------------------------------|
| Model training  | TensorFlow / Keras, Kaggle (P100 GPU)   |
| Preprocessing   | SRM filter bank (NumPy, SciPy)          |
| Data pipeline   | `tf.data` with `prefetch` + `AUTOTUNE`  |
| Web app         | Flask, HTML/CSS                         |
| Evaluation      | scikit-learn, Matplotlib, Seaborn       |

---

*Master's project — AI & Cybersecurity*