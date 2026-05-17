# Handwritten Arabic Digit Recognition — CNN

A convolutional neural network achieving **99.14% accuracy** on handwritten Arabic digit classification (digits ٠–٩), built with Python and PyTorch/TensorFlow.

**Author:** Hala Sabobeh

---

## Overview

Handwritten Arabic digit recognition is a challenging computer vision problem due to the diverse writing styles across different writers. This project trains a CNN on the **MADBase** (Modified Arabic Handwritten Digits Database) dataset and achieves near state-of-the-art accuracy.

The 10 Arabic digit classes correspond to:

| Arabic | ٠ | ١ | ٢ | ٣ | ٤ | ٥ | ٦ | ٧ | ٨ | ٩ |
|--------|---|---|---|---|---|---|---|---|---|---|
| Value  | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |

---

## Results

| Metric | Value |
|---|---|
| Test Accuracy | **99.14%** |
| Dataset | MADBase |
| Model | CNN |

---

## Project Structure

```
.
├── arabic_digit_recognition/   # Main package (model, training, utils)
├── Final+Project.pdf           # Project report
├── final_vision.pdf            # Final vision report
└── README.md
```

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/hala-sabobeh/Arabic-digit-recognition-AI-Computer-Vision.git
cd Arabic-digit-recognition-AI-Computer-Vision
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download the dataset**

This project uses the [MADBase dataset](http://datacenter.aucegypt.edu/shazeem/) — 70,000 handwritten Arabic digit images (60,000 training / 10,000 test), each 28×28 pixels.

Place the dataset under:
```
data/
├── train/
└── test/
```

---

## Usage

**Train the model:**
```bash
cd arabic_digit_recognition
python train.py
```

**Evaluate on the test set:**
```bash
python evaluate.py
```

---

## Model Architecture

The CNN is designed for 28×28 grayscale images with 10 output classes:

- Convolutional blocks with ReLU activation and max pooling
- Batch normalization for stable training
- Dropout for regularization
- Fully connected output layer with softmax

---

## Dataset

**MADBase** (Modified Arabic Handwritten Digits Database):
- 70,000 images total — 60,000 training, 10,000 test
- 10 classes (digits 0–9 in Arabic script)
- 28×28 grayscale images
- Collected from 700 writers

---

## Reports

Full methodology, architecture details, experiments, and analysis are in the included PDF reports:
- `Final+Project.pdf`
- `final_vision.pdf`
