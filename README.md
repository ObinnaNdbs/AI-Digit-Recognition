# **AI-Digit-Recognition**

A **TensorFlow** project for recognizing handwritten digits using a pre-trained MNIST model. This project demonstrates training a neural network to classify handwritten digits (0-9) and allows users to test predictions using custom images.

---

## **Table of Contents**
- [Overview](#overview)
- [Features](#features)
- [Setup](#setup)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## **Overview**
This project uses a **Convolutional Neural Network (CNN)** trained on the MNIST dataset for handwritten digit classification. Users can test predictions with custom handwritten images.

---

## **Features**
- Train a model on the MNIST dataset.
- Test predictions using custom digit images.
- Preprocessing pipeline for image normalization and resizing.
- Visualize predictions with Matplotlib.

---

## **Setup**
1. Clone this repository:
   ```bash
   git clone https://github.com/ObinnaNdbs/AI-Digit-Recognition.git
   cd AI-Digit-Recognition
2. Create and activate a virtual environment:

   ```bash
   Copy code
   python -m venv venv
   source venv/bin/activate        # Linux/Mac
   venv\Scripts\activate           # Windows
3. Clone this repository:
   ```bash
   pip install -r requirements.txt

---

## **Usage**
1. Train the model
      ```bash
      python scripts/train_model.py
2. Predict Custom Images
   - Add handwritten digit images to the images/ folder.
   - Run the prediction script:
   ```bash
      python scripts/predict_model.py

---

## **Dependencies**
- TensorFlow
- NumPy
- Pillow
- Matplotlib

---

## **Acknowledgements**
This project was self-taught, built using online resources like TensorFlow and MNIST tutorials, with inspiration from programming communities.

---

## **License**
This project is licensed under the MIT License.
