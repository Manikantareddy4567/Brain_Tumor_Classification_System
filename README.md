# 🧠 Brain Tumor Classification with Deep Learning

This project performs **automated brain tumor classification** using MRI images and a Convolutional Neural Network (CNN) built with TensorFlow. It is structured with modular components for data preparation, model training, evaluation, and utility support.

---

## 📚 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Results](#results)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## 🧾 Overview

Brain tumor detection from MRI scans is a critical task in medical imaging. This project classifies images into one of four categories:

- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

The system uses a CNN for training and evaluation with a focus on extendability and readability.

---

## ✅ Features

- 🧠 CNN model built using TensorFlow/Keras
- 🧹 Modular structure for training, evaluating, preprocessing
- 📦 Custom configuration management (`config/config.py`)
- 📊 Performance metrics: accuracy, classification report
- 💡 Easy to adapt with new datasets

---

## 🗂️ Project Structure

```bash
Brain_tumor_detection/
├── .vscode/                  # VSCode settings (optional)
├── config/
│   └── config.py             # Configuration constants (batch size, epochs, etc.)
├── model/
│   └── brain_tumor_model.h5  # Saved model after training
├── src/
│   ├── __init__.py
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation script
│   ├── model.py              # Model architecture definition
│   ├── utils.py              # Helper functions (if any)
│   └── data_preparation.py   # Data loading and preprocessing
├── main.py                   # Entry point to train and evaluate the model
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/brain-tumor-detection.git
cd brain-tumor-detection
```

### 2. Create Virtual Environment & Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Prepare Dataset

Ensure your dataset is structured as:

```
dataset/
├── glioma/
├── meningioma/
├── no_tumor/
└── pituitary/
```

Update the dataset path inside `src/data_preparation.py` as needed.

---

## ▶️ Usage

To **train and evaluate** the model, simply run:

```bash
python main.py
```

This will:
- Train the CNN using the dataset
- Save the trained model to `model/brain_tumor_model.h5`
- Evaluate the model on test data

---

## 📈 Results

Expected output includes:
- Accuracy and loss values
- Classification report (precision, recall, F1-score)
- Confusion matrix (if visualized)

Sample printed shapes:
```
Train X shape: (N, width, height, channels)
Train Y shape: (N, num_classes)
...
```

---

## 🙌 Acknowledgements

- TensorFlow & Keras for deep learning APIs
- OpenCV for image preprocessing
- Scikit-learn for evaluation metrics
- Kaggle/public datasets for MRI scans

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

*Project built for learning, research, and further innovation in healthcare AI.*
