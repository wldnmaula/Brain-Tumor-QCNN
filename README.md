


<p align="center">
  <img src="assets/banner.jpg" alt="Brain Tumour QCNN Banner" width="100%">
</p>


<p align="center">

  <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python Badge">
  <img src="https://img.shields.io/badge/PyTorch-DeepLearning-orange" alt="PyTorch Badge">
  <img src="https://img.shields.io/badge/PennyLane-QuantumML-purple" alt="PennyLane Badge">
  <img src="https://img.shields.io/badge/Gradio-UI-green" alt="Gradio Badge">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen" alt="Status Badge">
  <img src="https://img.shields.io/badge/License-OpenSource-blue" alt="License Badge">

</p>


# Brain Tumour Detection using Quantum Convolutional Neural Networks (QCNN)

This project demonstrates a hybrid **Quantum + Classical** deep learning model
for **brain tumour classification** using MRI images.
The architecture uses EfficientNet-B0 + a 4-qubit quantum variational layer.

---

## Features
- Hybrid QCNN architecture (EfficientNet + Quantum Layer)
- Binary brain tumor classification
- Quantum circuit built using PennyLane
- Gradio UI for predictions
- Safe-cleaning of model checkpoints
- GPU supported

---

## Tech Stack
Deep Learning:
- PyTorch
- EfficientNet-B0

Quantum:
- PennyLane
- Strongly Entangling Layers (4-qubit)

Utilities:
- NumPy, Pillow, Scikit-Learn, Torchvision, Gradio

---

## 📌 Dataset
Kaggle dataset by the author:
https://www.kaggle.com/datasets/skarthik112/karthik-braindataset-mri

Folder structure:
```
brain_Tumor_karr/
 ├── yes/    → tumour present
 └── no/     → no tumour
```

---

## 🧠 Sample Results
### Tumour Detected (YES)
![YES](assets/result_yes.png)

### No Tumour (NO)
![NO](assets/result_no.png)

---

## 🚀 Training the Model
```
python train_qcnn.py
```

## 🔍 Running Predictions (Gradio)
```
python predict_qcnn.py
```

---

## 📁 Repository Structure
```
Brain-Tumor-QCNN/
├── train_qcnn.py
├── predict_qcnn.py
├── README.md
├── requirements.txt
├── .gitignore
└── assets/
    ├── result_yes.png
    └── result_no.png
```

---

### Author
**S. Karthik (2025)** – Brain Tumour Detection using QCNN
