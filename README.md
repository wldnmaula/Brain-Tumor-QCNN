



# Brain Tumour Detection using Quantum Convolutional Neural Networks (QCNN)

This project demonstrates a hybrid **Quantum + Classical** deep learning model
for **brain tumour classification** using MRI images.
The architecture uses EfficientNet-B0 + a 4-qubit quantum variational layer.

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
