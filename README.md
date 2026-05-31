# 👁️ DrowsAI — Driver Drowsiness Detection

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://deeplearning-drowsinessdetection.streamlit.app/)
[![GitHub Pages](https://img.shields.io/badge/Site-GitHub%20Pages-00D4FF?style=for-the-badge&logo=github)](https://avish16.github.io/Deep_Learning-Drowsiness_Detection/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> CNN-based binary classifier for driver drowsiness detection using infrared eye images. Trained on 84,898 images from the MRL Eye Dataset. Achieves 98.6% test accuracy.

---

## 🌐 Links

| | |
|---|---|
| 🚀 **Live Demo** | [deeplearning-drowsinessdetection.streamlit.app](https://deeplearning-drowsinessdetection.streamlit.app/) |
| 📄 **GitHub Pages** | [avish16.github.io/Deep_Learning-Drowsiness_Detection](https://avish16.github.io/Deep_Learning-Drowsiness_Detection/) |
| 💻 **Repository** | [github.com/Avish16/Deep_Learning-Drowsiness_Detection](https://github.com/Avish16/Deep_Learning-Drowsiness_Detection) |

---

## ✨ Features

- **Face detection** — OpenCV Haar cascade detects face automatically
- **Eye region extraction** — Crops the eye strip from detected face geometry
- **CNN inference** — 3 trained models available (Baseline, Augmented, Deep)
- **Live demo** — Upload any face photo, get AWAKE/DROWSY prediction instantly
- **98.6% test accuracy** on 16,981 held-out infrared eye images

---

## 🏗️ Architecture

The detection pipeline follows this flow:
```
Upload Image
│
├─ Face Detection (OpenCV Haar Cascade)
│       └─ Blue bounding box drawn
│
├─ Eye Region Extraction (rows 22-48% of face height)
│       └─ Cyan bounding box drawn
│
├─ Preprocessing: grayscale → resize 64×64 → normalize /255
│
├─ CNN Inference → P(drowsy) sigmoid output
│
└─ P(drowsy) > threshold → DROWSY alert
                         → AWAKE otherwise
```

### CNN Model Structure
```
Input (64×64×1 grayscale)
│
├─ Conv2D(32, 3×3, ReLU) → MaxPool(2×2)
├─ Conv2D(64, 3×3, ReLU) → MaxPool(2×2)
├─ Conv2D(128, 3×3, ReLU) → MaxPool(2×2)
│  [Deep model adds Conv2D(256) block]
│
├─ Flatten → Dense(128, ReLU) → Dropout
└─ Dense(1, Sigmoid) → P(drowsy)
```

---

## 📊 Model Performance

| Model | Test Accuracy | Test Loss | Macro F1 | Parameters |
|---|---|---|---|---|
| baseline_cnn_mrl | 98.59% | 0.045 | ~0.99 | ~500K |
| cnn_mrl_driver_aug | 98.15% | 0.052 | ~0.98 | ~500K |
| **cnn_mrl_driver_aug_deep ★** | **98.39%** | **0.049** | **~0.98** | **~800K** |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning | TensorFlow 2.x / Keras |
| Computer Vision | OpenCV |
| Demo App | Streamlit |
| Image Processing | Pillow, NumPy |
| Analysis | pandas, matplotlib, scikit-learn |
| Face Detection | OpenCV Haar Cascades |

---

## 📦 Dataset

**MRL Eye Dataset** — 84,898 PNG grayscale infrared eye crops, binary labels (awake/sleepy), ~50/50 balanced.

| Split | Images |
|---|---|
| Train | 50,937 |
| Validation | 16,980 |
| Test | 16,981 |

[View on Kaggle →](https://www.kaggle.com/datasets/akashshingha850/mrl-eye-dataset)

---

## 🚀 Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/Avish16/Deep_Learning-Drowsiness_Detection
cd Deep_Learning-Drowsiness_Detection

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app/streamlit_app.py
```

App runs at **http://localhost:8501**

---

## ⚠️ Disclaimer

This is a **research and educational prototype**. Not validated for real-world vehicle deployment. Always consult safety engineering standards before using AI-based drowsiness detection in production systems.

---

## 👤 Author

**Avi Sharma**
- GitHub: [Avish16](https://github.com/Avish16)
- LinkedIn: [linkedin.com/in/avi-sharma-1716361b8](https://www.linkedin.com/in/avi-sharma-1716361b8/)

---

*IST-691 Deep Learning · Syracuse University*
