[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://deeplearning-drowsinessdetection.streamlit.app/)
[![GitHub Pages](https://img.shields.io/badge/Site-GitHub%20Pages-00D4FF?style=for-the-badge&logo=github)](https://avish16.github.io/Deep_Learning-Drowsiness_Detection/)

# Drowsiness detection from infrared eye images

## Open the model page & download weights

**GitHub Pages (metrics + download each trained model):**  
[https://Avish16.github.io/Deep_Learning-Drowsiness_Detection_Using_Infrared_Images/](https://Avish16.github.io/Deep_Learning-Drowsiness_Detection_Using_Infrared_Images/)

Pre-trained Keras checkpoints are also in the repo under [`weights/`](weights/). Copy the file you need into `models/` for local inference, or change `DEFAULT_MODEL_PATH` in `src/drowsiness/config.py`.

---

## Problem statement

Driver **drowsiness** is a major safety risk. This project focuses on a **proxy task**: from a **crop of the eye region**, classify whether the subject appears **awake** or **sleepy** (eyes open vs. closed / drowsy pattern). That supports downstream applications such as in-cabin alerts, provided real-world validation is done separately.

---

## Tech stack

| | |
|--|--|
| ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white) | Deep learning & training (`tensorflow>=2.13`, Keras Functional API) |
| ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) | Python 3.x |
| ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) | Arrays & numerics |
| ![Pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white) | Tables & EDA summaries |
| ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat) | Plots (distributions, training curves, ROC) |
| ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white) | Classification report, confusion matrix, ROC/AUC |
| ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white) | Video / image I/O, resizing, drawing overlays |
| ![Pillow](https://img.shields.io/badge/Pillow-3775A9?style=flat) | Image size inspection in EDA |
| **dlib** | Face detection + 68-point landmarks for eye crops at inference |

---

## Datasets

### MRL Eye Dataset (primary)

| | |
|--|--|
| **Name** | MRL Eye Dataset (infrared) |
| **Source** | [Kaggle — MRL Eye Dataset](https://www.kaggle.com/datasets/akashshingha850/mrl-eye-dataset) |
| **Image type** | PNG, **grayscale infrared** eye crops; **binary labels** `awake` vs `sleepy` |
| **Scale (from EDA notebook)** | **~84,898** PNGs indexed under `data/MRL/data`; **train 50,937** · **val 16,980** · **test 16,981** (roughly **50/50** awake vs sleepy per split) |

### Eye Dataset (optional / extension)

| | |
|--|--|
| **Name** | Eye Dataset (open / closed eyes) |
| **Source** | [Kaggle — Eye Dataset](https://www.kaggle.com/datasets/prasadvpatil/eye-dataset) |
| **Image type** | RGB folders: open/closed left and right eyes (used as a secondary path in EDA; modeling notebook focuses on MRL) |

---

## Exploratory analysis (from `notebooks/01_data_gathering.ipynb`)

What the EDA code actually does:

1. **Inventory** — Recursively collects all `*.png` under `data/MRL/data` and reports **total image count** (~85k in the logged run).
2. **Stratification check** — For each split (`train` / `val` / `test`) and class (`awake` / `sleepy`), counts images and computes **class ratios** (~0.506 awake / ~0.494 sleepy per split in the saved outputs).
3. **Resolution exploration** — Samples images with PIL, records **width × height**, aggregates **unique size frequencies**, and plots a **histogram of image widths** to see how consistent resolutions are before resizing.
4. **Qualitative review** — Displays **sample images** per split/class (awake vs sleepy) for a quick visual sanity check.
5. **TensorFlow pipeline prototype** — Builds `tf.keras.utils.image_dataset_from_directory` on train/val/test with **grayscale**, fixed **64×64** resize, **binary** labels, batching, and shows a **single batch shape** after preprocessing.
6. **Augmentation + `tf.data`** — Defines **horizontal `RandomFlip`** and small **`RandomRotation`** (0.05), applies them only when training; normalizes to **[0, 1]**; uses **`shuffle(1000)`**, **`cache()`**, and **`prefetch(AUTOTUNE)`** for efficient input pipelines (mirrors the training notebook).

---

## Modeling approach (`notebooks/02_baseline_cnn.ipynb`)

- **Input:** 64×64×1 grayscale, **binary** labels, **batch size 64**.
- **Augmentation (training):** `RandomFlip("horizontal")`, `RandomRotation(0.05)`; val/test without augmentation.
- **Baseline CNN (`baseline_cnn_mrl`):** Three **Conv2D** blocks (32 → 64 → 128 filters, 3×3, ReLU, `padding="same"`) each followed by **MaxPooling2D(2×2)** → **Flatten** → **Dense(128, relu)** → **Dense(1, sigmoid)**.
- **Training:** **Adam** (lr 1e-3), **binary cross-entropy**, metrics include **accuracy**; additional experiments in the same notebook train **augmented** and **deeper** variants and save `cnn_mrl_driver_aug.keras` and `cnn_mrl_driver_aug_deep.keras`.

---

## Evaluation metrics & results

On the **held-out test** set (**16,981** images, **2** classes), the baseline run reaches about **98.6% test accuracy** with **precision / recall / F1 ≈ 0.99** (macro/weighted). The notebook also reports **confusion matrices**, **ROC curves with AUC**, **misclassification examples**, and **threshold** sweeps for sleepy vs awake. See the **GitHub Pages** link at the top for a compact comparison of all three saved models and direct **downloads**.

---

## How to run

1. **EDA:** `notebooks/01_data_gathering.ipynb`
2. **Train / evaluate:** `notebooks/02_baseline_cnn.ipynb`
3. **Video:** place weights in `models/`, add `data/test_input/test_video.mp4`, then `python scripts/detect_video.py`
4. **Webcam:** `python scripts/detect_live.py`

---

## Conclusion & recommendations

The CNNs reach **strong metrics on the curated MRL test split**. Real deployment still needs checks on **lighting, pose, glasses, and domain shift**; consider **calibrating thresholds** on your own video and exploring **temporal** models for smoother alerts.

---

## Author

- **Avi Sharma**
- **GitHub:** [Avish16](https://github.com/Avish16)
- **LinkedIn:** [linkedin.com/in/avi-sharma-1716361b8](https://www.linkedin.com/in/avi-sharma-1716361b8/)