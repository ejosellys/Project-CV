# 🚦 Traffic Sign Detection in Indonesia using YOLOv8

This project focuses on detecting various types of **traffic signs commonly found in Indonesia** using the **YOLOv8 object detection model** by Ultralytics. The dataset, consisting of 30 traffic sign classes, was sourced from Roboflow.

---

## 🎯 Objective & Benefits

- **Objective**: To develop an automated detection system that recognizes Indonesian traffic signs using deep learning, specifically YOLOv8.
- **Benefits**:
  - 🚗 Assists in intelligent driving systems and autonomous vehicles
  - 📷 Enables traffic monitoring and road sign analysis
  - 🧠 Provides a benchmark for computer vision research in the Indonesian context

---

## 📊 Project Progress

| Phase | Description | Status | Progress |
|-------|-------------|--------|----------|
| 1 | **Dataset Collection & Preprocessing**<br>Downloaded from Roboflow and converted to YOLOv8 format | ✅ Completed | 100% |
| 2 | **Environment Setup & Dataset Configuration**<br>Package installation and creation of `data.yaml` | ✅ Completed | 100% |
| 3 | **YOLOv8 Model Training**<br>Trained YOLOv8s for 30 epochs | ✅ Completed | 100% |
| 4 | **Model Evaluation & Validation**<br>Assessed model with metrics like precision, recall, mAP | ✅ Completed | 100% |
| 5 | **Final Inference and Prediction**<br>Prediction on test image using `best.pt` weights | ✅ Completed | 100% |

---

## 🛠️ Tools & Technologies

- 🔍 YOLOv8 (ultralytics)
- 📦 Roboflow (Dataset Management)
- 💻 Google Colab / Python
- 🎞️ OpenCV & Matplotlib (Visualization)
- 🐍 Python 3.13.5

---

## 📁 Dataset Structure (YOLOv8)

```
Traffic-Sign-in-Indonesia-Detection-3/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

---

## 🚀 How to Run

### 1. Install Required Packages

```bash
pip install ultralytics roboflow
```

---

### 2. Download Dataset from Roboflow

```python
from roboflow import Roboflow

rf = Roboflow(api_key="BDCK1eDEowfgGG4scISW")
project = rf.workspace("putri-mawaring-wening-lwwcx").project("traffic-sign-in-indonesia-detection")
version = project.version(3)
dataset = version.download("yolov8")
```

---

### 3. Save Dataset Configuration

```python
fixed_yaml = """ 
names:
- Balai Pertolongan Pertama
- Banyak Anak-Anak
- Dilarang Belok Kanan
- Dilarang Berhenti
- Dilarang Berjalan Terus
- Dilarang Masuk
- Dilarang Mendahului
- Dilarang Parkir
- Dilarang Putar Balik
- Gereja
- Hati-Hati
- Jalur Penyebrangan
- Lampu Lalu Lintas
- Larangan Kecepatan - 30km-jam
- Larangan Kecepatan - 40km-jam
- Larangan Kendaraan MST - 10 Ton
- Masjid
- Pemberhentian Bus
- Perintah Ikuti Bundaran
- Perintah Jalur Sepeda
- Perintah Lajur Kiri
- Perintah Pilih Satu Jalur
- Persimpangan 3 Prioritas
- Persimpangan 3 Sisi Kanan Prioritas
- Persimpangan 3 Sisi Kiri Prioritas
- Persimpangan Empat
- Putar Balik
- Rumah Sakit
- SPBU
- Tempat Parkir
nc: 30
roboflow:
  license: CC BY 4.0
  project: traffic-sign-in-indonesia-detection
  url: https://universe.roboflow.com/putri-mawaring-wening-lwwcx/traffic-sign-in-indonesia-detection/dataset/3
  version: 3
  workspace: putri-mawaring-wening-lwwcx
train: train/images
val: valid/images
test: test/images
"""

with open("/content/Traffic-Sign-in-Indonesia-Detection-3/data.yaml", "w") as f:
    f.write(fixed_yaml)
```

---

### 4. Train the YOLOv8 Model

```python
from ultralytics import YOLO

model = YOLO('yolov8s.pt')
model.train(
    data="/content/Traffic-Sign-in-Indonesia-Detection-3/data.yaml",
    epochs=30,
    imgsz=640,
    batch=16
)
```

---

### 5. Visualize Detection Result

```python
from IPython.display import Image
Image(filename='https://github.com/ejosellys/Project-CV/blob/main/predict.png', width=800)
```

---

## 🖼️ Example Detection Output

![Detection Output](https://github.com/ejosellys/Project-CV/blob/main/predict.png)

---

> 📌 Note: This project can be expanded to include real-time detection (webcam/video) or deployed to a web application for broader accessibility.
