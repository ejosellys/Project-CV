# 🚦 Traffic Sign Detection in Indonesia with YOLOv8

Proyek ini bertujuan untuk mendeteksi berbagai jenis **rambu lalu lintas di Indonesia** menggunakan model **YOLOv8** dari Ultralytics. Dataset yang digunakan berasal dari Roboflow dan terdiri dari 30 kelas rambu lalu lintas yang umum ditemukan di Indonesia.

## 📊 Progress Proyek

| Fase | Deskripsi | Status | Persentase |
|------|-----------|--------|------------|
| 1 | **Pengumpulan & Pra-pemrosesan Dataset**<br>Download dataset Roboflow dan sesuaikan format ke YOLOv8 | ✅ Selesai | 100% |
| 2 | **Setup Environment & Konfigurasi Dataset**<br>Instalasi package dan penyusunan file `data.yaml` | ✅ Selesai | 100% |
| 3 | **Training Model YOLOv8**<br>Model YOLOv8s dilatih selama 30 epoch | ✅ Selesai | 100% |
| 4 | **Evaluasi Model & Validasi**<br>Pengujian hasil training (precision, recall, mAP) | ✅ Selesai | 100% |
| 5 | **Inferensi Deteksi Akhir**<br>Deteksi rambu menggunakan model `best.pt` | ✅ Selesai | 100% |

## 🛠️ Tools & Teknologi

- 🧠 YOLOv8 (`ultralytics`)
- 📦 Roboflow (dataset)
- 💻 Google Colab / Python
- 🎞️ OpenCV & Matplotlib (visualisasi)
- 🐍 Python 3.13.5

## 📁 Struktur Dataset (YOLOv8)

```
Traffic-Sign-in-Indonesia-Detection-3/
│
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

## 🚀 Cara Menjalankan

### 1. Instalasi Package

```bash
pip install ultralytics roboflow
```

### 2. Download Dataset Roboflow

```python
from roboflow import Roboflow

rf = Roboflow(api_key="BDCK1eDEowfgGG4scISW")
project = rf.workspace("putri-mawaring-wening-lwwcx").project("traffic-sign-in-indonesia-detection")
version = project.version(3)
dataset = version.download("yolov8")
```

### 3. Simpan Konfigurasi Dataset

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

### 4. Training Model

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

### 5. Inference (Uji Coba pada Gambar)

Setelah training selesai, kamu dapat melihat hasil visualisasi deteksi sebagai berikut:

```python
from IPython.display import Image
Image(filename='https://github.com/ejosellys/Project-CV/blob/main/predict.png', width=800)
```
