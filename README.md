# 🛡️ Real-Time Object Detection & Tracking
**CodeAlpha AI Internship - Task 4**

## 📌 Project Overview
This project implements a robust, real-time object detection and tracking system using the **MobileNet-SSD** (Single Shot Detector) architecture. It identifies and tracks 20 distinct VOC object classes (including people, bottles, vehicles, and furniture) via a live webcam feed integrated into a **Streamlit** web interface. Each detected object is assigned a unique tracking ID that persists across frames using a built-in **Centroid Tracker**.

## 🛠️ Technical Stack
* **Language:** Python 3.10 (64-bit)
* **Library:** OpenCV 4.11 (`cv2.dnn` module)
* **Framework:** Streamlit
* **Architecture:** MobileNet-SSD (Caffe Model)
* **Tracker:** Centroid Tracker (custom implementation)
* **Data Handling:** NumPy 1.26.4 (pinned for OpenCV compatibility)

## 🚀 Key Features
* **Live Inference:** Real-time per-frame detection using the OpenCV DNN backend at 300×300 input resolution.
* **Object Tracking:** Centroid-based tracker assigns a persistent ID to each object and follows it across frames.
* **Bounding Boxes:** Color-coded per VOC class with confidence scores displayed on-screen.
* **Live Tracking Table:** Displays tracking ID, class label, and centroid coordinates updated every frame.
* **Windows Optimization:** Implemented `CAP_DSHOW` DirectShow backend to resolve camera initialization latency.
* **Confidence Filtering:** Dynamic sidebar slider for thresholding — filters false positives in real time.
* **Environment Stability:** Resolved NumPy 2.x breaking changes by pinning to stable 1.26.4 dependencies.
* **Auto-Download:** Fetches the correct `MobileNetSSD_deploy.prototxt` automatically on first run.

## 📦 Detectable Classes (20 VOC Objects)
`aeroplane`, `bicycle`, `bird`, `boat`, `bottle`, `bus`, `car`, `cat`, `chair`, `cow`, `diningtable`, `dog`, `horse`, `motorbike`, `person`, `pottedplant`, `sheep`, `sofa`, `train`, `tvmonitor`.

## ⚙️ Setup & Installation
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/usmankhan89299/CodeAlpha_Object_Detection.git](https://github.com/usmankhan89299/CodeAlpha_Object_Detection.git)
   cd CodeAlpha_Object_Detection
