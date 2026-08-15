# 🦷 Dental Disease Identification Using YOLOv8

An AI-powered computer vision system for detecting dental abnormalities/diseases from dental X-ray images using **YOLOv8**, **Roboflow**, and **Gradio**.

## 📌 Overview

Dental diseases and abnormalities can be difficult to identify from X-ray images, especially when manual examination is required for a large number of images.

This project uses a deep learning-based object detection approach to identify annotated dental conditions in X-ray images.

The system:

1. Downloads a dental X-ray dataset from Roboflow.
2. Uses a YOLOv8 object detection model.
3. Trains the model on the annotated dataset.
4. Evaluates images using the trained model.
5. Displays detected regions with bounding boxes.
6. Provides the detection confidence for each prediction.
7. Offers an interactive Gradio interface for uploading X-ray images.

## 🧠 Model

The project uses **YOLOv8 Nano (`yolov8n.pt`)** as the object detection model.

YOLOv8 was selected because it provides a good balance between:

* Detection speed
* Model size
* Computational requirements
* Real-time inference capability

The notebook trains the model using:

* **Epochs:** 30
* **Image size:** 640 × 640
* **Batch size:** 16
* **Hardware:** NVIDIA T4 GPU through Google Colab

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data=dataset.location + "/data.yaml",
    epochs=30,
    imgsz=640,
    batch=16
)
```

## 🗂️ Dataset

The dataset is managed through **Roboflow** and downloaded in YOLOv8 format.

The project uses the dataset's `data.yaml` file to obtain the class definitions.

```python
with open("/content/Dental-X-ray-1/data.yaml", "r") as f:
    data = yaml.safe_load(f)

print("Classes:", data["names"])
```

The exact disease/abnormality class names are determined by the `data.yaml` associated with the Roboflow dataset.

## 🔄 Workflow

```text
Dental X-Ray Image
        │
        ▼
   Image Upload
        │
        ▼
    YOLOv8 Model
        │
        ▼
Object Detection
        │
        ├── Detected Class
        ├── Bounding Box
        └── Confidence Score
        │
        ▼
Annotated X-Ray Image
        │
        ▼
   Gradio Interface
```

## 🛠️ Technologies Used

| Technology   | Purpose                                     |
| ------------ | ------------------------------------------- |
| Python       | Core programming language                   |
| YOLOv8       | Dental abnormality detection                |
| Ultralytics  | YOLO training and inference                 |
| Roboflow     | Dataset management and export               |
| OpenCV       | Image processing                            |
| NumPy        | Numerical/image operations                  |
| Matplotlib   | Visualization                               |
| Gradio       | Interactive web interface                   |
| Google Colab | Model training                              |
| PyTorch      | Deep learning framework used by Ultralytics |

## 📦 Installation

Install the required packages:

```bash
pip install ultralytics roboflow
```

The notebook also uses:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
import yaml
```

## 📥 Dataset Setup

Create a Roboflow project containing annotated dental X-ray images.

Export/download the dataset in **YOLOv8 format**.

The notebook uses the Roboflow Python SDK:

```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
project = rf.workspace("YOUR_WORKSPACE").project("YOUR_PROJECT")
version = project.version(1)
dataset = version.download("yolov8")
```

### ⚠️ API Key Security

Never commit your Roboflow API key to GitHub.

Use an environment variable instead:

```python
import os
from roboflow import Roboflow

rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
```

## 🚀 Training

Load the pretrained YOLOv8 Nano model:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
```

Train the model:

```python
model.train(
    data=dataset.location + "/data.yaml",
    epochs=30,
    imgsz=640,
    batch=16
)
```

After training, the best model is saved by Ultralytics under:

```text
runs/detect/train/weights/best.pt
```

## 🔍 Inference

The trained model can be loaded using:

```python
model = YOLO("runs/detect/train/weights/best.pt")
```

An input image is passed to the model:

```python
results = model.predict(
    source=image,
    conf=0.7,
    imgsz=640
)
```

The system returns:

* Detected disease/abnormality class
* Confidence score
* Bounding box
* Annotated image

## 🖥️ Gradio Interface

The project includes an interactive Gradio interface where users can:

* Upload a dental X-ray image
* Select a confidence threshold
* View the prediction result
* View detected objects and confidence scores

Available confidence thresholds in the notebook are:

```text
0.3
0.5
0.7
0.8
0.9
```

Example interface:

```text
┌──────────────────────────────────────┐
│       Dental X-Ray Detection         │
├──────────────────────────────────────┤
│                                      │
│        Upload Dental X-Ray           │
│                                      │
│        Confidence: 0.7               │
│                                      │
├──────────────────────────────────────┤
│ Prediction Result                    │
│                                      │
│ [Annotated X-Ray Image]              │
│                                      │
│ Detected Objects                     │
│ → Detected: <class> (0.87)           │
└──────────────────────────────────────┘
```

## 📁 Project Structure

```text
Dental-Disease-Identification/
│
├── teeth_identification.ipynb
├── README.md
│
├── dataset/
│   ├── train/
│   ├── valid/
│   ├── test/
│   └── data.yaml
│
├── runs/
│   └── detect/
│       └── train/
│           └── weights/
│               └── best.pt
│
└── requirements.txt
```

## 📊 Detection Pipeline

The detection function converts the uploaded image into a NumPy array and passes it to the trained YOLO model.

```python
def detect_objects(img, confidence):
    img_np = np.array(img)

    results = model.predict(
        source=img_np,
        conf=confidence,
        imgsz=640
    )

    if results and results[0].boxes:
        output_img = results[0].plot()
```

For every detected object, the system extracts its class and confidence:

```python
for box in results[0].boxes:
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])

    label = model.names.get(
        cls_id,
        f"unknown({cls_id})"
    )

    print(label, conf)
```

## 🎯 Features

* 🦷 Dental X-ray analysis
* 🤖 YOLOv8-based object detection
* 📦 Roboflow dataset integration
* 🎯 Adjustable confidence threshold
* 📍 Bounding-box localization
* 📊 Detection confidence scores
* 🖥️ Interactive Gradio interface
* ⚡ GPU-supported training
* 📱 Easy image upload and prediction

## 🔮 Future Improvements

Possible improvements include:

* Train with a larger and more diverse dental X-ray dataset
* Compare YOLOv8 Nano with larger YOLOv8 models
* Add precision, recall, mAP@50, and mAP@50-95 evaluation
* Add confusion-matrix analysis
* Improve class balancing
* Apply image augmentation
* Add explainable AI techniques such as Grad-CAM
* Deploy the trained model as an API
* Build a dedicated web application
* Add patient-level prediction reports
* Validate the model on an independent clinical dataset

## ⚠️ Limitations

This project is an AI research/prototype system and should **not be considered a replacement for professional dental diagnosis**.

Model performance depends heavily on:

* Dataset quality
* Number of training images
* Annotation quality
* Class distribution
* Image quality
* Training configuration

Predictions should be reviewed by qualified dental professionals before being used for clinical decision-making.



### Project Summary

**Dental Disease Identification** is a deep-learning-based dental X-ray analysis system that uses YOLOv8 object detection to locate and identify annotated dental abnormalities. The model is trained using a Roboflow-managed dataset and deployed through an interactive Gradio interface for image-based prediction.
