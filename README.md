# 🛡️ Helmet Safety Detection System Using YOLOv8

An end-to-end Computer Vision object detection project developed as part of the **AIRI Team PITB AI Internship**. The project uses **YOLOv8n** to detect four PPE-related classes and assess helmet and safety-vest compliance from images and videos.

---

## 1. Project Overview

I built a Computer Vision object-detection system using YOLOv8 to detect Personal Protective Equipment (PPE) — specifically helmets and vests — in safety-related images and videos. The goal was to understand the complete AI workflow, from dataset preparation and annotation to model training, evaluation, inference, and deployment. Manual PPE monitoring is difficult when multiple people must be observed at once, so an automated detection system is useful for construction sites, industrial environments, and road safety monitoring. The model detects four classes — helmet, vest, without_helmet, and without_vest — and the final system draws bounding boxes with predicted labels and confidence scores on unseen images and video, with an interactive Streamlit demo for testing.

---

## 2. Tools and Technologies Used

- Python
- Google Colab
- Google Drive
- YOLOv8 (Ultralytics)
- OpenCV
- NumPy
- Pillow
- Matplotlib
- Streamlit

---

## 3. Dataset Preparation

The dataset was organized in YOLO format, with images and their corresponding annotation `.txt` files. Additional suitable images were incorporated and annotated to improve class representation. Bounding boxes were drawn around the target objects and exported in the YOLO annotation format:

```
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are normalized between 0 and 1.

**Classes (4 total):**

| Class ID | Class |
|----------|-------|
| 0 | helmet |
| 1 | vest |
| 2 | without_helmet |
| 3 | without_vest |

**Dataset structure:**

```
dataset/
│
├── data.yaml
│
├── images/
│   ├── train/
│   ├── val/
│   └── test/
│
└── labels/
    ├── train/
    ├── val/
    └── test/
```

**`data.yaml` configuration:**

```yaml
train: images/train
val: images/val
test: images/test

nc: 4

names:
  0: helmet
  1: vest
  2: without_helmet
  3: without_vest
```

---

## 4. Model Training

The model was trained on **Google Colab** using GPU acceleration.

| Parameter | Value |
|---|---|
| Model | YOLOv8n |
| Image Size | 640 × 640 |
| Epochs | 40 |
| Classes | 4 |
| Pretrained Weights | Yes |
| Framework | Ultralytics YOLO |
| Platform | Google Colab |
| Dataset Format | YOLO |

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(data="dataset/data.yaml", epochs=40, imgsz=640)
```

The best trained model checkpoint is saved as `models/best.pt`.

---

## 5. Evaluation Results

| Metric | Score |
|---|---|
| Precision | 0.9517 |
| Recall | 0.9016 |
| mAP@0.5 | 0.9503 |
| mAP@0.5:0.95 | 0.667 |

- **Precision** — proportion of predicted detections that were correct.
- **Recall** — proportion of actual objects successfully detected.
- **mAP@0.5** — mean Average Precision at an IoU threshold of 0.50.
- **mAP@0.5:0.95** — a stricter evaluation across IoU thresholds from 0.50 to 0.95.

Training results, confusion matrices, and precision/recall/F1 curves are stored in `outputs/training_results/yolov8n_baseline/`, including `results.csv`, `results.png`, `confusion_matrix.png`, `confusion_matrix_normalized.png`, and the `BoxP/BoxR/BoxF1/BoxPR` curve plots. The confusion matrices help identify where the model confuses the four PPE classes.

---

## 6. Inference Results

**Image inference:** The trained model was tested on unseen test images, producing bounding boxes, predicted class labels, and confidence scores. Prediction outputs (more than the required minimum of 15 images) are stored in `outputs/predictions/test_predictions/`.

**Video inference:** The system also processes video frame-by-frame using the trained YOLOv8 model, generating an annotated output video with bounding boxes, labels, and confidence scores — demonstrating the model's use beyond static images.

---

## 7. Error Analysis

Manual error analysis was performed on incorrect or weak predictions, covering false positives, false negatives, wrong class predictions, poor bounding boxes, small-object detection issues, poor lighting, and annotation mistakes.

**Common observations:**
- Small objects were harder to detect reliably.
- Similar-looking PPE classes (e.g., vest vs. without_vest) were occasionally confused.
- Occlusion and difficult viewing angles reduced accuracy.
- Poor image quality and low-light conditions led to missed or incorrect detections.
- Bounding-box inconsistencies in the training data affected some predictions.
- Limited examples of difficult/edge cases contributed to certain errors.

A detailed error-analysis table is included in the final project report (`report/final_report.pdf`).

---

## 8. What I Learned

- Defined a real-world Computer Vision problem and scoped it into a workable project.
- Prepared and organized a YOLO-format object detection dataset (train/val/test).
- Created and reviewed bounding-box annotations and learned the YOLO label format.
- Configured a `data.yaml` file for a multi-class detection task.
- Trained YOLOv8 on Google Colab using pretrained weights.
- Evaluated object detection performance using Precision, Recall, mAP@0.5, and mAP@0.5:0.95.
- Interpreted confusion matrices and training/validation curves.
- Performed manual error analysis and identified failure patterns (lighting, occlusion, small objects).
- Ran both image and video inference with a trained model.
- Built an interactive Streamlit demo for real-world testing.
- Learned how to structure and document a complete AI project for GitHub.

---

## 9. Future Improvements

- Add more images, especially for underrepresented/weak classes.
- Improve class balance across the dataset.
- Add more difficult examples (occlusion, small objects, low light).
- Improve annotation quality and bounding-box consistency.
- Train for more epochs and experiment with a larger YOLO architecture (e.g., YOLOv8s).
- Remove duplicate or highly similar images.
- Perform additional hyperparameter tuning.
- Integrate real-time CCTV/video stream support.
- Add object tracking across video frames.
- Deploy the Streamlit app as a public web service.
- Test the model on a larger, more diverse real-world dataset.

---

## 📦 Project Structure

```
Helmet_Safety_Detection/
│
├── README.md
├── requirements.txt
├── app.py
├── demo.zip
│
├── dataset/
│   ├── data.yaml
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
│
├── notebooks/
│   └── training_notebook.ipynb
│
├── models/
│   └── best.pt
│
├── outputs/
│   ├── training_results/
│   │   └── yolov8n_baseline/
│   ├── predictions/
│   │   └── test_predictions/
│   └── demo_results/
│
└── report/
    └── final_report.pdf
```

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/zunaisha011/Helmet_Safety_Detection.git
```

**2. Enter the project directory**
```bash
cd Helmet_Safety_Detection
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the Streamlit application**
```bash
streamlit run app.py
```

The Streamlit app supports:
- **Image detection** — upload an image, set a confidence threshold, run detection, and view detected-object counts.
- **Video detection** — upload a video, run detection, and view the annotated output along with maximum simultaneous detections.

---

## 📋 Requirements

Main dependencies (see `requirements.txt`):

```
streamlit
ultralytics==8.4.120
opencv-python
Pillow
numpy
```

---

## 📓 Notebook & Report

- Full training workflow: `notebooks/training_notebook.ipynb`
- Final internship report (methodology, dataset prep, training, evaluation, inference, error analysis, conclusions): `report/final_report.pdf`
