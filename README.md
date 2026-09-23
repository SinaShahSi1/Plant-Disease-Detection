# Plant Disease Classification and Detection

A deep learning-based system for **plant disease classification and detection** using the **PlantVillage** and **PlantDoc** datasets.

The project is organized into two main stages:

1. **Plant Disease Classification** — classification of plant leaf images into disease classes.
2. **Plant Disease Detection** — detection and localization of diseased leaves using object detection.

---

## Project Structure

```text
Plant-Disease-Detection/
│
├── Classification/
│   ├── plant_disease_classification_balanced.ipynb
│   └── plant_disease_classification_ensemble.ipynb
│
├── Detection/
│   └── plantdoc_yolov8n_detection.ipynb
│
└── README.md
```

---

# 1. Plant Disease Classification

The classification stage uses the **PlantVillage** dataset and focuses on classifying plant leaf images into their corresponding disease categories.

### Dataset Preparation

The original dataset contains an imbalanced number of images across different classes. To address this issue, the dataset is balanced using image augmentation.

The augmentation pipeline includes:

* Random horizontal flipping
* Random rotation up to ±20°
* Random brightness adjustment
* Random contrast adjustment
* Random saturation adjustment
* Random hue adjustment
* Resizing to `224 × 224`

Each class is augmented until it reaches the number of images of the largest original class.

### Dataset Split

The balanced dataset is divided into:

| Split      | Ratio |
| ---------- | ----: |
| Training   |   60% |
| Validation |   20% |
| Testing    |   20% |

A fixed random seed (`42`) is used for reproducibility.

### Models

Three pretrained convolutional neural networks are fine-tuned for the classification task:

* **ResNet-50**
* **EfficientNet-B0**
* **MobileNetV2**

All models use pretrained **ImageNet** weights, with their final classification layers replaced according to the number of classes in the PlantVillage dataset.

### Training Configuration

* Framework: **PyTorch**
* Input size: `224 × 224`
* Batch size: `32`
* Optimizer: **Adam**
* Learning rate: `0.001`
* Loss function: **Cross-Entropy Loss**
* Number of epochs: `7`
* GPU acceleration: CUDA when available

### Evaluation

The classification models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Classification report
* Confusion matrix

Training and validation accuracy/loss curves are also generated for comparison between the models.

---

# 2. Ensemble Classification

The second classification notebook combines the predictions of the trained classification models using a **weighted ensemble** approach.

The ensemble is designed to combine the complementary predictions of the individual models and improve the overall classification performance.

The ensemble experiments include combinations of:

* MobileNetV2
* EfficientNet-B0
* ResNet-50

The ensemble results are evaluated alongside the individual model performances.

---

# 3. Plant Disease Detection

The detection stage uses the **PlantDoc** dataset and an **YOLOv8n** object detection model.

Unlike the classification stage, which predicts the disease class of an entire image, the detection stage identifies and localizes diseased plant regions using bounding boxes.

### Dataset

The PlantDoc dataset is used for object detection.

The images are resized to:

```text
416 × 416
```

The dataset contains multiple plant and disease categories represented using object-detection annotations.

### Model

The baseline detection model is:

**YOLOv8n**

The model is initialized with pretrained weights and trained on the PlantDoc dataset.

### Training Configuration

Key training settings include:

* Model: YOLOv8n
* Image size: `416 × 416`
* Batch size: `8`
* Epochs: `400`
* Initial learning rate: `0.001`
* Weight decay: `0.0005`
* AMP: Disabled
* Mosaic closing: Disabled
* Early stopping patience: `10`

The best-performing checkpoint is saved and subsequently used for evaluation.

### Detection Evaluation

The detection notebook includes:

* Training and validation loss visualization
* Detection metric visualization
* Loading the best trained model
* Ground-truth bounding-box visualization
* Predicted bounding-box visualization
* Comparison of predictions with the original annotations

These visualizations provide a qualitative assessment of the model's ability to localize diseased plant regions.

---

# Datasets

This project uses two publicly available datasets:

### PlantVillage

Used for the **plant disease classification** stage.

The dataset provides labeled plant leaf images covering multiple crops and disease categories.

### PlantDoc

Used for the **plant disease detection** stage.

The dataset contains real-world plant images with object-level annotations, making it suitable for evaluating disease localization.

> The datasets are not included in this repository. They should be downloaded separately and the paths in the notebooks should be updated accordingly.

---

# Technologies

The project is implemented primarily using:

* Python
* PyTorch
* Torchvision
* Ultralytics YOLO
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Pillow
* tqdm

---

# How to Use

## 1. Clone the Repository

```bash
git clone https://github.com/SinaShahSi1/Plant-Disease-Detection.git
cd Plant-Disease-Detection
```

## 2. Install Dependencies

Create a Python environment and install the required packages:

```bash
pip install torch torchvision ultralytics numpy pandas matplotlib seaborn scikit-learn pillow tqdm
```

## 3. Download the Datasets

Download the PlantVillage and PlantDoc datasets and update the dataset paths in the corresponding notebooks.

## 4. Run the Notebooks

Open the notebooks using Jupyter Notebook, JupyterLab, or VS Code.

### Classification

```text
Classification/
├── plant_disease_classification_balanced.ipynb
└── plant_disease_classification_ensemble.ipynb
```

### Detection

```text
Detection/
└── plantdoc_yolov8n_detection.ipynb
```

---

# Project Pipeline

```text
                    Plant Disease System
                           │
             ┌─────────────┴─────────────┐
             │                           │
       Classification                Detection
             │                           │
       PlantVillage                  PlantDoc
             │                           │
     Dataset Balancing              YOLOv8n
             │                           │
     Train CNN Models              Object Detection
             │                           │
  ┌──────────┼──────────┐                 │
  │          │          │                 │
ResNet50  EfficientNet  MobileNetV2       │
  │          │          │                 │
  └──────────┼──────────┘                 │
             │                           │
      Weighted Ensemble                  │
             │                           │
             └─────────────┬─────────────┘
                           │
                 Plant Disease Analysis
```

---

# Results

The classification and detection notebooks contain the corresponding training curves, evaluation metrics, confusion matrices, and prediction visualizations.

Detailed numerical results can be found in the notebooks.

---

# Notes

* The notebooks were developed using GPU acceleration when CUDA is available.
* Dataset paths are currently configured for local execution and should be changed according to the user's environment.
* The repository contains the implementation and notebooks; the datasets themselves are not included.

---

## Author

**Sina ShahMohammadi**

GitHub: [SinaShahSi1](https://github.com/SinaShahSi1)
