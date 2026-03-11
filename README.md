# University Dissertation – Comparing YOLOv8 and YOLOv11 for Medicine Detection

## Project Overview

This dissertation investigates how well two state-of-the-art object detection models — **YOLOv8** and **YOLOv11** — can detect medicines on pharmacy shelves under varying levels of visual clutter. The goal is to determine whether clutter significantly degrades detection performance and whether one model handles clutter better than the other.

The project covers the full machine-learning pipeline:
1. **Dataset preparation** – A labelled medicine detection dataset was downloaded from Roboflow and used to fine-tune both models on Google Colab (A100 GPU).
2. **Clutter level definition** – Reference images with controlled amounts of clutter were photographed and classified into four levels (0 – 3) using pixel-coverage analysis.
3. **Model evaluation** – Both trained models were run on 100 test images (25 per clutter level). Bounding-box predictions were compared against ground-truth annotations to calculate Precision, Recall, F1-Score, and IoU.
4. **Statistical analysis** – Kruskal-Wallis and Dunn's Post-Hoc tests were used to assess whether performance differences across clutter levels were statistically significant.

---

## Technologies Used

| Category | Tools / Libraries |
|---|---|
| **Programming Language** | Python 3.13 |
| **Object Detection** | [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (YOLOv8 & YOLOv11) |
| **Deep Learning Framework** | PyTorch 2.6 |
| **Computer Vision** | OpenCV (`cv2`) |
| **Numerical Computing** | NumPy |
| **Data Manipulation** | Pandas |
| **Statistical Analysis** | SciPy (`kruskal`), Statsmodels (`pairwise_tukeyhsd`) |
| **Visualisation** | Matplotlib, Seaborn |
| **Image Editing (clutter masks)** | GIMP |
| **Dataset Source** | [Roboflow](https://roboflow.com/) |
| **Training Platform** | Google Colab (NVIDIA A100 GPU) |
| **Environment Management** | Anaconda / Conda |

---

## Repository Structure

```
University_Dissertation/
│
├── Analysis/                        # Evaluation pipeline
│   ├── Test.py                      # Runs inference, calculates Precision/Recall/F1/IoU, saves results CSV and output images
│   ├── KWH_Test.py                  # Kruskal-Wallis + Dunn's Post-Hoc statistical tests
│   ├── YOLOv8.pt                    # Trained YOLOv8 weights used for evaluation
│   ├── YOLOv11.pt                   # Trained YOLOv11 weights used for evaluation
│   ├── test_images/                 # Test images organised by clutter level (Level_0 – Level_3)
│   │   └── Rename_files.py          # Utility script for renaming test image files
│   └── ground_truth/                # YOLO-format annotation (.txt) files matching test images
│
├── Clutter_Calculator/              # Clutter level methodology and reference images
│   ├── Clutter_Calculator.py        # Calculates the % of pixels covered by clutter using OpenCV
│   ├── Calculating_clutter_levels.md # Step-by-step explanation of the clutter classification process
│   ├── Results.md                   # Clutter percentage results for each reference image
│   ├── Clutter_L1_ref.png           # Reference image for Level 1 clutter (~25%)
│   ├── Clutter_L2_ref.png           # Reference image for Level 2 clutter (~50%)
│   ├── Clutter_L3_ref.png           # Reference image for Level 3 clutter (~75%)
│   └── Clutter_mask*.png            # Binary masks (white = clutter, black = background)
│
├── YOLO8 V1.0/                      # YOLOv8 training run artefacts
│   ├── Weights/best.pt              # Best model weights from training
│   ├── Weights/last.pt              # Last epoch weights
│   ├── args.yaml                    # Training configuration (epochs: 100, batch: 16, imgsz: 640)
│   ├── results (1).csv              # Per-epoch training/validation metrics
│   └── *.png / *.jpg                # Training curves, confusion matrices, sample batches
│
├── YOLO11 V1.0/                     # YOLOv11 training run artefacts (same structure as YOLO8 V1.0)
│
├── output_images_v8_0.1/            # YOLOv8 inference output images (version 0.1)
├── output_images_v8_0.2/            # YOLOv8 inference output images (version 0.2 – latest)
├── output_images_v11_0.1/           # YOLOv11 inference output images (version 0.1)
├── output_images_v11_0.2/           # YOLOv11 inference output images (version 0.2 – latest)
│
├── YOLOv8 Medicine Detection.v1i.yolov8.zip   # Training dataset (YOLOv8 format)
├── YOLOv11 Medicine Detection.v1i.yolov11.zip # Training dataset (YOLOv11 format)
│
├── results_0.1.csv                  # Evaluation results – first test run
├── results_0.2.csv                  # Evaluation results – latest test run (use this one)
├── Google_Colab_logs.ipynb          # Jupyter notebook with the Google Colab training commands and logs
├── environment.txt                  # Conda environment file (Windows 64-bit, Python 3.13)
└── LICENSE
```

---

## Clutter Levels

Images were tested at four clutter levels defined by the percentage of the shelf area covered by non-medicine items:

| Level | Clutter Coverage | Description |
|---|---|---|
| **0** | 0 % | No clutter |
| **1** | ~25 % (±10 %) | Light clutter |
| **2** | ~50 % (±10 %) | Moderate clutter |
| **3** | ~75 % (±10 %) | Heavy clutter |

Clutter masks were created manually in **GIMP** and the percentage was calculated using the `Clutter_Calculator.py` script (OpenCV pixel counting).

---

## Setup & Installation

### 1. Recreate the Conda environment (Windows)

```bash
conda create --name dissertation --file environment.txt
conda activate dissertation
```

### 2. Install required Python packages

```bash
pip install ultralytics opencv-python pandas numpy scipy statsmodels matplotlib seaborn
```

> **Note:** The base `environment.txt` contains the core Python 3.13 Conda environment. The packages above must be installed separately with `pip` into that environment.

---

## How to Run

### Run the model evaluation

Update the file paths at the top of `Analysis/Test.py` to match your local directory structure, then run:

```bash
python Analysis/Test.py
```

This will:
- Load both trained models (`YOLOv8.pt` and `YOLOv11.pt`)
- Run inference on all test images in `Analysis/test_images/`
- Compare predictions to ground-truth annotations in `Analysis/ground_truth/`
- Save per-image metrics (Precision, Recall, F1-Score, IoU) to `results.csv`
- Save annotated output images to `output_images_v8/` and `output_images_v11/`

### Run the statistical analysis

Update the CSV path in `Analysis/KWH_Test.py`, then run:

```bash
python Analysis/KWH_Test.py
```

This will perform a **Kruskal-Wallis test** on F1-Score across clutter levels for each model and, if statistically significant differences are found, follow up with a **Dunn's Post-Hoc test**.

### Calculate clutter coverage for a new image

```bash
python Clutter_Calculator/Clutter_Calculator.py
```

---

## Results

Evaluation results are stored in `results_0.2.csv` (the most recent test run). Each row contains:

| Column | Description |
|---|---|
| `Model` | `YOLOv8` or `YOLOv11` |
| `Clutter_Level` | `Level_0` – `Level_3` |
| `Image_Name` | Test image filename |
| `Num_Detections` | Number of objects detected |
| `Avg_Confidence` | Mean detection confidence |
| `TP` / `FP` / `FN` | True Positives / False Positives / False Negatives (IoU threshold = 0.5) |
| `Precision` | TP / (TP + FP) |
| `Recall` | TP / (TP + FN) |
| `F1_Score` | Harmonic mean of Precision and Recall |
| `Avg_IOU` | Mean Intersection over Union across matched detections |

Output images in `output_images_v8_0.2/` and `output_images_v11_0.2/` show bounding boxes colour-coded as **green (TP)** or **red (FP)**, with the IoU score printed next to each box.

---

## Training

Both models were trained on Google Colab using an **NVIDIA A100 GPU**. The training commands and session logs are documented in `Google_Colab_logs.ipynb`.

Key training configuration (same for both models):

| Parameter | Value |
|---|---|
| Base model | `yolov8n.pt` / `yolo11n.pt` (nano) |
| Epochs | 100 |
| Batch size | 16 |
| Image size | 640 × 640 |
| Dataset | Roboflow – Medicine Detection v1 |
