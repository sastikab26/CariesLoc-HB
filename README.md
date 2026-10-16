CariesLoc-HB: A Lesion-Preserving Learning Framework for Dental Caries Localization in Intraoral RGB Images
This repository contains the Python implementation of CariesLoc-HB, a data-centric framework for localizing dental caries in intraoral RGB images using YOLO11m. It combines hard-background enhanced lesion-aware sampling with lesion-preserving augmentation to detect `primary_decay` and `permanent_decay`, without modifying the YOLO11m architecture.
---
Overview
The framework includes five main modules:
Preprocessing — Image–annotation validation and retention of background images with empty YOLO labels.
Hard-Background Enhanced Lesion-Aware Sampling — Capped repeat sampling prioritizing backgrounds, minority-class lesions, small lesions, and multiple lesions.
Lesion-Preserving Augmentation — Mild colour and geometric transformations applied during training.
Training — YOLO11m-based caries localization.
Testing — Overall and class-wise detection evaluation.
---
Repository Structure
```text
CariesLoc-HB/
├── 01_CariesLoc_Preprocessing.py
├── 02_CariesLoc_HB_Lesion_Sampling.py
├── 03_CariesLoc_Lesion_Augmentation.py
├── 04_CariesLoc_Training.py
├── 05_CariesLoc_Testing.py
├── README.md
├── requirements.txt
└── .gitignore
```
---
Environment Setup
```bash
conda create -n CariesLocHB python=3.10 -y
conda activate CariesLocHB
pip install -r requirements.txt
```
Verify GPU availability:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
For GPU training, install a PyTorch build compatible with your CUDA environment. Record the versions used in published experiments.
---
Dataset Organization
Provide an existing patient-wise split in YOLO format:
```text
dataset/
├── images/
│   ├── train/
│   ├── valid/
│   └── test/
├── labels/
│   ├── train/
│   ├── valid/
│   └── test/
└── data.yaml
```
YOLO annotations use `class_id x_center y_center width height` with normalized coordinates. Class `0` denotes `primary_decay`; class `1` denotes `permanent_decay`. Images without annotated lesions retain empty label files. Patient-wise splitting must be performed before using these scripts; the repository does not generate patient assignments.
---
Methodology
Preprocessing validates image–label pairs without aggressive colour or contrast enhancement. The sampling stage prioritizes challenging negative samples and underrepresented lesions, using a maximum repeat factor of 4 and a normalized small-lesion area threshold of 0.015.
During training, mild HSV and geometric changes preserve tooth appearance and bounding-box alignment; vertical flips, MixUp, and Copy-Paste are disabled. YOLO11m uses 1024 × 1024 inputs, AdamW, batch size 4, and up to 180 epochs.
---
Evaluation
The testing module reports precision, recall, F1-score, mAP@0.5, and mAP@0.5:0.95, alongside class-wise metrics. It can evaluate an independent external dataset using a frozen checkpoint without fine-tuning.
---
Code Availability
Source code is provided for research and reproducibility. Datasets, clinical images, and trained checkpoints are not distributed in this repository. Reported manuscript results should be verified using the original data partitions and experiment environment.
