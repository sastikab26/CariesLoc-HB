# CariesLoc-HB

**A Lesion-Preserving Learning Framework for Dental Caries Localization in Intraoral RGB Images**

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![YOLO11m](https://img.shields.io/badge/Model-YOLO11m-00A3E0)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

CariesLoc-HB detects and localizes dental caries in intraoral photographs using **YOLO11m**. It detects two classes, `primary_decay` and `permanent_decay`.

Instead of changing the model architecture, it improves the training data in two ways:

- **Hard-background lesion-aware sampling** shows the model difficult lesion-free images, rare lesions, small lesions and multi-lesion images more often during training.
- **Lesion-preserving augmentation** uses only mild colour and geometric changes, so teeth and lesions keep their natural appearance.

---

## Repository Structure

```text
CariesLoc-HB/
├── 01_CariesLoc_Preprocessing.py        # Check and clean the dataset
├── 02_CariesLoc_HB_Lesion_Sampling.py   # Build the weighted training list
├── 03_CariesLoc_Lesion_Augmentation.py  # Augmentation settings
├── 04_CariesLoc_Training.py             # Train YOLO11m
├── 05_CariesLoc_Testing.py              # Evaluate the trained model
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Installation

```bash
git clone https://github.com/sastikab26/CariesLoc-HB.git
cd CariesLoc-HB

conda create -n CariesLocHB python=3.10 -y
conda activate CariesLocHB
pip install -r requirements.txt
```

A GPU is recommended. Check that PyTorch can see it:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Dataset

Prepare your dataset in YOLO format, already split **by patient** into train, validation and test sets:

```text
dataset/
├── images/
│   ├── train/
│   ├── valid/
│   └── test/
└── labels/
    ├── train/
    ├── valid/
    └── test/
```

- Each label file has one line per lesion: `class_id x_center y_center width height` (normalized to 0–1).
- Class `0` is `primary_decay` and class `1` is `permanent_decay`.
- Images without lesions should have an **empty** label file. These are kept and used as background examples.

> This repository does not split the data for you. Make sure all images from one patient are in the same split.

---

## Usage

Run the scripts in order. Replace the example paths with your own.

**1. Preprocess the dataset**

```bash
python 01_CariesLoc_Preprocessing.py \
    --input-root dataset \
    --output-root dataset_clean
```

**2. Create the weighted training set**

```bash
python 02_CariesLoc_HB_Lesion_Sampling.py \
    --train-images dataset_clean/images/train \
    --train-labels dataset_clean/labels/train \
    --val-images dataset_clean/images/val \
    --test-images dataset_clean/images/test \
    --output-dir runs/sampling
```

**3. Augmentation**

Nothing to run. The training script loads these settings automatically. To view them:

```bash
python 03_CariesLoc_Lesion_Augmentation.py
```

**4. Train the model**

```bash
python 04_CariesLoc_Training.py \
    --data runs/sampling/sampling_data.yaml \
    --output-dir runs/train \
    --device 0
```

The best model is saved to `runs/train/cariesloc_hb_yolo11m/weights/best.pt`.

**5. Test the model**

```bash
python 05_CariesLoc_Testing.py \
    --weights runs/train/cariesloc_hb_yolo11m/weights/best.pt \
    --data runs/sampling/sampling_data.yaml \
    --split test \
    --out results/test
```

This reports precision, recall, F1-score, mAP@0.5 and mAP@0.5:0.95, both overall and for each class. The same script can evaluate an external dataset with the trained model, without any fine-tuning.

Run any script with `--help` to see all of its options.

---

## Training Setup

YOLO11m is trained with 1024 × 1024 images, the AdamW optimizer, a batch size of 4 and up to 180 epochs. Vertical flips, MixUp and Copy-Paste are turned off to keep lesions realistic.

---

## Data Availability

This repository contains source code only. Datasets, clinical images and trained model weights are not included.

---

## Disclaimer

This project is for research use only. It is not a medical device and should not be used for clinical diagnosis.

---

## Citation

If you use this code, please cite the associated paper. Citation details will be added after publication.

---

## License

This project is released under the [MIT License](LICENSE).
