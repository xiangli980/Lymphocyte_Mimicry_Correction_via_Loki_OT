# Lymphocyte Mimicry Correction via Region-Level Tissue Reasoning and Unbalanced Optimal Transport

[![MICCAI COMPAly 2024](https://img.shields.io/badge/MICCAI-COMPAly%20Workshop-blue)](https://sites.google.com/view/compaly)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation for the paper:

> **Lymphocyte Mimicry Correction via Region-Level Tissue Reasoning and Unbalanced Optimal Transport**  
> Accepted to MICCAI COMPAly Workshop

---

## Overview

Lymphocytes in histopathology images can be confused with other small round cells (mimicry), leading to inaccurate lymphocyte quantification. This work proposes a correction framework that:

1. **Region-Level Tissue Reasoning** — uses tissue context to reason about the local microenvironment and discriminate true lymphocytes from mimickers.
2. **Loki OT (Unbalanced Optimal Transport)** — applies unbalanced optimal transport to align cell-level feature distributions and correct misclassified detections.

The pipeline takes cell detection outputs and tissue segmentation masks as input and produces corrected lymphocyte labels.

---

## Repository Structure

```
.
├── notebooks/
│   ├── 01_data_exploration.ipynb        # Data loading and visualization
│   ├── 02_tissue_reasoning.ipynb        # Region-level tissue reasoning demo
│   └── 03_loki_ot_correction.ipynb      # Loki OT correction pipeline demo
├── src/
│   ├── __init__.py
│   ├── tissue_reasoning/
│   │   ├── __init__.py
│   │   ├── region_extractor.py          # Tissue region feature extraction
│   │   └── tissue_classifier.py         # Region-level tissue type classification
│   ├── loki_ot/
│   │   ├── __init__.py
│   │   ├── cost_matrix.py               # Cell-level cost matrix construction
│   │   └── unbalanced_ot.py             # Unbalanced optimal transport solver (Loki OT)
│   └── correction/
│       ├── __init__.py
│       └── pipeline.py                  # End-to-end correction pipeline
├── configs/
│   └── default.yaml                     # Default hyperparameters
├── requirements.txt                     # Python dependencies
└── README.md
```

---

## Installation

### Requirements

- Python 3.8 or higher
- PyTorch 1.12+
- CUDA 11.3+ (for GPU support, recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/xiangli980/Lymphocyte_Mimicry_Correction_via_Loki_OT.git
cd Lymphocyte_Mimicry_Correction_via_Loki_OT

# Create a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Running the Notebooks

The notebooks in `notebooks/` walk through the full pipeline step by step. Start Jupyter and open them in order:

```bash
jupyter notebook notebooks/
```

### Running the Correction Pipeline

```python
from src.correction.pipeline import CorrectionPipeline

pipeline = CorrectionPipeline(config="configs/default.yaml")
corrected_labels = pipeline.run(
    detection_csv="path/to/cell_detections.csv",
    tissue_mask="path/to/tissue_mask.png",
)
```

### Configuration

Edit `configs/default.yaml` to adjust hyperparameters such as:

- `tissue_reasoning.num_classes` — number of tissue types
- `loki_ot.epsilon` — entropic regularisation for OT
- `loki_ot.rho` — marginal relaxation for unbalanced OT

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{li2024lymphocyte,
  title     = {Lymphocyte Mimicry Correction via Region-Level Tissue Reasoning and Unbalanced Optimal Transport},
  author    = {Li, Xiang and others},
  booktitle = {MICCAI Workshop on Computational Pathology (COMPAly)},
  year      = {2024},
}
```

---

## License

This project is released under the [MIT License](LICENSE).
