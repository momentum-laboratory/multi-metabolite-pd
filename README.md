# 🧠 Quantitative Multi-Metabolite Imaging of Parkinson’s Disease using AI-Boosted Molecular MRI
<div align="center">

![Descriptive Alt Text](images/fig1_architecture.jpg)
[![arXiv](https://img.shields.io/badge/arXiv-2507.11329v1-b31b1b.svg)](https://arxiv.org/abs/2507.11329)
[![Python](https://img.shields.io/badge/language-Python-blue.svg)](https://www.python.org/)

【[Hagar Shmuely](https://github.com/hagarshmuely) | Michal Rivlin | [Or Perlman](https://github.com/operlman)】  
[Momentum Lab, Tel Aviv University](https://mri-ai.github.io/)
</div>
---

## 📌 Overview

This project demonstrates the inference pipeline described in the paper *"Quantitative Multi-Metabolite Imaging of Parkinson’s Disease using AI-Boosted Molecular MRI"*, focusing on:

- Inference on glutamate **phantoms**.
- Inference on **representative mouse brains** (pre-MPTP and post-MPTP groups) to visualize metabolite distributions and potential biomarkers.

All inference steps, along with interactive visualizations generated using Plotly, are contained in the [main.ipynb](main.ipynb) Jupyter notebook.  
You can open and run this notebook locally to explore the figures and outputs.

---

## ⚙️ Environment Setup

To set up the required Conda environment, run the following commands:

```bash
# Create a new environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate pd-multipool-cest
```
# Troubleshooting: for some systems the following packages might need to be externally installed:
```bash
conda install pytorch
pip install cmcrameri
```

---
## 🛠️ Utility Scripts

The project includes several utility Python modules that support different parts of the pipeline:

- `data_preprocessing.py`: Functions for loading, cleaning, and normalizing input data.
- `fig_gen.py`: Helpers for generating Plotly visualizations.
- `inference.py`: Core functions to run neural network inference on input data.
- `model.py`: Neural network model architecture.
- `normalization.py`: Functions for applying and reversing normalization on data.
- `colormaps.py`: Defines custom colormaps used in visualizations.

Users only need to run `main.ipynb`, but these scripts can be useful for deeper exploration or customization.

---

## 🗂️ Dataset Overview

### 🧪 Biological Data

- **19 treated mice**, each with:
  - `pre_mptp/`: baseline (pre-treatment) molecular MRI scans.
  - `post_mptp/`: post-treatment scans after MPTP-induced Parkinsonian model.

- **3 untreated mice**, each with:
  - All files stored right inside the mouse folder

Each scan instance includes:

| File              | Description                                |
|-------------------|--------------------------------------------|
| `highres.npy`     | High-resolution anatomical reference image |
| `mask.npy`        | Binary brain mask                          |
| `raw_mt.npy`      | Semisolid MT raw MRF                       |
| `raw_rnoe.npy`    | Relayed NOE raw MRF                        |
| `raw_amide.npy`   | Amide raw MRF                              |
| `raw_glu.npy`     | Glutamate raw MRF                          |
| `t1_map.npy`      | Quantitative T1 relaxation map             |
| `t2_map.npy`      | Quantitative T2 relaxation map             |


### 🔬 Phantom Data

- **3 phantom datasets**, each with:
  - `raw_glu.npy`: simulated glutamate-weighted signal.
  - `mask_1.npy`, `mask_2.npy`, `mask_3.npy`: binary masks for the 3 phantom vials.

---

## 🤖 Neural Network Weights

The `nn_weights/` directory contains trained neural network models and normalization parameters for both **in vitro** and **in vivo** datasets.

### Folder Structure
- `nn_weights/in_vitro/glu/`
- `nn_weights/in_vivo/amide_glu/`
- `nn_weights/in_vivo/mt/`
- `nn_weights/in_vivo/rnoe/`

Each of the subfolders contains:
  - `min_max_vals.npz`: NumPy file with normalization bounds (min and max per feature)
  - `trained_nn.pt`: PyTorch model weights (`state_dict`)


## 🔗 References
Shmuely, H., Rivlin, M., Perlman O. Quantitative multi-metabolite imaging of Parkinson’s disease using AI boosted
molecular MRI. npj Imaging, 2025. https://doi.org/10.1038/s44303-025-00130-x
