# SolarFilNet-V4: Physics-Informed Multi-Scale Neural Vision for Solar Filament Segmentation

Official Solution and Reproduction Repository for the **Kaggle Solar Filament Segmentation Challenge 2026** (MAGFiLO 1.0 Benchmark, sponsored by NSF / National Solar Observatory).

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Kaggle Public LB](https://img.shields.io/badge/Public%20LB-0.38-brightgreen.svg)](https://www.kaggle.com/competitions/filament-segmentation-2026)
[![Validation PQ](https://img.shields.io/badge/Val%20PQ-0.4340-success.svg)](#benchmark-performance)

---

## 🌟 Executive Summary

Solar filaments—dense chromospheric plasma suspended by magnetic field lines above photospheric neutral lines—are critical precursors to solar flares and Coronal Mass Ejections (CMEs). Segmenting them from ground-based GONG H-Alpha observations ($2048 \times 2048$ pixels) poses severe challenges due to low contrast, imaging artifacts, turbulent chromospheric backgrounds, and delicate, elongated barb topologies.

**SolarFilNet-V4** is a unified neural vision and physics-informed post-processing pipeline designed specifically for solar filament morphology:
1. **Curvilinear Geometry & Multi-Scale Vision**: Integrates a Pyramid Vision Transformer (`pvt_v2_b1`) with anisotropic $1 \times k$ and $k \times 1$ strip convolutions (`CurvilinearStripConv`) to mirror the elongated curvature of filament ribbons.
2. **Dual-Frequency Context (`DFFN`)**: Employs gated depthwise dilated feed-forward blocks to simultaneously capture fine spine threads and broad filament bodies.
3. **Physics-Informed Chromospheric Enhancement**: Computes local background chromospheric absorption and adaptive contrast suppression for active plage regions.
4. **$D_4$ Dihedral Symmetry ($8\times$ TTA)**: Exploits the natural rotation and reflection invariance of the solar disk.
5. **Solar Regime Adaptive Post-Processing**: Calibrates morphological joining and area filtering dynamically based on the Sun's activity level (Quiet vs. Active Sun).

---

## 🏆 Benchmark Performance

| Method | Backbone | Input Res | Val SQ | Val RQ | Val PQ | Public LB |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline UNet-R34 | ResNet-34 | 1024 | 0.6521 | 0.5360 | 0.3501 | 0.36 |
| SolarFilNet-V2 | MiT-B3 | 1024 | 0.6480 | 0.4480 | 0.2906 | 0.29 |
| SolarFilNet-V3 Baseline | PVT-v2-B1 | 1024 | 0.6812 | 0.5841 | 0.3978 | 0.37 |
| **SolarFilNet-V4 (Pure Neural)** | **PVT-v2-B1** | **1024** | **0.6974** | **0.6120** | **0.4268** | **0.37** |
| **SolarFilNet-V4 (Adaptive PP)** | **PVT-v2-B1** | **1024** | **0.6974** | **0.6224** | **0.4340** | **0.38** |

> **Official Host Reference**: The competition hosts noted that *"Any PQ score of greater than 0.35 is of great value to us."* SolarFilNet-V4 consistently surpasses this milestone on both local cross-validation and the competition Public Leaderboard.

---

## 📂 Repository Structure

```
SolarFilNet-V4-Submission/
├── README.md                      # Comprehensive documentation and pipeline description
├── requirements.txt               # Complete dependencies and package versions
├── LICENSE                        # MIT Open Source License
├── reproduce_submission.ipynb     # Self-contained Jupyter Notebook reproducing predictions
│
├── src/                           # Modular source code
│   ├── __init__.py                # Package exports
│   ├── model.py                   # SolarFilNetV4 architecture definition
│   ├── config.py                  # Model & inference hyperparameters
│   └── postproc.py                # Preprocessing, adaptive decoding & official PQ evaluator
│
├── weights/                       # Pre-trained model weights & configs
│   ├── ckpt_solarfilnet_v4.pt     # Official model checkpoint (59.1 MB, loaded directly)
│   └── best_postproc_adaptive.json# Calibrated quiet/active post-processing hyperparameters
│
└── output/                        # Submission artifacts
    └── submission_solarfilnet_v4_adaptive.csv # Verified competition submission file
```

---

## 🚀 Quickstart & Reproduction Guide

### 1. Installation

Clone this repository and install all required packages:
```bash
git clone https://github.com/<your-username>/SolarFilNet-V4.git
cd SolarFilNet-V4
pip install -r requirements.txt
```

### 2. Reproduce Predictions via Jupyter Notebook (Recommended)

Open and run **`reproduce_submission.ipynb`**:
```bash
jupyter notebook reproduce_submission.ipynb
```
The notebook automatically:
1. Verifies the compute device (GPU / CPU).
2. Instantiates `SolarFilNetV4` and loads `weights/ckpt_solarfilnet_v4.pt`.
3. Loads `weights/best_postproc_adaptive.json`.
4. Runs $8\times$ TTA inference across the test images.
5. Performs adaptive morphological decoding into COCO RLE format.
6. Audits the output schema against the official submission rubric.

---

## 🧠 Architectural Highlights

```
                                  Input Image (2048 x 2048)
                                              │
                              ┌───────────────┴───────────────┐
                              ▼                               ▼
                 Shallow Full-Res Branch (F0)     Physics-Informed Preprocessing
                  [Conv 3x3, BatchNorm, GELU]      [Ch0: Contrast, Ch1: Residue, Ch2: CLAHE]
                              │                               │
                              │                  PVT-v2-B1 Multi-Scale Backbone
                              │                  [Stages 1, 2, 3, 4 Features]
                              │                               │
                              │                  Curvilinear Bottleneck (k=15)
                              │                     + DFFN Residual Gate
                              │                               │
                              ▼                               ▼
                       Feature Pyramid Decoder (Levels 3 -> 2 -> 1)
                                              │
                                              ▼
                                Full-Resolution Decoder (D0)
                                              │
                       ┌──────────────────────┼──────────────────────┐
                       ▼                      ▼                      ▼
                   Mask Head              Spine Head             Edge Head
                (Filament Body)        (Center Skeleton)      (Barb Boundaries)
```

1. **Curvilinear Strip Convolution (`CurvilinearStripConv`)**:
   Standard isotropic square kernels ($3 \times 3$ or $5 \times 5$) fail to capture thin, meandering filament spines. We use coupled parallel $1 \times k$ and $k \times 1$ asymmetric strip convolutions ($k=15$) to trace extended solar magnetic flux ropes.
2. **Dual-Frequency Context (`DFFNBlock`)**:
   Gated depthwise convolutions with complementary receptive fields (local $3 \times 3$ and dilated $3 \times 3$, $d=2$) enable joint learning of high-frequency spine ridges and low-frequency filament envelopes.
3. **Multi-Task Topology Guidance**:
   The model is trained with composite supervision including Soft Dice Loss, Focal Tversky Loss, Centerline Dice (`clDice`), and Pixel-Contrastive Loss (`PCL`) to preserve continuity.

---

## 🔬 Physics-Informed Post-Processing

Solar filaments exist across distinct solar environments. In active regions, intense plages cause threshold saturation, while quiet regions require sensitive low-contrast delineation:
- **Quiet Sun**: Standard joining dilation ($7\text{ px}$), area cutoff ($200\text{ px}$), score threshold ($0.80$).
- **Active Sun ($\ge 0.65\%$ Plage Area)**: Reinforced joining dilation ($9\text{ px}$) to bridge fragments across bright plage backgrounds without inducing false-positive noise.

---

## 📊 Evaluation Metric

Predictions are evaluated using **Panoptic Quality ($PQ$)**, as defined by Kirillov et al. (CVPR 2019) and configured in the official host evaluation script:
$$\text{PQ} = \frac{\sum_{(p, g) \in \text{TP}} \text{IoU}(p, g)}{|\text{TP}| + \frac{1}{2}|\text{FP}| + \frac{1}{2}|\text{FN}|} = \text{SQ} \times \text{RQ}$$
where:
- $\text{SQ} = \frac{1}{|\text{TP}|}\sum_{(p, g) \in \text{TP}} \text{IoU}(p, g)$ (Segmentation Quality)
- $\text{RQ} = \frac{|\text{TP}|}{|\text{TP}| + \frac{1}{2}|\text{FP}| + \frac{1}{2}|\text{FN}|}$ (Recognition Quality)
- Matches are determined by $\text{IoU}(p, g) > 0.50$ per annotator layer.

---

## 📜 Citation & License

This codebase is licensed under the [MIT License](LICENSE).

```bibtex
@misc{solarfilnet2026,
  title={SolarFilNet-V4: Physics-Informed Multi-Scale Neural Vision for Solar Filament Segmentation},
  author={SolarFilNet Team},
  year={2026},
  publisher={GitHub},
  howpublished={\url{https://github.com/<your-username>/SolarFilNet-V4}}
}
```
