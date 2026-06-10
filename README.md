# Limited Strong-Keypoint Multi-Subregion Zero-Watermarking

This repository contains the Python implementation and experimental data used for the paper:

**Robust Zero-Watermarking via Limited Strong-Keypoint Matching and DWT-DCT Sub-Region Features**

The code is organized for reproducibility. It includes the proposed method, representative baseline implementations used in the comparison, attack protocols, metrics, and the image sets used in the experiments.

## Repository structure

```text
.
├── common/                         # Common attack, metric, IO, and Arnold utilities
├── methods/                        # Proposed method and representative baselines
├── data/
│   ├── watermark/                  # Binary watermark used in the experiments
│   ├── dataset2_medical_12/         # 12 medical images used for parameter/comparison tests
│   ├── dataset2_natural_12/         # 12 natural images used for parameter/ablation tests
│   ├── dataset1_medical_400/        # 400 medical images for large-scale evaluation
│   ├── dataset1_natural_600/        # 600 natural images for large-scale evaluation
│   └── same_modality/               # Same-modality medical subsets for impostor testing
├── run_param_sweep_revision.py      # Parameter analysis
├── run_large_scale_proposed.py      # Robustness, discriminability, runtime/storage on Dataset 1
├── run_hard_negative_discriminability.py
├── run_ablation_studies.py          # Ablation experiments
├── run_baseline_comparison_suite1.py# Comparison with representative methods
├── experiment_utils.py              # Shared experiment wrappers
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

## Installation

Python 3.12 was used in the experiments. Create an environment and install the required packages:

```bash
pip install -r requirements.txt
```

The proposed method uses SIFT. If your OpenCV build does not include SIFT, install `opencv-contrib-python`.

## Quick checks

Run a small parameter sweep on Dataset 2:

```bash
python run_param_sweep_revision.py --out-dir out_param_sweep --suite suite2
```

Run the large-scale proposed-method evaluation:

```bash
python run_large_scale_proposed.py --out-dir out_large_scale_1K --suite suite2 --NC-threshold 0.70
```

Run same-modality medical impostor evaluation:

```bash
python run_hard_negative_discriminability.py --out-dir out_same_modality
```

Run the ablation study:

```bash
python run_ablation_studies.py --out-dir out_ablation --suite suite2
```

Run the comparison with representative methods using the Nawaz-style medical-image protocol:

```bash
python run_baseline_comparison_suite1.py --out-dir out_baseline_suite1
```

## Notes on data

All images are resized to 512 x 512 by the experiment scripts. The watermark is located at `data/watermark/watermark.png` and is binarized to 32 x 32 by default.

The included datasets are provided only for academic reproducibility of the reported experiments. Please also respect the licenses and terms of the original image sources.

The natural-image subset of Dataset 1 is stored as data/dataset1_natural_600/dataset1_natural_600.zip. Please extract this file before running the large-scale Dataset 1 experiments. After extraction, the individual image files should be located directly in data/dataset1_natural_600/.

## Citation

If this repository is useful for your research, please cite the associated paper.
