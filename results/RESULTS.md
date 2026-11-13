# Results

Mean ± 95% CI half-width over **5 random draws** of the labelled images (the same draws for every method). Metric: **balanced accuracy** on the full official test set (chance = 1 / number of classes). Configuration: `outputs/config.json`.

## Overall ranking

Average rank across all dataset × label-budget settings (1 = best):

| Method | Average rank |
|---|---|
| ResNet-18 fine-tuned (ImageNet init) | 1.50 |
| Frozen ResNet-18 (ImageNet) + logistic regression | 2.31 |
| Frozen DINOv2 ViT-S/14 + logistic regression | 3.12 |
| Handcrafted (HOG+LBP+histogram) + Random Forest | 3.56 |
| Frozen CLIP ViT-B/32 + logistic regression | 4.50 |

## Balanced accuracy

### PneumoniaMNIST (chest X-ray) — 2 classes

| Method | 1-shot | 5-shot | 10-shot | 50-shot |
|---|---|---|---|---|
| HOG + RF | 0.647 ± 0.084 | 0.771 ± 0.073 | 0.796 ± 0.026 | 0.825 ± 0.017 |
| ResNet-18 fine-tuned | 0.656 ± 0.149 | 0.825 ± 0.049 | 0.849 ± 0.039 | 0.911 ± 0.018 |
| ResNet-18 frozen | 0.617 ± 0.148 | 0.840 ± 0.056 | 0.849 ± 0.030 | 0.881 ± 0.022 |
| DINOv2 frozen | 0.567 ± 0.115 | 0.721 ± 0.103 | 0.789 ± 0.036 | 0.837 ± 0.034 |
| CLIP frozen | 0.568 ± 0.092 | 0.650 ± 0.061 | 0.691 ± 0.062 | 0.695 ± 0.051 |

### DermaMNIST (skin) — 7 classes

| Method | 1-shot | 5-shot | 10-shot | 50-shot |
|---|---|---|---|---|
| HOG + RF | 0.179 ± 0.041 | 0.239 ± 0.019 | 0.290 ± 0.021 | 0.439 ± 0.029 |
| ResNet-18 fine-tuned | 0.257 ± 0.050 | 0.458 ± 0.047 | 0.488 ± 0.030 | 0.645 ± 0.023 |
| ResNet-18 frozen | 0.265 ± 0.041 | 0.357 ± 0.056 | 0.392 ± 0.055 | 0.504 ± 0.022 |
| DINOv2 frozen | 0.271 ± 0.030 | 0.351 ± 0.058 | 0.388 ± 0.031 | 0.538 ± 0.021 |
| CLIP frozen | 0.259 ± 0.070 | 0.352 ± 0.035 | 0.368 ± 0.023 | 0.480 ± 0.018 |

### BloodMNIST (blood cells) — 8 classes

| Method | 1-shot | 5-shot | 10-shot | 50-shot |
|---|---|---|---|---|
| HOG + RF | 0.345 ± 0.038 | 0.615 ± 0.041 | 0.737 ± 0.012 | 0.834 ± 0.006 |
| ResNet-18 fine-tuned | 0.580 ± 0.036 | 0.825 ± 0.030 | 0.868 ± 0.027 | 0.945 ± 0.007 |
| ResNet-18 frozen | 0.383 ± 0.022 | 0.587 ± 0.046 | 0.650 ± 0.024 | 0.763 ± 0.007 |
| DINOv2 frozen | 0.415 ± 0.034 | 0.636 ± 0.046 | 0.715 ± 0.025 | 0.814 ± 0.018 |
| CLIP frozen | 0.304 ± 0.084 | 0.476 ± 0.058 | 0.513 ± 0.041 | 0.633 ± 0.018 |

### BreastMNIST (breast ultrasound) — 2 classes

| Method | 1-shot | 5-shot | 10-shot | 50-shot |
|---|---|---|---|---|
| HOG + RF | 0.538 ± 0.104 | 0.654 ± 0.061 | 0.633 ± 0.138 | 0.730 ± 0.022 |
| ResNet-18 fine-tuned | 0.520 ± 0.082 | 0.674 ± 0.094 | 0.716 ± 0.036 | 0.805 ± 0.055 |
| ResNet-18 frozen | 0.553 ± 0.121 | 0.718 ± 0.068 | 0.703 ± 0.107 | 0.783 ± 0.022 |
| DINOv2 frozen | 0.491 ± 0.081 | 0.667 ± 0.060 | 0.635 ± 0.081 | 0.732 ± 0.020 |
| CLIP frozen | 0.491 ± 0.060 | 0.537 ± 0.044 | 0.579 ± 0.029 | 0.644 ± 0.036 |

## Macro AUC (one-vs-rest)

### PneumoniaMNIST (chest X-ray) — 2 classes

| Method | 1-shot | 5-shot | 10-shot | 50-shot |
|---|---|---|---|---|
| HOG + RF | 0.757 ± 0.048 | 0.850 ± 0.065 | 0.882 ± 0.011 | 0.925 ± 0.016 |
| ResNet-18 fine-tuned | 0.830 ± 0.113 | 0.907 ± 0.046 | 0.938 ± 0.024 | 0.972 ± 0.006 |
| ResNet-18 frozen | 0.833 ± 0.142 | 0.928 ± 0.007 | 0.934 ± 0.011 | 0.949 ± 0.013 |
| DINOv2 frozen | 0.662 ± 0.171 | 0.831 ± 0.079 | 0.886 ± 0.034 | 0.930 ± 0.015 |
| CLIP frozen | 0.651 ± 0.148 | 0.702 ± 0.109 | 0.776 ± 0.058 | 0.787 ± 0.044 |

### DermaMNIST (skin) — 7 classes

| Method | 1-shot | 5-shot | 10-shot | 50-shot |
|---|---|---|---|---|
| HOG + RF | 0.561 ± 0.058 | 0.642 ± 0.034 | 0.680 ± 0.037 | 0.809 ± 0.015 |
| ResNet-18 fine-tuned | 0.663 ± 0.071 | 0.803 ± 0.038 | 0.825 ± 0.013 | 0.905 ± 0.011 |
| ResNet-18 frozen | 0.657 ± 0.059 | 0.749 ± 0.043 | 0.769 ± 0.013 | 0.834 ± 0.013 |
| DINOv2 frozen | 0.667 ± 0.047 | 0.758 ± 0.026 | 0.784 ± 0.012 | 0.862 ± 0.015 |
| CLIP frozen | 0.642 ± 0.073 | 0.734 ± 0.031 | 0.753 ± 0.028 | 0.833 ± 0.017 |

### BloodMNIST (blood cells) — 8 classes

| Method | 1-shot | 5-shot | 10-shot | 50-shot |
|---|---|---|---|---|
| HOG + RF | 0.758 ± 0.040 | 0.899 ± 0.020 | 0.944 ± 0.007 | 0.974 ± 0.003 |
| ResNet-18 fine-tuned | 0.904 ± 0.030 | 0.977 ± 0.006 | 0.985 ± 0.004 | 0.997 ± 0.001 |
| ResNet-18 frozen | 0.805 ± 0.052 | 0.899 ± 0.011 | 0.916 ± 0.010 | 0.955 ± 0.003 |
| DINOv2 frozen | 0.802 ± 0.030 | 0.907 ± 0.013 | 0.932 ± 0.006 | 0.966 ± 0.004 |
| CLIP frozen | 0.700 ± 0.028 | 0.824 ± 0.032 | 0.863 ± 0.023 | 0.912 ± 0.008 |

### BreastMNIST (breast ultrasound) — 2 classes

| Method | 1-shot | 5-shot | 10-shot | 50-shot |
|---|---|---|---|---|
| HOG + RF | 0.550 ± 0.149 | 0.693 ± 0.038 | 0.678 ± 0.179 | 0.803 ± 0.018 |
| ResNet-18 fine-tuned | 0.542 ± 0.130 | 0.789 ± 0.055 | 0.784 ± 0.045 | 0.871 ± 0.035 |
| ResNet-18 frozen | 0.564 ± 0.213 | 0.799 ± 0.050 | 0.775 ± 0.157 | 0.839 ± 0.027 |
| DINOv2 frozen | 0.505 ± 0.112 | 0.721 ± 0.074 | 0.712 ± 0.101 | 0.815 ± 0.027 |
| CLIP frozen | 0.497 ± 0.125 | 0.574 ± 0.087 | 0.648 ± 0.024 | 0.703 ± 0.023 |

## Paired comparison: ResNet-18 frozen − ResNet-18 fine-tuned

Difference in balanced accuracy, paired by draw (positive = frozen features better).

| Dataset | 1-shot | 5-shot | 10-shot | 50-shot |
|---|---|---|---|---|
| PneumoniaMNIST (chest X-ray) | -0.039 ± 0.034 | +0.015 ± 0.072 | +0.001 ± 0.057 | -0.029 ± 0.025 |
| DermaMNIST (skin) | +0.007 ± 0.036 | -0.101 ± 0.063 | -0.096 ± 0.082 | -0.141 ± 0.024 |
| BloodMNIST (blood cells) | -0.197 ± 0.031 | -0.239 ± 0.052 | -0.218 ± 0.036 | -0.182 ± 0.010 |
| BreastMNIST (breast ultrasound) | +0.033 ± 0.048 | +0.044 ± 0.144 | -0.013 ± 0.125 | -0.022 ± 0.054 |

## Compute (training + prediction, excluding one-off feature extraction)

| Method | Total minutes |
|---|---|
| HOG + RF | 0.8 |
| ResNet-18 fine-tuned | 70.9 |
| ResNet-18 frozen | 0.0 |
| DINOv2 frozen | 0.0 |
| CLIP frozen | 0.0 |
