# 🔬 IDRiD Retinal Lesion Segmentation Pipeline

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Production-success.svg)

**Advanced deep learning pipeline for automated detection and segmentation of diabetic retinopathy lesions**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Results](#-results) • [Documentation](#-documentation)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-features)
- [Architecture](#-architecture)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage Examples](#-usage-examples)
- [Model Performance](#-model-performance)
- [Configuration](#-configuration)
- [Results Analysis](#-results-analysis)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [Citation](#-citation)
- [License](#-license)

---

## 🎯 Overview

This repository implements a state-of-the-art **patch-based U-Net segmentation pipeline** for detecting and segmenting four types of diabetic retinopathy lesions from fundus images using the **IDRiD (Indian Diabetic Retinopathy Image Dataset)**.

### 🏥 Clinical Significance

Diabetic retinopathy is a leading cause of blindness worldwide. Early detection and segmentation of retinal lesions can:
- Enable timely medical intervention
- Reduce healthcare costs through automation
- Assist ophthalmologists in diagnosis
- Monitor disease progression objectively

### 🎨 Lesion Types Detected

| Lesion Type | Abbreviation | Clinical Significance |
|-------------|--------------|----------------------|
| **Microaneurysms (MA)** | MA | Early indicator of diabetic retinopathy |
| **Haemorrhages (HE)** | HE | Sign of blood vessel damage |
| **Hard Exudates (EX)** | EX | Lipid deposits indicating vascular leakage |
| **Soft Exudates (SE)** | SE | Cotton-wool spots indicating ischemia |

---

## ✨ Features

### 🚀 Core Capabilities

- **Patch-Based Training**: Intelligent patch extraction with configurable lesion/background ratios
- **Advanced Augmentation**: 15+ augmentation techniques for robust generalization
- **Multi-Loss Functions**: Focal Loss, Dice Loss, Tversky Loss, and combinations
- **K-Fold Cross-Validation**: Robust 5-fold validation for reliable performance metrics
- **Sliding Window Inference**: Efficient full-image prediction with overlapping patches
- **CLAHE Enhancement**: Contrast-limited adaptive histogram equalization preprocessing
- **Automated Post-Processing**: Morphological operations and connected component analysis
- **Mixed Precision Training**: AMP support for faster training on modern GPUs
- **Comprehensive Metrics**: Dice, IoU, Sensitivity, Specificity, Precision, F1-Score

### 🛠️ Technical Highlights

```python
✓ Patch-based sampling (256×256 with 70/30 lesion/background ratio)
✓ Heavy augmentation pipeline (geometric + photometric)
✓ U-Net architecture with batch normalization
✓ Early stopping and learning rate scheduling
✓ Gradient clipping for training stability
✓ Automatic checkpoint management
✓ Rich visualization suite
✓ Results analysis toolkit
```

---

## 🏗️ Architecture

### U-Net Model

```
Input (3×256×256)
    ↓
┌─────────────────────┐
│  Encoder Path       │
│  ├─ Conv Block (64) │ ←─┐
│  ├─ MaxPool         │   │
│  ├─ Conv Block (128)│ ←─┤
│  ├─ MaxPool         │   │ Skip Connections
│  ├─ Conv Block (256)│ ←─┤
│  ├─ MaxPool         │   │
│  ├─ Conv Block (512)│ ←─┤
│  └─ MaxPool         │   │
└─────────────────────┘   │
         ↓                │
    Bottleneck (1024)     │
         ↓                │
┌─────────────────────┐   │
│  Decoder Path       │   │
│  ├─ UpConv          │ ──┘
│  ├─ Conv Block (512)│
│  ├─ UpConv          │
│  ├─ Conv Block (256)│
│  ├─ UpConv          │
│  ├─ Conv Block (128)│
│  ├─ UpConv          │
│  └─ Conv Block (64) │
└─────────────────────┘
         ↓
Output (1×256×256)
```

### Pipeline Workflow

```mermaid
graph LR
    A[Raw Images] --> B[CLAHE Preprocessing]
    B --> C[Patch Extraction]
    C --> D[Heavy Augmentation]
    D --> E[U-Net Training]
    E --> F[5-Fold CV]
    F --> G[Best Model Selection]
    G --> H[Sliding Window Inference]
    H --> I[Post-Processing]
    I --> J[Final Segmentation]
```

---

## 📊 Dataset

### IDRiD Dataset Structure

```
IDRiD/
├── A. Segmentation/
│   ├── 1. Original Images/
│   │   ├── a. Training Set/     # 54 images
│   │   └── b. Testing Set/      # 27 images
│   └── 2. All Segmentation Groundtruths/
│       ├── a. Training Set/
│       │   ├── 1. Microaneurysms/
│       │   ├── 2. Haemorrhages/
│       │   ├── 3. Hard Exudates/
│       │   └── 4. Soft Exudates/
│       └── b. Testing Set/
│           └── [same structure]
```

### Dataset Statistics

- **Training Images**: 54 high-resolution fundus images
- **Testing Images**: 27 high-resolution fundus images
- **Resolution**: 4288×2848 pixels (average)
- **Format**: JPG (images), TIF (masks)
- **Annotations**: Pixel-level expert annotations

**Download**: [IDRiD Dataset](https://idrid.grand-challenge.org/)

---

## 🔧 Installation

### Prerequisites

```bash
Python 3.8+
CUDA 11.0+ (for GPU acceleration)
8GB+ GPU memory recommended
```

### 1. Clone Repository

```bash
git clone https://github.com/kritanu/idrid-lesion-segmentation.git
cd idrid-lesion-segmentation
```

### 2. Create Virtual Environment

```bash
# Using conda (recommended)
conda create -n idrid python=3.8
conda activate idrid

# Or using venv
python -m venv idrid_env
source idrid_env/bin/activate  # Linux/Mac
# idrid_env\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

<details>
<summary><b>📦 View requirements.txt</b></summary>

```txt
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
opencv-python>=4.5.0
pillow>=9.0.0
matplotlib>=3.5.0
tqdm>=4.62.0
scipy>=1.7.0
scikit-learn>=1.0.0
albumentations>=1.3.0
```
</details>

### 4. Setup Dataset

```bash
# Update DATA_ROOT path in the script
# Line 35 in Config class:
DATA_ROOT = "/path/to/your/IDRiD/A.%20Segmentation/A. Segmentation"
```

---

## 🚀 Quick Start

### Training All Lesion Types

```bash
python idrid_segmentation.py --mode train_all
```

### Training Single Lesion Type

```bash
# Microaneurysms (Type 1)
python idrid_segmentation.py --mode train --lesion_type 1

# Haemorrhages (Type 2)
python idrid_segmentation.py --mode train --lesion_type 2

# Hard Exudates (Type 3)
python idrid_segmentation.py --mode train --lesion_type 3

# Soft Exudates (Type 4)
python idrid_segmentation.py --mode train --lesion_type 4
```

### Analyzing Results

```bash
python idrid_segmentation.py  # Run the results analyzer at the end
```

---

## 📖 Usage Examples

### Example 1: Custom Configuration Training

```python
from config import Config

# Modify configuration
Config.NUM_EPOCHS = 100
Config.PATCH_SIZE = 512
Config.LEARNING_RATE = 5e-5
Config.LOSS_TYPE = "focal_tversky"

# Train model
train_single_lesion(lesion_type=1, use_kfold=True)
```

### Example 2: Results Visualization

```python
from visualizer import Visualizer

# Plot training history
Visualizer.plot_training_history(history, lesion_type=1, fold=0)

# Create comparison grid
Visualizer.create_comparison_grid(
    test_images=test_images,
    predictions=predictions,
    ground_truths=ground_truths,
    lesion_type=1,
    num_samples=6
)
```

---

## 📈 Model Performance

### 🏆 Overall Performance Summary

<div align="center">

| 🎯 Lesion Type | 📉 Mean Val Loss | 🎲 Mean Dice | 🎯 Mean Sensitivity |
|:--------------|:----------------:|:------------:|:-------------------:|
| **Microaneurysms** | 0.5545 ± 0.0064 | **0.4564** ± 0.0079 | **0.5338** ± 0.0326 |
| **Haemorrhages** | 0.4747 ± 0.0333 | **0.5714** ± 0.0483 | **0.5877** ± 0.0404 |
| **Hard Exudates** | 0.3569 ± 0.0364 | **0.6815** ± 0.0259 | **0.6549** ± 0.0287 |
| **Soft Exudates** | 0.5542 ± 0.0964 | **0.7211** ± 0.1007 | **0.8123** ± 0.0679 |

</div>

---

### 🔬 Detailed Cross-Validation Results (5-Fold)

<details open>
<summary><h3>🔵 Microaneurysms (MA)</h3></summary>

<div align="center">

**Performance Metrics Across 5 Folds**

| Fold | Best Epoch | Validation Loss | Dice Score | Sensitivity | Total Epochs |
|:----:|:----------:|:---------------:|:----------:|:-----------:|:------------:|
| **Fold 0** | 76 | 0.5426 | **0.4709** 🏆 | 0.5302 | 80 |
| **Fold 1** | 65 | 0.5585 | 0.4524 | 0.4725 | 80 |
| **Fold 2** | 71 | 0.5576 | 0.4544 | 0.5489 | 80 |
| **Fold 3** | 72 | 0.5606 | 0.4472 | 0.5537 | 80 |
| **Fold 4** | 47 | 0.5533 | 0.4573 | **0.5637** 🏆 | 68 |

</div>

#### 📊 Statistical Summary

```
✓ Mean Loss:        0.5545 ± 0.0064
✓ Mean Dice:        0.4564 ± 0.0079
✓ Mean Sensitivity: 0.5338 ± 0.0326

★ Best Loss:        0.5426 (Fold 0, Epoch 76)
★ Best Dice:        0.4709 (Fold 0)
★ Best Sensitivity: 0.5637 (Fold 4)
```

#### 💡 Key Insights
- Most challenging lesion type due to small size (10-100 μm)
- Low standard deviation indicates consistent training
- Early stopping activated in Fold 4 (epoch 47)
- Dice score ~0.46 aligns with state-of-the-art for microaneurysms

</details>

---

<details>
<summary><h3>🔴 Haemorrhages (HE)</h3></summary>

<div align="center">

**Performance Metrics Across 5 Folds**

| Fold | Best Epoch | Validation Loss | Dice Score | Sensitivity | Total Epochs |
|:----:|:----------:|:---------------:|:----------:|:-----------:|:------------:|
| **Fold 0** | 45 | 0.4646 | 0.5798 | **0.6456** 🏆 | 73 |
| **Fold 1** | 31 | 0.4865 | 0.5646 | 0.5613 | 51 |
| **Fold 2** | 80 | **0.4384** 🏆 | **0.6403** 🏆 | 0.6187 | 80 |
| **Fold 3** | 55 | 0.4507 | 0.5825 | 0.5809 | 75 |
| **Fold 4** | 59 | 0.5331 | 0.4896 | 0.5319 | 78 |

</div>

#### 📊 Statistical Summary

```
✓ Mean Loss:        0.4747 ± 0.0333
✓ Mean Dice:        0.5714 ± 0.0483
✓ Mean Sensitivity: 0.5877 ± 0.0404

★ Best Loss:        0.4384 (Fold 2, Epoch 80)
★ Best Dice:        0.6403 (Fold 2)
★ Best Sensitivity: 0.6456 (Fold 0)
```

#### 💡 Key Insights
- Moderate performance with good sensitivity (58.77%)
- Fold 2 shows exceptional convergence to epoch 80
- Fold 1 demonstrates fastest learning (epoch 31)
- Higher variance suggests diverse hemorrhage presentations

</details>

---

<details>
<summary><h3>🟡 Hard Exudates (EX)</h3></summary>

<div align="center">

**Performance Metrics Across 5 Folds**

| Fold | Best Epoch | Validation Loss | Dice Score | Sensitivity | Total Epochs |
|:----:|:----------:|:---------------:|:----------:|:-----------:|:------------:|
| **Fold 0** | 64 | 0.3469 | 0.6857 | 0.6561 | 80 |
| **Fold 1** | 75 | **0.3173** 🏆 | **0.7104** 🏆 | **0.6910** 🏆 | 80 |
| **Fold 2** | 25 | 0.4111 | 0.6610 | 0.6734 | 45 |
| **Fold 3** | 51 | 0.3230 | 0.7068 | 0.6485 | 77 |
| **Fold 4** | 21 | 0.3863 | 0.6436 | 0.6057 | 42 |

</div>

#### 📊 Statistical Summary

```
✓ Mean Loss:        0.3569 ± 0.0364
✓ Mean Dice:        0.6815 ± 0.0259
✓ Mean Sensitivity: 0.6549 ± 0.0287

★ Best Loss:        0.3173 (Fold 1, Epoch 75)
★ Best Dice:        0.7104 (Fold 1)
★ Best Sensitivity: 0.6910 (Fold 1)
```

#### 💡 Key Insights
- **BEST PERFORMING LESION TYPE** 🏆
- Lowest validation loss across all lesion types
- Most consistent results (Dice std: 0.0259)
- Rapid convergence in Folds 2 & 4 (epochs 25 & 21)
- High-contrast lesions enable superior detection

</details>

---

<details>
<summary><h3>🟢 Soft Exudates (SE)</h3></summary>

<div align="center">

**Performance Metrics Across 5 Folds**

| Fold | Best Epoch | Validation Loss | Dice Score | Sensitivity | Total Epochs |
|:----:|:----------:|:---------------:|:----------:|:-----------:|:------------:|
| **Fold 0** | 59 | **0.4640** 🏆 | 0.6785 | **0.8705** 🏆 | 62 |
| **Fold 1** | 33 | 0.6100 | 0.7393 | 0.8674 | 60 |
| **Fold 2** | 44 | 0.7166 | 0.5597 | 0.7403 | 47 |
| **Fold 3** | 68 | 0.5081 | **0.8652** 🏆 | 0.8644 | 80 |
| **Fold 4** | 78 | 0.4723 | 0.7627 | 0.7190 | 80 |

</div>

#### 📊 Statistical Summary

```
✓ Mean Loss:        0.5542 ± 0.0964
✓ Mean Dice:        0.7211 ± 0.1007
✓ Mean Sensitivity: 0.8123 ± 0.0679

★ Best Loss:        0.4640 (Fold 0, Epoch 59)
★ Best Dice:        0.8652 (Fold 3)
★ Best Sensitivity: 0.8705 (Fold 0)
```

#### 💡 Key Insights
- **HIGHEST SENSITIVITY** 🎯 (81.23% mean)
- **HIGHEST DICE SCORE** in Fold 3 (0.8652)
- High variance indicates variable lesion presentation
- Excellent true positive rate (87% sensitivity in Fold 0)
- Cotton-wool spots successfully detected

</details>

---

### 📊 Comparative Analysis

<div align="center">

#### 🏅 Performance Ranking

| Metric | 🥇 Best | 🥈 Second | 🥉 Third | Fourth |
|:------:|:-------:|:---------:|:--------:|:------:|
| **Dice Score** | Soft Exudates<br>(0.7211) | Hard Exudates<br>(0.6815) | Haemorrhages<br>(0.5714) | Microaneurysms<br>(0.4564) |
| **Sensitivity** | Soft Exudates<br>(0.8123) | Hard Exudates<br>(0.6549) | Haemorrhages<br>(0.5877) | Microaneurysms<br>(0.5338) |
| **Validation Loss** | Hard Exudates<br>(0.3569) | Haemorrhages<br>(0.4747) | Soft Exudates<br>(0.5542) | Microaneurysms<br>(0.5545) |
| **Consistency** | Hard Exudates<br>(std: 0.0259) | Microaneurysms<br>(std: 0.0079) | Haemorrhages<br>(std: 0.0483) | Soft Exudates<br>(std: 0.1007) |

</div>

---

### 🎯 Key Takeaways

#### ✅ Strengths
- **Hard Exudates**: Most reliable and consistent performance
- **Soft Exudates**: Exceptional sensitivity for clinical screening
- **Training Stability**: All models converge within 80 epochs
- **Cross-Validation**: Robust results across 5 independent folds

#### ⚠️ Challenges
- **Microaneurysms**: Small lesion size (10-100 μm) limits detection
- **Variance**: Soft exudates show highest variability across folds
- **Class Imbalance**: Background pixels heavily outnumber lesion pixels

#### 🚀 Future Improvements
- Ensemble methods combining multiple fold models
- Attention mechanisms for small lesion detection
- Advanced post-processing for microaneurysms
- Multi-task learning across all lesion types simultaneously

---

## ⚙️ Configuration

### Key Hyperparameters

```python
# Model Architecture
IMG_SIZE = 1024              # Full image resize dimension
PATCH_SIZE = 256             # Training patch size
USE_PATCHES = True           # Enable patch-based training

# Training Strategy
NUM_EPOCHS = 80              # Maximum training epochs
LEARNING_RATE = 1e-4         # Initial learning rate
BATCH_SIZE = 16              # Patches per batch
WEIGHT_DECAY = 1e-5          # L2 regularization

# Loss Functions
LOSS_TYPE = "focal_dice"     # Options: focal_dice, tversky, focal_tversky
FOCAL_ALPHA = 0.25           # Focal loss weighting
FOCAL_GAMMA = 2.0            # Focal loss focusing parameter

# Data Sampling
LESION_PATCH_RATIO = 0.7     # 70% lesion, 30% background patches
MIN_LESION_PIXELS = 10       # Minimum pixels to consider lesion patch

# Regularization
EARLY_STOPPING_PATIENCE = 20 # Epochs before early stopping
REDUCE_LR_PATIENCE = 8       # Epochs before LR reduction
REDUCE_LR_FACTOR = 0.5       # LR reduction multiplier

# Preprocessing
USE_CLAHE = True             # CLAHE enhancement
CLAHE_CLIP_LIMIT = 2.0       # Contrast clipping
CLAHE_TILE_GRID_SIZE = (8,8) # CLAHE tile size

# Post-processing
THRESHOLD = 0.4              # Probability threshold
MIN_OBJECT_SIZE = 10         # Remove small objects (pixels)
MORPHOLOGY_KERNEL_SIZE = 3   # Morphological operations kernel

# Cross-Validation
USE_KFOLD = True             # Enable K-fold CV
NUM_FOLDS = 5                # Number of CV folds
```

### Modifying Configuration

```python
# In your training script
from config import Config

# Override specific parameters
Config.NUM_EPOCHS = 100
Config.LEARNING_RATE = 5e-5
Config.PATCH_SIZE = 512
Config.LOSS_TYPE = "focal_tversky"

# Run training
train_single_lesion(lesion_type=1, use_kfold=True)
```

---

## 📁 Project Structure

```
idrid-lesion-segmentation/
│
├── idrid_segmentation.py         # Main training script
├── config.py                      # Configuration class
├── requirements.txt               # Python dependencies
├── README.md                      # This file
│
├── idrid_improved_outputs/        # Training outputs
│   ├── checkpoints/               # Model checkpoints
│   │   ├── best_model_lesion1_fold0.pth
│   │   ├── best_model_lesion2_fold0.pth
│   │   └── ...
│   │
│   ├── results/                   # Training metrics
│   │   ├── lesion1_fold0_history.json
│   │   ├── lesion1_cv_summary.json
│   │   └── overall_summary.json
│   │
│   ├── visualizations/            # Plots and figures
│   │   ├── training_history_lesion1_fold0.png
│   │   ├── best_val_loss_lesion1.png
│   │   └── comparison_grid_lesion1.png
│   │
│   └── predictions/               # Test predictions
│       └── [prediction outputs]
│
└── data/                          # Dataset (not included)
    └── IDRiD/
        └── A. Segmentation/
            ├── 1. Original Images/
            └── 2. All Segmentation Groundtruths/
```

---

## 🔍 Results Analysis

### Built-in Results Analyzer

The pipeline includes a comprehensive results analyzer:

```python
from results_analyzer import ResultsAnalyzer

# Initialize analyzer
analyzer = ResultsAnalyzer()

# Display all lesion results
analyzer.display_all_lesions_summary()

# Display specific lesion
analyzer.display_single_lesion_results(1, "Microaneurysms")

# Plot loss curves
analyzer.plot_loss_curves(1, "Microaneurysms")
```

### Generated Outputs

The analyzer creates:
- **Fold-wise comparison tables**
- **Statistical summaries (mean, std, min, max)**
- **Loss curve visualizations**
- **Best epoch identification**
- **Cross-validation performance aggregation**

All results are automatically saved to:
- `idrid_improved_outputs/visualizations/`
- `idrid_improved_outputs/results/`

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include unit tests for new features
- Update README for significant changes

---

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{idrid_segmentation_2024,
  title={IDRiD Retinal Lesion Segmentation using Patch-Based U-Net},
  author={Kritanu Chattopadhyay},
  supervisor={Dr. Xiaoyu Cao},
  journal={GitHub Repository},
  year={2024},
  url={https://github.com/kritanu/idrid-lesion-segmentation}
}
```

### Dataset Citation

```bibtex
@article{porwal2018idrid,
  title={Indian diabetic retinopathy image dataset (IDRiD): 
         a database for diabetic retinopathy screening research},
  author={Porwal, Prasanna and Pachade, Samiksha and Kamble, Ravi 
          and Kokare, Manesh and Deshmukh, Girish and Sahasrabuddhe, Vivek 
          and Meriaudeau, Fabrice},
  journal={Data},
  volume={3},
  number={3},
  pages={25},
  year={2018},
  publisher={Multidisciplinary Digital Publishing Institute}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Kritanu Chattopadhyay

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 🙏 Acknowledgments

- **Dr. Xiaoyu Cao** - Research Supervisor and Mentor
- **IDRiD Dataset Team** - For providing the high-quality annotated dataset
- **PyTorch Community** - For the excellent deep learning framework
- **Albumentations** - For the powerful augmentation library
- **Medical Imaging Community** - For advancing automated diagnosis

---

## 📞 Contact

**Author**: Kritanu Chattopadhyay  
**Supervisor**: Dr. Xiaoyu Cao

- 📧 Email: kritanu@example.com
- 🐙 GitHub: [@kritanu](https://github.com/kritanu)
- 💼 LinkedIn: [Kritanu Chattopadhyay](https://linkedin.com/in/kritanu-chattopadhyay)

---

<div align="center">

### ⭐ Star this repository if you find it helpful!

**Made with ❤️ for advancing medical imaging AI**

[⬆ Back to Top](#-idrid-retinal-lesion-segmentation-pipeline)

</div>
