# MNIST Digit Pair Comparison - Stacking Ensemble Solution

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/onp-china/ML-EX1-Group2)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This project implements a **Stacking Ensemble** approach for binary classification of MNIST digit pairs, achieving **90.81% accuracy on validation set** and **89.85% on test_public set**.

## 🔗 Repository

**GitHub**: https://github.com/onp-china/ML-EX1-Group2

This repository contains the complete implementation, trained models, and all experimental results for the MNIST digit pair comparison task.

## 📊 Final Model Performance

### Performance Summary

| Model | Train Accuracy | Val Accuracy | Test Public Accuracy | Parameters |
|-------|---------------|--------------|---------------------|------------|
| resnet_optimized_1.12 | 96.02% | 88.74% | 88.45% | ~5M |
| resnet_fusion | 93.42% | 87.92% | 89.00% | ~5M |
| resnet_optimized | 91.37% | 87.80% | 88.20% | ~5M |
| seed_2025 | 95.00% | 87.39% | 87.85% | ~5M |
| seed_2023 | 94.72% | 87.02% | 87.95% | ~5M |
| seed_2024 | 93.77% | 86.71% | 86.75% | ~5M |
| fpn_model | 95.60% | 87.30% | 87.90% | 5,089,626 |
| resnet_fusion_seed42 | 89.74% | 85.46% | 85.25% | ~5M |
| resnet_fusion_seed456 | 88.76% | 84.88% | 83.50% | ~5M |
| resnet_fusion_seed123 | 88.33% | 84.56% | 84.80% | ~5M |
| **Stacking Ensemble** | **95.67%** | **90.81%** | **90.00%** | 10 models + LightGBM |

### Model Size and Training Time

- **Individual Model Size**: ~20 MB per model (PyTorch .pt file)
- **Total Ensemble Size**: ~200 MB (10 models)
- **Meta-learner Size**: <1 MB (LightGBM)
- **Training Time per Model**: 30-60 minutes on NVIDIA GPU (varies by architecture)
- **Total Training Time**: ~8-10 hours for all 10 models
- **Stacking Meta-learner Training**: ~2 minutes
- **Inference Time**: ~5 seconds for 8,000 samples on GPU

## 🏗️ Project Structure

```
mnist-demo/
├── src/                          # Source code
│   ├── __init__.py
│   ├── data_loader.py           # Data loading utilities
│   ├── model_loader.py          # Model loading utilities
│   ├── stacking_ensemble.py     # Stacking ensemble implementation
│   ├── augmentation.py          # Data augmentation utilities
│   ├── dynamic_ensemble.py      # Dynamic ensemble methods
│   ├── mc_dropout.py            # Monte Carlo dropout
│   ├── two_level_stacking.py   # Two-level stacking
│   ├── training_history.py      # Training history management
│   └── models/                  # Model architectures
│       ├── __init__.py
│       ├── simple_compare_cnn.py    # ResNet-based models
│       └── fpn_architecture_v2.py   # FPN model
├── models/                       # Trained model weights
│   ├── stage1_improvedv2/       # Initial improved model
│   ├── stage2_resnet_optimized/ # Optimized ResNet models
│   │   ├── resnet_optimized_1.12/
│   │   ├── resnet_fusion/
│   │   └── resnet_optimized/
│   ├── stage3_multi_seed/       # Multi-seed models
│   │   ├── seed_2023/, seed_2024/, seed_2025/
│   │   ├── fpn_model/
│   │   └── resnet_fusion_seed*/
│   ├── stage4_stacking/         # Stacking results
│   └── stage5_advanced_ensemble/ # Advanced ensemble methods
├── data/                         # Dataset files
│   ├── train.npz               # 50,000 training samples
│   ├── val.npz                 # 10,000 validation samples
│   ├── test_public.npz         # 2,000 public test samples
│   ├── test_public_labels.csv  # Public test labels
│   └── test_private.npz        # 8,000 private test samples
├── scripts/                      # Training and evaluation scripts
│   ├── training/                # Training scripts
│   │   ├── train_stage1_improvedv2.py
│   │   ├── train_stage2_resnet_optimized.py
│   │   ├── train_stage3_multi_seed.py
│   │   └── train_fpn_model.py
│   ├── generate_private_predictions.py  # Generate final predictions
│   ├── evaluate_10models_stacking.py    # Evaluate stacking
│   └── create_final_visualizations.py  # Create visualizations
├── docs/                         # Documentation
│   ├── TRAINING_GUIDE.md
│   ├── PREDICTION_SUMMARY.md
│   ├── MODEL_CLASSIFICATION.md
│   └── ...
├── outputs/                      # Results and visualizations
│   ├── results/                  # Evaluation results
│   │   ├── performance_table.csv
│   │   ├── 10models_stacking_evaluation.json
│   │   └── BEST_SOLUTION_test_public_90.30.json
│   ├── visualizations/          # Visualizations
│   │   ├── performance_table.png
│   │   ├── confusion_matrix.png
│   │   └── stacking_training_process.png
│   └── checkpoints/             # Training checkpoints
├── visualization/               # Visualization scripts
│   └── visualization/           # Model visualization code
├── requirements.txt             # Python dependencies
├── pred_private.csv             # Final predictions for test_private
├── check_submission.py          # Submission format validator
├── PROJECT_STRUCTURE.md         # Detailed project structure
└── README.md                    # This file
```

## 🚀 Quick Start

### 1. Environment Setup

#### Option A: Using pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Option B: Using conda

```bash
# Create conda environment
conda create -n mnist-demo python=3.10
conda activate mnist-demo

# Install dependencies
pip install -r requirements.txt
```

### Required Dependencies

- Python 3.8+
- PyTorch 2.0+ (with CUDA support recommended)
- NumPy 1.21.0+
- Pandas 1.3.0+
- scikit-learn 1.0.0+
- LightGBM 3.3.0+
- Matplotlib 3.5.0+
- Seaborn 0.11.0+
- tqdm 4.60.0+ (for progress bars)

### 2. Data Preparation

Place the following files in the `data/` directory:
- `train.npz` (50,000 samples)
- `val.npz` (10,000 samples)
- `test_public.npz` (2,000 samples)
- `test_public_labels.csv`
- `test_private.npz` (8,000 samples)

## 📈 Model Training

### Training Individual Models

The project uses 10 pre-trained models. To retrain from scratch:

```bash
# Train ResNet-based models (Stage 2)
python scripts/training/train_stage2_resnet_optimized.py

# Train multi-seed models (Stage 3)
python scripts/training/train_stage3_multi_seed.py
```

**Note**: Training all models from scratch takes approximately 8-10 hours on a modern GPU.

### Training Stacking Meta-learner

The stacking meta-learner is trained automatically during prediction generation:

```bash
python scripts/generate_private_predictions.py
```

This script:
1. Loads all 10 pre-trained models
2. Generates predictions on validation set
3. Trains LightGBM meta-learner using 5-fold cross-validation
4. Generates final predictions for `test_private.npz`

## 🔮 Generating Predictions

### Generate Predictions for test_private.npz

```bash
python scripts/generate_private_predictions.py
```

This creates `pred_private.csv` with predictions for the private test set.

### Validate Submission Format

```bash
python check_submission.py --data_dir data --pred pred_private.csv --test_file test_private.npz
```

Expected output:
```
OK: id coverage matches the test set.
CSV format looks good.
```

## 📊 Model Evaluation

### Evaluate on Validation and Test Public Sets

```bash
python scripts/evaluate_10models_stacking.py
```

This generates:
- Cross-validation scores (5-fold)
- Performance on validation set
- Performance on test_public set
- Detailed metrics in `outputs/results/10models_stacking_evaluation.json`

### Create Visualizations

```bash
python scripts/create_final_visualizations.py
```

This generates:
1. **Performance Table** (`outputs/visualizations/performance_table.png`)
   - Comparison of all models across train/val/test_public
   
2. **Confusion Matrix** (`outputs/visualizations/confusion_matrix.png`)
   - Stacking ensemble performance on validation set
   
3. **Stacking Training Process** (`outputs/visualizations/stacking_training_process.png`)
   - Cross-validation scores
   - Comparison with simple averaging

## 🧠 Model Architecture

### Base Models

The ensemble consists of 10 ResNet-based and FPN-based models:

1. **ResNet Variants** (9 models)
   - Architecture: Modified ResNet with BasicBlock
   - Layers: [2, 2, 2, 2] or [2, 2, 2]
   - Feature dimension: 256
   - Width multiplier: 1.0 or 1.12
   - Input: Two 28×28 grayscale images
   - Output: Binary classification (same digit or not)

2. **FPN Model** (1 model)
   - Architecture: Feature Pyramid Network
   - Layers: [2, 2, 2]
   - Feature dimension: 256
   - Width multiplier: 1.25
   - Multi-scale feature extraction

### Stacking Meta-learner

- **Algorithm**: LightGBM Classifier
- **Input**: Concatenated predictions from 10 base models (20 features)
- **Training**: 5-fold stratified cross-validation on validation set
- **Hyperparameters**:
  - `n_estimators`: 100
  - `learning_rate`: 0.05
  - `max_depth`: 5
  - `num_leaves`: 31

### Training Strategy

1. **Data Augmentation**: Random rotation, scaling, translation
2. **Loss Function**: Focal Loss with label smoothing
3. **Optimizer**: AdamW with weight decay
4. **Learning Rate Scheduler**: ReduceLROnPlateau
5. **Early Stopping**: Patience of 10 epochs
6. **Multi-seed Training**: Seeds 42, 2023, 2024, 2025 for robustness

## 📁 Output Files

### pred_private.csv

Format:
```csv
id,label
PRV_0000000,1
PRV_0000001,0
...
```

- **Rows**: 8,001 (including header)
- **Columns**: `id`, `label`
- **Label Distribution**: ~50% class 0, ~50% class 1

### Performance Metrics

Located in `outputs/results/`:
- `performance_table.csv`: Detailed performance comparison
- `10models_stacking_evaluation.json`: Full evaluation results
- `10models_final_report.md`: Comprehensive report

### Visualizations

Located in `outputs/visualizations/`:
- `performance_table.png`: Performance comparison table
- `confusion_matrix.png`: Confusion matrix on validation set
- `stacking_training_process.png`: Training process visualization

## 🎯 Key Results

### Stacking Ensemble Performance

- **Validation Set**: 90.81% accuracy (F1: 90.78%)
- **Test Public Set**: 89.85% accuracy (F1: 89.92%)
- **Improvement over Simple Average**: +2.21% on validation, +0.95% on test_public

### Cross-Validation Results

| Fold | Accuracy |
|------|----------|
| Fold 1 | 89.25% |
| Fold 2 | 90.75% |
| Fold 3 | 88.95% |
| Fold 4 | 90.25% |
| Fold 5 | 87.20% |
| **Mean** | **89.28% ± 1.23%** |

### Confusion Matrix (Validation Set)

|  | Predicted 0 | Predicted 1 |
|---|------------|------------|
| **True 0** | 4,557 | 443 |
| **True 1** | 476 | 4,524 |

**Accuracy**: 90.81%

## 🔬 Technical Details

### Why Stacking Works

1. **Model Diversity**: 10 models with different architectures, seeds, and hyperparameters
2. **Meta-learning**: LightGBM learns optimal weights for each model's predictions
3. **Ensemble Strength**: Combines complementary strengths of individual models
4. **Robustness**: Multi-seed training reduces variance

### Design Decisions

1. **10 Models**: Balance between performance and computational cost
2. **LightGBM**: Fast, accurate, handles feature interactions well
3. **5-Fold CV**: Ensures meta-learner doesn't overfit to validation set
4. **Focal Loss**: Addresses class imbalance and hard examples

## 📝 Citation

If you use this code, please cite:

```
MNIST Digit Pair Comparison - Stacking Ensemble Solution
Group 2
10-Model Ensemble with LightGBM Meta-learner
Validation Accuracy: 90.81% | Test Public Accuracy: 89.85%
Repository: https://github.com/onp-china/ML-EX1-Group2
```

## 📄 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- MNIST dataset creators
- PyTorch and scikit-learn communities
- LightGBM developers
- Stacking ensemble method community

## 📞 Contact

- **GitHub Issues**: https://github.com/onp-china/ML-EX1-Group2/issues
- **Repository**: https://github.com/onp-china/ML-EX1-Group2
- For questions or issues, please refer to the project documentation in `docs/`

---

**Last Updated**: January 2025

**Author**: Group 2 (ML-EX1-Group2)
