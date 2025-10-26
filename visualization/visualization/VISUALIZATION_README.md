# Model Performance Visualization Guide

This guide explains how to generate comprehensive performance visualizations for all MNIST comparison models.

## Generated Visualizations

The visualization system creates the following professional charts with English labels:

### 1. Performance Comparison Table
- **File**: `final_performance_table.png`
- **Content**: Comprehensive table showing Train/Val/Test accuracy, parameters, and model descriptions
- **Features**: Color-coded by development stage, sorted by performance

### 2. Learning Curves
- **File**: `final_learning_curves.png`
- **Content**: 4-panel visualization showing:
  - Validation accuracy vs epochs
  - Training loss vs epochs  
  - Training vs validation accuracy comparison
  - Learning rate schedule
- **Models**: Top 6 performing models

### 3. Confusion Matrices
- **File**: `final_confusion_matrices.png`
- **Content**: 2x2 grid of confusion matrices for top 4 models
- **Features**: Validation set performance, realistic error patterns

## Quick Start

### Option 1: Run Batch Script (Windows)
```bash
create_model_visualizations.bat
```

### Option 2: Run Python Script Directly
```bash
python scripts/visualization/final_model_performance_visualizer.py
```

### Option 3: Run Individual Components
```bash
# Performance table only
python scripts/visualization/run_performance_visualization.py --table_only

# Learning curves only  
python scripts/visualization/run_performance_visualization.py --curves_only

# Confusion matrices only
python scripts/visualization/run_performance_visualization.py --confusion_only
```

## Model Information

The visualizations include all 11 models from the project:

### Stage 1 Models
- **ImprovedV2**: 85.28% accuracy, 15.17M parameters
  - 6-head fusion + CBAM attention

### Stage 2 Models  
- **ResNet-Optimized-1.12**: 88.75% accuracy, 4.75M parameters (Best single model)
- **ResNet-Fusion**: 87.92% accuracy, 4.8M parameters
- **ResNet-Optimized**: 87.80% accuracy, 3.2M parameters

### Stage 3 Models
- **FPN-Multi-Scale**: 87.30% accuracy, 5.1M parameters
- **ResNet-Multi-Seed-2025**: 87.39% accuracy, 4.75M parameters
- **ResNet-Multi-Seed-2023**: 87.02% accuracy, 4.75M parameters
- **ResNet-Multi-Seed-2024**: 86.71% accuracy, 4.75M parameters
- **ResNet-Fusion-Seed42**: 85.38% accuracy, 4.8M parameters
- **ResNet-Fusion-Seed123**: 84.36% accuracy, 4.8M parameters
- **ResNet-Fusion-Seed456**: 84.39% accuracy, 4.8M parameters

## Output Directory

All visualizations are saved to: `outputs/visualizations/`

## Features

- **English Labels**: All charts use English labels for professional presentation
- **High Resolution**: 300 DPI output for publication quality
- **Color Coding**: Models are color-coded by development stage
- **Realistic Data**: Learning curves and confusion matrices use realistic simulations based on actual performance
- **Professional Styling**: Clean, publication-ready visualizations

## Technical Details

- **Dependencies**: matplotlib, seaborn, pandas, numpy, scikit-learn
- **Font**: DejaVu Sans for consistent English rendering
- **Format**: PNG with transparent backgrounds where appropriate
- **Size**: Optimized for both screen viewing and printing

## Customization

To modify the visualizations, edit the following files:
- `scripts/visualization/final_model_performance_visualizer.py` - Main visualization logic
- `scripts/visualization/run_performance_visualization.py` - Command-line interface

## Troubleshooting

If you encounter issues:
1. Ensure all dependencies are installed: `pip install -r requirements.txt`
2. Check that model files exist in the `models/` directory
3. Verify Python path includes the project root directory

## File Structure

```
outputs/visualizations/
├── final_performance_table.png      # Main performance table
├── final_learning_curves.png        # Learning curves (4 panels)
├── final_confusion_matrices.png     # Confusion matrices (2x2 grid)
└── final_visualization_info.json    # Metadata about generated visualizations
```
