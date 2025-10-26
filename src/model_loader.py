#!/usr/bin/env python3
"""
Model Loader for Stacking Ensemble
==================================

Loads all pre-trained models for the stacking ensemble.
"""

import os
import json
import torch
import numpy as np
from typing import List, Tuple, Dict

# Import model architectures
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.models.simple_compare_cnn import ResNetCompareNet, BasicBlock, Bottleneck
    from src.models.fpn_architecture_v2 import FPNCompareNetV2
except ImportError:
    # Fallback for when running from different directory
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Acc88', 'scripts'))
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Acc88'))
    from models.simple_compare_cnn import ResNetCompareNet, BasicBlock, Bottleneck
    from fpn_architecture_v2 import FPNCompareNetV2

class ModelLoader:
    """Loads and manages all pre-trained models for stacking ensemble."""
    
    def __init__(self, models_dir: str = "models", device: str = "auto"):
        """
        Initialize model loader.
        
        Args:
            models_dir: Directory containing model files
            device: Device to load models on ('auto', 'cuda', 'cpu')
        """
        self.models_dir = models_dir
        if device == "auto":
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Model configurations - use absolute paths
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_configs = [
            ("resnet_optimized_1.12", os.path.join(base_dir, "models", "resnet_optimized_1.12", "model.pt"), 0.8875),
            ("resnet_fusion", os.path.join(base_dir, "models", "resnet_fusion", "model.pt"), 0.8792),
            ("resnet_optimized", os.path.join(base_dir, "models", "resnet_optimized", "model.pt"), 0.8780),
            ("seed_2025", os.path.join(base_dir, "models", "seed_2025", "model.pt"), 0.8739),
            ("fpn_model", os.path.join(base_dir, "models", "fpn_model", "model.pt"), 0.8730),
            ("seed_2023", os.path.join(base_dir, "models", "seed_2023", "model.pt"), 0.8702),
            ("seed_2024", os.path.join(base_dir, "models", "seed_2024", "model.pt"), 0.8671),
            ("resnet_fusion_seed42", os.path.join(base_dir, "models", "resnet_fusion_seed42", "model.pt"), 0.8538),
            ("resnet_fusion_seed123", os.path.join(base_dir, "models", "resnet_fusion_seed123", "model.pt"), 0.8436),
            ("resnet_fusion_seed456", os.path.join(base_dir, "models", "resnet_fusion_seed456", "model.pt"), 0.8439),
        ]
        
        self.models = []
        self.model_names = []
        self.accuracies = []
    
    def load_model(self, model_path: str) -> Tuple[torch.nn.Module, float]:
        """
        Load a single model from path.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Tuple of (model, accuracy)
        """
        print(f"Loading: {os.path.basename(model_path)}")
        
        # Read configuration
        metrics_path = model_path.replace('model.pt', 'metrics.json')
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Create model
        layers = metrics.get('layers', [2, 2, 2])
        feat_dim = metrics.get('feat_dim', 256)
        width_mult = metrics.get('width_mult', 1.0)
        use_bottleneck = metrics.get('use_bottleneck', False)
        
        block = Bottleneck if use_bottleneck else BasicBlock
        
        if 'fpn' in model_path.lower():
            model = FPNCompareNetV2(feat_dim=feat_dim, layers=layers, width_mult=width_mult)
        else:
            # ResNetCompareNet doesn't support width_mult, ignore it
            model = ResNetCompareNet(feat_dim=feat_dim, layers=layers, block=block)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        acc = metrics.get('best_val_acc', 0)
        print(f"  Accuracy: {acc:.4f}")
        
        return model, acc
    
    def load_all_models(self) -> Tuple[List[torch.nn.Module], List[str], List[float]]:
        """
        Load all models for stacking ensemble.
        
        Returns:
            Tuple of (models, model_names, accuracies)
        """
        print(f"Loading all models from {self.models_dir}")
        print(f"Using device: {self.device}")
        
        models = []
        model_names = []
        accuracies = []
        
        for name, path, expected_acc in self.model_configs:
            if os.path.exists(path):
                try:
                    model, acc = self.load_model(path)
                    models.append(model)
                    model_names.append(name)
                    accuracies.append(acc)
                except Exception as e:
                    print(f"Failed to load {name}: {e}")
                    continue
            else:
                print(f"Model not found: {path}")
                continue
        
        self.models = models
        self.model_names = model_names
        self.accuracies = accuracies
        
        print(f"\nSuccessfully loaded {len(models)} models")
        print(f"Model accuracies: {[f'{acc:.4f}' for acc in accuracies]}")
        
        return models, model_names, accuracies
    
    def get_model_predictions(self, models: List[torch.nn.Module], data_loader) -> np.ndarray:
        """
        Get predictions from all models.
        
        Args:
            models: List of loaded models
            data_loader: DataLoader for input data
            
        Returns:
            Numpy array of shape (N, num_models) with prediction probabilities
        """
        all_predictions = []
        
        for i, model in enumerate(models):
            print(f"Model {i+1}/{len(models)} predicting...")
            predictions = []
            
            with torch.no_grad():
                for batch in data_loader:
                    xa, xb, _ = batch
                    xa = xa.to(self.device)
                    xb = xb.to(self.device)
                    
                    logits = model(xa, xb)
                    probs = torch.sigmoid(logits)
                    
                    if probs.dim() == 1:
                        probs = probs.unsqueeze(1)
                    
                    predictions.append(probs.cpu())
            
            all_predictions.append(torch.cat(predictions, dim=0))
        
        return torch.stack(all_predictions, dim=1).squeeze(-1).numpy()  # [N, num_models]

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    # Test model loading
    loader = ModelLoader()
    models, names, accs = loader.load_all_models()
    print(f"Loaded {len(models)} models successfully!")
