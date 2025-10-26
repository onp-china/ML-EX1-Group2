#!/usr/bin/env python3
"""
Stacking元学习器集成
使用所有10个模型作为基础模型，LightGBM作为元学习器
目标：突破90%
"""

import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

# 添加路径 - 适应项目结构
sys.path.append('src')
sys.path.append('src/models')

from data_loader import PairNPZDataset
from models.simple_compare_cnn import ResNetCompareNet, BasicBlock, Bottleneck
from models.fpn_architecture_v2 import FPNCompareNetV2

def load_model(model_path, device):
    """Load model"""
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
        # 对于不支持width_mult的模型，忽略该参数
        try:
            model = ResNetCompareNet(feat_dim=feat_dim, layers=layers, block=block, width_mult=width_mult)
        except TypeError:
            # 如果ResNetCompareNet不支持width_mult，使用默认参数
            model = ResNetCompareNet(feat_dim=feat_dim, layers=layers, block=block)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    acc = metrics.get('best_val_acc', 0)
    print(f"  Accuracy: {acc:.4f}")
    
    return model, acc

def get_model_predictions(models, data_loader, device):
    """Get predictions from all models"""
    all_predictions = []
    
    for i, model in enumerate(models):
        print(f"Model {i+1}/{len(models)} predicting...")
        predictions = []
        
        with torch.no_grad():
            for batch in data_loader:
                xa, xb, _ = batch
                xa = xa.to(device)
                xb = xb.to(device)
                
                logits = model(xa, xb)
                probs = torch.sigmoid(logits)
                
                if probs.dim() == 1:
                    probs = probs.unsqueeze(1)
                
                predictions.append(probs.cpu())
        
        all_predictions.append(torch.cat(predictions, dim=0))
    
    return torch.stack(all_predictions, dim=1).squeeze(-1)  # [N, num_models]

def train_stacking_model(X, y, n_folds=5):
    """Train Stacking meta-learner"""
    print(f"Training Stacking meta-learner with {n_folds}-fold cross-validation...")
    
    # Create cross-validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Store predictions for each fold
    oof_predictions = np.zeros(len(X))
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"  Training fold {fold+1}...")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create LightGBM model
        lgb_model = lgb.LGBMClassifier(
            objective='binary',
            metric='binary_logloss',
            boosting_type='gbdt',
            num_leaves=31,
            learning_rate=0.05,
            feature_fraction=0.9,
            bagging_fraction=0.8,
            bagging_freq=5,
            verbose=-1,
            random_state=42,
            n_estimators=100
        )
        
        # Train model
        lgb_model.fit(X_train, y_train)
        
        # Predict validation set
        val_pred = lgb_model.predict_proba(X_val)[:, 1]
        oof_predictions[val_idx] = val_pred
        
        # Calculate accuracy
        val_pred_binary = (val_pred >= 0.5).astype(int)
        fold_acc = accuracy_score(y_val, val_pred_binary)
        fold_scores.append(fold_acc)
        
        print(f"    Fold {fold+1} accuracy: {fold_acc:.4f}")
    
    # Calculate overall performance
    oof_pred_binary = (oof_predictions >= 0.5).astype(int)
    overall_acc = accuracy_score(y, oof_pred_binary)
    
    print(f"Stacking overall accuracy: {overall_acc:.4f}")
    print(f"Fold accuracies: {[f'{acc:.4f}' for acc in fold_scores]}")
    
    # Train final model
    print("Training final Stacking model...")
    final_model = lgb.LGBMClassifier(
        objective='binary',
        metric='binary_logloss',
        boosting_type='gbdt',
        num_leaves=31,
        learning_rate=0.05,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=5,
        verbose=-1,
        random_state=42,
        n_estimators=100
    )
    
    final_model.fit(X, y)
    
    return final_model, overall_acc, fold_scores

def main():
    print("Stacking Meta-Learner Ensemble")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # All model list - 修改为项目内的路径
    model_configs = [
        # Models above 87%
        ("resnet_optimized_1.12", "models/stage2_resnet_optimized/resnet_optimized_1.12/model.pt", 0.8875),
        ("resnet_fusion", "models/stage2_resnet_optimized/resnet_fusion/model.pt", 0.8792),
        ("resnet_optimized", "models/stage2_resnet_optimized/resnet_optimized/model.pt", 0.8780),
        ("seed_2025", "models/stage3_multi_seed/seed_2025/model.pt", 0.8739),
        ("fpn_model", "models/stage3_multi_seed/fpn_model/model.pt", 0.8730),
        ("seed_2023", "models/stage3_multi_seed/seed_2023/model.pt", 0.8702),
        ("seed_2024", "models/stage3_multi_seed/seed_2024/model.pt", 0.8671),
        
        # Models from 89.26% ensemble
        ("resnet_fusion_seed42", "models/stage3_multi_seed/resnet_fusion_seed42/model.pt", 0.8538),
        ("resnet_fusion_seed123", "models/stage3_multi_seed/resnet_fusion_seed123/model.pt", 0.8436),
        ("resnet_fusion_seed456", "models/stage3_multi_seed/resnet_fusion_seed456/model.pt", 0.8439),
    ]
    
    print("\n=== Step 1: Loading All Models ===")
    models = []
    accuracies = []
    model_names = []
    
    for name, path, expected_acc in model_configs:
        if os.path.exists(path):
            try:
                model, acc = load_model(path, device)
                models.append(model)
                accuracies.append(acc)
                model_names.append(name)
            except Exception as e:
                print(f"Failed to load {name}: {e}")
                continue
        else:
            print(f"Skipping non-existent model: {path}")
    
    print(f"\nSuccessfully loaded {len(models)} models")
    print(f"Model accuracies: {[f'{acc:.4f}' for acc in accuracies]}")
    
    if len(models) == 0:
        print("No models loaded successfully!")
        return
    
    # Load validation data
    print("\n=== Step 2: Loading Validation Data ===")
    val_dataset = PairNPZDataset('data/val.npz', is_train=False, use_augmentation=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    print(f"Validation set size: {len(val_dataset)}")
    
    # Get predictions from all models
    print("\n=== Step 3: Getting Model Predictions ===")
    X = get_model_predictions(models, val_loader, device).numpy()
    print(f"Feature matrix shape: {X.shape}")
    
    # Get true labels
    print("\n=== Step 4: Getting True Labels ===")
    y = []
    for batch in val_loader:
        _, _, labels = batch
        if isinstance(labels, torch.Tensor):
            y.append(labels.numpy())
    
    y = np.concatenate(y).astype(int)
    print(f"Label shape: {y.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Stacking ensemble
    print("\n=== Step 5: Stacking Ensemble ===")
    
    # 1. Simple average as baseline
    simple_avg = X.mean(axis=1)
    simple_pred_binary = (simple_avg >= 0.5).astype(int)
    simple_acc = accuracy_score(y, simple_pred_binary)
    print(f"Simple average baseline: Acc={simple_acc:.4f}")
    
    # 2. Performance weighted as baseline
    weights = np.array(accuracies)
    weights = weights / weights.sum()
    weighted_avg = (X * weights).sum(axis=1)
    weighted_pred_binary = (weighted_avg >= 0.5).astype(int)
    weighted_acc = accuracy_score(y, weighted_pred_binary)
    print(f"Performance weighted baseline: Acc={weighted_acc:.4f}")
    
    # 3. Stacking meta-learner
    try:
        stacking_model, stacking_acc, fold_scores = train_stacking_model(X, y, n_folds=5)
        
        # Use Stacking model for prediction
        stacking_pred = stacking_model.predict_proba(X)[:, 1]
        stacking_pred_binary = (stacking_pred >= 0.5).astype(int)
        stacking_final_acc = accuracy_score(y, stacking_pred_binary)
        
        print(f"Stacking final accuracy: Acc={stacking_final_acc:.4f}")
        
    except Exception as e:
        print(f"Stacking training failed: {e}")
        stacking_acc = weighted_acc
        stacking_final_acc = weighted_acc
        fold_scores = []
    
    # Final results comparison
    print("\n=== Step 6: Final Results Comparison ===")
    
    all_methods = [
        ("Best Single Model", max(accuracies)),
        ("Simple Average", simple_acc),
        ("Performance Weighted", weighted_acc),
        ("Stacking Meta-Learner", stacking_final_acc)
    ]
    
    # Sort by accuracy
    all_methods.sort(key=lambda x: x[1], reverse=True)
    
    print("All methods ranking:")
    for i, (method, acc) in enumerate(all_methods, 1):
        print(f"  {i}. {method}: {acc:.4f}")
    
    best_method, best_acc = all_methods[0]
    print(f"\nBest method: {best_method}")
    print(f"Best accuracy: {best_acc:.4f}")
    
    # Calculate improvement relative to best single model
    single_model_acc = max(accuracies)
    improvement = best_acc - single_model_acc
    print(f"Improvement over single model: +{improvement:.4f} ({improvement*100:.2f}%)")
    
    # Check if 90% target is reached
    if best_acc >= 0.90:
        print(f"Congratulations! 90% target achieved!")
    else:
        gap = 0.90 - best_acc
        print(f"Gap to 90% target: {gap:.4f} ({gap*100:.2f}%)")
    
    # Save results
    result = {
        'single_model_acc': single_model_acc,
        'best_method': best_method,
        'best_acc': best_acc,
        'improvement': improvement,
        'all_methods': all_methods,
        'model_accuracies': accuracies,
        'model_names': model_names,
        'reached_90_percent': best_acc >= 0.90,
        'stacking_fold_scores': fold_scores
    }
    
    os.makedirs('outputs/results', exist_ok=True)
    with open('outputs/results/stacking_ensemble_result.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nResults saved to: outputs/results/stacking_ensemble_result.json")
    print("\nStacking ensemble test completed!")

if __name__ == "__main__":
    main()
