#!/usr/bin/env python3
"""
评估10个模型的Stacking集成在验证集和test_public上的表现
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import pandas as pd

# 添加路径
sys.path.append('src')
sys.path.append('src/models')

from data_loader import PairNPZDataset
from models.simple_compare_cnn import ResNetCompareNet, BasicBlock, Bottleneck
from models.fpn_architecture_v2 import FPNCompareNetV2

def load_model(model_path, device):
    """Load model using metrics.json"""
    print(f"Loading: {os.path.basename(os.path.dirname(model_path))}", end='...')
    
    metrics_path = model_path.replace('model.pt', 'metrics.json')
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    layers = metrics.get('layers', [2, 2, 2])
    feat_dim = metrics.get('feat_dim', 256)
    width_mult = metrics.get('width_mult', 1.0)
    use_bottleneck = metrics.get('use_bottleneck', False)
    
    block = Bottleneck if use_bottleneck else BasicBlock
    
    if 'fpn' in model_path.lower():
        # FPN模型使用width_mult=1.25（即使metrics.json中没有记录）
        model = FPNCompareNetV2(feat_dim=feat_dim, layers=layers, width_mult=1.25)
    else:
        try:
            model = ResNetCompareNet(feat_dim=feat_dim, layers=layers, block=block, width_mult=width_mult)
        except TypeError:
            model = ResNetCompareNet(feat_dim=feat_dim, layers=layers, block=block)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    val_acc = metrics.get('best_val_acc', 0.0)
    print(f" OK ({val_acc:.4f})")
    
    return model, val_acc

def get_model_predictions(models, dataloader, device):
    """Get predictions from all models"""
    all_predictions = []
    all_labels = []
    
    # Collect labels first (only once)
    first_model = True
    
    for model in models:
        model_preds = []
        with torch.no_grad():
            for xa, xb, y in dataloader:
                xa, xb = xa.to(device), xb.to(device)
                logits = model(xa, xb)
                
                # Handle both 1D and 2D logits
                if logits.dim() == 1:
                    logits = logits.unsqueeze(1)
                
                # For binary classification with single output, convert to 2-class probabilities
                if logits.shape[1] == 1:
                    probs = torch.sigmoid(logits)
                    probs = torch.cat([1 - probs, probs], dim=1)
                else:
                    probs = F.softmax(logits, dim=1)
                
                model_preds.append(probs.cpu().numpy())
                
                # Collect labels only for the first model
                if first_model:
                    all_labels.append(y.numpy())
        
        first_model = False
        all_predictions.append(np.concatenate(model_preds, axis=0))
    
    # Stack: (n_models, n_samples, n_classes) -> (n_samples, n_models * n_classes)
    stacked = np.stack(all_predictions, axis=0)
    stacked = stacked.transpose(1, 0, 2).reshape(stacked.shape[1], -1)
    
    labels = np.concatenate(all_labels, axis=0) if all_labels else None
    
    return stacked, labels

def train_stacking_meta_learner(X, y, n_splits=5):
    """Train stacking meta-learner with cross-validation"""
    print(f"\n训练Stacking元学习器（{n_splits}折交叉验证）...")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        meta_learner = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=31,
            random_state=42,
            verbose=-1
        )
        meta_learner.fit(X_train, y_train)
        
        y_pred = meta_learner.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        cv_scores.append(acc)
        print(f"  Fold {fold}: {acc:.4f}")
    
    print(f"  平均准确率: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    # Train final model on all data
    print(f"  训练最终模型...")
    final_meta_learner = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=31,
        random_state=42,
        verbose=-1
    )
    final_meta_learner.fit(X, y)
    
    return final_meta_learner, cv_scores

def evaluate_on_dataset(models, meta_learner, dataloader, device, dataset_name):
    """Evaluate stacking ensemble on a dataset"""
    print(f"\n{'='*70}")
    print(f"评估 {dataset_name}")
    print(f"{'='*70}")
    
    # Get predictions
    print("获取模型预测...")
    X, y_true = get_model_predictions(models, dataloader, device)
    
    # Stacking prediction
    print("使用Stacking集成预测...")
    y_pred = meta_learner.predict(X)
    y_proba = meta_learner.predict_proba(X)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # Individual model predictions (simple average)
    individual_preds = X.reshape(len(X), -1, 2).mean(axis=1).argmax(axis=1)
    individual_acc = accuracy_score(y_true, individual_preds)
    
    print(f"\n结果:")
    print(f"  样本数量: {len(y_true)}")
    print(f"  Stacking准确率: {accuracy:.4f}")
    print(f"  Stacking F1分数: {f1:.4f}")
    print(f"  简单平均准确率: {individual_acc:.4f}")
    print(f"  提升: {(accuracy - individual_acc)*100:.2f}%")
    
    return {
        'dataset': dataset_name,
        'n_samples': len(y_true),
        'stacking_accuracy': float(accuracy),
        'stacking_f1': float(f1),
        'simple_average_accuracy': float(individual_acc),
        'improvement': float(accuracy - individual_acc)
    }

def main():
    print("="*70)
    print("评估10个模型的Stacking集成")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")
    
    # 定义10个模型路径
    model_paths = [
        'models/stage2_resnet_optimized/resnet_optimized_1.12/model.pt',
        'models/stage2_resnet_optimized/resnet_fusion/model.pt',
        'models/stage2_resnet_optimized/resnet_optimized/model.pt',
        'models/stage3_multi_seed/seed_2025/model.pt',
        'models/stage3_multi_seed/fpn_model/model.pt',
        'models/stage3_multi_seed/seed_2023/model.pt',
        'models/stage3_multi_seed/seed_2024/model.pt',
        'models/stage3_multi_seed/resnet_fusion_seed42/model.pt',
        'models/stage3_multi_seed/resnet_fusion_seed123/model.pt',
        'models/stage3_multi_seed/resnet_fusion_seed456/model.pt',
    ]
    
    # Step 1: Load models
    print("步骤1: 加载模型")
    print("-"*70)
    models = []
    for path in model_paths:
        try:
            model, val_acc = load_model(path, device)
            models.append(model)
        except Exception as e:
            print(f" 失败: {e}")
    
    print(f"\n成功加载 {len(models)} 个模型\n")
    
    if len(models) != 10:
        print(f"警告: 只加载了 {len(models)}/10 个模型")
    
    # Step 2: Load validation data and train meta-learner
    print("步骤2: 在验证集上训练Stacking元学习器")
    print("-"*70)
    val_dataset = PairNPZDataset('data/val.npz', is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    print(f"验证集: {len(val_dataset)} 样本")
    
    print("获取验证集预测...")
    X_val, y_val = get_model_predictions(models, val_loader, device)
    
    # Train meta-learner
    meta_learner, cv_scores = train_stacking_meta_learner(X_val, y_val)
    
    # Step 3: Evaluate on validation set
    val_results = evaluate_on_dataset(models, meta_learner, val_loader, device, "验证集 (val.npz)")
    
    # Step 4: Evaluate on test_public
    print("\n步骤3: 在test_public上评估")
    print("-"*70)
    test_public_dataset = PairNPZDataset(
        'data/test_public.npz', 
        is_train=False,
        labels_path='data/test_public_labels.csv'
    )
    test_public_loader = DataLoader(test_public_dataset, batch_size=64, shuffle=False, num_workers=0)
    print(f"test_public: {len(test_public_dataset)} 样本")
    
    test_public_results = evaluate_on_dataset(
        models, meta_learner, test_public_loader, device, "测试集 (test_public.npz)"
    )
    
    # Save results
    results = {
        'n_models': len(models),
        'cv_scores': [float(s) for s in cv_scores],
        'cv_mean': float(np.mean(cv_scores)),
        'cv_std': float(np.std(cv_scores)),
        'validation': val_results,
        'test_public': test_public_results
    }
    
    output_file = 'outputs/results/10models_stacking_evaluation.json'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print("完成！")
    print(f"{'='*70}")
    print(f"结果已保存到: {output_file}")
    print(f"\n总结:")
    print(f"  验证集准确率: {val_results['stacking_accuracy']:.4f}")
    print(f"  test_public准确率: {test_public_results['stacking_accuracy']:.4f}")

if __name__ == '__main__':
    main()

