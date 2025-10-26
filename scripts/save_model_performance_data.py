#!/usr/bin/env python3
"""
保存所有模型的性能数据，避免重复评估
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import lightgbm as lgb

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
    
    return model, metrics

def evaluate_model(model, dataloader, device):
    """Evaluate model on a dataset"""
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for xa, xb, y in dataloader:
            xa, xb = xa.to(device), xb.to(device)
            logits = model(xa, xb)
            
            if logits.dim() == 1:
                logits = logits.unsqueeze(1)
            
            if logits.shape[1] == 1:
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).long().squeeze()
            else:
                preds = torch.argmax(logits, dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.numpy())
    
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    
    return accuracy_score(y_true, y_pred), y_pred, y_true

def get_stacking_predictions(models, dataloader, device):
    """Get stacking ensemble predictions"""
    all_predictions = []
    all_labels = []
    first_model = True
    
    for model in models:
        model_preds = []
        with torch.no_grad():
            for xa, xb, y in dataloader:
                xa, xb = xa.to(device), xb.to(device)
                logits = model(xa, xb)
                
                if logits.dim() == 1:
                    logits = logits.unsqueeze(1)
                
                if logits.shape[1] == 1:
                    probs = torch.sigmoid(logits)
                    probs = torch.cat([1 - probs, probs], dim=1)
                else:
                    probs = F.softmax(logits, dim=1)
                
                model_preds.append(probs.cpu().numpy())
                
                if first_model:
                    all_labels.append(y.numpy())
        
        first_model = False
        all_predictions.append(np.concatenate(model_preds, axis=0))
    
    stacked = np.stack(all_predictions, axis=0)
    stacked = stacked.transpose(1, 0, 2).reshape(stacked.shape[1], -1)
    labels = np.concatenate(all_labels, axis=0)
    
    return stacked, labels

def main():
    print("="*70)
    print("保存模型性能数据")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")
    
    # 定义10个模型路径
    model_configs = [
        ('resnet_optimized_1.12', 'models/stage2_resnet_optimized/resnet_optimized_1.12/model.pt'),
        ('resnet_fusion', 'models/stage2_resnet_optimized/resnet_fusion/model.pt'),
        ('resnet_optimized', 'models/stage2_resnet_optimized/resnet_optimized/model.pt'),
        ('seed_2025', 'models/stage3_multi_seed/seed_2025/model.pt'),
        ('seed_2023', 'models/stage3_multi_seed/seed_2023/model.pt'),
        ('seed_2024', 'models/stage3_multi_seed/seed_2024/model.pt'),
        ('fpn_model', 'models/stage3_multi_seed/fpn_model/model.pt'),
        ('resnet_fusion_seed42', 'models/stage3_multi_seed/resnet_fusion_seed42/model.pt'),
        ('resnet_fusion_seed456', 'models/stage3_multi_seed/resnet_fusion_seed456/model.pt'),
        ('resnet_fusion_seed123', 'models/stage3_multi_seed/resnet_fusion_seed123/model.pt'),
    ]
    
    # 加载数据集
    print("加载数据集...")
    train_dataset = PairNPZDataset('data/train.npz', is_train=False)
    val_dataset = PairNPZDataset('data/val.npz', is_train=False)
    test_public_dataset = PairNPZDataset('data/test_public.npz', is_train=False, 
                                         labels_path='data/test_public_labels.csv')
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)
    test_public_loader = DataLoader(test_public_dataset, batch_size=128, shuffle=False, num_workers=0)
    
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")
    print(f"Test Public: {len(test_public_dataset)} 样本\n")
    
    # 加载模型并评估
    print("="*70)
    print("评估单个模型")
    print("="*70)
    
    models_list = []
    model_results = []
    
    for name, path in model_configs:
        print(f"\n评估 {name}...")
        model, metrics = load_model(path, device)
        models_list.append(model)
        
        train_acc, _, _ = evaluate_model(model, train_loader, device)
        val_acc, _, _ = evaluate_model(model, val_loader, device)
        test_acc, _, _ = evaluate_model(model, test_public_loader, device)
        
        result = {
            'name': name,
            'train_accuracy': float(train_acc),
            'val_accuracy': float(val_acc),
            'test_public_accuracy': float(test_acc),
            'parameters': metrics.get('total_params', 0)
        }
        model_results.append(result)
        
        print(f"  训练集: {train_acc:.4f}")
        print(f"  验证集: {val_acc:.4f}")
        print(f"  Test Public: {test_acc:.4f}")
    
    # 评估Stacking集成
    print("\n" + "="*70)
    print("评估Stacking集成")
    print("="*70)
    
    # 训练Stacking元学习器
    print("\n在验证集上训练Stacking元学习器...")
    X_val, y_val = get_stacking_predictions(models_list, val_loader, device)
    
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_val, y_val), 1):
        X_train_fold, X_val_fold = X_val[train_idx], X_val[val_idx]
        y_train_fold, y_val_fold = y_val[train_idx], y_val[val_idx]
        
        meta_learner = lgb.LGBMClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=5,
            num_leaves=31, random_state=42, verbose=-1
        )
        meta_learner.fit(X_train_fold, y_train_fold)
        y_pred = meta_learner.predict(X_val_fold)
        acc = accuracy_score(y_val_fold, y_pred)
        cv_scores.append(float(acc))
        print(f"  Fold {fold}: {acc:.4f}")
    
    print(f"  平均: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    # 训练最终元学习器
    print("\n训练最终元学习器...")
    final_meta_learner = lgb.LGBMClassifier(
        n_estimators=100, learning_rate=0.05, max_depth=5,
        num_leaves=31, random_state=42, verbose=-1
    )
    final_meta_learner.fit(X_val, y_val)
    
    # 在所有数据集上评估
    print("\n在所有数据集上评估Stacking...")
    
    # 训练集
    X_train, y_train = get_stacking_predictions(models_list, train_loader, device)
    train_pred = final_meta_learner.predict(X_train)
    train_acc_stack = accuracy_score(y_train, train_pred)
    train_f1_stack = f1_score(y_train, train_pred)
    
    # 验证集
    val_pred = final_meta_learner.predict(X_val)
    val_acc_stack = accuracy_score(y_val, val_pred)
    val_f1_stack = f1_score(y_val, val_pred)
    val_cm = confusion_matrix(y_val, val_pred)
    
    # Test Public
    X_test, y_test = get_stacking_predictions(models_list, test_public_loader, device)
    test_pred = final_meta_learner.predict(X_test)
    test_acc_stack = accuracy_score(y_test, test_pred)
    test_f1_stack = f1_score(y_test, test_pred)
    
    print(f"  训练集: {train_acc_stack:.4f}")
    print(f"  验证集: {val_acc_stack:.4f}")
    print(f"  Test Public: {test_acc_stack:.4f}")
    
    # 保存所有数据
    performance_data = {
        'individual_models': model_results,
        'stacking_ensemble': {
            'train_accuracy': float(train_acc_stack),
            'train_f1': float(train_f1_stack),
            'val_accuracy': float(val_acc_stack),
            'val_f1': float(val_f1_stack),
            'test_public_accuracy': float(test_acc_stack),
            'test_public_f1': float(test_f1_stack),
            'cv_scores': cv_scores,
            'cv_mean': float(np.mean(cv_scores)),
            'cv_std': float(np.std(cv_scores)),
            'confusion_matrix': val_cm.tolist()
        },
        'simple_average': {
            'train_accuracy': float(np.mean([r['train_accuracy'] for r in model_results])),
            'val_accuracy': float(np.mean([r['val_accuracy'] for r in model_results])),
            'test_public_accuracy': float(np.mean([r['test_public_accuracy'] for r in model_results]))
        }
    }
    
    # 保存到文件
    output_file = 'outputs/results/model_performance_data.json'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(performance_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print("完成！")
    print(f"{'='*70}")
    print(f"性能数据已保存到: {output_file}")
    print(f"\n摘要:")
    print(f"  单个模型数量: {len(model_results)}")
    print(f"  Stacking训练集: {train_acc_stack:.4f}")
    print(f"  Stacking验证集: {val_acc_stack:.4f}")
    print(f"  Stacking Test Public: {test_acc_stack:.4f}")

if __name__ == '__main__':
    main()

