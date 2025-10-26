#!/usr/bin/env python3
"""
生成test_private.npz的预测结果
使用最佳Stacking模型（验证集92.89%，测试集90.30%）
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

sys.path.append('src')
sys.path.append('src/models')

from models.simple_compare_cnn import ResNetCompareNet, BasicBlock, Bottleneck
from models.fpn_architecture_v2 import FPNCompareNetV2

class PrivateTestDataset(Dataset):
    """Private test dataset without labels"""
    
    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.x = data['x']  # Shape: (N, 28, 56)
        self.ids = data['id']  # IDs for submission
        
        print(f"Private test dataset loaded: {len(self.x)} samples")
        
        # Normalize
        self.x = self.x.astype(np.float32) / 255.0
        self.x = self.x.reshape(-1, 1, 28, 56)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        full_image = self.x[idx]
        left_image = full_image[:, :, :28]
        right_image = full_image[:, :, 28:]
        
        left_tensor = torch.from_numpy(left_image)
        right_tensor = torch.from_numpy(right_image)
        
        return left_tensor, right_tensor, self.ids[idx]

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
    
    acc = metrics.get('best_val_acc', 0)
    print(f" OK ({acc:.4f})")
    
    return model, acc

def get_model_predictions(models, data_loader, device):
    """Get predictions from all models"""
    all_predictions = []
    
    for model in models:
        model_preds = []
        for x1, x2, _ in data_loader:
            x1, x2 = x1.to(device), x2.to(device)
            with torch.no_grad():
                outputs = model(x1, x2)
                probs = torch.sigmoid(outputs).squeeze()
                model_preds.append(probs.cpu().numpy())
        
        all_predictions.append(np.concatenate(model_preds))
    
    return np.column_stack(all_predictions)

def train_stacking_on_val(val_predictions, val_labels, n_folds=5):
    """Train stacking meta-learner on validation set"""
    print("\n训练Stacking元学习器（在验证集上）...")
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    oof_predictions = np.zeros(len(val_predictions))
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(val_predictions, val_labels)):
        X_train, X_val = val_predictions[train_idx], val_predictions[val_idx]
        y_train, y_val = val_labels[train_idx], val_labels[val_idx]
        
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
        
        lgb_model.fit(X_train, y_train)
        val_pred = lgb_model.predict_proba(X_val)[:, 1]
        oof_predictions[val_idx] = val_pred
        
        from sklearn.metrics import accuracy_score
        val_pred_binary = (val_pred >= 0.5).astype(int)
        fold_acc = accuracy_score(y_val, val_pred_binary)
        fold_scores.append(fold_acc)
        print(f"  Fold {fold+1}: {fold_acc:.4f}")
    
    # Train final model on all validation data
    print("  训练最终模型...")
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
    
    final_model.fit(val_predictions, val_labels)
    
    return final_model

def main():
    print("=" * 70)
    print("生成test_private.npz预测结果")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")
    
    # 模型配置
    model_configs = [
        ("resnet_optimized_1.12", "models/stage2_resnet_optimized/resnet_optimized_1.12/model.pt"),
        ("resnet_fusion", "models/stage2_resnet_optimized/resnet_fusion/model.pt"),
        ("resnet_optimized", "models/stage2_resnet_optimized/resnet_optimized/model.pt"),
        ("seed_2025", "models/stage3_multi_seed/seed_2025/model.pt"),
        ("fpn_model", "models/stage3_multi_seed/fpn_model/model.pt"),
        ("seed_2023", "models/stage3_multi_seed/seed_2023/model.pt"),
        ("seed_2024", "models/stage3_multi_seed/seed_2024/model.pt"),
        ("resnet_fusion_seed42", "models/stage3_multi_seed/resnet_fusion_seed42/model.pt"),
        ("resnet_fusion_seed123", "models/stage3_multi_seed/resnet_fusion_seed123/model.pt"),
        ("resnet_fusion_seed456", "models/stage3_multi_seed/resnet_fusion_seed456/model.pt"),
    ]
    
    # 步骤1: 加载模型
    print("步骤1: 加载模型")
    print("-" * 70)
    models = []
    model_names = []
    
    for name, path in model_configs:
        if os.path.exists(path):
            try:
                model, acc = load_model(path, device)
                models.append(model)
                model_names.append(name)
            except Exception as e:
                print(f"  失败: {e}")
        else:
            print(f"  跳过: {name} (不存在)")
    
    print(f"\n成功加载 {len(models)} 个模型")
    
    if len(models) == 0:
        print("错误: 没有加载任何模型")
        return
    
    # 步骤2: 在验证集上训练Stacking
    print("\n步骤2: 在验证集上训练Stacking元学习器")
    print("-" * 70)
    
    # 加载验证集
    from data_loader import PairNPZDataset
    val_dataset = PairNPZDataset('data/val.npz', is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    print(f"验证集: {len(val_dataset)} 样本")
    
    # 获取验证集预测
    print("获取验证集预测...")
    X_val = get_model_predictions(models, val_loader, device)
    y_val = []
    for _, _, labels in val_loader:
        y_val.append(labels.numpy())
    y_val = np.concatenate(y_val).astype(int)
    
    # 训练Stacking
    stacking_model = train_stacking_on_val(X_val, y_val)
    
    # 步骤3: 加载private test数据
    print("\n步骤3: 加载test_private.npz")
    print("-" * 70)
    
    private_dataset = PrivateTestDataset('data/test_private.npz')
    private_loader = DataLoader(private_dataset, batch_size=64, shuffle=False)
    
    # 步骤4: 生成预测
    print("\n步骤4: 生成预测")
    print("-" * 70)
    
    print("获取模型预测...")
    X_private = get_model_predictions(models, private_loader, device)
    
    print("使用Stacking生成最终预测...")
    private_predictions = stacking_model.predict_proba(X_private)[:, 1]
    private_labels = (private_predictions >= 0.5).astype(int)
    
    # 获取IDs
    ids = []
    for _, _, id_val in private_loader:
        ids.extend(id_val)
    
    print(f"生成 {len(ids)} 个预测")
    
    # 步骤5: 保存为CSV
    print("\n步骤5: 保存预测结果")
    print("-" * 70)
    
    df = pd.DataFrame({
        'id': ids,
        'label': private_labels
    })
    
    output_file = 'pred_private.csv'
    df.to_csv(output_file, index=False)
    print(f"预测已保存到: {output_file}")
    
    # 显示统计信息
    print("\n预测统计:")
    print(f"  总样本数: {len(df)}")
    print(f"  标签分布:")
    print(f"    0: {(df['label'] == 0).sum()} ({(df['label'] == 0).sum() / len(df) * 100:.2f}%)")
    print(f"    1: {(df['label'] == 1).sum()} ({(df['label'] == 1).sum() / len(df) * 100:.2f}%)")
    
    # 显示前几行
    print("\n前10行预测:")
    print(df.head(10))
    
    print("\n" + "=" * 70)
    print("完成！")
    print("=" * 70)
    print(f"下一步: 运行验证脚本")
    print(f"  python check_submission.py --data_dir data --pred {output_file} --test_file test_private.npz")
    
    return df

if __name__ == '__main__':
    main()

