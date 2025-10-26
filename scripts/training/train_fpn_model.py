#!/usr/bin/env python3
"""
FPN (Feature Pyramid Network) 训练脚本
特征金字塔网络 - 多尺度特征融合

关键创新:
1. 特征金字塔结构
2. 多尺度特征融合
3. 自顶向下和自底向上路径
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# 添加路径
sys.path.append('src')
sys.path.append('src/models')

from data_loader import PairNPZDataset
from models.fpn_architecture_v2 import FPNCompareNetV2
from training_history import TrainingHistory

# 使用原本的FPNCompareNetV2模型

class FocalLoss(nn.Module):
    """Focal Loss - 关注难样本"""
    def __init__(self, alpha=1.0, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        
    def forward(self, inputs, targets):
        # 标签平滑
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # 计算BCE损失
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # 计算概率
        pt = torch.exp(-bce_loss)
        
        # 计算Focal Loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        return focal_loss.mean()

def train_epoch(model, dataloader, optimizer, criterion, device, scaler):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch_idx, (img1, img2, labels) in enumerate(tqdm(dataloader, desc="Training")):
        img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(img1, img2)
            loss = criterion(outputs.squeeze(), labels.float())
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
        # 计算准确率
        preds = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    return total_loss / len(dataloader), acc, f1

def validate(model, dataloader, criterion, device):
    """验证模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for img1, img2, labels in tqdm(dataloader, desc="Validation"):
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            
            outputs = model(img1, img2)
            loss = criterion(outputs.squeeze(), labels.float())
            
            total_loss += loss.item()
            
            # 计算准确率
            preds = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    return total_loss / len(dataloader), acc, f1

def main():
    parser = argparse.ArgumentParser(description='Train FPN Model')
    parser.add_argument('--epochs', type=int, default=60, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--feat_dim', type=int, default=256, help='Feature dimension')
    parser.add_argument('--layers', type=str, default='2,2,2', help='ResNet layers')
    parser.add_argument('--output_dir', type=str, default='models/stage3_multi_seed/fpn_model', help='Output directory')
    
    args = parser.parse_args()
    
    # 解析layers参数
    layers = [int(x) for x in args.layers.split(',')]
    
    print("=" * 80)
    print("FPN Model Training")
    print("=" * 80)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Feature dimension: {args.feat_dim}")
    print(f"ResNet layers: {layers}")
    print("=" * 80)
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 数据加载
    print("\nLoading data...")
    train_dataset = PairNPZDataset('data/train.npz', is_train=True, use_augmentation=True)
    val_dataset = PairNPZDataset('data/val.npz', is_train=False, use_augmentation=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # 模型创建
    print("\nCreating FPN model...")
    model = FPNCompareNetV2(feat_dim=args.feat_dim, layers=layers, width_mult=1.25)
    model.to(device)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = FocalLoss(alpha=1.0, gamma=2.0, label_smoothing=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # 混合精度训练
    scaler = GradScaler()
    
    # 初始化训练历史记录器
    history = TrainingHistory(
        model_name='FPN-Multi-Scale',
        output_dir=args.output_dir,
        use_tensorboard=True
    )
    
    # 训练循环
    print("\nStarting training...")
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 50)
        
        # 训练
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        
        # 验证
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
        
        # 学习率调度
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 计算梯度范数
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        
        # 记录训练历史
        history.add_epoch(
            epoch=epoch + 1,
            train_metrics={'loss': train_loss, 'accuracy': train_acc, 'f1': train_f1},
            val_metrics={'loss': val_loss, 'accuracy': val_acc, 'f1': val_f1},
            lr=current_lr,
            grad_norm=grad_norm
        )
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        print(f"Gradient Norm: {grad_norm:.4f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # 保存模型
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'model.pt'))
            
            # 保存配置
            config = {
                'feat_dim': args.feat_dim,
                'layers': layers,
                'best_val_acc': best_val_acc,
                'total_params': total_params,
                'trainable_params': trainable_params,
                'epoch': epoch + 1,
                'focal_loss_alpha': 1.0,
                'focal_loss_gamma': 2.0,
                'label_smoothing': 0.1
            }
            
            with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"New best model saved! Val Acc: {val_acc:.4f}")
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{args.patience}")
        
        # 早停
        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # 保存训练历史
    history.save_history()
    
    # 生成学习曲线图
    history.plot_learning_curves(save_path=os.path.join(args.output_dir, 'learning_curves.png'))
    
    # 关闭TensorBoard writer
    history.close()
    
    # 打印训练摘要
    summary = history.get_summary()
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f} ({best_val_acc:.2%})")
    print(f"Best epoch: {summary['best_epoch']}")
    print(f"Convergence epoch: {summary['convergence_epoch']}")
    print(f"Model saved to: {args.output_dir}")
    print(f"Training history saved to: {os.path.join(args.output_dir, 'training_history.json')}")
    print(f"TensorBoard logs saved to: {os.path.join(args.output_dir, 'tensorboard')}")
    print("=" * 80)

if __name__ == '__main__':
    main()
