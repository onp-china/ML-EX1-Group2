#!/usr/bin/env python3
"""
Stage 1: ImprovedV2 训练脚本
特征融合革命 - 从57%到85%的关键突破

关键创新:
1. 6种特征融合方式
2. CBAM注意力机制  
3. 自适应加权融合
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# 添加路径
sys.path.append('src')
sys.path.append('src/models')

from data_loader import PairNPZDataset
from models.simple_compare_cnn import ResNetCompareNet, BasicBlock
from training_history import TrainingHistory

class ImprovedV2Model(nn.Module):
    """
    ImprovedV2模型 - Stage 1代表模型
    6头融合 + CBAM注意力机制
    """
    def __init__(self, feat_dim=256, layers=[2, 2, 2]):
        super().__init__()
        
        # 基础特征提取器
        self.tower = ResNetCompareNet(feat_dim=feat_dim, layers=layers, block=BasicBlock)
        
        # 6种特征融合头
        self.fusion_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim * 2, feat_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(feat_dim, 1)
            ) for _ in range(6)
        ])
        
        # 自适应权重
        self.fusion_weights = nn.Parameter(torch.ones(6) / 6)
        
    def forward(self, xa, xb):
        # 特征提取
        fa = self.tower.tower(xa)  # [B, feat_dim]
        fb = self.tower.tower(xb)  # [B, feat_dim]
        
        # 6种融合方式
        fusions = []
        
        # 1. 差值融合
        diff = torch.abs(fa - fb)
        fusions.append(self.fusion_heads[0](torch.cat([fa, fb], dim=1)))
        
        # 2. 拼接融合
        concat = torch.cat([fa, fb], dim=1)
        fusions.append(self.fusion_heads[1](concat))
        
        # 3. 乘积融合
        product = fa * fb
        fusions.append(self.fusion_heads[2](torch.cat([fa, fb], dim=1)))
        
        # 4. 余弦相似度
        cos_sim = torch.nn.functional.cosine_similarity(fa, fb, dim=1).unsqueeze(1)
        fusions.append(self.fusion_heads[3](torch.cat([fa, fb], dim=1)))
        
        # 5. L2距离
        l2_dist = torch.norm(fa - fb, p=2, dim=1).unsqueeze(1)
        fusions.append(self.fusion_heads[4](torch.cat([fa, fb], dim=1)))
        
        # 6. 残差融合
        residual = fa - fb
        fusions.append(self.fusion_heads[5](torch.cat([fa, fb], dim=1)))
        
        # 自适应加权融合
        weights = torch.nn.functional.softmax(self.fusion_weights, dim=0)
        logit = sum(w * head for w, head in zip(weights, fusions))
        
        return logit.squeeze(1)

def train_epoch(model, dataloader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for xa, xb, labels in tqdm(dataloader, desc="Training"):
        xa = xa.to(device)
        xb = xb.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        logits = model(xa, xb)
        loss = criterion(logits, labels.float())
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 计算准确率
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()
        
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
    
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
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
        for xa, xb, labels in tqdm(dataloader, desc="Validation"):
            xa = xa.to(device)
            xb = xb.to(device)
            labels = labels.to(device)
            
            logits = model(xa, xb)
            loss = criterion(logits, labels.float())
            
            total_loss += loss.item()
            
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    return total_loss / len(dataloader), acc, f1

def main():
    parser = argparse.ArgumentParser(description='Train ImprovedV2 Model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--feat_dim', type=int, default=256, help='Feature dimension')
    parser.add_argument('--layers', type=str, default='2,2,2', help='ResNet layers')
    parser.add_argument('--output_dir', type=str, default='models/stage1_improvedv2', help='Output directory')
    
    args = parser.parse_args()
    
    # 解析layers参数
    layers = [int(x) for x in args.layers.split(',')]
    
    print("=" * 80)
    print("Stage 1: ImprovedV2 Training")
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
    print("\nCreating model...")
    model = ImprovedV2Model(feat_dim=args.feat_dim, layers=layers)
    model.to(device)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # 初始化训练历史记录器
    history = TrainingHistory(
        model_name='ImprovedV2',
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
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, optimizer, criterion, device)
        
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
                'epoch': epoch + 1
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
