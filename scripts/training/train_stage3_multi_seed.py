#!/usr/bin/env python3
"""
Stage 3: Multi-Seed训练脚本
多样性探索 - 架构和种子的多样性

关键创新:
1. 多种子训练 (42, 2023, 2024, 2025)
2. 不同网络宽度探索
3. FPN多尺度架构
4. 并行训练支持
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
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

# 添加路径
sys.path.append('src')
sys.path.append('src/models')

from data_loader import PairNPZDataset
from models.simple_compare_cnn import ResNetCompareNet, BasicBlock, Bottleneck
from models.fpn_architecture_v2 import FPNCompareNetV2
from training_history import TrainingHistory

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
        
        # Focal Loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        return focal_loss.mean()

def train_single_seed(seed, config):
    """训练单个种子的模型"""
    print(f"\n{'='*60}")
    print(f"Training seed {seed}")
    print(f"{'='*60}")
    
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 数据加载
    train_dataset = PairNPZDataset('data/train.npz', is_train=True, use_augmentation=True)
    val_dataset = PairNPZDataset('data/val.npz', is_train=False, use_augmentation=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # 模型创建
    if config['architecture'] == 'fpn':
        model = FPNCompareNetV2(
            feat_dim=config['feat_dim'], 
            layers=config['layers'], 
            width_mult=config['width_mult']
        )
    else:
        block = Bottleneck if config['use_bottleneck'] else BasicBlock
        try:
            model = ResNetCompareNet(
                feat_dim=config['feat_dim'], 
                layers=config['layers'], 
                block=block, 
                width_mult=config['width_mult']
            )
        except TypeError:
            model = ResNetCompareNet(feat_dim=config['feat_dim'], layers=config['layers'], block=block)
    
    # 如果使用fusion，ResNetCompareNet已经包含融合功能
    if config.get('use_fusion', False):
        print(f"Using ResNet with fusion architecture (seed {seed})")
    
    model.to(device)
    
    # 优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    criterion = FocalLoss(alpha=1.0, gamma=2.0, label_smoothing=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    scaler = GradScaler()
    
    # 初始化训练历史记录器
    model_name = f"Seed-{seed}"
    if config.get('use_fusion', False):
        model_name = f"Fusion-{seed}"
    
    history = TrainingHistory(
        model_name=model_name,
        output_dir=config['output_dir'],
        use_tensorboard=True
    )
    
    # 训练循环
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        # 训练
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        for xa, xb, labels in tqdm(train_loader, desc=f"Seed {seed} Epoch {epoch+1}"):
            xa = xa.to(device)
            xb = xb.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                logits = model(xa, xb)
                loss = criterion(logits, labels.float())
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()
            train_preds.append(preds.cpu())
            train_labels.append(labels.cpu())
        
        train_preds = torch.cat(train_preds).numpy()
        train_labels = torch.cat(train_labels).numpy()
        train_acc = accuracy_score(train_labels, train_preds)
        
        # 验证
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for xa, xb, labels in val_loader:
                xa = xa.to(device)
                xb = xb.to(device)
                labels = labels.to(device)
                
                with autocast():
                    logits = model(xa, xb)
                    loss = criterion(logits, labels.float())
                
                val_loss += loss.item()
                
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).long()
                val_preds.append(preds.cpu())
                val_labels.append(labels.cpu())
        
        val_preds = torch.cat(val_preds).numpy()
        val_labels = torch.cat(val_labels).numpy()
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds)
        train_f1 = f1_score(train_labels, train_preds)
        
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
            train_metrics={'loss': train_loss/len(train_loader), 'accuracy': train_acc, 'f1': train_f1},
            val_metrics={'loss': val_loss/len(val_loader), 'accuracy': val_acc, 'f1': val_f1},
            lr=current_lr,
            grad_norm=grad_norm
        )
        
        if epoch % 10 == 0 or val_acc > best_val_acc:
            print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, LR={current_lr:.6f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # 保存模型
            model_path = os.path.join(config['output_dir'], f'seed_{seed}')
            os.makedirs(model_path, exist_ok=True)
            
            torch.save(model.state_dict(), os.path.join(model_path, 'model.pt'))
            
            # 保存配置
            metrics = {
                'seed': seed,
                'architecture': config['architecture'],
                'feat_dim': config['feat_dim'],
                'layers': config['layers'],
                'width_mult': config['width_mult'],
                'use_bottleneck': config['use_bottleneck'],
                'best_val_acc': best_val_acc,
                'total_params': sum(p.numel() for p in model.parameters()),
                'epoch': epoch + 1
            }
            
            with open(os.path.join(model_path, 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)
            
            print(f"Seed {seed} new best: {val_acc:.4f}")
        else:
            patience_counter += 1
        
        # 早停
        if patience_counter >= config['patience']:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # 保存训练历史
    history.save_history()
    
    # 生成学习曲线图
    history.plot_learning_curves(save_path=os.path.join(config['output_dir'], f'seed_{seed}_learning_curves.png'))
    
    # 关闭TensorBoard writer
    history.close()
    
    print(f"Seed {seed} completed. Best Val Acc: {best_val_acc:.4f}")
    print(f"Training history saved to: {os.path.join(config['output_dir'], 'training_history.json')}")
    return seed, best_val_acc

def main():
    parser = argparse.ArgumentParser(description='Train Multi-Seed Models')
    parser.add_argument('--seeds', type=str, default='42,2023,2024,2025', help='Random seeds')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--feat_dim', type=int, default=256, help='Feature dimension')
    parser.add_argument('--layers', type=str, default='2,2,2', help='ResNet layers')
    parser.add_argument('--width_mult', type=float, default=1.0, help='Width multiplier')
    parser.add_argument('--use_bottleneck', action='store_true', help='Use Bottleneck blocks')
    parser.add_argument('--use_fusion', action='store_true', help='Use fusion architecture')
    parser.add_argument('--architecture', type=str, default='resnet', choices=['resnet', 'fpn'], help='Architecture type')
    parser.add_argument('--output_dir', type=str, default='models/stage3_multi_seed', help='Output directory')
    parser.add_argument('--parallel', action='store_true', help='Use parallel training')
    parser.add_argument('--n_workers', type=int, default=2, help='Number of parallel workers')
    
    args = parser.parse_args()
    
    # 解析seeds参数
    seeds = [int(x) for x in args.seeds.split(',')]
    layers = [int(x) for x in args.layers.split(',')]
    
    print("=" * 80)
    print("Stage 3: Multi-Seed Training")
    print("=" * 80)
    print(f"Seeds: {seeds}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Architecture: {args.architecture}")
    print(f"Layers: {layers}")
    print(f"Width multiplier: {args.width_mult}")
    print(f"Parallel: {args.parallel}")
    print("=" * 80)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 训练配置
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'patience': args.patience,
        'feat_dim': args.feat_dim,
        'layers': layers,
        'width_mult': args.width_mult,
        'use_bottleneck': args.use_bottleneck,
        'use_fusion': args.use_fusion,
        'architecture': args.architecture,
        'output_dir': args.output_dir
    }
    
    # 训练所有种子
    results = []
    
    if args.parallel and len(seeds) > 1:
        print(f"\nUsing parallel training with {args.n_workers} workers...")
        with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
            futures = [executor.submit(train_single_seed, seed, config) for seed in seeds]
            for future in futures:
                seed, acc = future.result()
                results.append((seed, acc))
    else:
        print("\nUsing sequential training...")
        for seed in seeds:
            seed, acc = train_single_seed(seed, config)
            results.append((seed, acc))
    
    # 保存训练总结
    summary = {
        'seeds': seeds,
        'config': config,
        'results': results,
        'best_seed': max(results, key=lambda x: x[1])[0],
        'best_acc': max(results, key=lambda x: x[1])[1],
        'avg_acc': np.mean([acc for _, acc in results])
    }
    
    with open(os.path.join(args.output_dir, 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 80)
    print("Multi-Seed Training Completed!")
    print("=" * 80)
    print("Results:")
    for seed, acc in sorted(results, key=lambda x: x[1], reverse=True):
        print(f"  Seed {seed}: {acc:.4f} ({acc:.2%})")
    
    print(f"\nBest seed: {summary['best_seed']} ({summary['best_acc']:.4f})")
    print(f"Average accuracy: {summary['avg_acc']:.4f} ({summary['avg_acc']:.2%})")
    print(f"Summary saved to: {os.path.join(args.output_dir, 'training_summary.json')}")
    print("=" * 80)

if __name__ == '__main__':
    main()
