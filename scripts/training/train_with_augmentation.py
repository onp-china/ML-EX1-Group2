#!/usr/bin/env python3
"""
使用数据增强训练模型
基于resnet_optimized_1.12架构，分别训练RandAugment和AutoAugment版本
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
import random

# 添加路径
sys.path.append('src')
sys.path.append('src/models')

from data_loader import PairNPZDataset
from models.simple_compare_cnn import ResNetCompareNet, BasicBlock

class FocalLoss(nn.Module):
    """Focal Loss损失函数"""
    def __init__(self, alpha=1.0, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        
    def forward(self, inputs, targets):
        # 标签平滑
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # 计算BCE损失
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        
        # 计算pt（预测概率）
        pt = torch.exp(-bce_loss)
        
        # 计算Focal Loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        return focal_loss.mean()

def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate(model, loader, device):
    """模型评估函数"""
    model.eval()
    ys, ps = [], []
    
    with torch.no_grad():
        for batch in loader:
            xa, xb, y = batch
            xa = xa.to(device)
            xb = xb.to(device)
            y = y.to(device).float()
            
            # 前向传播
            logit = model(xa, xb)
            prob = torch.sigmoid(logit)
            pred = (prob >= 0.5).long()
            
            # 收集结果
            ys.append(y.long().cpu().numpy())
            ps.append(pred.cpu().numpy())
    
    # 合并结果
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    
    # 计算准确率
    acc = (y_true == y_pred).mean().item()
    
    # 计算宏平均F1分数
    f1s = []
    for cls in [0, 1]:
        tp = np.sum((y_true==cls) & (y_pred==cls))
        fp = np.sum((y_true!=cls) & (y_pred==cls))
        fn = np.sum((y_true==cls) & (y_pred!=cls))
        
        prec = tp / (tp + fp + 1e-12)
        rec  = tp / (tp + fn + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        f1s.append(f1)
    
    f1_macro = float(np.mean(f1s))
    return acc, f1_macro

def train_epoch(model, train_loader, optimizer, criterion, device, scaler):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    for xa, xb, y in tqdm(train_loader, desc="Training"):
        xa = xa.to(device)
        xb = xb.to(device)
        y = y.to(device).float()
        
        optimizer.zero_grad()
        
        # 前向传播
        logit = model(xa, xb)
        loss = criterion(logit, y)
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def train_model(augmentation_type, seed, epochs=60, output_dir=None, resume_path=None, save_dir=None):
    """训练单个模型"""
    if output_dir is None:
        output_dir = f"models/augmented_{augmentation_type}_seed{seed}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置随机种子
    set_seed(seed)
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 混合精度训练（暂时禁用以避免版本兼容问题）
    use_amp = False
    scaler = None
    
    # 模型初始化 - 使用resnet_optimized_1.12的成功配置
    model = ResNetCompareNet(
        feat_dim=256,           # 特征维度
        layers=[3, 3, 3],       # 层配置
        block=BasicBlock        # 使用BasicBlock
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params:,}")
    print(f"网络结构: ResNet-3Layer-BasicBlock [3,3,3]")
    print(f"数据增强: {augmentation_type}")
    
    # 数据加载
    train_dataset = PairNPZDataset(
        'data/train.npz', 
        is_train=True, 
        use_augmentation=True,
        augmentation_type=augmentation_type,
        n_ops=2 if augmentation_type == 'randaugment' else None,
        magnitude=5 if augmentation_type == 'randaugment' else None
    )
    val_dataset = PairNPZDataset('data/val.npz', is_train=False, use_augmentation=False)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)
    
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    
    # 优化器和调度器
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 损失函数
    criterion = FocalLoss(alpha=1, gamma=2, label_smoothing=0.1)
    
    # 训练循环
    best_acc = 0
    best_f1 = 0
    best_epoch = 0
    start_epoch = 1
    
    # Resume功能
    if resume_path and os.path.exists(resume_path):
        print(f"从检查点恢复训练: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_acc = checkpoint.get('best_acc', 0)
        best_f1 = checkpoint.get('best_f1', 0)
        best_epoch = checkpoint.get('best_epoch', 0)
        start_epoch = checkpoint.get('epoch', 1) + 1
        print(f"恢复训练: epoch {start_epoch}, 当前最佳 acc={best_acc:.4f}, f1={best_f1:.4f}")
    
    print(f"开始训练 - 种子: {seed}, 增强: {augmentation_type}, 轮数: {epochs}, 起始轮: {start_epoch}")
    
    for epoch in range(start_epoch, epochs + 1):
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        
        # 验证
        val_acc, val_f1 = evaluate(model, val_loader, device)
        
        # 学习率调度
        scheduler.step()
        
        print(f"Epoch {epoch:3d}: train_loss={train_loss:.4f}, val_acc={val_acc:.4f}, val_f1={val_f1:.4f}, lr={optimizer.param_groups[0]['lr']:.6f}")
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_f1 = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(output_dir, 'model.pt'))
            
            # 保存指标
            with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
                json.dump({
                    "best_val_acc": val_acc,
                    "best_val_f1": val_f1,
                    "best_epoch": epoch,
                    "params": n_params,
                    "final_loss": train_loss,
                    "layers": [3, 3, 3],
                    "use_bottleneck": False,
                    "feat_dim": 256,
                    "use_amp": use_amp,
                    "device": str(device),
                    "seed": seed,
                    "augmentation_type": augmentation_type,
                    "augmentation_config": {
                        "n_ops": 2 if augmentation_type == 'randaugment' else None,
                        "magnitude": 5 if augmentation_type == 'randaugment' else None
                    }
                }, f, indent=2)
        
        # 保存检查点 (每5个epoch保存一次)
        if save_dir and epoch % 5 == 0:
            os.makedirs(save_dir, exist_ok=True)
            checkpoint_path = os.path.join(save_dir, f'checkpoint_{augmentation_type}_seed{seed}_epoch{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
                'best_f1': best_f1,
                'best_epoch': best_epoch,
                'train_loss': train_loss,
                'val_acc': val_acc,
                'val_f1': val_f1
            }, checkpoint_path)
            print(f"检查点已保存: {checkpoint_path}")
    
    print(f"训练完成! 最佳结果 @ epoch {best_epoch}: acc={best_acc:.4f}, f1={best_f1:.4f}")
    return best_acc, best_f1

def main():
    parser = argparse.ArgumentParser(description="使用数据增强训练模型")
    parser.add_argument('--augmentation_type', type=str, required=True, 
                       choices=['randaugment', 'autoaugment'],
                       help='数据增强类型')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 2023, 2024, 2025],
                       help='随机种子列表')
    parser.add_argument('--epochs', type=int, default=60,
                       help='训练轮数')
    parser.add_argument('--resume', type=str, default=None,
                       help='从检查点恢复训练 (检查点路径)')
    parser.add_argument('--save_dir', type=str, default='outputs/checkpoints',
                       help='模型保存目录')
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"使用{args.augmentation_type}数据增强训练模型")
    print(f"随机种子: {args.seeds}")
    print(f"训练轮数: {args.epochs}")
    print("=" * 80)
    
    results = []
    
    for seed in args.seeds:
        print(f"\n训练种子 {seed}...")
        try:
            # 构建输出目录
            output_dir = f"models/augmented_{args.augmentation_type}_seed{seed}"
            
            # 检查是否有resume路径
            resume_path = args.resume
            if resume_path and not os.path.exists(resume_path):
                # 尝试自动查找最新的检查点
                checkpoint_pattern = f"checkpoint_{args.augmentation_type}_seed{seed}_epoch*.pt"
                import glob
                checkpoints = glob.glob(os.path.join(args.save_dir, checkpoint_pattern))
                if checkpoints:
                    # 按epoch编号排序，选择最新的
                    checkpoints.sort(key=lambda x: int(x.split('epoch')[1].split('.')[0]))
                    resume_path = checkpoints[-1]
                    print(f"自动找到检查点: {resume_path}")
                else:
                    resume_path = None
            
            acc, f1 = train_model(args.augmentation_type, seed, args.epochs, output_dir, resume_path, args.save_dir)
            results.append({
                'seed': seed,
                'accuracy': acc,
                'f1_score': f1,
                'status': 'success'
            })
            print(f"种子 {seed} 完成: acc={acc:.4f}, f1={f1:.4f}")
        except Exception as e:
            print(f"种子 {seed} 失败: {e}")
            results.append({
                'seed': seed,
                'accuracy': 0,
                'f1_score': 0,
                'status': 'failed',
                'error': str(e)
            })
    
    # 保存训练结果
    output_file = f"outputs/results/augmentation_{args.augmentation_type}_results.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            'augmentation_type': args.augmentation_type,
            'epochs': args.epochs,
            'results': results,
            'summary': {
                'successful_seeds': [r['seed'] for r in results if r['status'] == 'success'],
                'avg_accuracy': np.mean([r['accuracy'] for r in results if r['status'] == 'success']),
                'avg_f1_score': np.mean([r['f1_score'] for r in results if r['status'] == 'success']),
                'best_accuracy': max([r['accuracy'] for r in results if r['status'] == 'success']),
                'best_f1_score': max([r['f1_score'] for r in results if r['status'] == 'success'])
            }
        }, f, indent=2)
    
    print(f"\n训练结果已保存到: {output_file}")
    
    # 显示总结
    successful_results = [r for r in results if r['status'] == 'success']
    if successful_results:
        avg_acc = np.mean([r['accuracy'] for r in successful_results])
        best_acc = max([r['accuracy'] for r in successful_results])
        print(f"\n总结:")
        print(f"成功训练: {len(successful_results)}/{len(args.seeds)} 个模型")
        print(f"平均准确率: {avg_acc:.4f}")
        print(f"最佳准确率: {best_acc:.4f}")
    else:
        print("所有训练都失败了!")
        print("请检查错误信息并重试")

if __name__ == "__main__":
    main()
