#!/usr/bin/env python3
"""
88.83%准确率模型多种子训练脚本
============================

功能：训练多个不同种子的88.83%准确率模型
特点：基于最佳配置参数，支持批量训练和进度监控

使用方法：
python train_88_83_multi_seed.py --seeds 42,123,456,789,999 --epochs 40
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from datetime import datetime

# 添加项目路径
sys.path.append('Acc88/scripts')
sys.path.append('Acc88')

from utils.data import PairNPZDataset
from models.simple_compare_cnn import ResNetCompareNet, BasicBlock
from utils.metrics import accuracy, f1_macro


def set_seed(seed):
    """设置随机种子确保结果可复现"""
    import random
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
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            img1, img2, labels = batch
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            
            outputs = model(img1, img2)
            probs = torch.sigmoid(outputs.squeeze())
            
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 寻找最佳阈值
    best_threshold = 0.5
    best_f1 = 0
    
    for threshold in np.arange(0.3, 0.7, 0.01):
        pred_labels = (all_preds > threshold).astype(int)
        f1 = f1_macro(all_labels, pred_labels)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    # 使用最佳阈值计算准确率
    pred_labels = (all_preds > best_threshold).astype(int)
    acc = accuracy(all_labels, pred_labels)
    
    return acc, best_f1, best_threshold


def train_single_seed(seed, epochs=40, out_dir=None):
    """训练单个种子的模型"""
    print(f"\n{'='*60}")
    print(f"开始训练种子 {seed} 的模型")
    print(f"{'='*60}")
    
    # 设置种子
    set_seed(seed)
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 设置输出目录
    if out_dir is None:
        out_dir = f"./outputs/seed_{seed}"
    os.makedirs(out_dir, exist_ok=True)
    
    # 加载数据
    print("加载数据...")
    train_ds = PairNPZDataset('Acc88/data/train.npz', is_train=True, use_augmentation=True)
    val_ds = PairNPZDataset('Acc88/data/val.npz', is_train=False, use_augmentation=False)
    
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"训练集大小: {len(train_ds)}")
    print(f"验证集大小: {len(val_ds)}")
    
    # 创建模型
    print("创建模型...")
    model = ResNetCompareNet(
        feat_dim=256,
        width_mult=1.0,  # 使用标准宽度
        layers=[2, 2, 2],
        block=BasicBlock
    ).to(device)
    
    # 计算参数量
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params:,}")
    
    # 损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # 训练状态跟踪
    best_val_acc = 0
    best_val_f1 = 0
    best_epoch = 0
    patience = 5
    patience_counter = 0
    
    print(f"\n开始训练...")
    start_time = time.time()
    
    # 训练循环
    for epoch in range(1, epochs + 1):
        # 训练阶段
        model.train()
        train_loss = 0
        train_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch in pbar:
            img1, img2, labels = batch
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(img1, img2)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{train_loss/train_batches:.4f}'
            })
        
        # 验证阶段
        val_acc, val_f1, best_threshold = evaluate(model, val_loader, device)
        
        # 更新学习率
        scheduler.step(val_f1)
        
        # 打印训练信息
        avg_train_loss = train_loss / train_batches
        print(f"Epoch {epoch:2d}: Train Loss={avg_train_loss:.4f}, "
              f"Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}, "
              f"Threshold={best_threshold:.3f}, LR={optimizer.param_groups[0]['lr']:.2e}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            
            # 保存模型
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'val_f1': val_f1,
                'threshold': best_threshold,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }, os.path.join(out_dir, 'model.pt'))
            
            # 保存指标
            metrics = {
                'best_val_acc': val_acc,
                'best_val_f1': val_f1,
                'best_epoch': epoch,
                'params': n_params,
                'final_loss': avg_train_loss,
                'layers': [2, 2, 2],
                'use_bottleneck': False,
                'feat_dim': 256,
                'width_mult': 1.0,
                'seed': seed,
                'threshold': best_threshold,
                'architecture': 'ResNetCompareNet',
                'optimizer': 'AdamW',
                'lr': 1e-3,
                'weight_decay': 5e-4,
                'scheduler': 'ReduceLROnPlateau',
                'training_time': time.time() - start_time
            }
            
            with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)
            
            print(f"[OK] 新的最佳模型已保存! (Acc: {val_acc:.4f}, F1: {val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[STOP] 早停触发! (patience={patience})")
                break
    
    # 训练完成
    training_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"种子 {seed} 训练完成!")
    print(f"最佳验证准确率: {best_val_acc:.4f}")
    print(f"最佳验证F1分数: {best_val_f1:.4f}")
    print(f"最佳轮次: {best_epoch}")
    print(f"训练时间: {training_time/60:.1f} 分钟")
    print(f"模型已保存到: {out_dir}")
    print(f"{'='*60}")
    
    return {
        'seed': seed,
        'best_val_acc': best_val_acc,
        'best_val_f1': best_val_f1,
        'best_epoch': best_epoch,
        'training_time': training_time,
        'out_dir': out_dir
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='88.83%准确率模型多种子训练脚本')
    parser.add_argument('--seeds', type=str, default='42,123,456,789,999', 
                       help='种子列表，用逗号分隔 (默认: 42,123,456,789,999)')
    parser.add_argument('--epochs', type=int, default=40, help='训练轮数 (默认: 40)')
    parser.add_argument('--output_base', type=str, default='./outputs', 
                       help='输出基础目录 (默认: ./outputs)')
    parser.add_argument('--resume', action='store_true', help='从checkpoint继续训练')
    
    args = parser.parse_args()
    
    # 解析种子列表
    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    
    print("="*80)
    print("88.83%准确率模型多种子训练脚本")
    print("="*80)
    print(f"训练种子: {seeds}")
    print(f"训练轮数: {args.epochs}")
    print(f"输出目录: {args.output_base}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # 创建输出基础目录
    os.makedirs(args.output_base, exist_ok=True)
    
    # 训练结果记录
    results = []
    start_time = time.time()
    
    # 逐个训练每个种子
    for i, seed in enumerate(seeds, 1):
        print(f"\n[{i}/{len(seeds)}] 开始训练种子 {seed}...")
        
        try:
            # 设置输出目录
            out_dir = os.path.join(args.output_base, f'seed_{seed}')
            
            # 训练单个种子
            result = train_single_seed(
                seed=seed,
                epochs=args.epochs,
                out_dir=out_dir
            )
            
            results.append(result)
            
            # 显示当前进度
            print(f"\n当前进度: {i}/{len(seeds)} 完成")
            print(f"已完成种子: {[r['seed'] for r in results]}")
            
        except Exception as e:
            print(f"[ERROR] 种子 {seed} 训练失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 训练完成，显示总结
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print("多种子训练完成!")
    print(f"{'='*80}")
    print(f"总训练时间: {total_time/3600:.1f} 小时")
    print(f"成功训练: {len(results)}/{len(seeds)} 个种子")
    
    if results:
        print(f"\n各种子结果:")
        print("-" * 60)
        print(f"{'种子':<8} {'验证准确率':<12} {'验证F1':<12} {'最佳轮次':<8} {'训练时间(分)':<12}")
        print("-" * 60)
        
        for result in results:
            print(f"{result['seed']:<8} {result['best_val_acc']:<12.4f} "
                  f"{result['best_val_f1']:<12.4f} {result['best_epoch']:<8} "
                  f"{result['training_time']/60:<12.1f}")
        
        # 统计信息
        accuracies = [r['best_val_acc'] for r in results]
        f1_scores = [r['best_val_f1'] for r in results]
        
        print("-" * 60)
        print(f"平均准确率: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"平均F1分数: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
        print(f"最高准确率: {max(accuracies):.4f} (种子 {results[np.argmax(accuracies)]['seed']})")
        print(f"最高F1分数: {max(f1_scores):.4f} (种子 {results[np.argmax(f1_scores)]['seed']})")
        
        # 达到目标的模型数量
        target_models = [r for r in results if r['best_val_acc'] >= 0.888]
        print(f"达到88.8%目标的模型: {len(target_models)}/{len(results)}")
        
        if target_models:
            print("达到目标的种子:", [r['seed'] for r in target_models])
    
    print(f"\n所有模型保存在: {args.output_base}")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    return 0


if __name__ == '__main__':
    exit(main())
