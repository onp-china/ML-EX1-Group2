#!/usr/bin/env python3
"""
简化的多种子训练脚本 - 直接在主进程中运行
基于resnet_optimized_1.12的成功配置
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# 添加路径
sys.path.append('resnet_optimized_1.12(1)/scripts')
sys.path.append('resnet_optimized_1.12(1)')

from utils.seed import set_seed
from utils.data import PairNPZDataset
from models.simple_compare_cnn import ResNetCompareNet, count_params

class FocalLoss(torch.nn.Module):
    """Focal Loss损失函数（带标签平滑）"""
    def __init__(self, alpha=1, gamma=2, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        
    def forward(self, inputs, targets):
        # 标签平滑
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # 计算BCE损失
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        
        # 计算pt（预测概率）
        pt = torch.exp(-bce_loss)
        
        # 计算Focal Loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        return focal_loss.mean()

def evaluate(model, loader, device):
    """模型评估函数"""
    model.eval()
    ys, ps = [], []
    
    with torch.no_grad():
        for batch in loader:
            xa, xb, y = batch
            xa = xa.to(device)
            xb = xb.to(device)
            
            if isinstance(y, (list, tuple)) and len(y) > 0:
                if isinstance(y[0], str):
                    # 测试集模式
                    logit = model(xa, xb)
                    prob = torch.sigmoid(logit)
                    pred = (prob >= 0.5).long()
                    return 0.0, 0.0
                else:
                    y = torch.tensor(y, dtype=torch.float32)
            elif isinstance(y, str):
                return 0.0, 0.0
            else:
                if not isinstance(y, torch.Tensor):
                    y = torch.tensor(y, dtype=torch.float32)
            
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

def train_single_seed(seed, data_dir, output_dir, epochs=80):
    """训练单个种子模型"""
    print(f"\n{'='*60}")
    print(f"开始训练种子 {seed}")
    print(f"输出目录: {output_dir}")
    print(f"训练轮数: {epochs}")
    print(f"{'='*60}")
    
    # 环境设置
    os.makedirs(output_dir, exist_ok=True)
    set_seed(seed)
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 混合精度训练
    use_amp = device.type == 'cuda'
    scaler = GradScaler() if use_amp else None
    
    # 模型初始化 - 使用resnet_optimized_1.12的成功配置
    from models.simple_compare_cnn import BasicBlock
    
    model = ResNetCompareNet(
        feat_dim=256,           # 特征维度
        layers=[3, 3, 3],       # 层配置
        block=BasicBlock        # 使用BasicBlock
    ).to(device)
    
    n_params = int(count_params(model))
    print(f"模型参数量: {n_params:,}")
    print(f"网络结构: ResNet-3Layer-BasicBlock [3,3,3]")
    
    # 数据加载
    train_path = os.path.join(data_dir, "train.npz")
    val_path = os.path.join(data_dir, "val.npz")
    
    train_ds = PairNPZDataset(train_path, is_train=True)
    val_ds = PairNPZDataset(val_path, is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2)
    
    # 优化器和调度器
    optim = AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optim, T_max=epochs)
    
    # 损失函数 - 使用Focal Loss
    criterion = FocalLoss(alpha=1, gamma=2, label_smoothing=0.1)
    
    # 训练状态
    best = {"acc": 0.0, "f1": 0.0, "epoch": -1}
    
    print(f"开始训练 - 种子: {seed}, 轮数: {epochs}")
    print(f"使用 Focal Loss")
    print(f"混合精度: {'启用' if use_amp else '禁用'}")
    
    # 训练循环
    for epoch in range(1, epochs + 1):
        # 训练阶段
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        total_loss = 0
        
        for xa, xb, y in pbar:
            xa = xa.to(device)
            xb = xb.to(device)
            y = y.to(device).float()
            
            # 前向传播
            if use_amp and scaler is not None:
                with autocast():
                    logit = model(xa, xb)
                    loss = criterion(logit, y)
            else:
                logit = model(xa, xb)
                loss = criterion(logit, y)
            
            # 反向传播
            optim.zero_grad()
            
            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optim.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=float(loss.item()))
        
        # 学习率调度
        scheduler.step()
        
        # 验证阶段
        acc, f1 = evaluate(model, val_loader, device)
        avg_loss = total_loss / len(train_loader)
        
        print(f"[Val] epoch={epoch} acc={acc:.4f} f1_macro={f1:.4f} "
              f"loss={avg_loss:.4f} lr={optim.param_groups[0]['lr']:.6f}")
        
        # 保存最佳模型
        if acc > best["acc"]:
            best = {"acc": acc, "f1": f1, "epoch": epoch}
            torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
            
            # 保存指标
            with open(os.path.join(output_dir, "metrics.json"), "w") as f:
                json.dump({
                    "best_val_acc": acc,
                    "best_val_f1": f1,
                    "best_epoch": epoch,
                    "params": n_params,
                    "final_loss": avg_loss,
                    "layers": [3, 3, 3],
                    "use_bottleneck": False,
                    "feat_dim": 256,
                    "use_amp": use_amp,
                    "device": str(device),
                    "seed": seed
                }, f, indent=2)
        
        # 保存checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best["acc"],
            'best_f1': best["f1"],
            'best_epoch': best["epoch"],
            'seed': seed
        }
        torch.save(checkpoint, os.path.join(output_dir, "checkpoint.pt"))
    
    # 训练完成
    print(f"种子 {seed} 训练完成!")
    print(f"最佳结果 @ epoch {best['epoch']}: acc={best['acc']:.4f}, f1={best['f1']:.4f}")
    
    return best["acc"], best["f1"]

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="简化的多种子训练脚本")
    parser.add_argument("--data_dir", type=str, default="resnet_optimized_1.12(1)/data", 
                       help="数据目录")
    parser.add_argument("--output_dir", type=str, default="multi_seed_models", 
                       help="输出目录")
    parser.add_argument("--seeds", type=int, nargs='+', default=[42, 2023, 2024, 2025], 
                       help="随机种子列表")
    parser.add_argument("--epochs", type=int, default=80, help="训练轮数")
    args = parser.parse_args()
    
    print("基于resnet_optimized_1.12的多种子训练")
    print("="*60)
    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"种子列表: {args.seeds}")
    print(f"训练轮数: {args.epochs}")
    print("="*60)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 训练结果
    results = {}
    
    # 串行训练
    for i, seed in enumerate(args.seeds, 1):
        print(f"\n进度: {i}/{len(args.seeds)}")
        
        # 检查是否已经训练过
        output_dir = os.path.join(args.output_dir, f"seed_{seed}")
        metrics_path = os.path.join(output_dir, "metrics.json")
        
        if os.path.exists(metrics_path):
            print(f"种子 {seed} 已存在，跳过训练")
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            acc = metrics.get('best_val_acc', 0)
            f1 = metrics.get('best_val_f1', 0)
            results[seed] = (acc, f1)
            continue
        
        # 训练单个种子
        acc, f1 = train_single_seed(seed, args.data_dir, output_dir, args.epochs)
        results[seed] = (acc, f1)
    
    # 输出总结
    print(f"\n{'='*60}")
    print("训练完成总结")
    print(f"{'='*60}")
    
    if results:
        print("各种子结果:")
        for seed, (acc, f1) in results.items():
            print(f"  种子 {seed}: acc={acc:.4f}, f1={f1:.4f}")
        
        # 计算统计信息
        accuracies = [acc for acc, f1 in results.values() if acc > 0]
        if accuracies:
            best_acc = max(accuracies)
            avg_acc = sum(accuracies) / len(accuracies)
            best_seed = max(results.keys(), key=lambda s: results[s][0])
            
            print(f"\n统计信息:")
            print(f"  最佳准确率: {best_acc:.4f} (种子 {best_seed})")
            print(f"  平均准确率: {avg_acc:.4f}")
            print(f"  成功训练: {len(accuracies)}/{len(args.seeds)} 个种子")
            
            # 保存总结
            summary = {
                "seeds": args.seeds,
                "epochs": args.epochs,
                "results": {str(seed): {"acc": acc, "f1": f1} for seed, (acc, f1) in results.items()},
                "best_accuracy": best_acc,
                "best_seed": best_seed,
                "average_accuracy": avg_acc,
                "successful_trains": len(accuracies)
            }
            
            summary_path = os.path.join(args.output_dir, "training_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"\n总结已保存到: {summary_path}")
            
            # 检查是否达到90%
            if best_acc >= 0.90:
                print(f"🎉 恭喜！已达到90%目标！")
            else:
                gap = 0.90 - best_acc
                print(f"距离90%目标还差: {gap:.4f} ({gap*100:.2f}%)")
    else:
        print("没有成功训练的模型")
    
    print(f"\n下一步建议:")
    print(f"1. 检查训练日志，确保所有模型都训练完成")
    print(f"2. 运行集成测试: python final_ensemble.py")
    print(f"3. 如果未达到90%，考虑训练更多种子或调整超参数")

if __name__ == "__main__":
    main()
