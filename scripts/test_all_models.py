#!/usr/bin/env python3
"""
测试所有模型并生成对比报告
用法: python scripts/test_all_models.py
"""

import json
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from src.data_loader import PairNPZDataset
from torch.utils.data import DataLoader
from src.models import ResNetCompareNet, FPNCompareNetV2, BasicBlock

def load_model(model_path, metrics_path, device):
    """加载模型"""
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    layers = metrics.get('layers', [2, 2, 2])
    feat_dim = metrics.get('feat_dim', 256)
    width_mult = metrics.get('width_mult', 1.0)
    
    if 'fpn' in model_path.lower():
        model = FPNCompareNetV2(feat_dim=feat_dim, layers=layers, width_mult=width_mult)
    else:
        try:
            model = ResNetCompareNet(feat_dim=feat_dim, layers=layers, block=BasicBlock, width_mult=width_mult)
        except TypeError:
            model = ResNetCompareNet(feat_dim=feat_dim, layers=layers, block=BasicBlock)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model

def evaluate_model(model, data_loader, device):
    """评估模型"""
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for xa, xb, labels in data_loader:
            xa = xa.to(device)
            xb = xb.to(device)
            
            logits = model(xa, xb)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()
            
            all_preds.append(preds.cpu())
            all_labels.append(labels)
    
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    return {
        'accuracy': float(accuracy_score(all_labels, all_preds)),
        'f1_score': float(f1_score(all_labels, all_preds))
    }

def main():
    print("=" * 80)
    print("测试所有模型")
    print("=" * 80)
    
    # 加载模型注册表
    with open('configs/model_registry.json', 'r') as f:
        registry = json.load(f)
    
    # 加载数据
    print("\n加载数据...")
    val_dataset = PairNPZDataset('data/val.npz', is_train=False, use_augmentation=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    print(f"验证集样本数: {len(val_dataset)}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 测试所有模型
    results_summary = []
    
    print("\n开始测试...")
    print("=" * 80)
    
    for model_info in tqdm(registry['models'], desc="测试进度"):
        try:
            # 加载模型
            model = load_model(
                model_info['path'],
                model_info['metrics_path'],
                device
            )
            
            # 评估
            results = evaluate_model(model, val_loader, device)
            
            # 记录
            results_summary.append({
                'id': model_info['id'],
                'name': model_info['name'],
                'stage': model_info['stage'],
                'expected': model_info['accuracy'],
                'actual': results['accuracy'],
                'difference': results['accuracy'] - model_info['accuracy'],
                'f1_score': results['f1_score'],
                'params': model_info.get('params', 'N/A'),
                'description': model_info['description']
            })
            
        except Exception as e:
            print(f"\n⚠️  {model_info['name']} 测试失败: {e}")
            continue
    
    # 生成报告
    print("\n" + "=" * 80)
    print("所有模型测试结果")
    print("=" * 80)
    print(f"{'模型':<32} {'阶段':^6} {'预期':^8} {'实际':^8} {'差异':^8} {'F1':^8}")
    print("-" * 80)
    
    for r in sorted(results_summary, key=lambda x: (x['stage'], -x['actual'])):
        status = "✅" if abs(r['difference']) < 0.01 else ("⚠️" if abs(r['difference']) < 0.02 else "❌")
        print(f"{status} {r['name']:<30} Stage{r['stage']} "
              f"{r['expected']:.4f}  {r['actual']:.4f}  "
              f"{r['difference']:+.4f}  {r['f1_score']:.4f}")
    
    print("=" * 80)
    
    # 按阶段统计
    print("\n按阶段统计:")
    print("-" * 80)
    for stage in sorted(set(r['stage'] for r in results_summary)):
        stage_models = [r for r in results_summary if r['stage'] == stage]
        avg_acc = sum(r['actual'] for r in stage_models) / len(stage_models)
        best_acc = max(r['actual'] for r in stage_models)
        print(f"Stage {stage}: {len(stage_models)}个模型, 平均 {avg_acc:.2%}, 最佳 {best_acc:.2%}")
    
    # 保存报告
    output_file = Path('outputs/metrics/all_models_comparison.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            'summary': results_summary,
            'device': str(device),
            'total_models': len(results_summary),
            'avg_accuracy': sum(r['actual'] for r in results_summary) / len(results_summary),
            'best_model': max(results_summary, key=lambda x: x['actual'])
        }, f, indent=2)
    
    print(f"\n详细报告已保存到: {output_file}")
    print("=" * 80)

if __name__ == '__main__':
    main()

