#!/usr/bin/env python3
"""
测试单个模型
用法: python scripts/test_single_model.py --model resnet_optimized_1.12
"""

import argparse
import json
import sys
import os
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.data_loader import PairNPZDataset
from torch.utils.data import DataLoader
from src.models import ResNetCompareNet, FPNCompareNetV2, BasicBlock

def load_model(model_path, metrics_path, device):
    """加载模型"""
    print(f"加载模型: {model_path}")
    
    # 读取配置
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # 创建模型
    layers = metrics.get('layers', [2, 2, 2])
    feat_dim = metrics.get('feat_dim', 256)
    width_mult = metrics.get('width_mult', 1.0)
    
    # 根据架构类型选择模型
    if 'fpn' in model_path.lower():
        model = FPNCompareNetV2(feat_dim=feat_dim, layers=layers, width_mult=width_mult)
    else:
        try:
            model = ResNetCompareNet(feat_dim=feat_dim, layers=layers, block=BasicBlock, width_mult=width_mult)
        except TypeError:
            model = ResNetCompareNet(feat_dim=feat_dim, layers=layers, block=BasicBlock)
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model, metrics

def evaluate_model(model, data_loader, device):
    """评估模型"""
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for xa, xb, labels in data_loader:
            xa = xa.to(device)
            xb = xb.to(device)
            
            logits = model(xa, xb)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()
            
            all_preds.append(preds.cpu())
            all_labels.append(labels)
            all_probs.append(probs.cpu())
    
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()
    
    # 计算指标
    results = {
        'accuracy': float(accuracy_score(all_labels, all_preds)),
        'f1_score': float(f1_score(all_labels, all_preds)),
        'precision': float(precision_score(all_labels, all_preds)),
        'recall': float(recall_score(all_labels, all_preds))
    }
    
    return results

def main():
    parser = argparse.ArgumentParser(description='测试单个模型')
    parser.add_argument('--model', type=str, required=True,
                       help='模型ID (见 configs/model_registry.json)')
    parser.add_argument('--data', type=str, default='data/val.npz',
                       help='测试数据路径')
    parser.add_argument('--output', type=str, default='outputs/predictions',
                       help='输出目录')
    args = parser.parse_args()
    
    # 加载模型注册表
    registry_path = Path('configs/model_registry.json')
    with open(registry_path, 'r') as f:
        registry = json.load(f)
    
    # 查找模型
    model_info = None
    for m in registry['models']:
        if m['id'] == args.model:
            model_info = m
            break
    
    if model_info is None:
        print(f"❌ 模型 '{args.model}' 未找到")
        print("\n可用模型:")
        for m in registry['models']:
            print(f"  - {m['id']:<30} Stage {m['stage']}  {m['accuracy']:.2%}  {m['description'][:50]}")
        return
    
    print("=" * 80)
    print(f"测试模型: {model_info['name']}")
    print(f"阶段: Stage {model_info['stage']}")
    print(f"预期准确率: {model_info['accuracy']:.4f} ({model_info['accuracy']:.2%})")
    print(f"描述: {model_info['description']}")
    print("=" * 80)
    
    # 加载数据
    print("\n加载数据...")
    val_dataset = PairNPZDataset(args.data, is_train=False, use_augmentation=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    print(f"验证集样本数: {len(val_dataset)}")
    
    # 加载模型
    print("\n加载模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    model, metrics = load_model(
        model_info['path'],
        model_info['metrics_path'],
        device
    )
    
    # 评估
    print("\n评估中...")
    results = evaluate_model(model, val_loader, device)
    
    # 显示结果
    print("\n" + "=" * 80)
    print("评估结果:")
    print("=" * 80)
    print(f"准确率 (Accuracy):  {results['accuracy']:.4f} ({results['accuracy']:.2%})")
    print(f"F1分数 (F1-Score):  {results['f1_score']:.4f}")
    print(f"精确率 (Precision): {results['precision']:.4f}")
    print(f"召回率 (Recall):    {results['recall']:.4f}")
    
    # 与预期对比
    diff = results['accuracy'] - model_info['accuracy']
    print(f"\n与预期对比:")
    if abs(diff) < 0.01:
        print(f"✅ 与预期一致 (误差 {diff:+.4f})")
    else:
        status = "⚠️" if abs(diff) < 0.02 else "❌"
        print(f"{status} 与预期有差异 (误差 {diff:+.4f})")
    
    # 保存结果
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result_file = output_dir / f"{args.model}_result.json"
    with open(result_file, 'w') as f:
        json.dump({
            'model_id': args.model,
            'model_name': model_info['name'],
            'stage': model_info['stage'],
            'results': results,
            'expected_accuracy': model_info['accuracy'],
            'difference': float(diff),
            'device': str(device)
        }, f, indent=2)
    
    print(f"\n结果已保存到: {result_file}")
    print("=" * 80)

if __name__ == '__main__':
    main()

