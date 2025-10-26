#!/usr/bin/env python3
"""
贝叶斯推理脚本
使用蒙特卡洛Dropout对多个模型进行贝叶斯集成
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import glob

# 添加路径
sys.path.append('src')
sys.path.append('src/models')

from data_loader import PairNPZDataset
from models.simple_compare_cnn import ResNetCompareNet, BasicBlock
from mc_dropout import MCDropoutWrapper, BayesianEnsemble, create_mc_dropout_models

def load_model_configs():
    """加载模型配置"""
    config_file = 'configs/model_registry.json'
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        # 默认配置
        return {
            "resnet_optimized_1.12": {
                "class": "ResNetCompareNet",
                "params": {
                    "feat_dim": 256,
                    "layers": [3, 3, 3],
                    "block": "BasicBlock"
                }
            }
        }

def find_model_files(model_dir: str = "models") -> list:
    """查找所有可用的模型文件"""
    model_files = []
    
    # 查找所有.pt文件
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            if file.endswith('.pt') and file != 'model.pt':  # 排除通用的model.pt
                model_files.append(os.path.join(root, file))
    
    return sorted(model_files)

def evaluate_bayesian_ensemble(ensemble, data_loader, device):
    """评估贝叶斯集成模型"""
    ensemble.models[0].set_training_state(False)  # 设置为评估模式
    
    all_predictions = []
    all_uncertainties = []
    all_labels = []
    
    print("进行贝叶斯推理...")
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="贝叶斯推理"):
            xa, xb, y = batch
            xa = xa.to(device)
            xb = xb.to(device)
            y = y.to(device).float()
            
            # 贝叶斯预测
            mean_pred, aleatoric_unc, epistemic_unc = ensemble.predict_with_uncertainty(xa, xb)
            
            # 转换为numpy
            pred_probs = mean_pred.cpu().numpy()
            pred_labels = (pred_probs >= 0.5).astype(int)
            total_uncertainty = (aleatoric_unc + epistemic_unc).cpu().numpy()
            
            all_predictions.append(pred_labels.flatten())
            all_uncertainties.append(total_uncertainty.flatten())
            all_labels.append(y.cpu().numpy())
    
    # 合并结果
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_predictions)
    uncertainties = np.concatenate(all_uncertainties)
    
    print(f"y_true shape: {y_true.shape}")
    print(f"y_pred shape: {y_pred.shape}")
    print(f"uncertainties shape: {uncertainties.shape}")
    
    # 计算指标
    accuracy = accuracy_score(y_true, y_pred)
    
    # 计算F1分数
    f1_scores = []
    for cls in [0, 1]:
        tp = np.sum((y_true==cls) & (y_pred==cls))
        fp = np.sum((y_true!=cls) & (y_pred==cls))
        fn = np.sum((y_true==cls) & (y_pred!=cls))
        
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        f1_scores.append(f1)
    
    f1_macro = np.mean(f1_scores)
    
    # 不确定性统计
    uncertainty_stats = {
        'mean': float(np.mean(uncertainties)),
        'std': float(np.std(uncertainties)),
        'min': float(np.min(uncertainties)),
        'max': float(np.max(uncertainties)),
        'median': float(np.median(uncertainties))
    }
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'uncertainty_stats': uncertainty_stats,
        'predictions': y_pred,
        'uncertainties': uncertainties
    }

def main():
    parser = argparse.ArgumentParser(description="贝叶斯推理脚本")
    parser.add_argument('--model_dir', type=str, default='models',
                       help='模型目录')
    parser.add_argument('--data_path', type=str, default='data/val.npz',
                       help='数据路径')
    parser.add_argument('--mc_samples', type=int, default=20,
                       help='蒙特卡洛采样次数')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                       help='Dropout率')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批次大小')
    parser.add_argument('--output_dir', type=str, default='outputs/bayesian',
                       help='输出目录')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("贝叶斯推理脚本")
    print("=" * 80)
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 查找模型文件
    model_files = find_model_files(args.model_dir)
    print(f"找到 {len(model_files)} 个模型文件:")
    for i, model_file in enumerate(model_files):
        print(f"  {i+1}. {model_file}")
    
    if not model_files:
        print("未找到模型文件!")
        return
    
    # 加载模型配置
    configs = load_model_configs()
    
    # 创建MC Dropout模型
    print(f"\n创建MC Dropout模型 (MC采样次数: {args.mc_samples})...")
    
    # 使用第一个模型的配置（假设所有模型使用相同架构）
    model_config = list(configs.values())[0]
    model_class = ResNetCompareNet  # 简化，假设都是ResNetCompareNet
    model_kwargs = {
        'feat_dim': 256,
        'layers': [3, 3, 3],
        'block': BasicBlock
    }
    
    try:
        mc_models = create_mc_dropout_models(
            model_files, 
            model_class, 
            model_kwargs,
            mc_samples=args.mc_samples,
            dropout_rate=args.dropout_rate
        )
        print(f"成功创建 {len(mc_models)} 个MC Dropout模型")
    except Exception as e:
        print(f"创建MC Dropout模型失败: {e}")
        return
    
    # 创建贝叶斯集成
    ensemble = BayesianEnsemble(mc_models)
    print(f"创建贝叶斯集成，包含 {len(mc_models)} 个模型")
    
    # 加载数据
    print(f"\n加载数据: {args.data_path}")
    dataset = PairNPZDataset(args.data_path, is_train=False, use_augmentation=False)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f"数据加载完成: {len(dataset)} 个样本")
    
    # 评估贝叶斯集成
    results = evaluate_bayesian_ensemble(ensemble, data_loader, device)
    
    # 打印结果
    print("\n" + "=" * 50)
    print("贝叶斯集成结果")
    print("=" * 50)
    print(f"准确率: {results['accuracy']:.4f}")
    print(f"F1分数: {results['f1_macro']:.4f}")
    print(f"\n不确定性统计:")
    for key, value in results['uncertainty_stats'].items():
        print(f"  {key}: {value:.4f}")
    
    # 保存结果
    output_file = os.path.join(args.output_dir, 'bayesian_results.json')
    save_results = {
        'accuracy': results['accuracy'],
        'f1_macro': results['f1_macro'],
        'uncertainty_stats': results['uncertainty_stats'],
        'model_files': model_files,
        'mc_samples': args.mc_samples,
        'dropout_rate': args.dropout_rate,
        'num_models': len(mc_models)
    }
    
    with open(output_file, 'w') as f:
        json.dump(save_results, f, indent=2)
    
    print(f"\n结果已保存到: {output_file}")
    
    # 分析不确定性分布
    uncertainties = results['uncertainties']
    high_uncertainty_indices = np.where(uncertainties > np.percentile(uncertainties, 90))[0]
    low_uncertainty_indices = np.where(uncertainties < np.percentile(uncertainties, 10))[0]
    
    print(f"\n不确定性分析:")
    print(f"高不确定性样本 (前10%): {len(high_uncertainty_indices)} 个")
    print(f"低不确定性样本 (后10%): {len(low_uncertainty_indices)} 个")
    
    # 计算高/低不确定性样本的准确率
    if len(high_uncertainty_indices) > 0:
        high_unc_acc = accuracy_score(
            results['predictions'][high_uncertainty_indices],
            results['predictions'][high_uncertainty_indices]
        )
        print(f"高不确定性样本准确率: {high_unc_acc:.4f}")
    
    if len(low_uncertainty_indices) > 0:
        low_unc_acc = accuracy_score(
            results['predictions'][low_uncertainty_indices],
            results['predictions'][low_uncertainty_indices]
        )
        print(f"低不确定性样本准确率: {low_unc_acc:.4f}")

if __name__ == "__main__":
    main()