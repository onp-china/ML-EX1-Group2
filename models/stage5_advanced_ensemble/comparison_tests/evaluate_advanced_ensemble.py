#!/usr/bin/env python3
"""
高级集成方法综合评估脚本
测试所有实现的优化方法
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
import glob
import warnings
warnings.filterwarnings('ignore')

# 添加路径
sys.path.append('src')
sys.path.append('src/models')

from data_loader import DataManager
from models.simple_compare_cnn import ResNetCompareNet, BasicBlock
from mc_dropout import MCDropoutWrapper, BayesianEnsemble
from two_level_stacking import TwoLevelStacking, evaluate_two_level_stacking
from dynamic_ensemble import DynamicEnsemble, evaluate_dynamic_ensemble

def load_trained_models(model_dir: str = 'models') -> list:
    """加载所有训练好的模型"""
    models = []
    model_files = glob.glob(os.path.join(model_dir, '**', '*.pt'), recursive=True)
    
    print(f"找到 {len(model_files)} 个模型文件")
    
    for model_file in model_files:
        try:
            # 加载模型
            checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)
            
            # 创建模型实例
            model = ResNetCompareNet(BasicBlock, [2, 2, 2, 2])
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.eval()
            models.append({
                'model': model,
                'path': model_file,
                'name': os.path.basename(model_file).replace('.pt', '')
            })
            
        except Exception as e:
            print(f"加载模型失败 {model_file}: {e}")
            continue
    
    print(f"成功加载 {len(models)} 个模型")
    return models

def get_model_predictions(models: list, dataloader: DataLoader, device: str = 'cpu') -> tuple:
    """获取所有模型的预测结果"""
    all_predictions = []
    all_labels = []
    
    print("获取模型预测...")
    
    with torch.no_grad():
        for batch_idx, (xa, xb, y) in enumerate(tqdm(dataloader)):
            xa, xb, y = xa.to(device), xb.to(device), y.to(device)
            
            batch_predictions = []
            for model_info in models:
                model = model_info['model'].to(device)
                model.eval()
                
                # 获取预测概率
                with torch.no_grad():
                    logits = model(xa, xb)
                    probs = torch.softmax(logits, dim=1)
                    pred_probs = probs[:, 1].cpu().numpy()  # 正类概率
                
                batch_predictions.append(pred_probs)
            
            all_predictions.append(np.column_stack(batch_predictions))
            all_labels.append(y.cpu().numpy())
    
    predictions = np.vstack(all_predictions)
    labels = np.concatenate(all_labels)
    
    print(f"预测结果形状: {predictions.shape}")
    print(f"标签形状: {labels.shape}")
    
    return predictions, labels

def evaluate_baseline_ensemble(predictions: np.ndarray, labels: np.ndarray) -> dict:
    """评估基础集成方法"""
    print("\n=== 基础集成方法评估 ===")
    
    # 简单平均
    simple_avg = np.mean(predictions, axis=1)
    simple_avg_pred = (simple_avg >= 0.5).astype(int)
    
    # 加权平均（基于验证集性能）
    # 这里使用简单的权重：每个模型的权重相等
    weights = np.ones(predictions.shape[1]) / predictions.shape[1]
    weighted_avg = np.sum(predictions * weights, axis=1)
    weighted_avg_pred = (weighted_avg >= 0.5).astype(int)
    
    # 投票
    vote_pred = (predictions >= 0.5).astype(int)
    majority_vote = (np.sum(vote_pred, axis=1) > predictions.shape[1] // 2).astype(int)
    
    results = {}
    
    for name, pred in [('Simple Average', simple_avg_pred), 
                      ('Weighted Average', weighted_avg_pred),
                      ('Majority Vote', majority_vote)]:
        acc = accuracy_score(labels, pred)
        f1 = f1_score(labels, pred)
        results[name] = {'accuracy': acc, 'f1_score': f1}
        print(f"{name}: Accuracy={acc:.4f}, F1={f1:.4f}")
    
    return results

def evaluate_mc_dropout_ensemble(models: list, dataloader: DataLoader, 
                                mc_samples: int = 10) -> dict:
    """评估蒙特卡洛Dropout集成"""
    print(f"\n=== 蒙特卡洛Dropout集成评估 (MC samples: {mc_samples}) ===")
    
    # 创建MC Dropout包装器
    mc_models = []
    for model_info in models:
        mc_model = MCDropoutWrapper(model_info['model'], mc_samples=mc_samples)
        mc_models.append(mc_model)
    
    # 创建贝叶斯集成
    bayesian_ensemble = BayesianEnsemble(mc_models)
    
    # 预测
    all_predictions = []
    all_labels = []
    all_uncertainties = []
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    with torch.no_grad():
        for xa, xb, y in tqdm(dataloader):
            xa, xb, y = xa.to(device), xb.to(device), y.to(device)
            
            # 贝叶斯预测
            pred_probs, uncertainty = bayesian_ensemble.predict_with_uncertainty(xa, xb)
            
            all_predictions.append(pred_probs.cpu().numpy())
            all_uncertainties.append(uncertainty.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    
    predictions = np.concatenate(all_predictions)
    uncertainties = np.concatenate(all_uncertainties)
    labels = np.concatenate(all_labels)
    
    # 计算指标
    pred_labels = (predictions >= 0.5).astype(int)
    accuracy = accuracy_score(labels, pred_labels)
    f1 = f1_score(labels, pred_labels)
    
    results = {
        'mc_dropout': {
            'accuracy': accuracy,
            'f1_score': f1,
            'avg_uncertainty': np.mean(uncertainties),
            'mc_samples': mc_samples
        }
    }
    
    print(f"MC Dropout: Accuracy={accuracy:.4f}, F1={f1:.4f}")
    print(f"平均不确定性: {np.mean(uncertainties):.4f}")
    
    return results

def evaluate_two_level_stacking_ensemble(predictions: np.ndarray, labels: np.ndarray) -> dict:
    """评估两层Stacking集成"""
    print("\n=== 两层Stacking集成评估 ===")
    
    try:
        # 使用两层Stacking
        results, stacking = evaluate_two_level_stacking(predictions, labels)
        
        print(f"两层Stacking: Accuracy={results['accuracy']:.4f}, F1={results['f1_macro']:.4f}")
        print(f"模型分组数: {results['n_groups']}")
        
        return {'two_level_stacking': results}
        
    except Exception as e:
        print(f"两层Stacking评估失败: {e}")
        return {'two_level_stacking': {'error': str(e)}}

def evaluate_dynamic_ensemble_methods(models: list, dataloader: DataLoader) -> dict:
    """评估动态集成方法"""
    print("\n=== 动态集成方法评估 ===")
    
    # 获取模型预测
    predictions, labels = get_model_predictions(models, dataloader)
    
    # 创建基础模型包装器
    class ModelWrapper:
        def __init__(self, predictions):
            self.predictions = predictions
        
        def predict_proba(self, X):
            # 返回预计算的预测概率
            return np.column_stack([1 - self.predictions, self.predictions])
    
    # 为每个模型创建包装器
    base_models = []
    for i in range(predictions.shape[1]):
        wrapper = ModelWrapper(predictions[:, i])
        base_models.append(wrapper)
    
    # 评估不同的动态集成策略
    results = evaluate_dynamic_ensemble(base_models, predictions, labels)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="高级集成方法综合评估")
    parser.add_argument('--model_dir', type=str, default='models',
                       help='模型目录')
    parser.add_argument('--data_path', type=str, default='data/val.npz',
                       help='验证数据路径')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--mc_samples', type=int, default=10,
                       help='蒙特卡洛采样次数')
    parser.add_argument('--output_file', type=str, default='outputs/results/advanced_ensemble_results.json',
                       help='结果输出文件')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    print("开始高级集成方法综合评估...")
    print(f"模型目录: {args.model_dir}")
    print(f"数据路径: {args.data_path}")
    
    # 加载数据
    data_manager = DataManager()
    dataloader = data_manager.load_val_data(batch_size=args.batch_size)
    
    # 加载模型
    models = load_trained_models(args.model_dir)
    if not models:
        print("没有找到可用的模型!")
        return
    
    # 获取模型预测
    predictions, labels = get_model_predictions(models, dataloader)
    
    # 评估结果
    all_results = {}
    
    # 1. 基础集成方法
    baseline_results = evaluate_baseline_ensemble(predictions, labels)
    all_results.update(baseline_results)
    
    # 2. 蒙特卡洛Dropout集成
    try:
        mc_results = evaluate_mc_dropout_ensemble(models, dataloader, args.mc_samples)
        all_results.update(mc_results)
    except Exception as e:
        print(f"蒙特卡洛Dropout评估失败: {e}")
        all_results['mc_dropout'] = {'error': str(e)}
    
    # 3. 两层Stacking集成
    stacking_results = evaluate_two_level_stacking_ensemble(predictions, labels)
    all_results.update(stacking_results)
    
    # 4. 动态集成方法
    try:
        dynamic_results = evaluate_dynamic_ensemble_methods(models, dataloader)
        all_results.update(dynamic_results)
    except Exception as e:
        print(f"动态集成评估失败: {e}")
        all_results['dynamic_ensemble'] = {'error': str(e)}
    
    # 保存结果
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== 综合评估结果 ===")
    print(f"结果已保存到: {args.output_file}")
    
    # 打印最佳结果
    best_accuracy = 0
    best_method = None
    
    for method, metrics in all_results.items():
        if isinstance(metrics, dict) and 'accuracy' in metrics:
            acc = metrics['accuracy']
            print(f"{method}: {acc:.4f}")
            if acc > best_accuracy:
                best_accuracy = acc
                best_method = method
    
    print(f"\n最佳方法: {best_method} (Accuracy: {best_accuracy:.4f})")

if __name__ == '__main__':
    main()