#!/usr/bin/env python3
"""
测试高级集成方法的简化脚本
使用模拟数据验证所有方法的功能
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# 添加路径
sys.path.append('src')

from mc_dropout import MCDropoutWrapper, BayesianEnsemble
from two_level_stacking import TwoLevelStacking, evaluate_two_level_stacking
from dynamic_ensemble import DynamicEnsemble, evaluate_dynamic_ensemble

class MockModel:
    """模拟模型类"""
    def __init__(self, name, noise_level=0.1, bias=0.0):
        self.name = name
        self.noise_level = noise_level
        self.bias = bias
        self.is_trained = True
    
    def predict_proba(self, X):
        """模拟预测概率"""
        # 简单的线性分类器模拟
        n_samples = X.shape[0]
        scores = np.random.randn(n_samples) * self.noise_level + self.bias
        probs = 1 / (1 + np.exp(-scores))
        return np.column_stack([1 - probs, probs])
    
    def predict(self, X):
        """模拟预测类别"""
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)

class MockMCDropoutModel:
    """模拟MC Dropout模型"""
    def __init__(self, base_model, mc_samples=10):
        self.base_model = base_model
        self.mc_samples = mc_samples
    
    def predict_with_uncertainty(self, X):
        """模拟MC Dropout预测"""
        n_samples = X.shape[0]
        predictions = []
        
        for _ in range(self.mc_samples):
            pred = self.base_model.predict_proba(X)[:, 1]
            predictions.append(pred)
        
        predictions = np.array(predictions).T  # (n_samples, mc_samples)
        mean_pred = np.mean(predictions, axis=1)
        uncertainty = np.std(predictions, axis=1)
        
        return torch.tensor(mean_pred), torch.tensor(uncertainty)

def generate_synthetic_data(n_samples=1000, n_features=20, noise_level=0.1):
    """生成合成数据"""
    np.random.seed(42)
    
    # 生成特征
    X = np.random.randn(n_samples, n_features)
    
    # 生成真实标签（基于特征的线性组合）
    true_weights = np.random.randn(n_features)
    true_scores = np.dot(X, true_weights)
    true_probs = 1 / (1 + np.exp(-true_scores))
    y_true = (true_probs >= 0.5).astype(int)
    
    # 添加噪声
    noise = np.random.normal(0, noise_level, n_samples)
    y_true = (true_probs + noise >= 0.5).astype(int)
    
    return X, y_true

def test_baseline_ensemble():
    """测试基础集成方法"""
    print("\n=== 测试基础集成方法 ===")
    
    # 生成数据
    X, y_true = generate_synthetic_data(1000, 20, 0.1)
    X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.2, random_state=42)
    
    # 创建模拟模型
    models = [
        MockModel("Model1", noise_level=0.1, bias=0.0),
        MockModel("Model2", noise_level=0.15, bias=0.1),
        MockModel("Model3", noise_level=0.12, bias=-0.05),
        MockModel("Model4", noise_level=0.08, bias=0.02),
        MockModel("Model5", noise_level=0.18, bias=-0.1)
    ]
    
    # 获取预测
    predictions = []
    for model in models:
        pred_probs = model.predict_proba(X_test)
        predictions.append(pred_probs[:, 1])  # 正类概率
    
    predictions = np.column_stack(predictions)
    
    # 基础集成方法
    results = {}
    
    # 1. 简单平均
    simple_avg = np.mean(predictions, axis=1)
    simple_avg_pred = (simple_avg >= 0.5).astype(int)
    results['Simple Average'] = {
        'accuracy': accuracy_score(y_test, simple_avg_pred),
        'f1_score': f1_score(y_test, simple_avg_pred)
    }
    
    # 2. 加权平均（基于训练集性能）
    train_predictions = []
    for model in models:
        train_pred = model.predict_proba(X_train)[:, 1]
        train_predictions.append(train_pred)
    
    train_predictions = np.column_stack(train_predictions)
    
    # 计算每个模型在训练集上的性能
    model_weights = []
    for i in range(len(models)):
        train_pred = (train_predictions[:, i] >= 0.5).astype(int)
        acc = accuracy_score(y_train, train_pred)
        model_weights.append(acc)
    
    # 归一化权重
    model_weights = np.array(model_weights)
    model_weights = model_weights / np.sum(model_weights)
    
    weighted_avg = np.sum(predictions * model_weights, axis=1)
    weighted_avg_pred = (weighted_avg >= 0.5).astype(int)
    results['Weighted Average'] = {
        'accuracy': accuracy_score(y_test, weighted_avg_pred),
        'f1_score': f1_score(y_test, weighted_avg_pred)
    }
    
    # 3. 投票
    vote_pred = (predictions >= 0.5).astype(int)
    majority_vote = (np.sum(vote_pred, axis=1) > len(models) // 2).astype(int)
    results['Majority Vote'] = {
        'accuracy': accuracy_score(y_test, majority_vote),
        'f1_score': f1_score(y_test, majority_vote)
    }
    
    print("基础集成方法结果:")
    for method, metrics in results.items():
        print(f"  {method}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}")
    
    return results

def test_mc_dropout_ensemble():
    """测试蒙特卡洛Dropout集成"""
    print("\n=== 测试蒙特卡洛Dropout集成 ===")
    
    # 生成数据
    X, y_true = generate_synthetic_data(1000, 20, 0.1)
    X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.2, random_state=42)
    
    # 创建模拟MC Dropout模型
    base_models = [
        MockModel("MC_Model1", noise_level=0.1),
        MockModel("MC_Model2", noise_level=0.12),
        MockModel("MC_Model3", noise_level=0.08)
    ]
    
    mc_models = [MockMCDropoutModel(model, mc_samples=10) for model in base_models]
    
    # 获取预测和不确定性
    all_predictions = []
    all_uncertainties = []
    
    for mc_model in mc_models:
        pred, uncertainty = mc_model.predict_with_uncertainty(X_test)
        all_predictions.append(pred.numpy())
        all_uncertainties.append(uncertainty.numpy())
    
    predictions = np.column_stack(all_predictions)
    uncertainties = np.column_stack(all_uncertainties)
    
    # 贝叶斯集成（简单平均）
    final_pred = np.mean(predictions, axis=1)
    final_pred_labels = (final_pred >= 0.5).astype(int)
    
    # 计算平均不确定性
    avg_uncertainty = np.mean(uncertainties)
    
    results = {
        'mc_dropout': {
            'accuracy': accuracy_score(y_test, final_pred_labels),
            'f1_score': f1_score(y_test, final_pred_labels),
            'avg_uncertainty': avg_uncertainty
        }
    }
    
    print(f"MC Dropout: Accuracy={results['mc_dropout']['accuracy']:.4f}, F1={results['mc_dropout']['f1_score']:.4f}")
    print(f"平均不确定性: {avg_uncertainty:.4f}")
    
    return results

def test_two_level_stacking():
    """测试两层Stacking"""
    print("\n=== 测试两层Stacking ===")
    
    # 生成数据
    X, y_true = generate_synthetic_data(1000, 20, 0.1)
    X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.2, random_state=42)
    
    # 创建模拟模型预测
    models = [
        MockModel("Stack_Model1", noise_level=0.1),
        MockModel("Stack_Model2", noise_level=0.12),
        MockModel("Stack_Model3", noise_level=0.08),
        MockModel("Stack_Model4", noise_level=0.15),
        MockModel("Stack_Model5", noise_level=0.09),
        MockModel("Stack_Model6", noise_level=0.11)
    ]
    
    # 获取模型预测
    train_predictions = []
    test_predictions = []
    
    for model in models:
        train_pred = model.predict_proba(X_train)[:, 1]
        test_pred = model.predict_proba(X_test)[:, 1]
        train_predictions.append(train_pred)
        test_predictions.append(test_pred)
    
    train_predictions = np.column_stack(train_predictions)
    test_predictions = np.column_stack(test_predictions)
    
    # 使用训练集训练两层Stacking
    try:
        results, stacking = evaluate_two_level_stacking(train_predictions, y_train)
        
        # 在测试集上预测
        test_pred_proba = stacking.predict_proba(test_predictions)
        test_pred = (test_pred_proba[:, 1] >= 0.5).astype(int)
        
        test_accuracy = accuracy_score(y_test, test_pred)
        test_f1 = f1_score(y_test, test_pred)
        
        print(f"两层Stacking: Accuracy={test_accuracy:.4f}, F1={test_f1:.4f}")
        print(f"模型分组数: {results['n_groups']}")
        
        return {'two_level_stacking': {
            'accuracy': test_accuracy,
            'f1_score': test_f1,
            'n_groups': results['n_groups']
        }}
        
    except Exception as e:
        print(f"两层Stacking测试失败: {e}")
        return {'two_level_stacking': {'error': str(e)}}

def test_dynamic_ensemble():
    """测试动态集成"""
    print("\n=== 测试动态集成 ===")
    
    # 生成数据
    X, y_true = generate_synthetic_data(1000, 20, 0.1)
    X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.2, random_state=42)
    
    # 创建模拟模型
    models = [
        MockModel("Dyn_Model1", noise_level=0.1),
        MockModel("Dyn_Model2", noise_level=0.12),
        MockModel("Dyn_Model3", noise_level=0.08),
        MockModel("Dyn_Model4", noise_level=0.15)
    ]
    
    # 获取预测
    predictions = []
    for model in models:
        pred_probs = model.predict_proba(X_test)
        predictions.append(pred_probs[:, 1])
    
    predictions = np.column_stack(predictions)
    
    # 测试不同的动态集成策略
    try:
        results = evaluate_dynamic_ensemble(models, X_test, y_test)
        
        print("动态集成方法结果:")
        for strategy, metrics in results.items():
            print(f"  {strategy}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}")
        
        return results
        
    except Exception as e:
        print(f"动态集成测试失败: {e}")
        return {'dynamic_ensemble': {'error': str(e)}}

def main():
    """主函数"""
    print("开始测试高级集成方法...")
    
    all_results = {}
    
    # 1. 基础集成方法
    baseline_results = test_baseline_ensemble()
    all_results.update(baseline_results)
    
    # 2. 蒙特卡洛Dropout集成
    mc_results = test_mc_dropout_ensemble()
    all_results.update(mc_results)
    
    # 3. 两层Stacking
    stacking_results = test_two_level_stacking()
    all_results.update(stacking_results)
    
    # 4. 动态集成
    dynamic_results = test_dynamic_ensemble()
    all_results.update(dynamic_results)
    
    # 保存结果
    output_file = 'outputs/results/advanced_methods_test_results.json'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== 综合测试结果 ===")
    print(f"结果已保存到: {output_file}")
    
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
    print("\n所有高级集成方法测试完成!")

if __name__ == '__main__':
    main()
