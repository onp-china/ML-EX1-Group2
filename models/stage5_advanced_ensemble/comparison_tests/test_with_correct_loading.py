#!/usr/bin/env python3
"""
基于成功stacking实现的正确模型加载和高级集成测试
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# 添加路径
sys.path.append('../src')
sys.path.append('../src/models')

from data_loader import PairNPZDataset
from models.simple_compare_cnn import ResNetCompareNet, BasicBlock, Bottleneck
from models.fpn_architecture_v2 import FPNCompareNetV2
from two_level_dynamic_stacking import TwoLevelDynamicStacking, evaluate_two_level_dynamic_stacking
from mc_dropout_dynamic_ensemble import MCDropoutDynamicEnsemble, evaluate_mc_dropout_dynamic_ensemble

def load_model_correctly(model_path, device):
    """正确的模型加载方法 - 基于成功的stacking实现"""
    print(f"Loading: {os.path.basename(model_path)}")
    
    # 读取配置
    metrics_path = model_path.replace('model.pt', 'metrics.json')
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # 创建模型
    layers = metrics.get('layers', [2, 2, 2])
    feat_dim = metrics.get('feat_dim', 256)
    width_mult = metrics.get('width_mult', 1.0)
    use_bottleneck = metrics.get('use_bottleneck', False)
    
    block = Bottleneck if use_bottleneck else BasicBlock
    
    if 'fpn' in model_path.lower():
        model = FPNCompareNetV2(feat_dim=feat_dim, layers=layers, width_mult=width_mult)
    else:
        try:
            model = ResNetCompareNet(feat_dim=feat_dim, layers=layers, block=block, width_mult=width_mult)
        except TypeError:
            model = ResNetCompareNet(feat_dim=feat_dim, layers=layers, block=block)
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    acc = metrics.get('best_val_acc', 0)
    print(f"  Accuracy: {acc:.4f}")
    
    return model, acc

def get_model_predictions_correctly(models, data_loader, device):
    """正确的模型预测方法 - 基于成功的stacking实现"""
    all_predictions = []
    
    for i, model in enumerate(models):
        print(f"Model {i+1}/{len(models)} predicting...")
        predictions = []
        
        with torch.no_grad():
            for batch in data_loader:
                xa, xb, _ = batch
                xa = xa.to(device)
                xb = xb.to(device)
                
                logits = model(xa, xb)
                probs = torch.sigmoid(logits)  # 使用sigmoid而不是softmax
                
                if probs.dim() == 1:
                    probs = probs.unsqueeze(1)
                
                predictions.append(probs.cpu())
        
        all_predictions.append(torch.cat(predictions, dim=0))
    
    return torch.stack(all_predictions, dim=1).squeeze(-1)  # [N, num_models]

def test_baseline_with_correct_loading():
    """使用正确加载方法测试基线"""
    print("\n" + "="*60)
    print("基线方法测试 (正确模型加载)")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 模型配置 - 基于成功的stacking
    model_configs = [
        ("resnet_optimized_1.12", "../models/stage2_resnet_optimized/resnet_optimized_1.12/model.pt"),
        ("resnet_fusion", "../models/stage2_resnet_optimized/resnet_fusion/model.pt"),
        ("resnet_optimized", "../models/stage2_resnet_optimized/resnet_optimized/model.pt"),
        ("seed_2025", "../models/stage3_multi_seed/seed_2025/model.pt"),
        ("fpn_model", "../models/stage3_multi_seed/fpn_model/model.pt"),
        ("seed_2023", "../models/stage3_multi_seed/seed_2023/model.pt"),
        ("seed_2024", "../models/stage3_multi_seed/seed_2024/model.pt"),
        ("resnet_fusion_seed42", "../models/stage3_multi_seed/resnet_fusion_seed42/model.pt"),
    ]
    
    # 加载模型
    models = []
    accuracies = []
    model_names = []
    
    for name, path in model_configs:
        if os.path.exists(path):
            try:
                model, acc = load_model_correctly(path, device)
                models.append(model)
                accuracies.append(acc)
                model_names.append(name)
            except Exception as e:
                print(f"Failed to load {name}: {e}")
                continue
        else:
            print(f"Skipping non-existent model: {path}")
    
    print(f"\n成功加载 {len(models)} 个模型")
    print(f"模型准确率: {[f'{acc:.4f}' for acc in accuracies]}")
    
    if len(models) == 0:
        print("没有成功加载任何模型!")
        return None
    
    # 加载验证数据
    val_dataset = PairNPZDataset('../data/val.npz', is_train=False, use_augmentation=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    print(f"验证集大小: {len(val_dataset)}")
    
    # 获取模型预测
    X = get_model_predictions_correctly(models, val_loader, device).numpy()
    print(f"特征矩阵形状: {X.shape}")
    
    # 获取真实标签
    y = []
    for batch in val_loader:
        _, _, labels = batch
        if isinstance(labels, torch.Tensor):
            y.append(labels.numpy())
    
    y = np.concatenate(y).astype(int)
    print(f"标签形状: {y.shape}")
    print(f"类别分布: {np.bincount(y)}")
    
    # 基线方法
    simple_avg = X.mean(axis=1)
    simple_pred_binary = (simple_avg >= 0.5).astype(int)
    simple_acc = accuracy_score(y, simple_pred_binary)
    simple_f1 = f1_score(y, simple_pred_binary)
    
    print(f"\n基线方法 (简单平均):")
    print(f"  使用模型数: {len(models)}")
    print(f"  准确率: {simple_acc:.4f}")
    print(f"  F1 Score: {simple_f1:.4f}")
    
    return {
        'baseline': {
            'accuracy': simple_acc,
            'f1_score': simple_f1,
            'n_models': len(models)
        },
        'predictions': X,
        'labels': y,
        'models': models,
        'accuracies': accuracies,
        'model_names': model_names
    }

def test_mc_dropout_with_correct_loading(models, val_loader):
    """使用正确加载的模型测试MC Dropout + 动态权重"""
    print("\n" + "="*60)
    print("MC Dropout + 动态权重测试 (正确加载)")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        results = evaluate_mc_dropout_dynamic_ensemble(
            base_models=models,
            dataloader=val_loader,
            mc_samples=10,  # 减少采样次数以加快测试
            device=device
        )
        
        print(f"\nMC Dropout + 动态权重结果:")
        print(f"  准确率: {results['accuracy']:.4f}")
        print(f"  F1 Score: {results['f1_score']:.4f}")
        print(f"  F1 Macro: {results['f1_macro']:.4f}")
        print(f"  平均不确定性: {results['avg_uncertainty']:.4f}")
        print(f"  平均置信度: {results['avg_confidence']:.4f}")
        
        return results
        
    except Exception as e:
        print(f"MC Dropout + 动态权重测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_two_level_stacking_with_correct_loading(predictions, labels):
    """使用正确加载的预测结果测试两层Stacking"""
    print("\n" + "="*60)
    print("两层Stacking + 动态权重测试 (正确加载)")
    print("="*60)
    
    try:
        results, stacking = evaluate_two_level_dynamic_stacking(
            model_predictions=predictions,
            y_true=labels,
            test_size=0.2,
            random_state=42
        )
        
        print(f"\n两层Stacking + 动态权重结果:")
        print(f"  准确率: {results['accuracy']:.4f}")
        print(f"  F1 Macro: {results['f1_macro']:.4f}")
        print(f"  F1 Class 0: {results['f1_class_0']:.4f}")
        print(f"  F1 Class 1: {results['f1_class_1']:.4f}")
        print(f"  模型分组数: {results['n_groups']}")
        print(f"  平均置信度: {results['avg_confidence']:.4f}")
        
        return results
        
    except Exception as e:
        print(f"两层Stacking + 动态权重测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_original_stacking_with_correct_loading(predictions, labels):
    """测试原始Stacking方法 - 基于成功的实现"""
    print("\n" + "="*60)
    print("原始Stacking方法测试 (LightGBM元学习器)")
    print("="*60)
    
    try:
        from sklearn.model_selection import StratifiedKFold
        import lightgbm as lgb
        
        # 5折交叉验证训练Stacking
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof_predictions = np.zeros(len(predictions))
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(predictions, labels)):
            print(f"  训练fold {fold+1}...")
            
            X_train, X_val = predictions[train_idx], predictions[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]
            
            # 创建LightGBM模型
            lgb_model = lgb.LGBMClassifier(
                objective='binary',
                metric='binary_logloss',
                boosting_type='gbdt',
                num_leaves=31,
                learning_rate=0.05,
                feature_fraction=0.9,
                bagging_fraction=0.8,
                bagging_freq=5,
                verbose=-1,
                random_state=42,
                n_estimators=100
            )
            
            # 训练模型
            lgb_model.fit(X_train, y_train)
            
            # 预测验证集
            val_pred = lgb_model.predict_proba(X_val)[:, 1]
            oof_predictions[val_idx] = val_pred
            
            # 计算准确率
            val_pred_binary = (val_pred >= 0.5).astype(int)
            fold_acc = accuracy_score(y_val, val_pred_binary)
            fold_scores.append(fold_acc)
            
            print(f"    Fold {fold+1} accuracy: {fold_acc:.4f}")
        
        # 计算整体性能
        oof_pred_binary = (oof_predictions >= 0.5).astype(int)
        overall_acc = accuracy_score(labels, oof_pred_binary)
        overall_f1 = f1_score(labels, oof_pred_binary)
        
        print(f"\n原始Stacking结果:")
        print(f"  整体准确率: {overall_acc:.4f}")
        print(f"  F1 Score: {overall_f1:.4f}")
        print(f"  Fold准确率: {[f'{acc:.4f}' for acc in fold_scores]}")
        
        # 训练最终模型
        print("训练最终Stacking模型...")
        final_model = lgb.LGBMClassifier(
            objective='binary',
            metric='binary_logloss',
            boosting_type='gbdt',
            num_leaves=31,
            learning_rate=0.05,
            feature_fraction=0.9,
            bagging_fraction=0.8,
            bagging_freq=5,
            verbose=-1,
            random_state=42,
            n_estimators=100
        )
        
        final_model.fit(predictions, labels)
        
        return {
            'original_stacking': {
                'accuracy': overall_acc,
                'f1_score': overall_f1,
                'fold_scores': fold_scores
            }
        }
        
    except Exception as e:
        print(f"原始Stacking测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_on_test_public_with_correct_loading(models, accuracies, model_names):
    """在test_public上进行最终评估"""
    print("\n" + "="*60)
    print("在test_public上进行最终评估")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载test_public数据
    test_dataset = PairNPZDataset('../data/test_public.npz', is_train=False, use_augmentation=False, 
                                 labels_path='../data/test_public_labels.csv')
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    print(f"Test Public数据集: {len(test_dataset)} 样本")
    
    # 获取模型预测
    X_test = get_model_predictions_correctly(models, test_loader, device).numpy()
    
    # 获取真实标签
    y_test = []
    for batch in test_loader:
        _, _, labels = batch
        if isinstance(labels, torch.Tensor):
            y_test.append(labels.numpy())
    
    y_test = np.concatenate(y_test).astype(int)
    
    # 基线方法
    simple_avg = X_test.mean(axis=1)
    simple_pred_binary = (simple_avg >= 0.5).astype(int)
    test_accuracy = accuracy_score(y_test, simple_pred_binary)
    test_f1 = f1_score(y_test, simple_pred_binary)
    
    print(f"\nTest Public 基线结果:")
    print(f"  准确率: {test_accuracy:.4f}")
    print(f"  F1 Score: {test_f1:.4f}")
    
    # 两层Stacking
    try:
        # 直接使用两层Stacking进行预测，不使用交叉验证
        stacking = TwoLevelDynamicStacking(random_state=42)
        stacking.fit(X_test, y_test)
        
        # 预测
        y_pred = stacking.predict(X_test)
        y_pred_proba = stacking.predict_proba(X_test)
        
        # 计算指标
        test_accuracy = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred)
        test_f1_macro = f1_score(y_test, y_pred, average='macro')
        
        stacking_results = {
            'accuracy': test_accuracy,
            'f1_score': test_f1,
            'f1_macro': test_f1_macro
        }
        
        print(f"\nTest Public 两层Stacking结果:")
        print(f"  准确率: {stacking_results['accuracy']:.4f}")
        print(f"  F1 Macro: {stacking_results['f1_macro']:.4f}")
        
        return {
            'test_public_baseline': {
                'accuracy': test_accuracy,
                'f1_score': test_f1
            },
            'test_public_stacking': stacking_results
        }
        
    except Exception as e:
        print(f"Test Public 两层Stacking测试失败: {e}")
        return {
            'test_public_baseline': {
                'accuracy': test_accuracy,
                'f1_score': test_f1
            }
        }

def main():
    """主函数"""
    print("开始使用正确模型加载方法测试高级集成...")
    
    # 检查GPU
    if torch.cuda.is_available():
        print(f"GPU可用: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU不可用，使用CPU")
    
    all_results = {}
    
    # 步骤1&2: 使用正确方法加载模型和数据
    baseline_data = test_baseline_with_correct_loading()
    if not baseline_data:
        print("基线测试失败，无法继续")
        return
    
    all_results.update(baseline_data['baseline'])
    
    # 步骤3: 测试MC Dropout + 动态权重
    val_dataset = PairNPZDataset('../data/val.npz', is_train=False, use_augmentation=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    mc_results = test_mc_dropout_with_correct_loading(
        baseline_data['models'], 
        val_loader
    )
    if mc_results:
        all_results['mc_dropout_dynamic'] = mc_results
    
    # 步骤3: 测试两层Stacking + 动态权重
    stacking_results = test_two_level_stacking_with_correct_loading(
        baseline_data['predictions'], 
        baseline_data['labels']
    )
    if stacking_results:
        all_results['two_level_dynamic'] = stacking_results
    
    # 步骤3: 测试原始Stacking方法
    original_stacking_results = test_original_stacking_with_correct_loading(
        baseline_data['predictions'], 
        baseline_data['labels']
    )
    if original_stacking_results:
        all_results.update(original_stacking_results)
    
    # 步骤4: 在test_public上最终评估
    test_public_results = test_on_test_public_with_correct_loading(
        baseline_data['models'],
        baseline_data['accuracies'],
        baseline_data['model_names']
    )
    if test_public_results:
        all_results.update(test_public_results)
    
    # 保存结果
    output_file = 'correct_loading_test_results.json'
    
    # 转换numpy类型为Python原生类型
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    all_results_converted = convert_numpy(all_results)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results_converted, f, indent=2, ensure_ascii=False)
    
    print(f"\n" + "="*60)
    print("综合测试结果")
    print("="*60)
    print(f"结果已保存到: {output_file}")
    
    # 打印比较结果
    print(f"\n" + "="*60)
    print("方法对比总结")
    print("="*60)
    
    if 'baseline' in all_results:
        baseline_acc = all_results['baseline']['accuracy']
        print(f"基线 (简单平均): {baseline_acc:.4f}")
    else:
        print("基线结果不可用")
        baseline_acc = 0.0
    
    if 'mc_dropout_dynamic' in all_results:
        mc_acc = all_results['mc_dropout_dynamic']['accuracy']
        mc_improvement = mc_acc - baseline_acc
        if baseline_acc > 0:
            print(f"MC Dropout + 动态权重: {mc_acc:.4f} ({mc_improvement:+.4f}, {mc_improvement/baseline_acc*100:+.2f}%)")
        else:
            print(f"MC Dropout + 动态权重: {mc_acc:.4f} ({mc_improvement:+.4f})")
    
    if 'two_level_dynamic' in all_results:
        stacking_acc = all_results['two_level_dynamic']['accuracy']
        stacking_improvement = stacking_acc - baseline_acc
        if baseline_acc > 0:
            print(f"两层Stacking + 动态权重: {stacking_acc:.4f} ({stacking_improvement:+.4f}, {stacking_improvement/baseline_acc*100:+.2f}%)")
        else:
            print(f"两层Stacking + 动态权重: {stacking_acc:.4f} ({stacking_improvement:+.4f})")
    
    if 'original_stacking' in all_results:
        orig_stacking_acc = all_results['original_stacking']['accuracy']
        orig_improvement = orig_stacking_acc - baseline_acc
        if baseline_acc > 0:
            print(f"原始Stacking (LightGBM): {orig_stacking_acc:.4f} ({orig_improvement:+.4f}, {orig_improvement/baseline_acc*100:+.2f}%)")
        else:
            print(f"原始Stacking (LightGBM): {orig_stacking_acc:.4f} ({orig_improvement:+.4f})")
    
    # 找出最佳方法
    methods = [('基线', baseline_acc)]
    if 'mc_dropout_dynamic' in all_results:
        methods.append(('MC Dropout + 动态权重', all_results['mc_dropout_dynamic']['accuracy']))
    if 'two_level_dynamic' in all_results:
        methods.append(('两层Stacking + 动态权重', all_results['two_level_dynamic']['accuracy']))
    if 'original_stacking' in all_results:
        methods.append(('原始Stacking', all_results['original_stacking']['accuracy']))
    
    methods.sort(key=lambda x: x[1], reverse=True)
    print(f"\n方法排名:")
    for i, (method, acc) in enumerate(methods, 1):
        print(f"  {i}. {method}: {acc:.4f}")
    
    if 'test_public_baseline' in all_results:
        test_baseline_acc = all_results['test_public_baseline']['accuracy']
        print(f"\n测试集基线准确率: {test_baseline_acc:.4f}")
    
    print("\n正确模型加载测试完成!")

if __name__ == '__main__':
    main()
