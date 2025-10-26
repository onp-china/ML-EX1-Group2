#!/usr/bin/env python3
"""
两层Stacking + 动态权重组合集成方法
结合分层集成和动态权重分配
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import lightgbm as lgb
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class TwoLevelDynamicStacking:
    """
    两层Stacking + 动态权重组合集成类
    """
    
    def __init__(self, 
                 first_level_models: List[str] = None,
                 second_level_model: str = 'lightgbm',
                 n_folds: int = 5,
                 confidence_method: str = 'entropy',
                 uncertainty_threshold: float = 0.3,
                 random_state: int = 42):
        """
        初始化两层动态Stacking
        
        Args:
            first_level_models: 第一层模型类型列表
            second_level_model: 第二层模型类型
            n_folds: 交叉验证折数
            confidence_method: 置信度计算方法
            uncertainty_threshold: 不确定性阈值
            random_state: 随机种子
        """
        self.first_level_models = first_level_models or ['lightgbm', 'rf', 'lr']
        self.second_level_model = second_level_model
        self.n_folds = n_folds
        self.confidence_method = confidence_method
        self.uncertainty_threshold = uncertainty_threshold
        self.random_state = random_state
        
        # 存储训练好的模型
        self.first_level_meta_learners = []
        self.second_level_meta_learner = None
        self.model_groups = []
        
        # 动态权重相关
        self.dynamic_weights_enabled = True
        self.weight_history = []
    
    def _create_meta_learner(self, model_type: str, **kwargs):
        """创建元学习器"""
        if model_type == 'lightgbm':
            return lgb.LGBMClassifier(
                objective='binary',
                metric='binary_logloss',
                boosting_type='gbdt',
                num_leaves=31,
                learning_rate=0.05,
                feature_fraction=0.9,
                bagging_fraction=0.8,
                bagging_freq=5,
                verbose=-1,
                random_state=self.random_state,
                **kwargs
            )
        elif model_type == 'rf':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                **kwargs
            )
        elif model_type == 'lr':
            return LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _group_models(self, model_predictions: np.ndarray, n_groups: int = None) -> List[List[int]]:
        """将模型分组"""
        n_models = model_predictions.shape[1]
        
        if n_groups is None:
            n_groups = max(2, min(n_models // 2, 5))
        
        group_size = n_models // n_groups
        groups = []
        
        for i in range(n_groups):
            start_idx = i * group_size
            if i == n_groups - 1:
                end_idx = n_models
            else:
                end_idx = (i + 1) * group_size
            groups.append(list(range(start_idx, end_idx)))
        
        return groups
    
    def _extract_features(self, 
                         model_predictions: np.ndarray, 
                         model_group: List[int],
                         include_uncertainty: bool = True) -> np.ndarray:
        """从模型预测中提取特征"""
        group_predictions = model_predictions[:, model_group]
        
        features = []
        
        # 基础特征：每个模型的预测概率
        features.append(group_predictions)
        
        # 统计特征
        features.append(np.mean(group_predictions, axis=1, keepdims=True))
        features.append(np.std(group_predictions, axis=1, keepdims=True))
        features.append(np.max(group_predictions, axis=1, keepdims=True))
        features.append(np.min(group_predictions, axis=1, keepdims=True))
        
        # 排序特征
        sorted_predictions = np.sort(group_predictions, axis=1)
        features.append(sorted_predictions)
        
        # 差异特征
        if len(model_group) > 1:
            for i in range(len(model_group) - 1):
                for j in range(i + 1, len(model_group)):
                    diff = group_predictions[:, i:i+1] - group_predictions[:, j:j+1]
                    features.append(diff)
        
        # 不确定性特征
        if include_uncertainty:
            uncertainty = np.std(group_predictions, axis=1, keepdims=True)
            features.append(uncertainty)
            features.append(uncertainty ** 2)
        
        return np.concatenate(features, axis=1)
    
    def _calculate_confidence(self, predictions: np.ndarray) -> np.ndarray:
        """计算预测置信度"""
        if predictions.ndim == 2:
            predictions = np.stack([1 - predictions, predictions], axis=-1)
        
        if self.confidence_method == 'entropy':
            entropy = -np.sum(predictions * np.log(predictions + 1e-12), axis=-1)
            confidence = 1.0 / (1.0 + entropy)
        elif self.confidence_method == 'max_prob':
            max_probs = np.max(predictions, axis=-1)
            confidence = max_probs
        elif self.confidence_method == 'variance':
            variance = np.var(predictions, axis=-1)
            confidence = 1.0 / (1.0 + variance)
        else:
            raise ValueError(f"Unknown confidence method: {self.confidence_method}")
        
        return confidence
    
    def _calculate_dynamic_weights(self, 
                                 first_level_predictions: np.ndarray,
                                 confidence: np.ndarray) -> np.ndarray:
        """计算动态权重"""
        n_samples, n_groups = first_level_predictions.shape
        
        # 基础权重：置信度越高，权重越大
        base_weights = confidence
        
        # 对于低置信度样本，使用更平均的权重
        low_confidence_mask = confidence < self.uncertainty_threshold
        
        if np.any(low_confidence_mask):
            # 低置信度样本使用平均权重
            avg_weights = np.ones((n_samples, n_groups)) / n_groups
            base_weights[low_confidence_mask] = avg_weights[low_confidence_mask]
        
        # 归一化权重
        weights = base_weights / np.sum(base_weights, axis=1, keepdims=True)
        
        return weights
    
    def fit(self, 
            X: np.ndarray, 
            y: np.ndarray,
            model_groups: List[List[int]] = None) -> 'TwoLevelDynamicStacking':
        """训练两层动态Stacking模型"""
        print("开始训练两层动态Stacking...")
        
        # 直接使用X作为模型预测结果
        model_predictions = X
        
        # 确定模型分组
        if model_groups is None:
            self.model_groups = self._group_models(model_predictions)
        else:
            self.model_groups = model_groups
        
        print(f"模型分组: {self.model_groups}")
        
        # 第一层：训练每个分组的元学习器
        first_level_predictions = []
        
        for group_idx, model_group in enumerate(self.model_groups):
            print(f"训练第一层分组 {group_idx + 1}/{len(self.model_groups)}")
            
            # 提取特征
            X_features = self._extract_features(model_predictions, model_group)
            
            # 交叉验证训练
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
            group_predictions = np.zeros(len(X_features))
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_features)):
                X_train, X_val = X_features[train_idx], X_features[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # 训练元学习器
                meta_learner = self._create_meta_learner(
                    self.first_level_models[group_idx % len(self.first_level_models)]
                )
                meta_learner.fit(X_train, y_train)
                
                # 预测验证集
                val_pred = meta_learner.predict_proba(X_val)[:, 1]
                group_predictions[val_idx] = val_pred
                
                # 保存模型（用于最终预测）
                if fold == 0:
                    self.first_level_meta_learners.append(meta_learner)
            
            first_level_predictions.append(group_predictions)
        
        # 合并第一层预测结果
        first_level_features = np.column_stack(first_level_predictions)
        
        # 计算置信度
        confidence = self._calculate_confidence(first_level_features)
        
        # 计算动态权重
        dynamic_weights = self._calculate_dynamic_weights(first_level_features, confidence)
        
        # 应用动态权重到第一层特征
        weighted_features = first_level_features * dynamic_weights
        
        # 添加不确定性特征
        first_level_uncertainty = np.std(first_level_features, axis=1, keepdims=True)
        weighted_features = np.concatenate([
            weighted_features,
            first_level_uncertainty,
            first_level_uncertainty ** 2,
            confidence,  # 添加置信度作为特征
            dynamic_weights  # 添加权重作为特征
        ], axis=1)
        
        # 第二层：训练最终集成模型
        print("训练第二层元学习器...")
        self.second_level_meta_learner = self._create_meta_learner(self.second_level_model)
        self.second_level_meta_learner.fit(weighted_features, y)
        
        # 存储权重历史
        self.weight_history.append({
            'first_level_features': first_level_features,
            'dynamic_weights': dynamic_weights,
            'confidence': confidence
        })
        
        print("两层动态Stacking训练完成!")
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        # 直接使用X作为模型预测结果
        model_predictions = X
        
        # 第一层预测
        first_level_predictions = []
        
        for group_idx, model_group in enumerate(self.model_groups):
            # 提取特征
            X_features = self._extract_features(model_predictions, model_group)
            
            # 使用训练好的元学习器预测
            meta_learner = self.first_level_meta_learners[group_idx]
            group_pred = meta_learner.predict_proba(X_features)[:, 1]
            first_level_predictions.append(group_pred)
        
        # 合并第一层预测
        first_level_features = np.column_stack(first_level_predictions)
        
        # 计算置信度和动态权重
        confidence = self._calculate_confidence(first_level_features)
        dynamic_weights = self._calculate_dynamic_weights(first_level_features, confidence)
        
        # 应用动态权重
        weighted_features = first_level_features * dynamic_weights
        
        # 添加不确定性特征
        first_level_uncertainty = np.std(first_level_features, axis=1, keepdims=True)
        weighted_features = np.concatenate([
            weighted_features,
            first_level_uncertainty,
            first_level_uncertainty ** 2,
            confidence,
            dynamic_weights
        ], axis=1)
        
        # 第二层预测
        second_level_pred = self.second_level_meta_learner.predict_proba(weighted_features)
        
        return second_level_pred
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别"""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)
    
    def predict_with_weights(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """预测并返回权重信息"""
        # 直接使用X作为模型预测结果
        model_predictions = X
        
        # 第一层预测
        first_level_predictions = []
        
        for group_idx, model_group in enumerate(self.model_groups):
            X_features = self._extract_features(model_predictions, model_group)
            meta_learner = self.first_level_meta_learners[group_idx]
            group_pred = meta_learner.predict_proba(X_features)[:, 1]
            first_level_predictions.append(group_pred)
        
        first_level_features = np.column_stack(first_level_predictions)
        
        # 计算置信度和动态权重
        confidence = self._calculate_confidence(first_level_features)
        dynamic_weights = self._calculate_dynamic_weights(first_level_features, confidence)
        
        # 应用动态权重
        weighted_features = first_level_features * dynamic_weights
        
        # 添加不确定性特征
        first_level_uncertainty = np.std(first_level_features, axis=1, keepdims=True)
        weighted_features = np.concatenate([
            weighted_features,
            first_level_uncertainty,
            first_level_uncertainty ** 2,
            confidence,
            dynamic_weights
        ], axis=1)
        
        # 第二层预测
        second_level_pred = self.second_level_meta_learner.predict_proba(weighted_features)
        
        return second_level_pred, dynamic_weights, confidence
    
    def save(self, filepath: str):
        """保存模型"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'first_level_meta_learners': self.first_level_meta_learners,
            'second_level_meta_learner': self.second_level_meta_learner,
            'model_groups': self.model_groups,
            'first_level_models': self.first_level_models,
            'second_level_model': self.second_level_model,
            'n_folds': self.n_folds,
            'confidence_method': self.confidence_method,
            'uncertainty_threshold': self.uncertainty_threshold,
            'random_state': self.random_state,
            'weight_history': self.weight_history
        }
        
        joblib.dump(model_data, filepath)
        print(f"模型已保存到: {filepath}")
    
    def load(self, filepath: str):
        """加载模型"""
        model_data = joblib.load(filepath)
        
        self.first_level_meta_learners = model_data['first_level_meta_learners']
        self.second_level_meta_learner = model_data['second_level_meta_learner']
        self.model_groups = model_data['model_groups']
        self.first_level_models = model_data['first_level_models']
        self.second_level_model = model_data['second_level_model']
        self.n_folds = model_data['n_folds']
        self.confidence_method = model_data['confidence_method']
        self.uncertainty_threshold = model_data['uncertainty_threshold']
        self.random_state = model_data['random_state']
        self.weight_history = model_data.get('weight_history', [])
        
        print(f"模型已从 {filepath} 加载")

def evaluate_two_level_dynamic_stacking(model_predictions: np.ndarray,
                                       y_true: np.ndarray,
                                       test_size: float = 0.2,
                                       random_state: int = 42) -> Dict:
    """
    评估两层动态Stacking性能
    
    Args:
        model_predictions: 基础模型预测结果 (n_samples, n_models)
        y_true: 真实标签 (n_samples,)
        test_size: 测试集比例
        random_state: 随机种子
        
    Returns:
        评估结果字典
    """
    from sklearn.model_selection import train_test_split
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        model_predictions, y_true, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y_true
    )
    
    # 训练两层动态Stacking
    stacking = TwoLevelDynamicStacking(random_state=random_state)
    # 直接传递模型预测结果，不需要额外的检查
    stacking.model_predictions = X_train
    stacking.fit(X_train, y_train)
    
    # 预测
    y_pred = stacking.predict(X_test)
    y_pred_proba = stacking.predict_proba(X_test)
    
    # 获取权重信息
    _, weights, confidence = stacking.predict_with_weights(X_test)
    
    # 计算指标
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # 计算每个类别的F1分数
    f1_scores = []
    for cls in [0, 1]:
        tp = np.sum((y_test==cls) & (y_pred==cls))
        fp = np.sum((y_test!=cls) & (y_pred==cls))
        fn = np.sum((y_test==cls) & (y_pred!=cls))
        
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1_cls = 2 * prec * rec / (prec + rec + 1e-12)
        f1_scores.append(f1_cls)
    
    f1_macro = np.mean(f1_scores)
    
    results = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_class_0': f1_scores[0],
        'f1_class_1': f1_scores[1],
        'n_models': model_predictions.shape[1],
        'n_groups': len(stacking.model_groups),
        'test_size': len(X_test),
        'avg_confidence': np.mean(confidence),
        'std_confidence': np.std(confidence),
        'avg_weight_entropy': np.mean([-np.sum(w * np.log(w + 1e-12)) for w in weights])
    }
    
    return results, stacking

if __name__ == "__main__":
    # 测试代码
    print("测试两层动态Stacking...")
    
    # 生成模拟数据
    np.random.seed(42)
    n_samples, n_models = 1000, 8
    
    # 模拟基础模型预测结果
    model_predictions = np.random.rand(n_samples, n_models)
    y_true = np.random.randint(0, 2, n_samples)
    
    # 评估
    results, stacking = evaluate_two_level_dynamic_stacking(model_predictions, y_true)
    
    print("评估结果:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")
    
    print("两层动态Stacking测试完成!")
