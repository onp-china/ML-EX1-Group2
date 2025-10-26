#!/usr/bin/env python3
"""
动态集成权重实现
基于模型置信度和不确定性动态调整集成权重
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union
from sklearn.metrics import accuracy_score, f1_score
import warnings

class ConfidenceBasedWeights:
    """
    基于置信度的动态权重计算
    """
    
    def __init__(self, 
                 confidence_method: str = 'entropy',
                 temperature: float = 1.0,
                 min_weight: float = 0.01,
                 max_weight: float = 1.0):
        """
        初始化置信度权重计算器
        
        Args:
            confidence_method: 置信度计算方法 ['entropy', 'max_prob', 'variance']
            temperature: 温度参数，用于softmax
            min_weight: 最小权重
            max_weight: 最大权重
        """
        self.confidence_method = confidence_method
        self.temperature = temperature
        self.min_weight = min_weight
        self.max_weight = max_weight
        
    def calculate_confidence(self, predictions: np.ndarray) -> np.ndarray:
        """
        计算预测置信度
        
        Args:
            predictions: 模型预测概率 (n_samples, n_models, n_classes) 或 (n_samples, n_models)
            
        Returns:
            置信度分数 (n_samples, n_models)
        """
        if predictions.ndim == 2:
            # 二分类情况，转换为概率
            predictions = np.stack([1 - predictions, predictions], axis=-1)
        
        n_samples, n_models, n_classes = predictions.shape
        
        if self.confidence_method == 'entropy':
            # 基于熵的置信度：熵越低，置信度越高
            entropy = -np.sum(predictions * np.log(predictions + 1e-12), axis=-1)
            confidence = 1.0 / (1.0 + entropy)  # 转换为置信度
        elif self.confidence_method == 'max_prob':
            # 基于最大概率的置信度
            max_probs = np.max(predictions, axis=-1)
            confidence = max_probs
        elif self.confidence_method == 'variance':
            # 基于方差的置信度：方差越小，置信度越高
            variance = np.var(predictions, axis=-1)
            confidence = 1.0 / (1.0 + variance)
        else:
            raise ValueError(f"Unknown confidence method: {self.confidence_method}")
        
        return confidence
    
    def calculate_weights(self, 
                         predictions: np.ndarray,
                         uncertainties: Optional[np.ndarray] = None) -> np.ndarray:
        """
        计算动态权重
        
        Args:
            predictions: 模型预测概率 (n_samples, n_models, n_classes) 或 (n_samples, n_models)
            uncertainties: 不确定性分数 (n_samples, n_models)，可选
            
        Returns:
            动态权重 (n_samples, n_models)
        """
        # 计算置信度
        confidence = self.calculate_confidence(predictions)
        
        # 如果有不确定性信息，结合使用
        if uncertainties is not None:
            # 不确定性越低，权重越高
            uncertainty_weights = 1.0 / (1.0 + uncertainties)
            confidence = confidence * uncertainty_weights
        
        # 应用温度参数
        confidence = confidence / self.temperature
        
        # 使用softmax计算权重
        weights = self._softmax(confidence, axis=1)
        
        # 应用权重范围限制
        weights = np.clip(weights, self.min_weight, self.max_weight)
        
        # 重新归一化
        weights = weights / np.sum(weights, axis=1, keepdims=True)
        
        return weights
    
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Softmax函数"""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

class UncertaintyBasedWeights:
    """
    基于不确定性的动态权重计算
    """
    
    def __init__(self, 
                 uncertainty_method: str = 'mc_dropout',
                 uncertainty_threshold: float = 0.5,
                 weight_decay: float = 0.1):
        """
        初始化不确定性权重计算器
        
        Args:
            uncertainty_method: 不确定性计算方法 ['mc_dropout', 'ensemble_variance', 'prediction_variance']
            uncertainty_threshold: 不确定性阈值
            weight_decay: 权重衰减参数
        """
        self.uncertainty_method = uncertainty_method
        self.uncertainty_threshold = uncertainty_threshold
        self.weight_decay = weight_decay
    
    def calculate_uncertainty(self, 
                            predictions: np.ndarray,
                            mc_samples: Optional[int] = None) -> np.ndarray:
        """
        计算预测不确定性
        
        Args:
            predictions: 模型预测 (n_samples, n_models, n_classes) 或 (n_samples, n_models)
            mc_samples: 蒙特卡洛采样次数（用于mc_dropout方法）
            
        Returns:
            不确定性分数 (n_samples, n_models)
        """
        if predictions.ndim == 2:
            # 二分类情况
            predictions = np.stack([1 - predictions, predictions], axis=-1)
        
        n_samples, n_models, n_classes = predictions.shape
        
        if self.uncertainty_method == 'mc_dropout':
            # 蒙特卡洛Dropout不确定性
            if mc_samples is None:
                mc_samples = predictions.shape[0]  # 使用样本数作为默认值
            
            # 模拟MC Dropout：对每个模型的预测添加噪声
            uncertainties = []
            for i in range(n_models):
                model_preds = predictions[:, i, :]
                # 计算预测的方差作为不确定性
                pred_variance = np.var(model_preds, axis=1)
                uncertainties.append(pred_variance)
            
            uncertainties = np.column_stack(uncertainties)
            
        elif self.uncertainty_method == 'ensemble_variance':
            # 集成方差不确定性
            mean_pred = np.mean(predictions, axis=1, keepdims=True)
            variance = np.mean((predictions - mean_pred) ** 2, axis=1)
            uncertainties = np.tile(variance, (n_models, 1)).T
            
        elif self.uncertainty_method == 'prediction_variance':
            # 预测方差不确定性
            uncertainties = np.var(predictions, axis=-1)
            
        else:
            raise ValueError(f"Unknown uncertainty method: {self.uncertainty_method}")
        
        return uncertainties
    
    def calculate_weights(self, 
                         predictions: np.ndarray,
                         uncertainties: Optional[np.ndarray] = None) -> np.ndarray:
        """
        基于不确定性计算权重
        
        Args:
            predictions: 模型预测概率
            uncertainties: 不确定性分数，如果为None则计算
            
        Returns:
            动态权重 (n_samples, n_models)
        """
        if uncertainties is None:
            uncertainties = self.calculate_uncertainty(predictions)
        
        # 不确定性越低，权重越高
        inverse_uncertainty = 1.0 / (1.0 + uncertainties)
        
        # 应用权重衰减
        weights = inverse_uncertainty * (1.0 - self.weight_decay)
        
        # 归一化
        weights = weights / np.sum(weights, axis=1, keepdims=True)
        
        return weights

class AdaptiveEnsemble:
    """
    自适应集成类
    结合多种权重计算方法
    """
    
    def __init__(self, 
                 weight_methods: List[str] = None,
                 method_weights: List[float] = None,
                 confidence_threshold: float = 0.7,
                 uncertainty_threshold: float = 0.3):
        """
        初始化自适应集成
        
        Args:
            weight_methods: 权重计算方法列表
            method_weights: 各方法的权重
            confidence_threshold: 置信度阈值
            uncertainty_threshold: 不确定性阈值
        """
        self.weight_methods = weight_methods or ['confidence', 'uncertainty']
        self.method_weights = method_weights or [0.6, 0.4]
        self.confidence_threshold = confidence_threshold
        self.uncertainty_threshold = uncertainty_threshold
        
        # 初始化权重计算器
        self.confidence_weights = ConfidenceBasedWeights()
        self.uncertainty_weights = UncertaintyBasedWeights()
        
        # 存储历史性能
        self.model_performance_history = []
        self.adaptive_weights = None
    
    def calculate_adaptive_weights(self, 
                                 predictions: np.ndarray,
                                 uncertainties: Optional[np.ndarray] = None,
                                 historical_performance: Optional[np.ndarray] = None) -> np.ndarray:
        """
        计算自适应权重
        
        Args:
            predictions: 模型预测概率 (n_samples, n_models, n_classes) 或 (n_samples, n_models)
            uncertainties: 不确定性分数
            historical_performance: 历史性能分数 (n_models,)
            
        Returns:
            自适应权重 (n_samples, n_models)
        """
        n_samples, n_models = predictions.shape[:2]
        weights = np.zeros((n_samples, n_models))
        
        # 方法1：基于置信度的权重
        if 'confidence' in self.weight_methods:
            conf_weights = self.confidence_weights.calculate_weights(predictions, uncertainties)
            weights += self.method_weights[0] * conf_weights
        
        # 方法2：基于不确定性的权重
        if 'uncertainty' in self.weight_methods:
            unc_weights = self.uncertainty_weights.calculate_weights(predictions, uncertainties)
            weights += self.method_weights[1] * unc_weights
        
        # 方法3：基于历史性能的权重
        if historical_performance is not None and 'performance' in self.weight_methods:
            perf_weights = self._calculate_performance_weights(historical_performance, n_samples)
            weights += self.method_weights[2] * perf_weights
        
        # 归一化
        weights = weights / np.sum(weights, axis=1, keepdims=True)
        
        return weights
    
    def _calculate_performance_weights(self, 
                                     historical_performance: np.ndarray, 
                                     n_samples: int) -> np.ndarray:
        """基于历史性能计算权重"""
        # 性能越高，权重越高
        performance_weights = historical_performance / np.sum(historical_performance)
        
        # 扩展到所有样本
        weights = np.tile(performance_weights, (n_samples, 1))
        
        return weights
    
    def predict_with_adaptive_weights(self, 
                                    predictions: np.ndarray,
                                    uncertainties: Optional[np.ndarray] = None,
                                    historical_performance: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用自适应权重进行预测
        
        Args:
            predictions: 模型预测概率
            uncertainties: 不确定性分数
            historical_performance: 历史性能分数
            
        Returns:
            (最终预测, 权重)
        """
        # 计算自适应权重
        weights = self.calculate_adaptive_weights(
            predictions, uncertainties, historical_performance
        )
        
        # 加权平均预测
        if predictions.ndim == 2:
            # 二分类情况
            final_predictions = np.sum(predictions * weights, axis=1)
        else:
            # 多分类情况
            final_predictions = np.sum(predictions * weights[:, :, np.newaxis], axis=1)
        
        return final_predictions, weights
    
    def update_performance_history(self, 
                                 model_performances: np.ndarray,
                                 window_size: int = 10):
        """
        更新模型性能历史
        
        Args:
            model_performances: 当前性能分数 (n_models,)
            window_size: 历史窗口大小
        """
        self.model_performance_history.append(model_performances)
        
        # 保持窗口大小
        if len(self.model_performance_history) > window_size:
            self.model_performance_history = self.model_performance_history[-window_size:]
        
        # 计算平均性能
        if len(self.model_performance_history) > 0:
            avg_performance = np.mean(self.model_performance_history, axis=0)
            self.adaptive_weights = avg_performance / np.sum(avg_performance)

class DynamicEnsemble:
    """
    动态集成主类
    """
    
    def __init__(self, 
                 base_models: List = None,
                 weight_strategy: str = 'adaptive',
                 confidence_method: str = 'entropy',
                 uncertainty_method: str = 'mc_dropout'):
        """
        初始化动态集成
        
        Args:
            base_models: 基础模型列表
            weight_strategy: 权重策略 ['adaptive', 'confidence', 'uncertainty', 'fixed']
            confidence_method: 置信度计算方法
            uncertainty_method: 不确定性计算方法
        """
        self.base_models = base_models or []
        self.weight_strategy = weight_strategy
        self.confidence_method = confidence_method
        self.uncertainty_method = uncertainty_method
        
        # 初始化权重计算器
        if weight_strategy == 'adaptive':
            self.weight_calculator = AdaptiveEnsemble()
        elif weight_strategy == 'confidence':
            self.weight_calculator = ConfidenceBasedWeights(confidence_method)
        elif weight_strategy == 'uncertainty':
            self.weight_calculator = UncertaintyBasedWeights(uncertainty_method)
        else:
            self.weight_calculator = None
    
    def predict(self, 
                X: np.ndarray,
                return_weights: bool = False,
                return_uncertainties: bool = False) -> Union[np.ndarray, Tuple]:
        """
        动态集成预测
        
        Args:
            X: 输入特征
            return_weights: 是否返回权重
            return_uncertainties: 是否返回不确定性
            
        Returns:
            预测结果，可选权重和不确定性
        """
        if not self.base_models:
            raise ValueError("没有基础模型")
        
        # 获取所有模型的预测
        all_predictions = []
        all_uncertainties = []
        
        for model in self.base_models:
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)
                if pred.ndim == 2 and pred.shape[1] == 2:
                    pred = pred[:, 1]  # 二分类取正类概率
                all_predictions.append(pred)
            else:
                pred = model.predict(X)
                all_predictions.append(pred)
            
            # 计算不确定性（如果模型支持）
            if hasattr(model, 'predict_with_uncertainty'):
                uncertainty = model.predict_with_uncertainty(X)
                all_uncertainties.append(uncertainty)
        
        predictions = np.column_stack(all_predictions)
        uncertainties = np.column_stack(all_uncertainties) if all_uncertainties else None
        
        # 计算动态权重
        if self.weight_calculator is not None:
            if self.weight_strategy == 'adaptive':
                final_pred, weights = self.weight_calculator.predict_with_adaptive_weights(
                    predictions, uncertainties
                )
            else:
                weights = self.weight_calculator.calculate_weights(predictions, uncertainties)
                final_pred = np.sum(predictions * weights, axis=1)
        else:
            # 固定权重（简单平均）
            weights = np.ones((len(X), len(self.base_models))) / len(self.base_models)
            final_pred = np.mean(predictions, axis=1)
        
        # 准备返回结果
        result = [final_pred]
        if return_weights:
            result.append(weights)
        if return_uncertainties:
            result.append(uncertainties)
        
        if len(result) == 1:
            return result[0]
        else:
            return tuple(result)

def evaluate_dynamic_ensemble(base_models: List,
                            X_test: np.ndarray,
                            y_test: np.ndarray,
                            weight_strategies: List[str] = None) -> Dict:
    """
    评估动态集成性能
    
    Args:
        base_models: 基础模型列表
        X_test: 测试特征
        y_test: 测试标签
        weight_strategies: 权重策略列表
        
    Returns:
        评估结果字典
    """
    if weight_strategies is None:
        weight_strategies = ['adaptive', 'confidence', 'uncertainty', 'fixed']
    
    results = {}
    
    for strategy in weight_strategies:
        print(f"评估策略: {strategy}")
        
        # 创建动态集成
        ensemble = DynamicEnsemble(
            base_models=base_models,
            weight_strategy=strategy
        )
        
        # 预测
        y_pred = ensemble.predict(X_test)
        
        # 计算指标
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results[strategy] = {
            'accuracy': accuracy,
            'f1_score': f1
        }
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Score: {f1:.4f}")
    
    return results

if __name__ == "__main__":
    # 测试代码
    print("测试动态集成...")
    
    # 生成模拟数据
    np.random.seed(42)
    n_samples, n_features = 1000, 20
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    
    # 模拟基础模型
    class MockModel:
        def __init__(self, noise_level=0.1):
            self.noise_level = noise_level
        
        def predict_proba(self, X):
            # 简单的线性分类器模拟
            scores = np.dot(X, np.random.randn(X.shape[1])) + np.random.normal(0, self.noise_level, len(X))
            probs = 1 / (1 + np.exp(-scores))
            return np.column_stack([1 - probs, probs])
    
    # 创建多个基础模型
    base_models = [MockModel(noise_level=0.1 + i*0.05) for i in range(5)]
    
    # 评估动态集成
    results = evaluate_dynamic_ensemble(base_models, X, y)
    
    print("\n评估结果:")
    for strategy, metrics in results.items():
        print(f"{strategy}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}")
    
    print("动态集成测试完成!")