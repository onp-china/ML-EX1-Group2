#!/usr/bin/env python3
"""
MC Dropout + 动态权重组合集成方法
结合不确定性估计和动态权重分配
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

class MCDropoutDynamicEnsemble:
    """
    MC Dropout + 动态权重组合集成类
    """
    
    def __init__(self, 
                 base_models: List[nn.Module],
                 mc_samples: int = 20,
                 dropout_rate: float = 0.5,
                 confidence_method: str = 'entropy',
                 uncertainty_threshold: float = 0.3,
                 device: str = 'cuda'):
        """
        初始化MC Dropout动态集成
        
        Args:
            base_models: 基础模型列表
            mc_samples: 蒙特卡洛采样次数
            dropout_rate: Dropout率
            confidence_method: 置信度计算方法 ['entropy', 'max_prob', 'variance']
            uncertainty_threshold: 不确定性阈值
            device: 设备
        """
        self.base_models = base_models
        self.mc_samples = mc_samples
        self.dropout_rate = dropout_rate
        self.confidence_method = confidence_method
        self.uncertainty_threshold = uncertainty_threshold
        self.device = device
        
        # 确保所有模型在正确设备上
        for model in self.base_models:
            model.to(device)
            model.eval()
        
        # 为每个模型添加Dropout层（如果没有的话）
        self._add_dropout_layers()
    
    def _add_dropout_layers(self):
        """为模型添加Dropout层"""
        for model in self.base_models:
            # 检查是否已有Dropout层
            has_dropout = any(isinstance(module, nn.Dropout) for module in model.modules())
            if not has_dropout:
                # 在最后的全连接层前添加Dropout
                for name, module in model.named_modules():
                    if isinstance(module, nn.Linear) and 'fc' in name.lower():
                        # 找到最后一个全连接层
                        parent_module = model
                        for part in name.split('.')[:-1]:
                            parent_module = getattr(parent_module, part)
                        setattr(parent_module, 'dropout', nn.Dropout(self.dropout_rate))
                        break
    
    def _calculate_confidence(self, predictions: np.ndarray) -> np.ndarray:
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
            confidence = 1.0 / (1.0 + entropy)
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
    
    def _calculate_uncertainty(self, mc_predictions: np.ndarray) -> np.ndarray:
        """
        计算MC Dropout不确定性
        
        Args:
            mc_predictions: MC采样预测结果 (n_samples, n_models, mc_samples)
            
        Returns:
            不确定性分数 (n_samples, n_models)
        """
        # 计算每个模型预测的标准差作为不确定性
        uncertainty = np.std(mc_predictions, axis=2)
        return uncertainty
    
    def _calculate_dynamic_weights(self, 
                                 confidence: np.ndarray,
                                 uncertainty: np.ndarray) -> np.ndarray:
        """
        基于置信度和不确定性计算动态权重
        
        Args:
            confidence: 置信度分数 (n_samples, n_models)
            uncertainty: 不确定性分数 (n_samples, n_models)
            
        Returns:
            动态权重 (n_samples, n_models)
        """
        n_samples, n_models = confidence.shape
        
        # 基础权重：置信度越高，权重越大
        base_weights = confidence
        
        # 不确定性调整：不确定性越高，权重越小
        uncertainty_penalty = 1.0 / (1.0 + uncertainty)
        adjusted_weights = base_weights * uncertainty_penalty
        
        # 对于高不确定性样本，使用更保守的权重分配
        high_uncertainty_mask = uncertainty > self.uncertainty_threshold
        
        # 高不确定性样本：使用更平均的权重
        if np.any(high_uncertainty_mask):
            # 计算平均权重
            avg_weights = np.ones((n_samples, n_models)) / n_models
            
            # 对高不确定性样本使用平均权重
            adjusted_weights[high_uncertainty_mask] = avg_weights[high_uncertainty_mask]
        
        # 归一化权重
        weights = adjusted_weights / np.sum(adjusted_weights, axis=1, keepdims=True)
        
        return weights
    
    def predict_with_uncertainty(self, 
                                xa: torch.Tensor, 
                                xb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        使用MC Dropout和动态权重进行预测
        
        Args:
            xa: 左图像 (batch_size, channels, height, width)
            xb: 右图像 (batch_size, channels, height, width)
            
        Returns:
            (最终预测概率, 不确定性, 动态权重)
        """
        batch_size = xa.size(0)
        n_models = len(self.base_models)
        
        # 存储所有模型的MC采样结果
        all_mc_predictions = []
        
        with torch.no_grad():
            for model in self.base_models:
                model.eval()
                # 启用Dropout进行MC采样
                model.train()  # 启用Dropout
                
                mc_predictions = []
                for _ in range(self.mc_samples):
                    # 前向传播
                    logits = model(xa, xb)
                    # 确保logits是2D的 (batch_size, num_classes)
                    if logits.dim() == 1:
                        logits = logits.unsqueeze(0)
                    probs = F.softmax(logits, dim=1)
                    # 检查是否有两个类别
                    if probs.shape[1] == 2:
                        pred_probs = probs[:, 1].cpu().numpy()  # 正类概率
                    else:
                        # 如果只有一个类别，使用sigmoid
                        pred_probs = torch.sigmoid(logits).squeeze().cpu().numpy()
                    mc_predictions.append(pred_probs)
                
                # 恢复eval模式
                model.eval()
                
                # 存储MC采样结果
                mc_predictions = np.array(mc_predictions).T  # (batch_size, mc_samples)
                all_mc_predictions.append(mc_predictions)
        
        # 转换为numpy数组
        all_mc_predictions = np.array(all_mc_predictions).transpose(1, 0, 2)  # (batch_size, n_models, mc_samples)
        
        # 计算每个模型的平均预测和不确定性
        mean_predictions = np.mean(all_mc_predictions, axis=2)  # (batch_size, n_models)
        uncertainty = self._calculate_uncertainty(all_mc_predictions)  # (batch_size, n_models)
        
        # 计算置信度
        confidence = self._calculate_confidence(mean_predictions)  # (batch_size, n_models)
        
        # 计算动态权重
        dynamic_weights = self._calculate_dynamic_weights(confidence, uncertainty)
        
        # 加权平均预测
        final_predictions = np.sum(mean_predictions * dynamic_weights, axis=1)
        
        # 计算最终不确定性（加权平均）
        final_uncertainty = np.sum(uncertainty * dynamic_weights, axis=1)
        
        return (torch.tensor(final_predictions, device=self.device),
                torch.tensor(final_uncertainty, device=self.device),
                torch.tensor(dynamic_weights, device=self.device))
    
    def predict(self, xa: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        """预测类别"""
        pred_probs, _, _ = self.predict_with_uncertainty(xa, xb)
        return (pred_probs >= 0.5).long()
    
    def predict_proba(self, xa: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        """预测概率"""
        pred_probs, _, _ = self.predict_with_uncertainty(xa, xb)
        return torch.stack([1 - pred_probs, pred_probs], dim=1)

def evaluate_mc_dropout_dynamic_ensemble(base_models: List[nn.Module],
                                        dataloader,
                                        mc_samples: int = 20,
                                        device: str = 'cuda') -> Dict:
    """
    评估MC Dropout + 动态权重集成
    
    Args:
        base_models: 基础模型列表
        dataloader: 数据加载器
        mc_samples: MC采样次数
        device: 设备
        
    Returns:
        评估结果字典
    """
    print(f"评估MC Dropout + 动态权重集成 (MC samples: {mc_samples})")
    
    # 创建集成模型
    ensemble = MCDropoutDynamicEnsemble(
        base_models=base_models,
        mc_samples=mc_samples,
        device=device
    )
    
    all_predictions = []
    all_labels = []
    all_uncertainties = []
    all_weights = []
    
    print("开始预测...")
    with torch.no_grad():
        for batch_idx, (xa, xb, y) in enumerate(dataloader):
            xa, xb, y = xa.to(device), xb.to(device), y.to(device)
            
            # 预测
            pred_probs, uncertainty, weights = ensemble.predict_with_uncertainty(xa, xb)
            
            all_predictions.append(pred_probs.cpu().numpy())
            all_uncertainties.append(uncertainty.cpu().numpy())
            all_weights.append(weights.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"已处理 {batch_idx + 1} 个批次")
    
    # 合并结果
    predictions = np.concatenate(all_predictions)
    uncertainties = np.concatenate(all_uncertainties)
    weights = np.concatenate(all_weights)
    labels = np.concatenate(all_labels)
    
    # 计算指标
    pred_labels = (predictions >= 0.5).astype(int)
    accuracy = accuracy_score(labels, pred_labels)
    f1 = f1_score(labels, pred_labels)
    
    # 计算每个类别的F1分数
    f1_scores = []
    for cls in [0, 1]:
        tp = np.sum((labels==cls) & (pred_labels==cls))
        fp = np.sum((labels!=cls) & (pred_labels==cls))
        fn = np.sum((labels==cls) & (pred_labels!=cls))
        
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1_cls = 2 * prec * rec / (prec + rec + 1e-12)
        f1_scores.append(f1_cls)
    
    f1_macro = np.mean(f1_scores)
    
    results = {
        'accuracy': accuracy,
        'f1_score': f1,
        'f1_macro': f1_macro,
        'f1_class_0': f1_scores[0],
        'f1_class_1': f1_scores[1],
        'avg_uncertainty': np.mean(uncertainties),
        'std_uncertainty': np.std(uncertainties),
        'avg_confidence': np.mean(1.0 / (1.0 + uncertainties)),
        'n_models': len(base_models),
        'mc_samples': mc_samples
    }
    
    print(f"MC Dropout + 动态权重结果:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  F1 Macro: {f1_macro:.4f}")
    print(f"  平均不确定性: {np.mean(uncertainties):.4f}")
    print(f"  平均置信度: {np.mean(1.0 / (1.0 + uncertainties)):.4f}")
    
    return results

if __name__ == "__main__":
    # 测试代码
    print("测试MC Dropout + 动态权重集成...")
    
    # 生成模拟数据
    torch.manual_seed(42)
    np.random.seed(42)
    
    batch_size = 32
    n_samples = 1000
    
    # 模拟图像数据
    xa = torch.randn(n_samples, 1, 28, 56)
    xb = torch.randn(n_samples, 1, 28, 56)
    y = torch.randint(0, 2, (n_samples,))
    
    # 创建模拟模型
    class MockModel(nn.Module):
        def __init__(self, name):
            super().__init__()
            self.name = name
            self.conv = nn.Conv2d(1, 32, 3, padding=1)
            self.fc = nn.Linear(32 * 28 * 56, 2)
            self.dropout = nn.Dropout(0.5)
        
        def forward(self, xa, xb):
            # 简单的特征提取
            feat_a = F.relu(self.conv(xa))
            feat_b = F.relu(self.conv(xb))
            
            # 特征融合
            combined = torch.cat([feat_a, feat_b], dim=1)
            combined = combined.view(combined.size(0), -1)
            
            # 分类
            out = self.fc(self.dropout(combined))
            return out
    
    # 创建多个模拟模型
    models = [MockModel(f"Model_{i}") for i in range(3)]
    
    # 创建数据加载器
    dataset = torch.utils.data.TensorDataset(xa, xb, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # 评估
    results = evaluate_mc_dropout_dynamic_ensemble(models, dataloader, mc_samples=10)
    
    print("测试完成!")
