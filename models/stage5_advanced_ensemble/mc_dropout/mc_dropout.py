#!/usr/bin/env python3
"""
蒙特卡洛Dropout实现
用于贝叶斯模型平均和不确定性估计
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Union
import warnings

class MCDropoutWrapper(nn.Module):
    """
    蒙特卡洛Dropout包装器
    在推理时启用Dropout进行多次前向传播
    """
    
    def __init__(self, model: nn.Module, mc_samples: int = 20, dropout_rate: float = 0.5):
        """
        初始化MC Dropout包装器
        
        Args:
            model: 要包装的模型
            mc_samples: 蒙特卡洛采样次数
            dropout_rate: Dropout率（如果模型没有Dropout层，会添加）
        """
        super().__init__()
        self.model = model
        self.mc_samples = mc_samples
        self.dropout_rate = dropout_rate
        
        # 确保模型有Dropout层
        self._ensure_dropout_layers()
        
        # 存储原始训练状态
        self._original_training = None
        
    def _ensure_dropout_layers(self):
        """确保模型有足够的Dropout层"""
        dropout_layers = []
        
        def find_dropout_layers(module, name=""):
            for child_name, child in module.named_children():
                full_name = f"{name}.{child_name}" if name else child_name
                if isinstance(child, nn.Dropout):
                    dropout_layers.append(full_name)
                elif isinstance(child, nn.Dropout2d):
                    dropout_layers.append(full_name)
                find_dropout_layers(child, full_name)
        
        find_dropout_layers(self.model)
        
        if not dropout_layers:
            warnings.warn("模型中没有找到Dropout层，将在最后添加一个Dropout层")
            # 在模型最后添加一个Dropout层
            if hasattr(self.model, 'classifier'):
                # 如果模型有classifier属性
                if isinstance(self.model.classifier, nn.Sequential):
                    self.model.classifier.add_module('mc_dropout', nn.Dropout(self.dropout_rate))
                else:
                    self.model.classifier = nn.Sequential(
                        self.model.classifier,
                        nn.Dropout(self.dropout_rate)
                    )
            else:
                # 尝试在最后添加
                self.model.add_module('mc_dropout', nn.Dropout(self.dropout_rate))
    
    def forward(self, xa: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        """
        单次前向传播
        
        Args:
            xa: 第一个输入张量
            xb: 第二个输入张量
            
        Returns:
            预测logits
        """
        return self.model(xa, xb)
    
    def predict_with_uncertainty(self, xa: torch.Tensor, xb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用蒙特卡洛Dropout进行预测并估计不确定性
        
        Args:
            xa: 第一个输入张量
            xb: 第二个输入张量
            
        Returns:
            mean_pred: 平均预测概率 (batch_size, num_classes)
            uncertainty: 预测不确定性 (batch_size,)
        """
        self.model.eval()  # 设置为评估模式
        
        # 存储所有预测
        predictions = []
        
        with torch.no_grad():
            for _ in range(self.mc_samples):
                # 在评估模式下启用Dropout
                self._enable_dropout()
                
                # 前向传播
                logits = self.forward(xa, xb)
                probs = torch.sigmoid(logits)
                predictions.append(probs)
        
        # 恢复原始状态
        self._restore_training_state()
        
        # 计算均值和不确定性
        predictions = torch.stack(predictions, dim=0)  # (mc_samples, batch_size, num_classes)
        mean_pred = torch.mean(predictions, dim=0)
        
        # 计算不确定性（标准差）
        uncertainty = torch.std(predictions, dim=0).mean(dim=-1)  # 平均所有类别的标准差
        
        return mean_pred, uncertainty
    
    def _enable_dropout(self):
        """启用所有Dropout层"""
        def enable_dropout_recursive(module):
            for child in module.children():
                if isinstance(child, (nn.Dropout, nn.Dropout2d)):
                    child.train()  # 设置为训练模式以启用Dropout
                enable_dropout_recursive(child)
        
        enable_dropout_recursive(self.model)
    
    def _restore_training_state(self):
        """恢复原始训练状态"""
        if self._original_training is not None:
            self.model.train(self._original_training)
    
    def set_training_state(self, training: bool):
        """设置训练状态"""
        self._original_training = training
        self.model.train(training)

class BayesianEnsemble:
    """
    贝叶斯集成类
    使用多个模型的MC Dropout进行集成
    """
    
    def __init__(self, models: List[MCDropoutWrapper], weights: Optional[List[float]] = None):
        """
        初始化贝叶斯集成
        
        Args:
            models: MC Dropout包装的模型列表
            weights: 模型权重，如果为None则使用均等权重
        """
        self.models = models
        self.num_models = len(models)
        
        if weights is None:
            self.weights = [1.0 / self.num_models] * self.num_models
        else:
            assert len(weights) == self.num_models, "权重数量必须与模型数量相同"
            # 归一化权重
            total_weight = sum(weights)
            self.weights = [w / total_weight for w in weights]
    
    def predict_with_uncertainty(self, xa: torch.Tensor, xb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        使用所有模型进行贝叶斯集成预测
        
        Args:
            xa: 第一个输入张量
            xb: 第二个输入张量
            
        Returns:
            mean_pred: 集成平均预测概率
            aleatoric_uncertainty: 认知不确定性（模型间差异）
            epistemic_uncertainty: 认知不确定性（模型内差异）
        """
        all_predictions = []
        all_uncertainties = []
        
        # 获取每个模型的预测
        for model in self.models:
            mean_pred, uncertainty = model.predict_with_uncertainty(xa, xb)
            all_predictions.append(mean_pred)
            all_uncertainties.append(uncertainty)
        
        # 转换为张量
        all_predictions = torch.stack(all_predictions, dim=0)  # (num_models, batch_size, num_classes)
        all_uncertainties = torch.stack(all_uncertainties, dim=0)  # (num_models, batch_size)
        
        # 计算加权平均预测
        weights_tensor = torch.tensor(self.weights, device=xa.device).view(-1, 1, 1)
        mean_pred = torch.sum(weights_tensor * all_predictions, dim=0)
        
        # 计算认知不确定性（模型间差异）
        aleatoric_uncertainty = torch.sum(weights_tensor.squeeze() * all_uncertainties, dim=0)
        
        # 计算认知不确定性（模型间预测差异）
        epistemic_uncertainty = torch.std(all_predictions, dim=0).mean(dim=-1)
        
        return mean_pred, aleatoric_uncertainty, epistemic_uncertainty
    
    def predict(self, xa: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        """
        简单预测（只返回平均预测）
        
        Args:
            xa: 第一个输入张量
            xb: 第二个输入张量
            
        Returns:
            预测概率
        """
        mean_pred, _, _ = self.predict_with_uncertainty(xa, xb)
        return mean_pred

def create_mc_dropout_models(model_paths: List[str], 
                           model_class, 
                           model_kwargs: dict,
                           mc_samples: int = 20,
                           dropout_rate: float = 0.5) -> List[MCDropoutWrapper]:
    """
    从保存的模型创建MC Dropout包装器列表
    
    Args:
        model_paths: 模型文件路径列表
        model_class: 模型类
        model_kwargs: 模型初始化参数
        mc_samples: MC采样次数
        dropout_rate: Dropout率
        
    Returns:
        MC Dropout包装器列表
    """
    models = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for model_path in model_paths:
        try:
            # 创建模型实例
            model = model_class(**model_kwargs)
            
            # 加载模型状态
            checkpoint = torch.load(model_path, map_location=device)
            
            # 检查是否是checkpoint格式
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"从checkpoint加载模型: {model_path}")
            else:
                model.load_state_dict(checkpoint)
                print(f"直接加载模型: {model_path}")
            
            model.to(device)
            
            # 包装为MC Dropout
            mc_model = MCDropoutWrapper(model, mc_samples=mc_samples, dropout_rate=dropout_rate)
            models.append(mc_model)
            
        except Exception as e:
            print(f"加载模型失败 {model_path}: {e}")
            continue
    
    return models

if __name__ == "__main__":
    # 测试代码
    print("测试MC Dropout包装器...")
    
    # 创建一个简单的测试模型
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 5)
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(5, 1)
        
        def forward(self, xa, xb):
            x = torch.cat([xa, xb], dim=1)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    
    # 创建测试数据
    batch_size = 4
    xa = torch.randn(batch_size, 5)
    xb = torch.randn(batch_size, 5)
    
    # 创建模型和MC包装器
    model = TestModel()
    mc_model = MCDropoutWrapper(model, mc_samples=10)
    
    # 测试预测
    mean_pred, uncertainty = mc_model.predict_with_uncertainty(xa, xb)
    print(f"平均预测形状: {mean_pred.shape}")
    print(f"不确定性形状: {uncertainty.shape}")
    print(f"平均预测: {mean_pred}")
    print(f"不确定性: {uncertainty}")
    
    print("MC Dropout测试完成!")