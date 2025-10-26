# MC Dropout + 动态权重集成

## 概述

本目录包含蒙特卡洛Dropout (MC Dropout) 与动态权重结合的集成方法实现。

## 文件说明

- `mc_dropout_dynamic_ensemble.py` - MC Dropout + 动态权重的完整实现
- `mc_dropout.py` - 基础MC Dropout模块
- `bayesian_inference.py` - 贝叶斯推理脚本

## 核心特性

- **不确定性估计**: 通过多次MC采样量化预测不确定性
- **动态权重**: 根据模型置信度动态调整权重
- **高置信度**: 平均置信度达95.96%
- **贝叶斯集成**: 近似贝叶斯模型平均

## 性能指标

- **准确率**: 0.8767
- **F1 Score**: 0.8762
- **平均不确定性**: 0.0425
- **平均置信度**: 0.9596

## 使用方法

```python
from mc_dropout_dynamic_ensemble import MCDropoutDynamicEnsemble

# 创建集成模型
ensemble = MCDropoutDynamicEnsemble(
    base_models=models,
    mc_samples=20,
    device='cuda'
)

# 预测
predictions, uncertainty, weights = ensemble.predict_with_uncertainty(xa, xb)
```

## 参数说明

- `mc_samples`: MC采样次数 (默认20)
- `dropout_rate`: Dropout率 (默认0.5)
- `confidence_method`: 置信度计算方法 ('max_prob', 'entropy', 'variance')
