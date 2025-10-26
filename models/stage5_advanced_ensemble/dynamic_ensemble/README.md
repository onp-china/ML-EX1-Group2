# 动态权重集成

## 概述

本目录包含动态权重集成方法的实现，根据模型预测的置信度或不确定性动态调整集成权重。

## 文件说明

- `dynamic_ensemble.py` - 动态权重集成模块

## 核心特性

- **多种权重策略**: 支持置信度、不确定性、均匀权重策略
- **动态调整**: 根据每个样本的预测情况调整权重
- **灵活配置**: 可自定义权重计算策略

## 权重策略

### 1. 置信度策略 (confidence)
- 基于每个模型预测的最高概率
- 置信度高的模型获得更高权重

### 2. 不确定性策略 (uncertainty)
- 基于预测概率的熵
- 不确定性低的模型获得更高权重

### 3. 均匀策略 (uniform)
- 所有模型权重相等
- 作为基线参考

## 使用方法

```python
from dynamic_ensemble import DynamicEnsemble

# 创建动态集成
ensemble = DynamicEnsemble(strategy='confidence', num_classes=2)

# 预测
predictions = ensemble.ensemble_predict(model_probas)
```

## 参数说明

- `strategy`: 权重策略 ('confidence', 'uncertainty', 'uniform')
- `num_classes`: 分类类别数量
