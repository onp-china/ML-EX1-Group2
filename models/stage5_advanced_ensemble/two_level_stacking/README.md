# 两层Stacking + 动态权重集成

## 概述

本目录包含两层Stacking与动态权重结合的集成方法实现。

## 文件说明

- `two_level_stacking.py` - 基础两层Stacking模块
- `two_level_dynamic_stacking.py` - 两层Stacking + 动态权重实现

## 核心特性

- **分层集成**: 第一层使用分组的元学习器，第二层使用最终元学习器
- **动态权重**: 根据预测置信度动态调整权重
- **强泛化能力**: 在测试集上表现最佳 (0.9240)
- **模型分组**: 自动将基础模型分组处理

## 架构设计

```
基础模型预测 → 第一层元学习器(分组) → 动态权重调整 → 第二层元学习器 → 最终预测
```

## 性能指标

- **验证集准确率**: 0.8890
- **测试集准确率**: 0.9240 (最佳)
- **平均置信度**: 0.8242
- **模型分组数**: 4组

## 使用方法

```python
from two_level_dynamic_stacking import TwoLevelDynamicStacking

# 创建两层Stacking
stacking = TwoLevelDynamicStacking(
    dynamic_weight_strategy='confidence',
    random_state=42
)

# 训练
stacking.fit(X_train, y_train)

# 预测
predictions = stacking.predict(X_test)
```

## 参数说明

- `dynamic_weight_strategy`: 动态权重策略 ('confidence', 'uncertainty', 'uniform')
- `meta_learner_class_1`: 第一层元学习器类
- `meta_learner_class_2`: 第二层元学习器类
- `random_state`: 随机种子
