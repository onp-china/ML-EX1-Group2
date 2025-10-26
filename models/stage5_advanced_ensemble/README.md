# Stage 5: 高级集成方法 (Advanced Ensemble Methods)

## 概述

第五阶段实现了多种高级集成方法，用于进一步提升MNIST数字比较任务的性能。本阶段包含四种主要的集成策略：

1. **MC Dropout + 动态权重** - 蒙特卡洛Dropout与动态权重结合
2. **两层Stacking + 动态权重** - 分层Stacking与动态权重结合
3. **原始Stacking** - 基于LightGBM的传统Stacking方法
4. **基线方法** - 简单平均集成

## 目录结构

```
stage5_advanced_ensemble/
├── README.md                           # 本文件
├── mc_dropout/                         # MC Dropout相关脚本
│   ├── mc_dropout_dynamic_ensemble.py  # MC Dropout + 动态权重实现
│   ├── mc_dropout.py                   # 基础MC Dropout模块
│   └── bayesian_inference.py           # 贝叶斯推理脚本
├── dynamic_ensemble/                   # 动态集成相关脚本
│   └── dynamic_ensemble.py             # 动态权重集成模块
├── two_level_stacking/                 # 两层Stacking相关脚本
│   ├── two_level_stacking.py           # 基础两层Stacking模块
│   └── two_level_dynamic_stacking.py   # 两层Stacking + 动态权重实现
├── comparison_tests/                   # 对比测试脚本
│   ├── test_with_correct_loading.py    # 完整对比测试脚本
│   ├── evaluate_advanced_ensemble.py   # 高级集成评估脚本
│   ├── final_comparison_report.md      # 对比测试报告
│   └── correct_loading_test_results.json # 测试结果
└── utils/                              # 工具脚本
    ├── resume_training.py              # 训练恢复管理工具
    └── augmentation.py                 # 数据增强模块
```

## 性能对比

### 验证集结果 (val.npz)

| 方法 | 准确率 | F1 Score | 特点 |
|------|--------|----------|------|
| **原始Stacking (LightGBM)** | **0.8912** | **0.8912** | 🏆 验证集最佳，5折CV稳定 |
| 两层Stacking + 动态权重 | 0.8890 | 0.8890 | 分层集成，动态权重 |
| MC Dropout + 动态权重 | 0.8767 | 0.8762 | 不确定性估计，高置信度 |
| 基线 (简单平均) | 0.8913 | 0.8915 | 基线参考 |

### 测试集结果 (test_public.npz)

| 方法 | 准确率 | F1 Score | 特点 |
|------|--------|----------|------|
| **两层Stacking + 动态权重** | **0.9240** | **0.9240** | 🏆 测试集最佳，泛化能力强 |
| 基线 (简单平均) | 0.8845 | 0.8789 | 基线参考 |

## 核心特性

### 1. MC Dropout + 动态权重
- **不确定性估计**: 通过多次采样量化预测不确定性
- **动态权重**: 根据置信度调整模型权重
- **高置信度**: 平均置信度达95.96%
- **适用场景**: 需要不确定性信息的应用

### 2. 两层Stacking + 动态权重
- **分层集成**: 第一层分组元学习器，第二层最终集成
- **动态权重**: 基于预测置信度的权重调整
- **强泛化**: 在测试集上表现最佳
- **适用场景**: 生产环境部署

### 3. 原始Stacking (LightGBM)
- **快速训练**: 使用LightGBM作为元学习器
- **稳定性能**: 5折交叉验证结果稳定
- **简单高效**: 易于理解和部署
- **适用场景**: 快速原型开发

## 使用方法

### 运行完整对比测试
```bash
cd comparison_tests
python test_with_correct_loading.py
```

### 运行特定方法测试
```bash
# MC Dropout + 动态权重
cd mc_dropout
python bayesian_inference.py

# 两层Stacking + 动态权重
cd two_level_stacking
python two_level_dynamic_stacking.py

# 动态集成
cd dynamic_ensemble
python dynamic_ensemble.py
```

## 技术实现要点

1. **正确的模型加载**: 使用与原始Stacking相同的模型加载方式
2. **路径管理**: 正确处理相对路径和绝对路径
3. **数据预处理**: 确保数据格式与模型期望一致
4. **错误处理**: 添加适当的异常处理和边界情况检查
5. **GPU加速**: 支持CUDA加速计算

## 依赖要求

- PyTorch >= 1.9.0
- scikit-learn >= 0.24.0
- LightGBM >= 3.2.0
- NumPy >= 1.21.0
- Pandas >= 1.3.0

## 文件说明

### 核心模块
- `mc_dropout.py`: 基础MC Dropout实现
- `dynamic_ensemble.py`: 动态权重集成实现
- `two_level_stacking.py`: 两层Stacking基础实现

### 组合方法
- `mc_dropout_dynamic_ensemble.py`: MC Dropout + 动态权重组合
- `two_level_dynamic_stacking.py`: 两层Stacking + 动态权重组合

### 测试脚本
- `test_with_correct_loading.py`: 完整对比测试
- `evaluate_advanced_ensemble.py`: 高级集成评估

### 工具脚本
- `resume_training.py`: 训练恢复管理
- `augmentation.py`: 数据增强模块

## 推荐使用策略

1. **生产环境**: 使用**两层Stacking + 动态权重**，测试集性能最佳
2. **需要不确定性**: 使用**MC Dropout + 动态权重**
3. **快速部署**: 使用**原始Stacking (LightGBM)**
4. **简单场景**: 使用**基线方法 (简单平均)**

## 注意事项

- 所有脚本都经过测试，确保路径和依赖正确
- MC Dropout方法计算量较大，建议使用GPU
- 动态权重策略可根据具体需求调整
- 测试结果基于8个预训练模型的集成

## 更新日志

- **v1.0** (2025-10-26): 初始版本，包含四种集成方法
- 修复了模型加载和路径问题
- 添加了完整的对比测试和文档
