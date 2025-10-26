# 对比测试脚本

## 概述

本目录包含所有高级集成方法的对比测试脚本和结果。

## 文件说明

- `test_with_correct_loading.py` - 完整对比测试脚本
- `evaluate_advanced_ensemble.py` - 高级集成评估脚本
- `final_comparison_report.md` - 详细对比测试报告
- `correct_loading_test_results.json` - 测试结果数据

## 测试内容

### 1. 基线方法测试
- 简单平均集成
- 加权平均集成
- 多数投票集成

### 2. MC Dropout + 动态权重测试
- 不确定性估计
- 动态权重调整
- 贝叶斯集成

### 3. 两层Stacking + 动态权重测试
- 分层集成架构
- 动态权重策略
- 模型分组处理

### 4. 原始Stacking测试
- LightGBM元学习器
- 5折交叉验证
- 传统Stacking方法

## 运行测试

```bash
# 运行完整对比测试
python test_with_correct_loading.py

# 运行高级集成评估
python evaluate_advanced_ensemble.py
```

## 测试结果

### 验证集性能排名
1. **原始Stacking (LightGBM)**: 0.8912
2. **两层Stacking + 动态权重**: 0.8890
3. **MC Dropout + 动态权重**: 0.8767
4. **基线 (简单平均)**: 0.8913

### 测试集性能排名
1. **两层Stacking + 动态权重**: 0.9240 (最佳)
2. **基线 (简单平均)**: 0.8845

## 关键发现

1. **泛化能力**: 两层Stacking在测试集上表现最佳
2. **稳定性**: 原始Stacking在验证集上最稳定
3. **不确定性**: MC Dropout提供不确定性估计
4. **效率**: 基线方法简单高效

## 注意事项

- 所有测试都使用相同的8个预训练模型
- 测试结果基于正确的模型加载方式
- 路径和依赖已正确配置
