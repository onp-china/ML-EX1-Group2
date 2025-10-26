# Training History and Visualization Guide

本指南介绍如何使用新的训练历史记录功能来生成真实的模型性能可视化。

## 新功能概述

### ✅ 已实现的功能

1. **训练历史记录** - 记录每个epoch的详细指标
2. **TensorBoard支持** - 实时可视化训练过程
3. **真实学习曲线** - 基于实际训练数据生成图表
4. **多模型对比** - 比较不同模型的训练过程
5. **自动可视化** - 训练完成后自动生成图表

## 训练历史记录

### 记录的数据

每个epoch记录以下指标：
- **训练指标**: 损失、准确率、F1分数
- **验证指标**: 损失、准确率、F1分数
- **学习率**: 当前学习率
- **梯度范数**: 梯度大小
- **最佳性能**: 最佳验证准确率及对应epoch

### 输出文件

训练完成后，每个模型目录包含：
```
models/stage*/model_name/
├── model.pt                    # 最佳模型权重
├── metrics.json               # 最终性能指标
├── training_history.json      # 详细训练历史
├── learning_curves.png        # 学习曲线图
└── tensorboard/               # TensorBoard日志
    ├── events.out.tfevents.*
    └── ...
```

## 使用方法

### 1. 训练单个模型

```bash
# ImprovedV2模型
python scripts/training/train_stage1_improvedv2.py --epochs 50

# ResNet优化模型
python scripts/training/train_stage2_resnet_optimized.py --epochs 60

# FPN模型
python scripts/training/train_fpn_model.py --epochs 60
```

### 2. 批量训练（推荐）

```bash
# Windows
train_with_history.bat

# 选择训练选项：
# 1. Stage 1 - ImprovedV2
# 2. Stage 2 - ResNet Optimized  
# 3. FPN Model
# 4. All models
```

### 3. 查看TensorBoard

```bash
# 查看所有模型
python scripts/visualization/tensorboard_launcher.py --all

# 查看特定模型
python scripts/visualization/tensorboard_launcher.py --model improvedv2

# 自定义端口
python scripts/visualization/tensorboard_launcher.py --port 6007
```

### 4. 生成真实数据可视化

```bash
# 生成基于真实训练数据的可视化
python scripts/visualization/real_data_visualizer.py
```

## 可视化图表

### 1. 学习曲线 (Learning Curves)

**文件**: `learning_curves.png`

包含4个面板：
- **训练/验证损失**: 显示模型学习过程
- **训练/验证准确率**: 显示性能提升
- **训练/验证F1分数**: 显示分类质量
- **学习率调度**: 显示学习率变化

### 2. 性能对比 (Performance Comparison)

**文件**: `real_performance_comparison.png`

包含4个面板：
- **最佳验证准确率**: 模型性能排名
- **训练vs验证准确率**: 过拟合分析
- **收敛速度**: 训练效率对比
- **最佳epoch**: 性能峰值位置

### 3. 训练摘要表格 (Training Summary Table)

**文件**: `real_training_summary_table.png`

包含关键指标：
- 最佳验证准确率
- 最佳训练准确率
- 最佳epoch
- 总训练轮数
- 收敛epoch
- 过拟合程度

## TensorBoard功能

### 实时监控

训练过程中可以实时查看：
- 损失曲线
- 准确率曲线
- 学习率变化
- 梯度范数
- 模型对比

### 访问方式

1. 启动TensorBoard: `python scripts/visualization/tensorboard_launcher.py --all`
2. 打开浏览器: `http://localhost:6006`
3. 选择不同模型进行对比

## 技术细节

### 训练历史记录器

```python
from training_history import TrainingHistory

# 初始化
history = TrainingHistory(
    model_name='ModelName',
    output_dir='output/path',
    use_tensorboard=True
)

# 记录每个epoch
history.add_epoch(
    epoch=epoch_num,
    train_metrics={'loss': loss, 'accuracy': acc, 'f1': f1},
    val_metrics={'loss': val_loss, 'accuracy': val_acc, 'f1': val_f1},
    lr=learning_rate,
    grad_norm=gradient_norm
)

# 保存和可视化
history.save_history()
history.plot_learning_curves()
history.close()
```

### 数据格式

训练历史JSON格式：
```json
{
  "model_name": "ModelName",
  "created_at": "2024-01-01T00:00:00",
  "total_epochs": 50,
  "best_metrics": {
    "epoch": 45,
    "train_acc": 0.9234,
    "val_acc": 0.8875,
    "val_f1": 0.8874
  },
  "epochs": [1, 2, 3, ...],
  "train_losses": [0.5, 0.4, 0.3, ...],
  "val_accuracies": [0.6, 0.7, 0.8, ...],
  "learning_rates": [0.001, 0.001, 0.0005, ...]
}
```

## 优势

### 相比模拟数据

1. **真实性**: 基于实际训练过程
2. **准确性**: 反映真实的学习模式
3. **可重现性**: 可以复现训练过程
4. **详细性**: 包含更多训练细节

### 学术价值

1. **论文发表**: 真实的学习曲线更有说服力
2. **实验分析**: 可以分析训练动态
3. **模型对比**: 客观比较不同模型
4. **问题诊断**: 识别训练问题

## 故障排除

### 常见问题

1. **TensorBoard无法启动**
   ```bash
   pip install tensorboard
   ```

2. **没有训练历史数据**
   - 确保使用新的训练脚本
   - 检查模型目录结构

3. **可视化生成失败**
   - 检查依赖包: `pip install matplotlib seaborn pandas`
   - 确保有训练历史文件

### 依赖包

```bash
pip install torch torchvision
pip install tensorboard
pip install matplotlib seaborn pandas
pip install scikit-learn tqdm
```

## 下一步

1. **训练所有模型** - 使用 `train_with_history.bat`
2. **查看TensorBoard** - 实时监控训练过程
3. **生成可视化** - 创建论文级别的图表
4. **分析结果** - 比较不同模型的性能

这些真实的学习曲线和性能数据将大大提升您论文的可信度和学术价值！
