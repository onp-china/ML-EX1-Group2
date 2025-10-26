# 🎨 模型迭代可视化

> **一键生成专业的模型进化历程可视化图表**

---

## 🚀 快速开始

### 一键可视化

```bash
# Windows
visualize_models.bat

# Linux/Mac
./visualize_models.sh

# Python
python scripts/visualization/quick_visualize.py
```

### 安装依赖

```bash
pip install -r requirements.txt
```

---

## 📊 生成的图表

### 1. 进化时间线
- **文件**: `model_evolution_timeline.png`
- **内容**: 5个阶段的完整进化历程
- **用途**: 整体概述，适合PPT首页

### 2. 性能曲线
- **文件**: `performance_evolution_curve.png`
- **内容**: 性能提升曲线 + 改进幅度柱状图
- **用途**: 数据展示，适合技术报告

### 3. 模型架构对比
- **文件**: `model_architecture_comparison.png`
- **内容**: 4个主要阶段的架构对比
- **用途**: 技术细节，适合学术论文

### 4. 集成学习可视化
- **文件**: `stacking_ensemble_architecture.png`
- **内容**: 10个基础模型到最终集成的流程
- **用途**: 最终方案展示，适合演示

### 5. 技术雷达图
- **文件**: `technology_radar_chart.png`
- **内容**: 8个维度的技术能力对比
- **用途**: 技术对比，适合分析报告

---

## 🎯 使用场景

### 学术报告
- 进化时间线 → 整体概述
- 性能曲线 → 数据支撑
- 技术雷达图 → 技术对比

### 技术演示
- 模型架构对比 → 技术细节
- 集成学习可视化 → 最终方案
- 性能曲线 → 效果展示

### 论文插图
- 性能曲线 (Figure 1)
- 模型架构对比 (Figure 2)
- 集成学习可视化 (Figure 3)

---

## 🔧 自定义选项

### 创建特定图表

```bash
# 只创建时间线
python scripts/visualization/create_visualizations.py --timeline

# 只创建性能曲线
python scripts/visualization/create_visualizations.py --performance

# 创建多个图表
python scripts/visualization/create_visualizations.py --timeline --performance --ensemble
```

### 自定义输出目录

```bash
python scripts/visualization/create_visualizations.py --output_dir my_charts
```

---

## 📁 输出文件

```
outputs/visualizations/
├── model_evolution_timeline.png      # 进化时间线
├── performance_evolution_curve.png   # 性能曲线
├── model_architecture_comparison.png # 架构对比
├── stacking_ensemble_architecture.png # 集成学习
├── technology_radar_chart.png        # 技术雷达图
└── visualization_info.json          # 可视化信息
```

---

## ⚠️ 注意事项

1. **依赖要求**: 需要安装 matplotlib 和 seaborn
2. **中文显示**: 自动配置中文字体支持
3. **文件格式**: 输出PNG格式，300 DPI分辨率
4. **内存使用**: 建议4GB+内存用于图表生成

---

## 🎉 特色功能

- ✅ **一键生成**: 一个命令创建所有图表
- ✅ **专业质量**: 300 DPI高分辨率输出
- ✅ **中文支持**: 完美支持中文标签和标题
- ✅ **多种图表**: 5种不同类型的可视化
- ✅ **自定义选项**: 灵活的输出配置
- ✅ **详细文档**: 完整的使用指南

---

**开始创建你的专业可视化图表吧！** 🎨

