# 项目信息

## 📦 文件夹说明

这是MNIST数字比较项目的**演示/复现文件夹**，包含了从基线57%到最终93%的完整模型进化历程。

---

## 🎯 设计目标

1. **统一结构**: 所有模型按进化阶段组织
2. **易于测试**: 提供一键测试脚本
3. **完整文档**: 从快速开始到技术细节
4. **可复现**: 所有模型权重和配置都已包含

---

## 📁 核心文件说明

### 数据集 (data/)

| 文件 | 大小 | 说明 |
|------|------|------|
| train.npz | ~5.5MB | 训练集 (10000样本) |
| val.npz | ~1.1MB | 验证集 (2000样本) |
| test_public.npz | ~1.1MB | 测试集 (2000样本) |

**数据格式**: 
- 图像: 28×56 (两个28×28拼接)
- 标签: 0(不同) / 1(相同)

---

### 模型权重 (models/)

**Stage 1: 特征融合革命**
- `stage1_improvedv2/`: ImprovedV2模型 (85.28%)

**Stage 2: 深度优化突破**
- `stage2_resnet_optimized/resnet_optimized_1.12/`: 最佳单模型 (88.75%) ⭐
- `stage2_resnet_optimized/resnet_fusion/`: Fusion版本 (87.92%)
- `stage2_resnet_optimized/resnet_optimized/`: 标准版本 (87.80%)

**Stage 3: 多样性探索**
- `stage3_multi_seed/seed_*/`: 多种子训练模型 (86-87%)
- `stage3_multi_seed/fpn_model/`: FPN架构 (87.30%)
- `stage3_multi_seed/resnet_fusion_seed*/`: Fusion多种子 (84-85%)

**Stage 4: Stacking集成**
- `stage4_stacking/`: Stacking结果和元学习器 (93.09%)

---

### 源代码 (src/)

| 文件 | 功能 |
|------|------|
| `data_loader.py` | 数据加载、预处理、增强 |
| `model_loader.py` | 模型加载、权重管理 |
| `stacking_ensemble.py` | Stacking集成实现 |
| `models/simple_compare_cnn.py` | ResNet架构定义 |
| `models/fpn_architecture_v2.py` | FPN架构定义 |

---

### 配置文件 (configs/)

**model_registry.json**: 模型注册表
- 包含所有模型的元信息
- 路径、准确率、架构、参数等
- 测试脚本依赖此文件

---

### 测试脚本 (scripts/)

| 脚本 | 功能 | 用法 |
|------|------|------|
| `test_single_model.py` | 测试单个模型 | `python scripts/test_single_model.py --model resnet_optimized_1.12` |
| `test_all_models.py` | 测试所有模型 | `python scripts/test_all_models.py` |

---

### 快速测试 (quick_test.bat)

一键测试脚本，依次执行：
1. 测试最佳单模型
2. 测试ImprovedV2
3. 测试所有模型并生成对比报告

---

## 🔧 技术栈

| 组件 | 版本要求 |
|------|---------|
| Python | 3.8+ |
| PyTorch | 2.0+ |
| NumPy | 1.21+ |
| scikit-learn | 1.0+ |
| LightGBM | 3.3+ |
| CUDA | 11.0+ (可选) |

---

## 📊 性能指标

| 阶段 | 最佳模型 | 准确率 | 提升 |
|------|---------|--------|------|
| Stage 0 | Baseline | 57.11% | - |
| Stage 1 | ImprovedV2 | 85.28% | +28.17% |
| Stage 2 | ResNet-Opt | 88.75% | +3.47% |
| Stage 3 | Multi-seed | 87.39% | 架构多样化 |
| Stage 4 | Stacking | 93.09% | +4.34% |

**测试集性能**: 90.30%

---

## 🚀 快速使用

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 快速测试
```bash
# Windows
quick_test.bat

# 或者手动测试
python scripts/test_single_model.py --model resnet_optimized_1.12
```

### 3. 查看结果
- 终端输出: 准确率、F1等指标
- JSON文件: `outputs/predictions/*.json`
- 对比报告: `outputs/metrics/all_models_comparison.json`

---

## 📖 文档索引

1. **README.md**: 项目概览、快速开始、模型列表
2. **PRESENTATION_GUIDE.md**: 10分钟汇报演讲指南
3. **PROJECT_INFO.md**: 本文档，文件夹详细说明

---

## ⚙️ 自定义与扩展

### 添加新模型

1. 将模型文件放入对应阶段目录
2. 在 `configs/model_registry.json` 注册
3. 运行测试验证

### 修改测试参数

编辑脚本中的参数：
```python
# test_single_model.py
batch_size = 64  # 调整批次大小
device = 'cuda'  # 强制使用GPU/CPU
```

---

## 🔍 故障排查

### 模型加载失败
- 检查 `model_registry.json` 中的路径是否正确
- 确认 `.pt` 文件存在
- 查看 `metrics.json` 配置是否完整

### CUDA错误
- 安装对应版本的PyTorch: `torch.cuda.is_available()`
- 或者使用CPU模式（自动降级）

### 准确率异常
- 小幅差异（<1%）正常
- 大幅差异检查数据集是否正确

---

## 📈 性能优化建议

### 单模型使用
- 推荐: `resnet_optimized_1.12` (88.75%, 4.75M参数)
- 平衡性能和速度

### 追求极致
- 使用Stacking (93.09%)
- 需要加载10个模型，推理慢10倍

### 资源受限
- 使用 `resnet_optimized` (87.80%, 3.2M参数)
- 或 `seed_2023` (87.02%, 3.8M参数)

---

## 🎓 学习价值

本项目展示了：

1. **系统优化方法论**
   - 问题驱动的迭代
   - 数据支撑的决策
   - 渐进式提升策略

2. **深度学习技术栈**
   - ResNet残差网络
   - SE注意力机制
   - Focal Loss
   - 特征融合方法
   - 集成学习 (Stacking)

3. **工程实践经验**
   - 模型管理与版本控制
   - 可复现性设计
   - 性能监控与对比
   - 文档化与演示

---

## 📞 技术支持

遇到问题？
1. 查看 README.md 的"常见问题"章节
2. 检查环境配置: `pip list | grep torch`
3. 验证数据完整性: `ls data/`
4. 查看错误日志定位问题

---

**项目状态**: ✅ 完成  
**最后更新**: 2024年10月24日  
**维护者**: MNIST-Compare Team

