# 模型分类整理

> **按模型类型和种子分类 - 相同模型不同种子算一个**

---

## 📊 模型分类表

根据你提供的表格，按照"相同模型使用不一样种子的算一个"的原则重新分类：

### 1. ResNet-Optimized系列 (2个变体)

| 分类 | 模型名称 | 架构 | 参数量 | 准确率 | 特点 | 种子 |
|------|---------|------|--------|--------|------|------|
| **ResNet-Optimized-1.12** | resnet_optimized_1.12 | ResNet [3,3,3] | 4.75M | 88.75% | Focal+AMP | - |
| **ResNet-Optimized** | resnet_optimized | ResNet [2,2,2] | 3.2M | 87.80% | 标准配置 | - |

**说明**: 这是同一个模型架构的两个不同版本，不是种子差异

### 2. ResNet-Fusion系列 (4个种子)

| 分类 | 模型名称 | 架构 | 参数量 | 准确率 | 特点 | 种子 |
|------|---------|------|--------|--------|------|------|
| **ResNet-Fusion** | resnet_fusion | ResNet [2,2,2] | 4.8M | 87.92% | 5头融合 | - |
| **ResNet-Fusion-Seed42** | resnet_fusion_seed42 | ResNet [2,2,2] | 4.8M | 85.38% | Fusion42 | 42 |
| **ResNet-Fusion-Seed123** | resnet_fusion_seed123 | ResNet [2,2,2] | 4.8M | 84.36% | Fusion123 | 123 |
| **ResNet-Fusion-Seed456** | resnet_fusion_seed456 | ResNet [2,2,2] | 4.8M | 84.39% | Fusion456 | 456 |

**说明**: 这是同一个ResNet-Fusion模型的不同种子版本

### 3. ResNet-Multi-Seed系列 (4个种子)

| 分类 | 模型名称 | 架构 | 参数量 | 准确率 | 特点 | 种子 |
|------|---------|------|--------|--------|------|------|
| **ResNet-Multi-Seed-2025** | seed_2025 | ResNet [2,2,2] | 4.8M | 87.39% | seed=2025 | 2025 |
| **ResNet-Multi-Seed-2023** | seed_2023 | ResNet [2,2,2] | 3.8M | 87.02% | 轻量化 | 2023 |
| **ResNet-Multi-Seed-2024** | seed_2024 | ResNet [2,2,2] | 6.5M | 86.71% | 宽化 | 2024 |
| **ResNet-Multi-Seed-42** | seed_42 | ResNet [2,2,2] | 4.8M | - | 基础版本 | 42 |

**说明**: 这是同一个ResNet模型的不同种子和配置版本

### 4. FPN系列 (1个)

| 分类 | 模型名称 | 架构 | 参数量 | 准确率 | 特点 | 种子 |
|------|---------|------|--------|--------|------|------|
| **FPN-Multi-Scale** | fpn_model | FPN | 5.1M | 87.30% | 多尺度 | - |

**说明**: 这是特征金字塔网络架构

---

## 🎯 重新分类后的模型类型

### 按模型架构分类 (4个主要类型)

1. **ResNet-Optimized系列** (2个变体)
   - ResNet-Optimized-1.12: 深度版本 [3,3,3]
   - ResNet-Optimized: 标准版本 [2,2,2]

2. **ResNet-Fusion系列** (4个种子)
   - 基础版本: resnet_fusion
   - 多种子版本: seed42, seed123, seed456

3. **ResNet-Multi-Seed系列** (4个种子)
   - 不同种子: 2025, 2023, 2024, 42
   - 不同配置: 标准、轻量化、宽化

4. **FPN系列** (1个)
   - FPN-Multi-Scale: 多尺度特征金字塔

### 按种子分类

| 种子 | 模型数量 | 模型列表 |
|------|---------|---------|
| **无种子** | 3个 | resnet_optimized_1.12, resnet_fusion, fpn_model |
| **种子42** | 2个 | resnet_fusion_seed42, seed_42 |
| **种子123** | 1个 | resnet_fusion_seed123 |
| **种子456** | 1个 | resnet_fusion_seed456 |
| **种子2023** | 1个 | seed_2023 |
| **种子2024** | 1个 | seed_2024 |
| **种子2025** | 1个 | seed_2025 |

---

## 📁 文件组织建议

基于重新分类，建议的文件组织：

```
models/
├── resnet_optimized/           # ResNet-Optimized系列
│   ├── resnet_optimized_1.12/ # 深度版本
│   └── resnet_optimized/      # 标准版本
│
├── resnet_fusion/             # ResNet-Fusion系列
│   ├── base/                  # 基础版本
│   ├── seed_42/               # 种子42
│   ├── seed_123/              # 种子123
│   └── seed_456/              # 种子456
│
├── resnet_multi_seed/         # ResNet-Multi-Seed系列
│   ├── seed_2025/             # 种子2025
│   ├── seed_2023/             # 种子2023 (轻量化)
│   ├── seed_2024/             # 种子2024 (宽化)
│   └── seed_42/               # 种子42 (基础)
│
└── fpn/                      # FPN系列
    └── multi_scale/           # 多尺度版本
```

---

## 🔄 训练脚本对应关系

### 按模型类型训练

1. **ResNet-Optimized系列**:
   ```bash
   # 深度版本
   python scripts/training/train_stage2_resnet_optimized.py --model_name resnet_optimized_1.12 --layers 3,3,3
   
   # 标准版本
   python scripts/training/train_stage2_resnet_optimized.py --model_name resnet_optimized --layers 2,2,2
   ```

2. **ResNet-Fusion系列**:
   ```bash
   # 基础版本
   python scripts/training/train_stage2_resnet_optimized.py --model_name resnet_fusion --use_fusion
   
   # 多种子版本
   python scripts/training/train_stage3_multi_seed.py --seeds 42,123,456 --use_fusion
   ```

3. **ResNet-Multi-Seed系列**:
   ```bash
   # 多种子训练
   python scripts/training/train_stage3_multi_seed.py --seeds 2025,2023,2024,42
   ```

4. **FPN系列**:
   ```bash
   # FPN训练
   python scripts/training/train_stage3_multi_seed.py --architecture fpn --seeds 42
   ```

---

## 📊 性能分析

### 按模型类型性能排序

| 排名 | 模型类型 | 最佳准确率 | 模型名称 |
|------|---------|-----------|---------|
| 1 | ResNet-Optimized-1.12 | 88.75% | resnet_optimized_1.12 |
| 2 | ResNet-Fusion | 87.92% | resnet_fusion |
| 3 | ResNet-Optimized | 87.80% | resnet_optimized |
| 4 | ResNet-Multi-Seed | 87.39% | seed_2025 |
| 5 | FPN | 87.30% | fpn_model |

### 按种子性能分析

| 种子 | 最佳准确率 | 模型名称 | 特点 |
|------|-----------|---------|------|
| 无种子 | 88.75% | resnet_optimized_1.12 | 深度网络 |
| 2025 | 87.39% | seed_2025 | 标准配置 |
| 2023 | 87.02% | seed_2023 | 轻量化 |
| 2024 | 86.71% | seed_2024 | 宽化 |
| 42 | 85.38% | resnet_fusion_seed42 | Fusion版本 |

---

## 🎯 总结

重新分类后，实际上有：

- **4个主要模型架构类型**
- **10个具体模型实例** (包含不同种子)
- **7个不同的随机种子**

这种分类方式更清晰地展示了：
1. 模型架构的多样性
2. 种子对性能的影响
3. 不同配置的效果差异

**关键发现**: ResNet-Optimized-1.12 (深度版本) 是性能最好的单模型，而ResNet-Fusion系列通过多种子训练提供了很好的多样性。
