# 复现验证清单

> **确保完整复现实验的检查清单**

---

## ✅ 模型完整性检查

### 10个基础模型验证

根据你提供的表格，需要验证以下10个模型：

| ID | 模型名称 | 路径 | 状态 | 验证命令 |
|----|---------|------|------|---------|
| M1 | resnet_optimized_1.12 | `models/stage2_resnet_optimized/resnet_optimized_1.12/model.pt` | ⏳ | `python scripts/test_single_model.py --model resnet_optimized_1.12` |
| M2 | resnet_fusion | `models/stage2_resnet_optimized/resnet_fusion/model.pt` | ⏳ | `python scripts/test_single_model.py --model resnet_fusion` |
| M3 | resnet_optimized | `models/stage2_resnet_optimized/resnet_optimized/model.pt` | ⏳ | `python scripts/test_single_model.py --model resnet_optimized` |
| M4 | seed_2025 | `models/stage3_multi_seed/seed_2025/model.pt` | ⏳ | `python scripts/test_single_model.py --model seed_2025` |
| M5 | fpn_model | `models/stage3_multi_seed/fpn_model/model.pt` | ⏳ | `python scripts/test_single_model.py --model fpn_model` |
| M6 | seed_2023 | `models/stage3_multi_seed/seed_2023/model.pt` | ⏳ | `python scripts/test_single_model.py --model seed_2023` |
| M7 | seed_2024 | `models/stage3_multi_seed/seed_2024/model.pt` | ⏳ | `python scripts/test_single_model.py --model seed_2024` |
| M8 | resnet_fusion_seed42 | `models/stage3_multi_seed/resnet_fusion_seed42/model.pt` | ⏳ | `python scripts/test_single_model.py --model resnet_fusion_seed42` |
| M9 | resnet_fusion_seed123 | `models/stage3_multi_seed/resnet_fusion_seed123/model.pt` | ⏳ | `python scripts/test_single_model.py --model resnet_fusion_seed123` |
| M10 | resnet_fusion_seed456 | `models/stage3_multi_seed/resnet_fusion_seed456/model.pt` | ⏳ | `python scripts/test_single_model.py --model resnet_fusion_seed456` |

### 预期性能指标

| 模型 | 目标准确率 | 参数量 | 架构 |
|------|-----------|--------|------|
| M1 | 88.75% | 4.75M | ResNet [3,3,3] |
| M2 | 87.92% | 4.8M | ResNet [2,2,2] + 5头融合 |
| M3 | 87.80% | 3.2M | ResNet [2,2,2] |
| M4 | 87.39% | 4.8M | ResNet [2,2,2] |
| M5 | 87.30% | 5.1M | FPN |
| M6 | 87.02% | 3.8M | ResNet [2,2,2] |
| M7 | 86.71% | 6.5M | ResNet [2,2,2] |
| M8 | 85.38% | 4.8M | ResNet [2,2,2] + Fusion |
| M9 | 84.36% | 4.8M | ResNet [2,2,2] + Fusion |
| M10 | 84.39% | 4.8M | ResNet [2,2,2] + Fusion |

---

## 🔍 快速验证命令

### 1. 检查文件存在性

```bash
# 检查所有模型权重文件
find models/ -name "model.pt" | wc -l
# 应该输出: 10

# 检查具体文件
ls models/stage2_resnet_optimized/*/model.pt
ls models/stage3_multi_seed/*/model.pt
```

### 2. 测试所有模型

```bash
# 测试所有模型性能
python scripts/test_all_models.py

# 预期输出应该显示所有10个模型的准确率
```

### 3. 验证Stacking集成

```bash
# 运行Stacking集成
python scripts/training/run_stacking.py

# 预期输出: 93.09% 验证准确率
```

---

## 🚀 完整复现流程

### 步骤1: 环境准备

```bash
cd mnist-demo
pip install -r requirements.txt
```

### 步骤2: 删除现有权重 (可选)

```bash
# Windows
clear_weights.bat

# Linux/Mac
./clear_weights.sh
```

### 步骤3: 重新训练所有模型

```bash
# 方法1: 一键训练
train_all_stages.bat  # Windows
./train_all_stages.sh # Linux/Mac

# 方法2: 分阶段训练
python scripts/training/train_stage1_improvedv2.py
python scripts/training/train_stage2_resnet_optimized.py --layers 3,3,3
python scripts/training/train_stage3_multi_seed.py --seeds 42,2023,2024,2025
python scripts/training/run_stacking.py
```

### 步骤4: 验证结果

```bash
# 测试所有模型
python scripts/test_all_models.py

# 预期看到所有10个模型的准确率
```

---

## 📊 成功标志

### 模型训练成功标志

1. **所有10个模型权重文件存在**
2. **每个模型的准确率接近预期值** (±1%误差可接受)
3. **Stacking集成达到93.09%验证准确率**

### 预期输出示例

```bash
================================================================================
所有模型测试结果
================================================================================
模型                                阶段      预期      实际      差异        F1
--------------------------------------------------------------------------------
✅ ResNet-Optimized-1.12          Stage2  0.8875  0.8875  +0.0000  0.8872
✅ ResNet-Fusion                  Stage2  0.8792  0.8790  -0.0002  0.8788
✅ ResNet-Optimized               Stage2  0.8780  0.8780  +0.0000  0.8778
✅ Multi-Seed 2025                Stage3  0.8739  0.8741  +0.0002  0.8738
✅ FPN Multi-Scale                Stage3  0.8730  0.8730  +0.0000  0.8728
✅ Multi-Seed 2023                Stage3  0.8702  0.8702  +0.0000  0.8700
✅ Multi-Seed 2024                Stage3  0.8671  0.8671  +0.0000  0.8669
✅ ResNet-Fusion Seed 42          Stage3  0.8538  0.8538  +0.0000  0.8536
✅ ResNet-Fusion Seed 123         Stage3  0.8436  0.8436  +0.0000  0.8434
✅ ResNet-Fusion Seed 456         Stage3  0.8439  0.8439  +0.0000  0.8437
================================================================================

Stacking final accuracy: 0.9309 (93.09%)
```

---

## ⚠️ 常见问题解决

### 问题1: 模型文件不存在

**解决方案**:
```bash
# 检查目录结构
tree models/ -I "__pycache__"

# 重新创建目录
mkdir -p models/stage2_resnet_optimized/{resnet_optimized_1.12,resnet_fusion,resnet_optimized}
mkdir -p models/stage3_multi_seed/{seed_2025,fpn_model,seed_2023,seed_2024,resnet_fusion_seed42,resnet_fusion_seed123,resnet_fusion_seed456,seed_42}
```

### 问题2: 准确率不匹配

**可能原因**:
- 随机种子不同
- PyTorch版本不同
- 硬件差异

**解决方案**:
```bash
# 设置固定随机种子
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# 在训练脚本中添加
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

### 问题3: Stacking集成失败

**解决方案**:
```bash
# 检查所有基础模型
python -c "
import os
models = [
    'models/stage2_resnet_optimized/resnet_optimized_1.12/model.pt',
    'models/stage2_resnet_optimized/resnet_fusion/model.pt',
    'models/stage2_resnet_optimized/resnet_optimized/model.pt',
    'models/stage3_multi_seed/seed_2025/model.pt',
    'models/stage3_multi_seed/fpn_model/model.pt',
    'models/stage3_multi_seed/seed_2023/model.pt',
    'models/stage3_multi_seed/seed_2024/model.pt',
    'models/stage3_multi_seed/resnet_fusion_seed42/model.pt',
    'models/stage3_multi_seed/resnet_fusion_seed123/model.pt',
    'models/stage3_multi_seed/resnet_fusion_seed456/model.pt'
]
for model in models:
    print(f'{model}: {\"存在\" if os.path.exists(model) else \"不存在\"}')
"
```

---

## 🎯 最终确认

### 复现成功的标准

1. ✅ **所有10个基础模型权重文件存在**
2. ✅ **每个模型的准确率在预期范围内** (±1%)
3. ✅ **Stacking集成达到93.09%验证准确率**
4. ✅ **可以删除权重后重新训练**
5. ✅ **完整的训练代码可用**

### 验证命令

```bash
# 最终验证
echo "检查模型文件数量:"
find models/ -name "model.pt" | wc -l

echo "测试所有模型:"
python scripts/test_all_models.py

echo "运行Stacking集成:"
python scripts/training/run_stacking.py

echo "复现验证完成！"
```

---
