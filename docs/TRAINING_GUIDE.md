# 训练指南

> **从零开始训练所有模型 - 完整复现指南**

本指南将带你从零开始训练所有阶段的模型，完整复现从57%到93%的优化历程。

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 验证环境
python -c "import torch; print('PyTorch版本:', torch.__version__)"
python -c "import lightgbm; print('LightGBM版本:', lightgbm.__version__)"
```

### 2. 数据准备

确保数据文件存在:
```bash
ls data/
# 应该看到: train.npz, val.npz, test_public.npz
```

### 3. 一键训练所有模型

```bash
# 训练所有阶段模型
bash train_all_stages.sh

# 或者分阶段训练
python scripts/training/train_stage1_improvedv2.py
python scripts/training/train_stage2_resnet_optimized.py --layers 3,3,3
python scripts/training/train_stage3_multi_seed.py --seeds 42,2023,2024,2025
python scripts/training/run_stacking.py
```

---

## 📊 分阶段训练详解

### Stage 1: 特征融合革命 (57%→85%)

**目标**: 实现6头融合 + CBAM注意力机制

```bash
python scripts/training/train_stage1_improvedv2.py \
    --epochs 50 \
    --batch_size 64 \
    --lr 1e-3 \
    --feat_dim 256 \
    --layers 2,2,2
```

**关键创新**:
- 6种特征融合方式 (差值、拼接、乘积、余弦相似度、L2距离、残差)
- 自适应加权融合
- CBAM注意力机制

**预期结果**: 85.28% 验证准确率

---

### Stage 2: 深度优化突破 (85%→89%)

**目标**: 实现ResNet + Focal Loss + 混合精度训练

#### 2.1 训练最佳单模型 (ResNet-Optimized-1.12)

```bash
python scripts/training/train_stage2_resnet_optimized.py \
    --model_name resnet_optimized_1.12 \
    --layers 3,3,3 \
    --epochs 100 \
    --batch_size 64 \
    --lr 1e-3 \
    --width_mult 1.0
```

**关键特性**:
- ResNet[3,3,3] 深度网络
- Focal Loss (α=1.0, γ=2.0)
- 混合精度训练 (AMP)
- SE-Module通道注意力

**预期结果**: 88.75% 验证准确率

#### 2.2 训练ResNet-Fusion

```bash
python scripts/training/train_stage2_resnet_optimized.py \
    --model_name resnet_fusion \
    --layers 2,2,2 \
    --use_fusion \
    --epochs 80
```

**预期结果**: 87.92% 验证准确率

#### 2.3 训练ResNet-Optimized

```bash
python scripts/training/train_stage2_resnet_optimized.py \
    --model_name resnet_optimized \
    --layers 2,2,2 \
    --epochs 80
```

**预期结果**: 87.80% 验证准确率

---

### Stage 3: 多样性探索 (84%→88%)

**目标**: 实现多种子训练 + 架构多样性

#### 3.1 训练多种子ResNet模型

```bash
python scripts/training/train_stage3_multi_seed.py \
    --seeds 42,2023,2024,2025 \
    --epochs 50 \
    --batch_size 64 \
    --layers 2,2,2 \
    --width_mult 1.0
```

**预期结果**:
- Seed 2025: 87.39%
- Seed 2023: 87.02%
- Seed 2024: 86.71%
- Seed 42: 85.38%

#### 3.2 训练FPN多尺度模型

```bash
python scripts/training/train_stage3_multi_seed.py \
    --seeds 42 \
    --architecture fpn \
    --epochs 50 \
    --layers 2,2,2 \
    --width_mult 1.0
```

**预期结果**: 87.30% 验证准确率

#### 3.3 训练Fusion多种子模型

```bash
python scripts/training/train_stage3_multi_seed.py \
    --seeds 42,123,456 \
    --epochs 50 \
    --layers 2,2,2 \
    --use_fusion
```

**预期结果**:
- Seed 42: 85.38%
- Seed 123: 84.36%
- Seed 456: 84.39%

---

### Stage 4: Stacking集成 (89%→93%)

**目标**: 实现LightGBM元学习器集成

```bash
python scripts/training/run_stacking.py
```

**关键特性**:
- 10个异构基础模型
- LightGBM元学习器
- 5折交叉验证
- 多种基线对比

**预期结果**: 93.09% 验证准确率

---

## 🔧 训练配置详解

### 通用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 50-100 | 训练轮数 |
| `--batch_size` | 64 | 批次大小 |
| `--lr` | 1e-3 | 学习率 |
| `--weight_decay` | 5e-4 | 权重衰减 |
| `--patience` | 5-10 | 早停容忍度 |

### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--feat_dim` | 256 | 特征维度 |
| `--layers` | 2,2,2 | ResNet层数 |
| `--width_mult` | 1.0 | 宽度倍数 |
| `--use_bottleneck` | False | 使用Bottleneck块 |
| `--use_fusion` | False | 使用5头融合 |

### 训练策略

| 策略 | 说明 | 效果 |
|------|------|------|
| **Focal Loss** | 关注难样本 | +0.8% |
| **混合精度** | 加速训练 | 2x速度 |
| **梯度裁剪** | 稳定训练 | 避免梯度爆炸 |
| **标签平滑** | 正则化 | +0.2% |
| **早停** | 防止过拟合 | 节省时间 |

---

## 📈 训练监控

### 实时监控

```bash
# 使用tensorboard监控训练过程
tensorboard --logdir=outputs/logs

# 或者查看训练日志
tail -f outputs/logs/training.log
```

### 关键指标

**训练指标**:
- 训练损失 (Training Loss)
- 验证准确率 (Validation Accuracy)
- 学习率 (Learning Rate)
- 梯度范数 (Gradient Norm)

**性能指标**:
- 准确率 (Accuracy)
- F1分数 (F1-Score)
- 精确率 (Precision)
- 召回率 (Recall)

### 模型保存

**自动保存**:
- 最佳模型: `models/stage*/model_name/model.pt`
- 配置信息: `models/stage*/model_name/metrics.json`
- 训练日志: `outputs/logs/training.log`

**手动保存**:
```python
# 保存检查点
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'best_val_acc': best_val_acc,
}, 'checkpoint.pt')

# 加载检查点
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

---

## ⚡ 性能优化

### 1. 硬件优化

**GPU配置**:
```bash
# 设置GPU内存增长
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# 使用多GPU训练
python -m torch.distributed.launch --nproc_per_node=2 train_stage2_resnet_optimized.py
```

**内存优化**:
```python
# 梯度检查点
model.gradient_checkpointing_enable()

# 清理GPU缓存
torch.cuda.empty_cache()
```

### 2. 训练优化

**混合精度训练**:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    logits = model(xa, xb)
    loss = criterion(logits, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**数据并行**:
```python
# 单机多GPU
model = nn.DataParallel(model)

# 分布式训练
model = nn.parallel.DistributedDataParallel(model)
```

### 3. 超参数优化

**网格搜索**:
```python
# 学习率搜索
lrs = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
for lr in lrs:
    train_model(lr=lr)
```

**贝叶斯优化**:
```python
import optuna

def objective(trial):
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    return train_model(lr=lr, weight_decay=weight_decay)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

---

## 🐛 常见问题

### 1. 内存不足

**问题**: `CUDA out of memory`

**解决方案**:
```bash
# 减小批次大小
python train_stage2_resnet_optimized.py --batch_size 32

# 使用梯度累积
python train_stage2_resnet_optimized.py --gradient_accumulation_steps 2

# 清理GPU缓存
python -c "import torch; torch.cuda.empty_cache()"
```

### 2. 训练不收敛

**问题**: 损失不下降或准确率不提升

**解决方案**:
```bash
# 降低学习率
python train_stage2_resnet_optimized.py --lr 1e-4

# 增加权重衰减
python train_stage2_resnet_optimized.py --weight_decay 1e-3

# 检查数据加载
python -c "from data_loader import PairNPZDataset; print(len(PairNPZDataset('data/train.npz')))"
```

### 3. 模型性能差

**问题**: 准确率远低于预期

**解决方案**:
```bash
# 检查数据预处理
python -c "from data_loader import PairNPZDataset; dataset = PairNPZDataset('data/train.npz'); print(dataset[0])"

# 检查模型架构
python -c "from models.simple_compare_cnn import ResNetCompareNet; model = ResNetCompareNet(); print(model)"

# 检查损失函数
python -c "from torch.nn import BCEWithLogitsLoss; criterion = BCEWithLogitsLoss(); print(criterion)"
```

### 4. 训练速度慢

**问题**: 训练时间过长

**解决方案**:
```bash
# 使用混合精度
python train_stage2_resnet_optimized.py --use_amp

# 增加批次大小
python train_stage2_resnet_optimized.py --batch_size 128

# 使用多进程数据加载
python train_stage2_resnet_optimized.py --num_workers 4
```

---

## 📊 训练结果验证

### 1. 单模型验证

```bash
# 测试单个模型
python scripts/test_single_model.py --model resnet_optimized_1.12

# 预期输出
# 准确率 (Accuracy):  0.8875 (88.75%)
# F1分数 (F1-Score):  0.8872
```

### 2. 全模型对比

```bash
# 测试所有模型
python scripts/test_all_models.py

# 预期输出
# ResNet-Optimized-1.12: 88.75%
# ResNet-Fusion: 87.92%
# Multi-Seed 2025: 87.39%
# ...
```

### 3. Stacking集成验证

```bash
# 运行Stacking集成
python scripts/training/run_stacking.py

# 预期输出
# Stacking final accuracy: 0.9309 (93.09%)
```

---

## 🎯 训练目标

### 性能目标

| 阶段 | 目标准确率 | 实际达到 | 状态 |
|------|-----------|---------|------|
| Stage 1 | 85% | 85.28% | ✅ |
| Stage 2 | 88% | 88.75% | ✅ |
| Stage 3 | 87% | 87.39% | ✅ |
| Stage 4 | 90% | 93.09% | ✅ |

### 时间目标

| 模型 | 预期时间 | 硬件要求 |
|------|---------|---------|
| ImprovedV2 | 30分钟 | 8GB GPU |
| ResNet-Opt | 2小时 | 8GB GPU |
| Multi-Seed | 4小时 | 8GB GPU |
| Stacking | 10分钟 | 8GB GPU |

---

## 🚀 进阶技巧

### 1. 自定义损失函数

```python
class CustomLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets)
        focal_loss = self.alpha * (1 - torch.sigmoid(inputs)) ** 2 * bce_loss
        return focal_loss
```

### 2. 自定义数据增强

```python
class CustomAugmentation:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomResizedCrop(28, scale=(0.9, 1.0))
        ])
    
    def __call__(self, xa, xb):
        return self.transform(xa), self.transform(xb)
```

### 3. 模型蒸馏

```python
def distillation_loss(student_logits, teacher_logits, labels, temperature=3.0):
    soft_loss = F.kl_div(
        F.log_softmax(student_logits/temperature, dim=1),
        F.softmax(teacher_logits/temperature, dim=1),
        reduction='batchmean'
    )
    hard_loss = F.cross_entropy(student_logits, labels)
    return 0.7 * soft_loss + 0.3 * hard_loss
```

---

## 📚 学习资源

### 1. 理论基础

- **ResNet**: 残差网络原理
- **Focal Loss**: 难样本挖掘
- **混合精度**: AMP训练技术
- **集成学习**: Stacking方法

### 2. 实践技巧

- **超参数调优**: 网格搜索、贝叶斯优化
- **模型调试**: 梯度检查、激活可视化
- **性能分析**: 时间分析、内存分析

### 3. 工具推荐

- **监控**: TensorBoard, Weights & Biases
- **优化**: Optuna, Ray Tune
- **调试**: PyTorch Profiler, torchviz

---

**开始你的训练之旅吧！** 🚀

通过这个完整的训练指南，你可以：
- 理解每个阶段的技术创新
- 掌握现代深度学习训练技巧
- 复现从57%到93%的优化历程
- 为你的项目提供参考
