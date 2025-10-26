# 训练代码说明

> **各阶段模型训练脚本详解**

本目录包含了从基线到最终Stacking集成的所有训练代码，展示了模型架构的演进和训练策略的优化。

---

## 📁 训练脚本概览

### Stage 1: 特征融合革命 (57%→85%)

**暂无独立训练脚本** - ImprovedV2模型通过实验迭代获得

**关键创新**:
- 6种特征融合方式
- CBAM注意力机制
- 自适应加权融合

---

### Stage 2: 深度优化突破 (85%→89%)

#### `train_88_83_multi_seed.py` ⭐ **核心训练脚本**

**用途**: 训练ResNet系列模型，目标88.83%准确率

**关键特性**:
- ResNetCompareNet架构
- Focal Loss损失函数
- AdamW优化器
- 混合精度训练 (AMP)
- 早停机制

**模型配置**:
```python
# 架构参数
layers = [2, 2, 2]  # 或 [3, 3, 3]
feat_dim = 256
width_mult = 1.0

# 训练参数
optimizer = AdamW(lr=1e-3, weight_decay=5e-4)
criterion = FocalLoss(alpha=1.0, gamma=2.0)
scheduler = ReduceLROnPlateau(patience=3)
```

**使用方法**:
```bash
# 训练ResNet-Optimized-1.12 (最佳单模型)
python scripts/training/train_88_83_multi_seed.py --layers 3,3,3 --epochs 100

# 训练ResNet-Fusion
python scripts/training/train_88_83_multi_seed.py --layers 2,2,2 --use_fusion

# 训练ResNet-Optimized
python scripts/training/train_88_83_multi_seed.py --layers 2,2,2
```

---

### Stage 3: 多样性探索 (84%→88%)

#### `train_multi_seed_optimized.py` ⭐ **多种子训练**

**用途**: 使用不同随机种子训练模型，增加多样性

**关键特性**:
- 多种子训练 (42, 2023, 2024, 2025)
- 不同网络宽度探索
- 并行训练支持

**使用方法**:
```bash
# 训练所有种子
python scripts/training/train_multi_seed_optimized.py --seeds 42,2023,2024,2025

# 训练单个种子
python scripts/training/train_multi_seed_optimized.py --seeds 2025
```

#### `train_simple_multi_seed.py` **简化多种子训练**

**用途**: 轻量级多种子训练脚本

**特点**:
- 更简单的配置
- 快速实验
- 适合调试

#### `train_efficientnet_symmetric.py` **EfficientNet训练**

**用途**: 训练EfficientNet架构模型

**关键特性**:
- EfficientNet-B0架构
- 对称数据增强
- 标签平滑

**使用方法**:
```bash
python scripts/training/train_efficientnet_symmetric.py --epochs 50
```

---

### Stage 4: Stacking集成 (89%→93%)

#### `run_stacking.py` ⭐ **Stacking集成训练**

**用途**: 训练LightGBM元学习器，集成10个基础模型

**关键特性**:
- 10个异构基础模型
- LightGBM元学习器
- 5折交叉验证
- 多种基线对比

**使用方法**:
```bash
# 运行Stacking集成
python scripts/training/run_stacking.py

# 指定不同的基础模型
python scripts/training/run_stacking.py --models resnet_optimized_1.12,resnet_fusion
```

**输出**:
- 集成性能对比
- 5折CV结果
- 最终Stacking模型

---

## 🔧 训练环境配置

### 依赖包
```bash
pip install torch torchvision
pip install numpy scikit-learn
pip install lightgbm
pip install tqdm
```

### 硬件要求
- **GPU**: 推荐8GB+显存
- **内存**: 16GB+ RAM
- **存储**: 10GB+ 可用空间

### 数据准备
确保以下数据文件存在:
```
data/
├── train.npz      # 训练集
├── val.npz        # 验证集
└── test_public.npz # 测试集
```

---

## 📊 训练流程详解

### 1. 单模型训练流程

```python
# 1. 数据加载
dataset = PairNPZDataset('data/train.npz', is_train=True)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# 2. 模型创建
model = ResNetCompareNet(feat_dim=256, layers=[2,2,2])

# 3. 训练配置
optimizer = AdamW(model.parameters(), lr=1e-3)
criterion = FocalLoss(alpha=1.0, gamma=2.0)
scheduler = ReduceLROnPlateau(optimizer, patience=3)

# 4. 训练循环
for epoch in range(100):
    train_one_epoch(model, dataloader, optimizer, criterion)
    val_acc = validate(model, val_dataloader)
    scheduler.step(val_acc)
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pt')
```

### 2. 多种子训练流程

```python
# 1. 定义种子列表
seeds = [42, 2023, 2024, 2025]

# 2. 为每个种子训练模型
for seed in seeds:
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 训练模型
    model = train_model(seed=seed)
    
    # 保存模型
    save_model(model, f'models/seed_{seed}/')
```

### 3. Stacking集成流程

```python
# 1. 加载所有基础模型
models = load_all_base_models()

# 2. 获取预测结果
X = get_predictions(models, val_data)  # [N, num_models]

# 3. 训练元学习器
lgb_model = LGBMClassifier()
lgb_model.fit(X, y)

# 4. 集成预测
final_pred = lgb_model.predict_proba(X)[:, 1]
```

---

## 🎯 关键训练技巧

### 1. 损失函数选择

**BCEWithLogitsLoss** (基线):
```python
criterion = nn.BCEWithLogitsLoss()
```

**Focal Loss** (Stage 2+):
```python
criterion = FocalLoss(alpha=1.0, gamma=2.0)
# 关注难样本，降低简单样本权重
```

### 2. 优化器配置

**AdamW** (推荐):
```python
optimizer = AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=5e-4  # L2正则化
)
```

### 3. 学习率调度

**ReduceLROnPlateau**:
```python
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='max',      # 监控验证准确率
    factor=0.5,      # 学习率衰减因子
    patience=3       # 容忍轮数
)
```

### 4. 早停策略

```python
patience = 5
patience_counter = 0

if val_acc > best_acc:
    best_acc = val_acc
    patience_counter = 0
    torch.save(model.state_dict(), 'best_model.pt')
else:
    patience_counter += 1
    if patience_counter >= patience:
        print("Early stopping!")
        break
```

### 5. 混合精度训练

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

---

## 📈 性能监控

### 训练指标

**关键指标**:
- 训练损失 (Training Loss)
- 验证准确率 (Validation Accuracy)
- 学习率 (Learning Rate)
- 梯度范数 (Gradient Norm)

**监控工具**:
```python
# 使用tqdm显示进度
from tqdm import tqdm

for epoch in tqdm(range(epochs), desc="Training"):
    # 训练代码
    pass

# 使用wandb记录指标 (可选)
import wandb
wandb.log({"val_acc": val_acc, "train_loss": train_loss})
```

### 模型保存策略

**检查点保存**:
```python
# 每个epoch保存
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'best_val_acc': best_val_acc,
}, f'checkpoint_epoch_{epoch}.pt')

# 最佳模型保存
if val_acc > best_val_acc:
    torch.save(model.state_dict(), 'best_model.pt')
```

---

## 🚀 快速开始

### 1. 训练最佳单模型

```bash
cd mnist-demo
python scripts/training/train_88_83_multi_seed.py \
    --layers 3,3,3 \
    --epochs 100 \
    --batch_size 64 \
    --lr 1e-3
```

### 2. 训练多种子模型

```bash
python scripts/training/train_multi_seed_optimized.py \
    --seeds 42,2023,2024,2025 \
    --epochs 50
```

### 3. 运行Stacking集成

```bash
python scripts/training/run_stacking.py
```

---

## ⚠️ 注意事项

### 1. 路径配置

确保脚本中的路径正确:
```python
# 数据路径
data_path = 'data/train.npz'

# 模型保存路径
model_path = 'models/stage2_resnet_optimized/'
```

### 2. 设备选择

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
```

### 3. 内存管理

```python
# 清理GPU缓存
torch.cuda.empty_cache()

# 使用梯度检查点 (大模型)
model.gradient_checkpointing_enable()
```

### 4. 随机种子

```python
# 设置随机种子确保可复现性
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
```

---

## 📚 进阶技巧

### 1. 超参数搜索

```python
# 使用Optuna进行超参数优化
import optuna

def objective(trial):
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    
    # 训练模型
    model = train_model(lr=lr, weight_decay=weight_decay)
    return evaluate_model(model)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

### 2. 模型蒸馏

```python
# 使用教师模型指导学生模型
teacher_model = load_teacher_model()
student_model = create_student_model()

# 蒸馏损失
def distillation_loss(student_logits, teacher_logits, labels, temperature=3.0):
    soft_loss = F.kl_div(
        F.log_softmax(student_logits/temperature, dim=1),
        F.softmax(teacher_logits/temperature, dim=1),
        reduction='batchmean'
    )
    hard_loss = F.cross_entropy(student_logits, labels)
    return 0.7 * soft_loss + 0.3 * hard_loss
```

### 3. 数据增强

```python
# 自定义数据增强
class CustomAugmentation:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomResizedCrop(28, scale=(0.9, 1.0))
        ])
    
    def __call__(self, xa, xb):
        return self.transform(xa), self.transform(xb)
```

---

## 🎓 学习建议

### 1. 理解模型架构

- 研究 `src/models/simple_compare_cnn.py` 中的ResNetCompareNet
- 理解残差连接、注意力机制、特征融合

### 2. 分析训练过程

- 观察损失曲线和准确率变化
- 理解早停和学习率调度的作用

### 3. 实验不同配置

- 尝试不同的网络深度和宽度
- 测试不同的损失函数和优化器

### 4. 对比不同方法

- 单模型 vs 集成
- 不同架构的性能差异

---

**训练代码是理解模型架构和优化策略的关键！** 🚀

通过研究这些训练脚本，你可以深入了解：
- 模型是如何构建的
- 网络是如何连接的
- 训练策略是如何优化的
- 性能是如何提升的
