# 工具脚本

## 概述

本目录包含第五阶段使用的工具脚本和辅助模块。

## 文件说明

- `resume_training.py` - 训练恢复管理工具
- `augmentation.py` - 数据增强模块

## 工具功能

### 1. 训练恢复管理 (resume_training.py)

用于管理和查找训练检查点，支持：
- 列出所有可用检查点
- 按增强类型和随机种子过滤
- 获取最新检查点路径
- 检查点信息解析

#### 使用方法
```bash
# 列出所有检查点
python resume_training.py --list

# 获取最新检查点
python resume_training.py --latest

# 按类型过滤
python resume_training.py --list --augmentation_type randaugment
```

### 2. 数据增强 (augmentation.py)

提供多种数据增强方法：
- RandAugment
- AutoAugment
- 自定义增强策略

#### 使用方法
```python
from augmentation import get_augmentation_transforms

# RandAugment
aug = get_augmentation_transforms('randaugment', num_ops=2, magnitude=9)

# AutoAugment
aug = get_augmentation_transforms('autoaugment', policy='IMAGENET')

# 应用增强
aug_img1, aug_img2 = aug(img1, img2)
```

## 参数说明

### resume_training.py
- `--save_dir`: 检查点保存目录
- `--list`: 列出所有检查点
- `--latest`: 获取最新检查点
- `--augmentation_type`: 按增强类型过滤
- `--seed`: 按随机种子过滤

### augmentation.py
- `augmentation_type`: 增强类型 ('randaugment', 'autoaugment', 'none')
- `num_ops`: RandAugment操作数量
- `magnitude`: RandAugment操作强度
- `policy`: AutoAugment策略

## 依赖要求

- torch >= 1.9.0
- torchvision >= 0.10.0
- numpy >= 1.21.0
