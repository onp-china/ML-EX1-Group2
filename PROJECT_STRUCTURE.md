# Project Structure

## 📁 Root Directory (Clean & Organized)

```
mnist-demo/
├── README.md                    # 主文档（环境设置、训练、预测）
├── requirements.txt             # Python依赖
├── pred_private.csv            # 最终预测文件 ✅
├── check_submission.py         # 格式验证脚本
├── 提交需求.md                  # 提交要求
├── 最终结果总结.md               # 最终结果总结
│
├── src/                        # 源代码 ✅
│   ├── data_loader.py          # 数据加载
│   ├── model_loader.py         # 模型加载
│   ├── stacking_ensemble.py    # Stacking集成
│   ├── augmentation.py         # 数据增强
│   ├── mc_dropout.py           # MC Dropout
│   ├── dynamic_ensemble.py     # 动态集成
│   ├── two_level_stacking.py   # 两层Stacking
│   └── models/                 # 模型架构
│       ├── simple_compare_cnn.py    # ResNet模型
│       └── fpn_architecture_v2.py   # FPN模型
│
├── data/                       # 数据集
│   ├── train.npz              # 训练集 (50,000)
│   ├── val.npz                # 验证集 (10,000)
│   ├── test_public.npz        # 公开测试集 (2,000)
│   ├── test_public_labels.csv # 公开测试集标签
│   └── test_private.npz       # 私有测试集 (8,000)
│
├── models/                     # 训练好的模型 (10个)
│   ├── stage2_resnet_optimized/
│   │   ├── resnet_optimized_1.12/  # 88.75%
│   │   ├── resnet_fusion/          # 87.92%
│   │   └── resnet_optimized/       # 87.80%
│   └── stage3_multi_seed/
│       ├── seed_2025/              # 87.39%
│       ├── seed_2023/              # 87.02%
│       ├── seed_2024/              # 86.71%
│       ├── fpn_model/              # 86.58%
│       ├── resnet_fusion_seed42/   # 85.38%
│       ├── resnet_fusion_seed456/  # 84.39%
│       └── resnet_fusion_seed123/  # 84.36%
│
├── scripts/                    # 脚本
│   ├── generate_private_predictions.py  # 生成预测
│   ├── evaluate_10models_stacking.py    # 评估性能
│   ├── create_final_visualizations.py   # 创建可视化
│   └── training/               # 训练脚本
│       ├── train_stage2_resnet_optimized.py
│       └── train_stage3_multi_seed.py
│
├── outputs/                    # 输出结果
│   ├── results/
│   │   ├── performance_table.csv       # 性能表格
│   │   └── 10models_stacking_evaluation.json
│   └── visualizations/         # 可视化图表 ✅
│       ├── performance_table.png       # 性能对比
│       ├── confusion_matrix.png        # 混淆矩阵
│       └── stacking_training_process.png  # 训练过程
│
├── docs/                       # 文档（已整理）
│   ├── SUBMISSION_CHECKLIST.md        # 提交清单
│   ├── FINAL_RESULTS_REPORT.md        # 最终报告
│   ├── TRAINING_GUIDE.md              # 训练指南
│   └── ...                            # 其他文档
│
└── configs/                    # 配置文件
    └── model_registry.json     # 模型配置
```

## 🎯 提交必需文件

### 核心文件（根目录）
1. ✅ `src/` - 源代码目录
2. ✅ `requirements.txt` - 依赖文件
3. ✅ `README.md` - 完整文档
4. ✅ `pred_private.csv` - 预测文件

### 可视化文件
5. ✅ `outputs/visualizations/performance_table.png`
6. ✅ `outputs/visualizations/confusion_matrix.png`
7. ✅ `outputs/visualizations/stacking_training_process.png`

## 📊 关键性能指标

- **训练集**: 95.67%
- **验证集**: 90.76%
- **Test Public**: 90.00% 🎯

## 🚀 快速使用

### 生成预测
```bash
python scripts/generate_private_predictions.py
```

### 验证格式
```bash
python check_submission.py --data_dir data --pred pred_private.csv --test_file test_private.npz
```

### 评估性能
```bash
python scripts/evaluate_10models_stacking.py
```

### 创建可视化
```bash
python scripts/create_final_visualizations.py
```

## 📝 文档说明

所有详细文档已整理到 `docs/` 目录：
- 提交清单
- 训练指南
- 最终报告
- 验证清单
- 等等...

---

**状态**: ✅ 准备就绪，可以提交
**准确率**: 90.00% (Test Public)

