#!/usr/bin/env python3
"""
从保存的性能数据生成可视化图表
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 设置样式
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_performance_data():
    """加载保存的性能数据"""
    data_file = 'outputs/results/model_performance_data.json'
    if not os.path.exists(data_file):
        print(f"错误: 未找到性能数据文件 {data_file}")
        print("请先运行: python scripts/save_model_performance_data.py")
        return None
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"成功加载性能数据: {data_file}")
    return data

def create_performance_table(data):
    """创建性能表格"""
    print("\n" + "="*70)
    print("创建性能表格")
    print("="*70)
    
    # 准备数据
    rows = []
    for model in data['individual_models']:
        rows.append({
            'Model': model['name'],
            'Train Accuracy': f"{model['train_accuracy']:.4f}",
            'Val Accuracy': f"{model['val_accuracy']:.4f}",
            'Test Public Accuracy': f"{model['test_public_accuracy']:.4f}",
            'Parameters': f"{model['parameters']:,}" if model['parameters'] > 0 else 'N/A'
        })
    
    # 添加Stacking行
    stacking = data['stacking_ensemble']
    rows.append({
        'Model': 'Stacking Ensemble',
        'Train Accuracy': f"{stacking['train_accuracy']:.4f}",
        'Val Accuracy': f"{stacking['val_accuracy']:.4f}",
        'Test Public Accuracy': f"{stacking['test_public_accuracy']:.4f}",
        'Parameters': '10 models + LightGBM'
    })
    
    df = pd.DataFrame(rows)
    
    # 保存CSV
    output_csv = 'outputs/results/performance_table.csv'
    df.to_csv(output_csv, index=False)
    print(f"性能表格CSV已保存: {output_csv}")
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns,
                     cellLoc='center', loc='center',
                     colWidths=[0.25, 0.15, 0.15, 0.2, 0.25])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # 样式化表头
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 样式化Stacking行
    for i in range(len(df.columns)):
        table[(len(df), i)].set_facecolor('#FFF9C4')
        table[(len(df), i)].set_text_props(weight='bold')
    
    plt.title('Model Performance Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    output_img = 'outputs/visualizations/performance_table.png'
    os.makedirs(os.path.dirname(output_img), exist_ok=True)
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"性能表格图片已保存: {output_img}")
    plt.close()
    
    return df

def create_confusion_matrix(data):
    """创建混淆矩阵"""
    print("\n" + "="*70)
    print("创建混淆矩阵")
    print("="*70)
    
    cm = np.array(data['stacking_ensemble']['confusion_matrix'])
    accuracy = data['stacking_ensemble']['val_accuracy']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'],
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix on Validation Set\n(Stacking Ensemble)',
              fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # 添加准确率文本
    plt.text(1, -0.3, f'Accuracy: {accuracy:.4f}',
             ha='center', va='center', fontsize=12, fontweight='bold',
             transform=ax.transAxes)
    
    plt.tight_layout()
    
    output_path = 'outputs/visualizations/confusion_matrix.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"混淆矩阵已保存: {output_path}")
    plt.close()

def create_stacking_training_visualization(data):
    """创建Stacking训练过程可视化"""
    print("\n" + "="*70)
    print("创建Stacking训练可视化")
    print("="*70)
    
    cv_scores = data['stacking_ensemble']['cv_scores']
    stacking = data['stacking_ensemble']
    simple_avg = data['simple_average']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 图1: 交叉验证分数
    folds = [f'Fold {i+1}' for i in range(len(cv_scores))]
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(cv_scores)))
    
    bars = ax1.bar(folds, cv_scores, color=colors, edgecolor='black', linewidth=1.5)
    ax1.axhline(y=np.mean(cv_scores), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {np.mean(cv_scores):.4f}')
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Cross-Validation Fold', fontsize=12, fontweight='bold')
    ax1.set_title('Stacking Meta-Learner: 5-Fold Cross-Validation',
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0.85, 0.92])
    
    # 添加数值标签
    for bar, score in zip(bars, cv_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.4f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 图2: 性能对比
    datasets = ['Train', 'Validation', 'Test Public']
    stacking_accs = [
        stacking['train_accuracy'],
        stacking['val_accuracy'],
        stacking['test_public_accuracy']
    ]
    simple_accs = [
        simple_avg['train_accuracy'],
        simple_avg['val_accuracy'],
        simple_avg['test_public_accuracy']
    ]
    
    x = np.arange(len(datasets))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, simple_accs, width, label='Simple Average',
                    color='#FF9800', edgecolor='black', linewidth=1.5)
    bars2 = ax2.bar(x + width/2, stacking_accs, width, label='Stacking Ensemble',
                    color='#4CAF50', edgecolor='black', linewidth=1.5)
    
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax2.set_title('Stacking vs Simple Average Performance',
                  fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0.85, 1.0])
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = 'outputs/visualizations/stacking_training_process.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Stacking训练可视化已保存: {output_path}")
    plt.close()

def create_accuracy_comparison(data):
    """创建准确率对比图"""
    print("\n" + "="*70)
    print("创建准确率对比图")
    print("="*70)
    
    stacking = data['stacking_ensemble']
    
    # 柱状图
    datasets = ['Training', 'Validation', 'Test Public']
    accuracies = [
        stacking['train_accuracy'] * 100,
        stacking['val_accuracy'] * 100,
        stacking['test_public_accuracy'] * 100
    ]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#4CAF50', '#2196F3', '#FF9800']
    bars = ax.bar(datasets, accuracies, color=colors, edgecolor='black', linewidth=2, alpha=0.8)
    
    # 添加数值标签
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Dataset', fontsize=14, fontweight='bold')
    ax.set_title('Stacking Ensemble Performance Across Datasets',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim([85, 100])
    ax.axhline(y=90, color='red', linestyle='--', linewidth=2, alpha=0.7, label='90% Threshold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.tick_params(axis='both', labelsize=12)
    
    plt.tight_layout()
    
    output_path = 'outputs/visualizations/accuracy_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"准确率对比图已保存: {output_path}")
    plt.close()

def main():
    print("="*70)
    print("从保存的数据生成可视化图表")
    print("="*70)
    
    # 加载数据
    data = load_performance_data()
    if data is None:
        return
    
    # 生成所有可视化
    df = create_performance_table(data)
    create_confusion_matrix(data)
    create_stacking_training_visualization(data)
    create_accuracy_comparison(data)
    
    print("\n" + "="*70)
    print("所有可视化已生成完成！")
    print("="*70)
    print("\n生成的文件:")
    print("  - outputs/results/performance_table.csv")
    print("  - outputs/visualizations/performance_table.png")
    print("  - outputs/visualizations/confusion_matrix.png")
    print("  - outputs/visualizations/stacking_training_process.png")
    print("  - outputs/visualizations/accuracy_comparison.png")
    
    print("\n性能摘要:")
    print(f"  训练集: {data['stacking_ensemble']['train_accuracy']:.4f}")
    print(f"  验证集: {data['stacking_ensemble']['val_accuracy']:.4f}")
    print(f"  Test Public: {data['stacking_ensemble']['test_public_accuracy']:.4f}")

if __name__ == '__main__':
    main()

