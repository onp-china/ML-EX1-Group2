#!/usr/bin/env python3
"""
模型迭代可视化工具
展示从57%到93%的完整进化历程
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import seaborn as sns
from datetime import datetime
import pandas as pd

# 添加路径
sys.path.append('src')
sys.path.append('src/models')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ModelEvolutionVisualizer:
    """模型进化可视化器"""
    
    def __init__(self, output_dir='outputs/visualizations'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 模型进化数据
        self.evolution_data = self._load_evolution_data()
        
        # 颜色方案
        self.colors = {
            'stage1': '#FF6B6B',  # 红色 - 特征融合革命
            'stage2': '#4ECDC4',  # 青色 - 深度优化突破
            'stage3': '#45B7D1',  # 蓝色 - 多样性探索
            'stage4': '#96CEB4',  # 绿色 - Stacking集成
            'baseline': '#D3D3D3',  # 灰色 - 基线
            'text': '#2C3E50',    # 深蓝灰 - 文字
            'accent': '#E74C3C'   # 红色 - 强调
        }
    
    def _load_evolution_data(self):
        """加载模型进化数据"""
        return {
            'stages': [
                {
                    'name': 'Stage 0: Baseline',
                    'accuracy': 57.11,
                    'models': ['Simple CNN'],
                    'key_innovation': '基础CNN架构',
                    'color': self.colors['baseline']
                },
                {
                    'name': 'Stage 1: 特征融合革命',
                    'accuracy': 85.28,
                    'models': ['ImprovedV2 (seed 42)'],
                    'key_innovation': '6头融合 + CBAM注意力',
                    'improvement': '+28.17%',
                    'color': self.colors['stage1']
                },
                {
                    'name': 'Stage 2: 深度优化突破',
                    'accuracy': 88.75,
                    'models': ['ResNet-Optimized-1.12', 'ResNet-Fusion', 'ResNet-Optimized'],
                    'key_innovation': 'Focal Loss + 混合精度训练',
                    'improvement': '+3.47%',
                    'color': self.colors['stage2']
                },
                {
                    'name': 'Stage 3: 多样性探索',
                    'accuracy': 87.39,
                    'models': ['Multi-Seed 2025', 'FPN Multi-Scale', 'Multi-Seed 2023', 'Multi-Seed 2024'],
                    'key_innovation': '多种子 + 架构多样性',
                    'improvement': '架构多样化',
                    'color': self.colors['stage3']
                },
                {
                    'name': 'Stage 4: Stacking集成',
                    'accuracy': 93.09,
                    'models': ['Stacking Ensemble'],
                    'key_innovation': 'LightGBM元学习器',
                    'improvement': '+4.34%',
                    'color': self.colors['stage4']
                }
            ],
            'total_improvement': 35.98,
            'final_accuracy': 93.09,
            'baseline_accuracy': 57.11
        }
    
    def create_evolution_timeline(self):
        """创建进化时间线图"""
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # 设置背景
        ax.set_facecolor('#F8F9FA')
        
        # 绘制时间线
        y_positions = [0.8, 0.6, 0.4, 0.2, 0.05]
        
        for i, stage in enumerate(self.evolution_data['stages']):
            y = y_positions[i]
            
            # 绘制阶段框
            box = FancyBboxPatch(
                (0.05, y - 0.08), 0.9, 0.15,
                boxstyle="round,pad=0.02",
                facecolor=stage['color'],
                edgecolor='white',
                linewidth=2,
                alpha=0.8
            )
            ax.add_patch(box)
            
            # 添加阶段信息
            ax.text(0.1, y, stage['name'], fontsize=14, fontweight='bold', 
                   color='white', va='center')
            ax.text(0.1, y - 0.03, f"准确率: {stage['accuracy']:.2f}%", 
                   fontsize=12, color='white', va='center')
            
            if 'improvement' in stage:
                ax.text(0.1, y - 0.06, f"提升: {stage['improvement']}", 
                       fontsize=11, color='white', va='center')
            
            # 添加关键创新
            ax.text(0.5, y, stage['key_innovation'], fontsize=11, 
                   color='white', va='center', ha='left')
            
            # 添加模型列表
            models_text = ', '.join(stage['models'])
            ax.text(0.5, y - 0.03, f"模型: {models_text}", fontsize=10, 
                   color='white', va='center', ha='left')
            
            # 绘制连接线
            if i < len(self.evolution_data['stages']) - 1:
                ax.plot([0.5, 0.5], [y - 0.08, y_positions[i+1] + 0.07], 
                       color='#BDC3C7', linewidth=2, alpha=0.6)
        
        # 设置坐标轴
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # 添加标题
        ax.text(0.5, 0.95, 'MNIST数字比较模型进化历程', 
               fontsize=20, fontweight='bold', ha='center', color=self.colors['text'])
        ax.text(0.5, 0.92, f'从 {self.evolution_data["baseline_accuracy"]:.1f}% 到 {self.evolution_data["final_accuracy"]:.1f}% (+{self.evolution_data["total_improvement"]:.1f}%)', 
               fontsize=14, ha='center', color=self.colors['text'])
        
        # 添加图例
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor=self.colors['baseline'], label='基线'),
            plt.Rectangle((0, 0), 1, 1, facecolor=self.colors['stage1'], label='特征融合革命'),
            plt.Rectangle((0, 0), 1, 1, facecolor=self.colors['stage2'], label='深度优化突破'),
            plt.Rectangle((0, 0), 1, 1, facecolor=self.colors['stage3'], label='多样性探索'),
            plt.Rectangle((0, 0), 1, 1, facecolor=self.colors['stage4'], label='Stacking集成')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/model_evolution_timeline.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return f'{self.output_dir}/model_evolution_timeline.png'
    
    def create_performance_curve(self):
        """创建性能曲线图"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # 提取数据
        stages = [stage['name'] for stage in self.evolution_data['stages']]
        accuracies = [stage['accuracy'] for stage in self.evolution_data['stages']]
        colors = [stage['color'] for stage in self.evolution_data['stages']]
        
        # 上图：性能曲线
        ax1.plot(range(len(stages)), accuracies, 'o-', linewidth=3, markersize=8, 
                color=self.colors['accent'], markerfacecolor='white', markeredgewidth=2)
        
        # 填充区域
        ax1.fill_between(range(len(stages)), accuracies, alpha=0.3, color=self.colors['accent'])
        
        # 添加数值标签
        for i, (stage, acc) in enumerate(zip(stages, accuracies)):
            ax1.annotate(f'{acc:.1f}%', (i, acc), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=10, fontweight='bold')
        
        # 设置坐标轴
        ax1.set_xticks(range(len(stages)))
        ax1.set_xticklabels([stage.split(':')[0] for stage in stages], rotation=45)
        ax1.set_ylabel('准确率 (%)', fontsize=12)
        ax1.set_title('模型性能进化曲线', fontsize=16, fontweight='bold', pad=20)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(50, 100)
        
        # 添加改进幅度标注
        for i in range(1, len(accuracies)):
            improvement = accuracies[i] - accuracies[i-1]
            ax1.annotate(f'+{improvement:.1f}%', 
                        ((i-1+i)/2, (accuracies[i-1]+accuracies[i])/2),
                        ha='center', va='center', fontsize=9, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
        
        # 下图：改进幅度柱状图
        improvements = [0] + [accuracies[i] - accuracies[i-1] for i in range(1, len(accuracies))]
        bars = ax2.bar(range(len(stages)), improvements, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
        
        # 添加数值标签
        for i, (bar, imp) in enumerate(zip(bars, improvements)):
            if imp > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                        f'+{imp:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax2.set_xticks(range(len(stages)))
        ax2.set_xticklabels([stage.split(':')[0] for stage in stages], rotation=45)
        ax2.set_ylabel('改进幅度 (%)', fontsize=12)
        ax2.set_title('各阶段性能改进幅度', fontsize=16, fontweight='bold', pad=20)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/performance_evolution_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return f'{self.output_dir}/performance_evolution_curve.png'
    
    def create_model_architecture_diagram(self):
        """创建模型架构对比图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # 模型架构数据
        architectures = [
            {
                'name': 'Stage 1: ImprovedV2',
                'components': ['Input (28×56)', 'CNN Layers', '6 Fusion Heads', 'CBAM Attention', 'Output'],
                'accuracy': 85.28,
                'color': self.colors['stage1']
            },
            {
                'name': 'Stage 2: ResNet-Optimized',
                'components': ['Input (28×56)', 'ResNet [3,3,3]', 'SE-Module', 'Focal Loss', 'AMP Training', 'Output'],
                'accuracy': 88.75,
                'color': self.colors['stage2']
            },
            {
                'name': 'Stage 3: Multi-Seed',
                'components': ['Input (28×56)', 'ResNet [2,2,2]', 'Multiple Seeds', 'FPN Variants', 'Ensemble', 'Output'],
                'accuracy': 87.39,
                'color': self.colors['stage3']
            },
            {
                'name': 'Stage 4: Stacking',
                'components': ['10 Base Models', 'Feature Matrix', 'LightGBM', '5-Fold CV', 'Meta-Learning', 'Output'],
                'accuracy': 93.09,
                'color': self.colors['stage4']
            }
        ]
        
        for i, arch in enumerate(architectures):
            ax = axes[i]
            
            # 绘制架构框
            y_positions = np.linspace(0.1, 0.9, len(arch['components']))
            
            for j, component in enumerate(arch['components']):
                # 绘制组件框
                box = FancyBboxPatch(
                    (0.1, y_positions[j] - 0.08), 0.8, 0.15,
                    boxstyle="round,pad=0.02",
                    facecolor=arch['color'],
                    edgecolor='white',
                    linewidth=2,
                    alpha=0.7
                )
                ax.add_patch(box)
                
                # 添加组件名称
                ax.text(0.5, y_positions[j], component, fontsize=10, 
                       ha='center', va='center', color='white', fontweight='bold')
                
                # 绘制连接线
                if j < len(arch['components']) - 1:
                    ax.plot([0.5, 0.5], [y_positions[j] - 0.08, y_positions[j+1] + 0.07], 
                           color='#BDC3C7', linewidth=2, alpha=0.6)
            
            # 设置坐标轴
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            # 添加标题
            ax.text(0.5, 0.95, f"{arch['name']}\n准确率: {arch['accuracy']:.2f}%", 
                   fontsize=12, ha='center', va='top', fontweight='bold', color=self.colors['text'])
        
        plt.suptitle('各阶段模型架构对比', fontsize=18, fontweight='bold', y=0.95)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/model_architecture_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return f'{self.output_dir}/model_architecture_comparison.png'
    
    def create_ensemble_visualization(self):
        """创建集成学习可视化"""
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # 基础模型数据
        base_models = [
            {'name': 'ResNet-Opt-1.12', 'acc': 88.75, 'color': '#E74C3C'},
            {'name': 'ResNet-Fusion', 'acc': 87.92, 'color': '#E67E22'},
            {'name': 'ResNet-Opt', 'acc': 87.80, 'color': '#F39C12'},
            {'name': 'Seed-2025', 'acc': 87.39, 'color': '#F1C40F'},
            {'name': 'FPN-Model', 'acc': 87.30, 'color': '#2ECC71'},
            {'name': 'Seed-2023', 'acc': 87.02, 'color': '#1ABC9C'},
            {'name': 'Seed-2024', 'acc': 86.71, 'color': '#3498DB'},
            {'name': 'Fusion-42', 'acc': 85.38, 'color': '#9B59B6'},
            {'name': 'Fusion-123', 'acc': 84.36, 'color': '#E91E63'},
            {'name': 'Fusion-456', 'acc': 84.39, 'color': '#FF5722'}
        ]
        
        # 绘制基础模型
        x_positions = np.linspace(0.1, 0.9, len(base_models))
        y_position = 0.3
        
        for i, model in enumerate(base_models):
            # 绘制模型框
            box = FancyBboxPatch(
                (x_positions[i] - 0.04, y_position - 0.05), 0.08, 0.1,
                boxstyle="round,pad=0.01",
                facecolor=model['color'],
                edgecolor='white',
                linewidth=2,
                alpha=0.8
            )
            ax.add_patch(box)
            
            # 添加模型名称和准确率
            ax.text(x_positions[i], y_position, f"{model['name']}\n{model['acc']:.1f}%", 
                   fontsize=8, ha='center', va='center', color='white', fontweight='bold')
            
            # 绘制到集成器的连接线
            ax.plot([x_positions[i], 0.5], [y_position + 0.05, 0.6], 
                   color=model['color'], linewidth=2, alpha=0.6)
        
        # 绘制LightGBM集成器
        ensemble_box = FancyBboxPatch(
            (0.4, 0.55), 0.2, 0.1,
            boxstyle="round,pad=0.02",
            facecolor=self.colors['stage4'],
            edgecolor='white',
            linewidth=3,
            alpha=0.9
        )
        ax.add_patch(ensemble_box)
        
        ax.text(0.5, 0.6, 'LightGBM\nMeta-Learner', fontsize=12, ha='center', va='center', 
               color='white', fontweight='bold')
        
        # 绘制最终输出
        output_box = FancyBboxPatch(
            (0.45, 0.75), 0.1, 0.08,
            boxstyle="round,pad=0.02",
            facecolor=self.colors['accent'],
            edgecolor='white',
            linewidth=3,
            alpha=0.9
        )
        ax.add_patch(output_box)
        
        ax.text(0.5, 0.79, '93.09%\nFinal', fontsize=12, ha='center', va='center', 
               color='white', fontweight='bold')
        
        # 绘制从集成器到输出的连接线
        ax.plot([0.5, 0.5], [0.65, 0.75], color=self.colors['accent'], linewidth=4, alpha=0.8)
        
        # 添加标题和说明
        ax.text(0.5, 0.95, 'Stacking集成学习架构', fontsize=18, ha='center', va='top', 
               fontweight='bold', color=self.colors['text'])
        ax.text(0.5, 0.9, '10个基础模型 + LightGBM元学习器 = 93.09%准确率', fontsize=14, 
               ha='center', va='top', color=self.colors['text'])
        
        # 添加性能对比
        ax.text(0.1, 0.15, '性能对比:', fontsize=12, fontweight='bold', color=self.colors['text'])
        ax.text(0.1, 0.12, '最佳单模型: 88.75%', fontsize=10, color=self.colors['text'])
        ax.text(0.1, 0.09, '简单平均: 89.26%', fontsize=10, color=self.colors['text'])
        ax.text(0.1, 0.06, 'Stacking集成: 93.09%', fontsize=10, fontweight='bold', color=self.colors['accent'])
        ax.text(0.1, 0.03, '相对提升: +4.34%', fontsize=10, fontweight='bold', color=self.colors['accent'])
        
        # 设置坐标轴
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/stacking_ensemble_architecture.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return f'{self.output_dir}/stacking_ensemble_architecture.png'
    
    def create_technology_radar(self):
        """创建技术雷达图"""
        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
        
        # 技术维度
        categories = ['模型深度', '特征融合', '注意力机制', '损失函数', '训练技巧', '集成学习', '数据增强', '正则化']
        
        # 各阶段技术评分
        stage_scores = {
            'Stage 0': [2, 1, 1, 2, 2, 1, 1, 2],
            'Stage 1': [3, 5, 4, 3, 3, 2, 2, 3],
            'Stage 2': [5, 4, 4, 5, 5, 2, 3, 4],
            'Stage 3': [4, 3, 3, 4, 4, 3, 4, 4],
            'Stage 4': [4, 4, 4, 4, 4, 5, 4, 4]
        }
        
        # 计算角度
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        # 绘制各阶段
        colors = [self.colors['baseline'], self.colors['stage1'], 
                 self.colors['stage2'], self.colors['stage3'], self.colors['stage4']]
        
        for i, (stage, scores) in enumerate(stage_scores.items()):
            scores += scores[:1]  # 闭合图形
            ax.plot(angles, scores, 'o-', linewidth=2, label=stage, color=colors[i])
            ax.fill(angles, scores, alpha=0.25, color=colors[i])
        
        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=12)
        ax.set_ylim(0, 5)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_yticklabels(['1', '2', '3', '4', '5'], fontsize=10)
        ax.grid(True)
        
        # 添加标题和图例
        ax.set_title('各阶段技术能力雷达图', size=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/technology_radar_chart.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return f'{self.output_dir}/technology_radar_chart.png'
    
    def create_all_visualizations(self):
        """创建所有可视化图表"""
        print("🎨 开始创建模型迭代可视化...")
        
        visualizations = {}
        
        # 1. 进化时间线
        print("📈 创建进化时间线...")
        visualizations['timeline'] = self.create_evolution_timeline()
        
        # 2. 性能曲线
        print("📊 创建性能曲线...")
        visualizations['performance'] = self.create_performance_curve()
        
        # 3. 模型架构对比
        print("🏗️ 创建模型架构对比...")
        visualizations['architecture'] = self.create_model_architecture_diagram()
        
        # 4. 集成学习可视化
        print("🤝 创建集成学习可视化...")
        visualizations['ensemble'] = self.create_ensemble_visualization()
        
        # 5. 技术雷达图
        print("🎯 创建技术雷达图...")
        visualizations['radar'] = self.create_technology_radar()
        
        # 保存可视化信息
        viz_info = {
            'created_at': datetime.now().isoformat(),
            'total_improvement': self.evolution_data['total_improvement'],
            'final_accuracy': self.evolution_data['final_accuracy'],
            'baseline_accuracy': self.evolution_data['baseline_accuracy'],
            'visualizations': visualizations
        }
        
        with open(f'{self.output_dir}/visualization_info.json', 'w', encoding='utf-8') as f:
            json.dump(viz_info, f, indent=2, ensure_ascii=False)
        
        print("✅ 所有可视化图表创建完成！")
        print(f"📁 输出目录: {self.output_dir}")
        
        return visualizations

def main():
    """主函数"""
    print("🚀 MNIST数字比较模型迭代可视化工具")
    print("=" * 50)
    
    # 创建可视化器
    visualizer = ModelEvolutionVisualizer()
    
    # 创建所有可视化
    visualizations = visualizer.create_all_visualizations()
    
    # 显示结果
    print("\n📊 生成的可视化图表:")
    for name, path in visualizations.items():
        print(f"  - {name}: {path}")
    
    print(f"\n🎉 可视化完成！请查看 {visualizer.output_dir} 目录")

if __name__ == '__main__':
    main()

