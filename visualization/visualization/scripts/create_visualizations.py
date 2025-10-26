#!/usr/bin/env python3
"""
一键创建所有模型迭代可视化
"""

import os
import sys
import argparse

# 添加路径
sys.path.append('src')
sys.path.append('src/models')

from model_evolution_visualizer import ModelEvolutionVisualizer

def main():
    parser = argparse.ArgumentParser(description='创建模型迭代可视化')
    parser.add_argument('--output_dir', type=str, default='outputs/visualizations',
                       help='输出目录')
    parser.add_argument('--timeline', action='store_true', help='创建进化时间线')
    parser.add_argument('--performance', action='store_true', help='创建性能曲线')
    parser.add_argument('--architecture', action='store_true', help='创建架构对比')
    parser.add_argument('--ensemble', action='store_true', help='创建集成学习可视化')
    parser.add_argument('--radar', action='store_true', help='创建技术雷达图')
    parser.add_argument('--all', action='store_true', help='创建所有可视化')
    
    args = parser.parse_args()
    
    # 创建可视化器
    visualizer = ModelEvolutionVisualizer(output_dir=args.output_dir)
    
    if args.all or (not any([args.timeline, args.performance, args.architecture, args.ensemble, args.radar])):
        # 创建所有可视化
        visualizations = visualizer.create_all_visualizations()
    else:
        # 创建指定的可视化
        visualizations = {}
        
        if args.timeline:
            print("📈 创建进化时间线...")
            visualizations['timeline'] = visualizer.create_evolution_timeline()
        
        if args.performance:
            print("📊 创建性能曲线...")
            visualizations['performance'] = visualizer.create_performance_curve()
        
        if args.architecture:
            print("🏗️ 创建模型架构对比...")
            visualizations['architecture'] = visualizer.create_model_architecture_diagram()
        
        if args.ensemble:
            print("🤝 创建集成学习可视化...")
            visualizations['ensemble'] = visualizer.create_ensemble_visualization()
        
        if args.radar:
            print("🎯 创建技术雷达图...")
            visualizations['radar'] = visualizer.create_technology_radar()
    
    print("\n📊 生成的可视化图表:")
    for name, path in visualizations.items():
        print(f"  - {name}: {path}")
    
    print(f"\n🎉 可视化完成！请查看 {args.output_dir} 目录")

if __name__ == '__main__':
    main()

