#!/usr/bin/env python3
"""
快速可视化脚本 - 一键生成所有图表
"""

import os
import sys

# 添加路径
sys.path.append('src')
sys.path.append('src/models')

def main():
    print("🎨 MNIST数字比较模型迭代可视化")
    print("=" * 50)
    
    try:
        from model_evolution_visualizer import ModelEvolutionVisualizer
        
        # 创建可视化器
        visualizer = ModelEvolutionVisualizer()
        
        # 创建所有可视化
        print("开始创建可视化图表...")
        visualizations = visualizer.create_all_visualizations()
        
        print("\n✅ 可视化完成！")
        print("📁 输出目录: outputs/visualizations/")
        print("\n📊 生成的图表:")
        for name, path in visualizations.items():
            print(f"  - {name}: {path}")
        
        print("\n🚀 使用方法:")
        print("  - 查看图表: 打开 outputs/visualizations/ 目录")
        print("  - 在报告中引用: 直接使用生成的PNG文件")
        print("  - 自定义输出: python scripts/visualization/create_visualizations.py --output_dir 你的目录")
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保安装了必要的依赖包:")
        print("  pip install matplotlib seaborn pandas numpy")
    except Exception as e:
        print(f"❌ 错误: {e}")
        print("请检查代码和依赖是否正确安装")

if __name__ == '__main__':
    main()

