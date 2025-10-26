#!/bin/bash
echo "=========================================="
echo "MNIST数字比较模型迭代可视化"
echo "=========================================="

echo "开始创建可视化图表..."

python scripts/visualization/quick_visualize.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 可视化创建失败！"
    echo "请检查Python环境和依赖包是否正确安装"
    echo ""
    echo "安装依赖:"
    echo "pip install matplotlib seaborn pandas numpy"
    exit 1
fi

echo ""
echo "=========================================="
echo "可视化完成！"
echo "=========================================="
echo "查看结果: outputs/visualizations/"
echo ""
echo "生成的图表:"
echo "- model_evolution_timeline.png      (进化时间线)"
echo "- performance_evolution_curve.png   (性能曲线)"
echo "- model_architecture_comparison.png (架构对比)"
echo "- stacking_ensemble_architecture.png (集成学习)"
echo "- technology_radar_chart.png        (技术雷达图)"
echo "=========================================="

