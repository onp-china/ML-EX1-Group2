#!/usr/bin/env python3
"""
第五阶段高级集成方法快速启动脚本
"""

import os
import sys
import subprocess
import argparse

def run_comparison_test():
    """运行完整对比测试"""
    print("="*60)
    print("运行第五阶段高级集成方法对比测试")
    print("="*60)
    
    # 切换到comparison_tests目录
    os.chdir('comparison_tests')
    
    try:
        # 运行测试脚本
        result = subprocess.run([sys.executable, 'test_with_correct_loading.py'], 
                              capture_output=True, text=True, encoding='utf-8')
        
        print("测试输出:")
        print(result.stdout)
        
        if result.stderr:
            print("错误信息:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("\n✅ 对比测试完成成功!")
        else:
            print(f"\n❌ 测试失败，退出码: {result.returncode}")
            
    except Exception as e:
        print(f"❌ 运行测试时出错: {e}")
    finally:
        # 返回上级目录
        os.chdir('..')

def run_specific_test(method):
    """运行特定方法测试"""
    print(f"运行 {method} 方法测试...")
    
    if method == 'mc_dropout':
        os.chdir('mc_dropout')
        try:
            subprocess.run([sys.executable, 'bayesian_inference.py'])
        except Exception as e:
            print(f"MC Dropout测试失败: {e}")
        finally:
            os.chdir('..')
    
    elif method == 'two_level_stacking':
        os.chdir('two_level_stacking')
        try:
            subprocess.run([sys.executable, 'two_level_dynamic_stacking.py'])
        except Exception as e:
            print(f"两层Stacking测试失败: {e}")
        finally:
            os.chdir('..')
    
    elif method == 'dynamic_ensemble':
        os.chdir('dynamic_ensemble')
        try:
            subprocess.run([sys.executable, 'dynamic_ensemble.py'])
        except Exception as e:
            print(f"动态集成测试失败: {e}")
        finally:
            os.chdir('..')
    
    else:
        print(f"未知的方法: {method}")

def show_results():
    """显示测试结果"""
    print("="*60)
    print("第五阶段测试结果")
    print("="*60)
    
    results_file = 'comparison_tests/correct_loading_test_results.json'
    if os.path.exists(results_file):
        import json
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        print("\n验证集结果:")
        if 'accuracy' in results:
            print(f"  基线 (简单平均): {results['accuracy']:.4f}")
        
        if 'mc_dropout_dynamic' in results:
            print(f"  MC Dropout + 动态权重: {results['mc_dropout_dynamic']['accuracy']:.4f}")
        
        if 'two_level_dynamic' in results:
            print(f"  两层Stacking + 动态权重: {results['two_level_dynamic']['accuracy']:.4f}")
        
        if 'original_stacking' in results:
            print(f"  原始Stacking (LightGBM): {results['original_stacking']['accuracy']:.4f}")
        
        print("\n测试集结果:")
        if 'test_public_baseline' in results:
            print(f"  基线: {results['test_public_baseline']['accuracy']:.4f}")
        
        if 'test_public_stacking' in results:
            print(f"  两层Stacking: {results['test_public_stacking']['accuracy']:.4f}")
    else:
        print("未找到测试结果文件")

def main():
    parser = argparse.ArgumentParser(description="第五阶段高级集成方法快速启动")
    parser.add_argument('--method', type=str, choices=['mc_dropout', 'two_level_stacking', 'dynamic_ensemble'],
                       help='运行特定方法测试')
    parser.add_argument('--results', action='store_true',
                       help='显示测试结果')
    parser.add_argument('--all', action='store_true',
                       help='运行完整对比测试')
    
    args = parser.parse_args()
    
    if args.results:
        show_results()
    elif args.method:
        run_specific_test(args.method)
    elif args.all:
        run_comparison_test()
    else:
        print("第五阶段高级集成方法")
        print("="*40)
        print("可用选项:")
        print("  --all                   运行完整对比测试")
        print("  --method <method>       运行特定方法测试")
        print("  --results               显示测试结果")
        print("\n方法选项:")
        print("  mc_dropout              MC Dropout + 动态权重")
        print("  two_level_stacking      两层Stacking + 动态权重")
        print("  dynamic_ensemble        动态集成")
        print("\n示例:")
        print("  python run_comparison.py --all")
        print("  python run_comparison.py --method mc_dropout")
        print("  python run_comparison.py --results")

if __name__ == '__main__':
    main()
