#!/usr/bin/env python3
"""
监控训练进度
"""

import os
import time
import json
from datetime import datetime

def check_training_progress():
    """检查训练进度"""
    models_to_check = [
        'models/stage1_improvedv2',
        'models/stage2_resnet_optimized/resnet_optimized_1.12',
        'models/stage2_resnet_optimized/resnet_fusion',
        'models/stage2_resnet_optimized/resnet_optimized',
        'models/stage3_multi_seed/fpn_model',
        'models/stage3_multi_seed/seed_2025',
        'models/stage3_multi_seed/seed_2023',
        'models/stage3_multi_seed/seed_2024',
        'models/stage3_multi_seed/resnet_fusion_seed42',
        'models/stage3_multi_seed/resnet_fusion_seed123',
        'models/stage3_multi_seed/resnet_fusion_seed456'
    ]
    
    print(f"\n{'='*80}")
    print(f"训练进度监控 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    completed = 0
    in_progress = 0
    not_started = 0
    
    for model_dir in models_to_check:
        model_name = model_dir.split('/')[-1]
        
        # 检查是否有训练历史文件
        history_file = os.path.join(model_dir, 'training_history.json')
        metrics_file = os.path.join(model_dir, 'metrics.json')
        
        if os.path.exists(history_file) and os.path.exists(metrics_file):
            try:
                with open(history_file, 'r') as f:
                    history = json.load(f)
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                total_epochs = len(history.get('epochs', []))
                best_val_acc = metrics.get('best_val_acc', 0)
                
                print(f"[完成] {model_name:25} - 完成 ({total_epochs} epochs, 最佳准确率: {best_val_acc:.4f})")
                completed += 1
                
            except Exception as e:
                print(f"[错误] {model_name:25} - 数据读取错误: {e}")
                in_progress += 1
        elif os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    history = json.load(f)
                total_epochs = len(history.get('epochs', []))
                print(f"[训练中] {model_name:25} - 训练中 ({total_epochs} epochs)")
                in_progress += 1
            except:
                print(f"[训练中] {model_name:25} - 训练中")
                in_progress += 1
        else:
            print(f"[未开始] {model_name:25} - 未开始")
            not_started += 1
    
    print(f"\n统计:")
    print(f"  已完成: {completed}")
    print(f"  训练中: {in_progress}")
    print(f"  未开始: {not_started}")
    print(f"  总计: {completed + in_progress + not_started}")
    
    return completed, in_progress, not_started

def main():
    """主函数"""
    print("开始监控训练进度...")
    
    while True:
        completed, in_progress, not_started = check_training_progress()
        
        if completed == 11:  # 所有模型都完成
            print(f"\n所有模型训练完成!")
            break
        elif in_progress == 0 and not_started > 0:
            print(f"\n没有正在训练的模型，但还有 {not_started} 个模型未开始")
            break
        
        print(f"\n等待30秒后再次检查...")
        time.sleep(30)

if __name__ == '__main__':
    main()
