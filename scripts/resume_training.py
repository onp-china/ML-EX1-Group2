#!/usr/bin/env python3
"""
训练恢复管理脚本
帮助查找和管理训练检查点
"""

import os
import glob
import argparse
import json
from datetime import datetime

def list_checkpoints(save_dir, augmentation_type=None, seed=None):
    """列出可用的检查点"""
    if not os.path.exists(save_dir):
        print(f"检查点目录不存在: {save_dir}")
        return []
    
    # 构建搜索模式
    if augmentation_type and seed:
        pattern = f"checkpoint_{augmentation_type}_seed{seed}_epoch*.pt"
    elif augmentation_type:
        pattern = f"checkpoint_{augmentation_type}_seed*_epoch*.pt"
    else:
        pattern = "checkpoint_*_seed*_epoch*.pt"
    
    checkpoints = glob.glob(os.path.join(save_dir, pattern))
    
    if not checkpoints:
        print("未找到检查点文件")
        return []
    
    # 解析检查点信息
    checkpoint_info = []
    for cp in checkpoints:
        filename = os.path.basename(cp)
        parts = filename.replace('.pt', '').split('_')
        
        if len(parts) >= 4:
            aug_type = parts[1]
            seed_num = int(parts[2].replace('seed', ''))
            epoch_num = int(parts[3].replace('epoch', ''))
            
            # 获取文件修改时间
            mtime = os.path.getmtime(cp)
            mtime_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
            
            checkpoint_info.append({
                'path': cp,
                'augmentation_type': aug_type,
                'seed': seed_num,
                'epoch': epoch_num,
                'modified_time': mtime_str
            })
    
    # 按种子和epoch排序
    checkpoint_info.sort(key=lambda x: (x['seed'], x['epoch']))
    
    return checkpoint_info

def show_checkpoints(checkpoint_info):
    """显示检查点信息"""
    if not checkpoint_info:
        return
    
    print(f"{'序号':<4} {'增强类型':<12} {'种子':<6} {'轮数':<6} {'修改时间':<20} {'路径'}")
    print("-" * 80)
    
    for i, info in enumerate(checkpoint_info, 1):
        print(f"{i:<4} {info['augmentation_type']:<12} {info['seed']:<6} {info['epoch']:<6} {info['modified_time']:<20} {info['path']}")

def get_latest_checkpoint(save_dir, augmentation_type, seed):
    """获取指定增强类型和种子的最新检查点"""
    checkpoints = list_checkpoints(save_dir, augmentation_type, seed)
    if checkpoints:
        return checkpoints[-1]['path']
    return None

def main():
    parser = argparse.ArgumentParser(description="训练恢复管理")
    parser.add_argument('--save_dir', type=str, default='outputs/checkpoints',
                       help='检查点保存目录')
    parser.add_argument('--augmentation_type', type=str, 
                       choices=['randaugment', 'autoaugment'],
                       help='过滤特定增强类型')
    parser.add_argument('--seed', type=int, help='过滤特定种子')
    parser.add_argument('--latest', action='store_true',
                       help='显示最新的检查点')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("训练检查点管理")
    print("=" * 80)
    
    # 列出检查点
    checkpoints = list_checkpoints(args.save_dir, args.augmentation_type, args.seed)
    
    if args.latest and checkpoints:
        # 按修改时间排序，显示最新的
        checkpoints.sort(key=lambda x: os.path.getmtime(x['path']), reverse=True)
        print("最新检查点:")
        show_checkpoints(checkpoints[:1])
        
        # 提供恢复命令
        latest = checkpoints[0]
        print(f"\n恢复命令:")
        print(f"python scripts/training/train_with_augmentation.py \\")
        print(f"    --augmentation_type {latest['augmentation_type']} \\")
        print(f"    --seeds {latest['seed']} \\")
        print(f"    --resume {latest['path']}")
    else:
        show_checkpoints(checkpoints)
    
    if checkpoints:
        print(f"\n找到 {len(checkpoints)} 个检查点")
        print("\n使用 --latest 查看最新检查点和恢复命令")
    else:
        print("\n未找到检查点文件")
        print("开始新的训练:")
        print("python scripts/training/train_with_augmentation.py --augmentation_type randaugment --seeds 42")

if __name__ == "__main__":
    main()
