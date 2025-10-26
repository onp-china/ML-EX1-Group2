#!/usr/bin/env python3
"""
TensorBoard Launcher
启动TensorBoard来可视化训练过程
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def find_tensorboard_logs(base_dir='models'):
    """查找所有TensorBoard日志目录"""
    log_dirs = []
    
    for root, dirs, files in os.walk(base_dir):
        if 'tensorboard' in dirs:
            log_path = os.path.join(root, 'tensorboard')
            model_name = os.path.basename(root)
            log_dirs.append((model_name, log_path))
    
    return log_dirs

def launch_tensorboard(log_dir, port=6006, host='localhost'):
    """启动TensorBoard"""
    try:
        cmd = [
            'tensorboard',
            '--logdir', log_dir,
            '--port', str(port),
            '--host', host
        ]
        
        print(f"Starting TensorBoard...")
        print(f"Command: {' '.join(cmd)}")
        print(f"URL: http://{host}:{port}")
        print("Press Ctrl+C to stop TensorBoard")
        
        subprocess.run(cmd, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"Error starting TensorBoard: {e}")
        print("Make sure TensorBoard is installed: pip install tensorboard")
    except KeyboardInterrupt:
        print("\nTensorBoard stopped by user")

def launch_multi_model_tensorboard(log_dirs, port=6006, host='localhost'):
    """启动多模型TensorBoard"""
    try:
        # 创建日志目录映射
        logdir_spec = ','.join([f"{name}:{path}" for name, path in log_dirs])
        
        cmd = [
            'tensorboard',
            '--logdir_spec', logdir_spec,
            '--port', str(port),
            '--host', host
        ]
        
        print(f"Starting TensorBoard for multiple models...")
        print(f"Command: {' '.join(cmd)}")
        print(f"URL: http://{host}:{port}")
        print("Available models:")
        for name, path in log_dirs:
            print(f"  - {name}: {path}")
        print("Press Ctrl+C to stop TensorBoard")
        
        subprocess.run(cmd, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"Error starting TensorBoard: {e}")
        print("Make sure TensorBoard is installed: pip install tensorboard")
    except KeyboardInterrupt:
        print("\nTensorBoard stopped by user")

def main():
    parser = argparse.ArgumentParser(description='Launch TensorBoard for model training visualization')
    parser.add_argument('--port', type=int, default=6006, help='TensorBoard port')
    parser.add_argument('--host', type=str, default='localhost', help='TensorBoard host')
    parser.add_argument('--model', type=str, help='Specific model to visualize')
    parser.add_argument('--all', action='store_true', help='Show all models')
    parser.add_argument('--base_dir', type=str, default='models', help='Base directory to search for logs')
    
    args = parser.parse_args()
    
    print("TensorBoard Launcher")
    print("=" * 50)
    
    # 查找日志目录
    log_dirs = find_tensorboard_logs(args.base_dir)
    
    if not log_dirs:
        print("No TensorBoard logs found!")
        print(f"Searched in: {args.base_dir}")
        print("Make sure models have been trained with TensorBoard logging enabled.")
        return
    
    print(f"Found {len(log_dirs)} models with TensorBoard logs:")
    for i, (name, path) in enumerate(log_dirs):
        print(f"  {i+1}. {name}: {path}")
    
    if args.model:
        # 启动特定模型
        model_logs = [(name, path) for name, path in log_dirs if args.model.lower() in name.lower()]
        if not model_logs:
            print(f"Model '{args.model}' not found!")
            return
        
        print(f"\nLaunching TensorBoard for {args.model}...")
        launch_tensorboard(model_logs[0][1], args.port, args.host)
    
    elif args.all or len(log_dirs) > 1:
        # 启动多模型TensorBoard
        print(f"\nLaunching TensorBoard for all models...")
        launch_multi_model_tensorboard(log_dirs, args.port, args.host)
    
    else:
        # 启动第一个模型
        print(f"\nLaunching TensorBoard for {log_dirs[0][0]}...")
        launch_tensorboard(log_dirs[0][1], args.port, args.host)

if __name__ == '__main__':
    main()
