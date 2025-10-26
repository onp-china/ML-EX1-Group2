#!/usr/bin/env python3
"""
Training History Recorder
记录训练过程中的各种指标，支持TensorBoard和JSON保存
"""

import os
import json
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

class TrainingHistory:
    """训练历史记录器"""
    
    def __init__(self, model_name, output_dir, use_tensorboard=True):
        self.model_name = model_name
        self.output_dir = output_dir
        self.use_tensorboard = use_tensorboard
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化TensorBoard
        if self.use_tensorboard:
            self.tb_writer = SummaryWriter(log_dir=os.path.join(output_dir, 'tensorboard'))
        
        # 初始化数据存储
        self.reset()
    
    def reset(self):
        """重置历史数据"""
        self.epochs = []
        self.train_losses = []
        self.train_accuracies = []
        self.train_f1_scores = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_f1_scores = []
        self.learning_rates = []
        self.gradient_norms = []
        
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.best_metrics = {}
    
    def add_epoch(self, epoch, train_metrics, val_metrics, lr, grad_norm=None):
        """添加一个epoch的数据"""
        self.epochs.append(epoch)
        
        # 训练指标
        self.train_losses.append(train_metrics.get('loss', 0.0))
        self.train_accuracies.append(train_metrics.get('accuracy', 0.0))
        self.train_f1_scores.append(train_metrics.get('f1', 0.0))
        
        # 验证指标
        self.val_losses.append(val_metrics.get('loss', 0.0))
        self.val_accuracies.append(val_metrics.get('accuracy', 0.0))
        self.val_f1_scores.append(val_metrics.get('f1', 0.0))
        
        # 学习率和梯度
        self.learning_rates.append(lr)
        if grad_norm is not None:
            self.gradient_norms.append(grad_norm)
        
        # 更新最佳指标
        if val_metrics.get('accuracy', 0.0) > self.best_val_acc:
            self.best_val_acc = val_metrics.get('accuracy', 0.0)
            self.best_epoch = epoch
            self.best_metrics = {
                'epoch': epoch,
                'train_loss': train_metrics.get('loss', 0.0),
                'train_acc': train_metrics.get('accuracy', 0.0),
                'train_f1': train_metrics.get('f1', 0.0),
                'val_loss': val_metrics.get('loss', 0.0),
                'val_acc': val_metrics.get('accuracy', 0.0),
                'val_f1': val_metrics.get('f1', 0.0),
                'lr': lr
            }
        
        # 记录到TensorBoard
        if self.use_tensorboard:
            self._log_to_tensorboard(epoch, train_metrics, val_metrics, lr, grad_norm)
    
    def _log_to_tensorboard(self, epoch, train_metrics, val_metrics, lr, grad_norm=None):
        """记录到TensorBoard"""
        # 损失
        self.tb_writer.add_scalar('Loss/Train', train_metrics.get('loss', 0.0), epoch)
        self.tb_writer.add_scalar('Loss/Validation', val_metrics.get('loss', 0.0), epoch)
        
        # 准确率
        self.tb_writer.add_scalar('Accuracy/Train', train_metrics.get('accuracy', 0.0), epoch)
        self.tb_writer.add_scalar('Accuracy/Validation', val_metrics.get('accuracy', 0.0), epoch)
        
        # F1分数
        self.tb_writer.add_scalar('F1/Train', train_metrics.get('f1', 0.0), epoch)
        self.tb_writer.add_scalar('F1/Validation', val_metrics.get('f1', 0.0), epoch)
        
        # 学习率
        self.tb_writer.add_scalar('Learning_Rate', lr, epoch)
        
        # 梯度范数
        if grad_norm is not None:
            self.tb_writer.add_scalar('Gradient_Norm', grad_norm, epoch)
        
        # 学习曲线对比
        self.tb_writer.add_scalars('Learning_Curves', {
            'Train_Acc': train_metrics.get('accuracy', 0.0),
            'Val_Acc': val_metrics.get('accuracy', 0.0)
        }, epoch)
    
    def save_history(self):
        """保存训练历史到JSON文件"""
        history_data = {
            'model_name': self.model_name,
            'created_at': datetime.now().isoformat(),
            'total_epochs': len(self.epochs),
            'best_metrics': self.best_metrics,
            'epochs': self.epochs,
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'train_f1_scores': self.train_f1_scores,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'val_f1_scores': self.val_f1_scores,
            'learning_rates': self.learning_rates,
            'gradient_norms': self.gradient_norms
        }
        
        # 保存到JSON
        history_file = os.path.join(self.output_dir, 'training_history.json')
        with open(history_file, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        print(f"Training history saved to: {history_file}")
        return history_file
    
    def plot_learning_curves(self, save_path=None):
        """绘制学习曲线"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        ax1.plot(self.epochs, self.train_losses, label='Train Loss', linewidth=2)
        ax1.plot(self.epochs, self.val_losses, label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 准确率曲线
        ax2.plot(self.epochs, self.train_accuracies, label='Train Acc', linewidth=2)
        ax2.plot(self.epochs, self.val_accuracies, label='Val Acc', linewidth=2)
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # F1分数曲线
        ax3.plot(self.epochs, self.train_f1_scores, label='Train F1', linewidth=2)
        ax3.plot(self.epochs, self.val_f1_scores, label='Val F1', linewidth=2)
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('F1 Score')
        ax3.set_title('Training and Validation F1 Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 学习率曲线
        ax4.plot(self.epochs, self.learning_rates, label='Learning Rate', linewidth=2, color='red')
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('Learning Rate')
        ax4.set_title('Learning Rate Schedule')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
        
        plt.suptitle(f'{self.model_name} - Learning Curves', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Learning curves saved to: {save_path}")
        
        plt.show()
        return fig
    
    def get_summary(self):
        """获取训练摘要"""
        return {
            'model_name': self.model_name,
            'total_epochs': len(self.epochs),
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'final_train_acc': self.train_accuracies[-1] if self.train_accuracies else 0.0,
            'final_val_acc': self.val_accuracies[-1] if self.val_accuracies else 0.0,
            'convergence_epoch': self._find_convergence_epoch()
        }
    
    def _find_convergence_epoch(self, patience=5):
        """找到收敛的epoch（连续patience个epoch没有改善）"""
        if len(self.val_accuracies) < patience:
            return len(self.val_accuracies)
        
        best_acc = max(self.val_accuracies)
        best_epoch = self.val_accuracies.index(best_acc)
        
        # 检查是否在最后patience个epoch内
        if best_epoch >= len(self.val_accuracies) - patience:
            return best_epoch + 1
        
        return len(self.val_accuracies)
    
    def close(self):
        """关闭TensorBoard writer"""
        if self.use_tensorboard:
            self.tb_writer.close()

def load_training_history(history_file):
    """加载训练历史"""
    with open(history_file, 'r') as f:
        return json.load(f)

def compare_models_learning_curves(history_files, output_dir):
    """比较多个模型的学习曲线"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C']
    
    for i, history_file in enumerate(history_files):
        history = load_training_history(history_file)
        model_name = history['model_name']
        epochs = history['epochs']
        color = colors[i % len(colors)]
        
        # 损失曲线
        ax1.plot(epochs, history['train_losses'], 
                label=f'{model_name} (Train)', linewidth=2, alpha=0.8, linestyle='--', color=color)
        ax1.plot(epochs, history['val_losses'], 
                label=f'{model_name} (Val)', linewidth=2, alpha=0.8, color=color)
        
        # 准确率曲线
        ax2.plot(epochs, history['train_accuracies'], 
                label=f'{model_name} (Train)', linewidth=2, alpha=0.8, linestyle='--', color=color)
        ax2.plot(epochs, history['val_accuracies'], 
                label=f'{model_name} (Val)', linewidth=2, alpha=0.8, color=color)
        
        # F1分数曲线
        ax3.plot(epochs, history['train_f1_scores'], 
                label=f'{model_name} (Train)', linewidth=2, alpha=0.8, linestyle='--', color=color)
        ax3.plot(epochs, history['val_f1_scores'], 
                label=f'{model_name} (Val)', linewidth=2, alpha=0.8, color=color)
        
        # 学习率曲线
        ax4.plot(epochs, history['learning_rates'], 
                label=model_name, linewidth=2, alpha=0.8, color=color)
    
    # 设置标签和标题
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss Comparison')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy Comparison')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('F1 Score')
    ax3.set_title('Training and Validation F1 Score Comparison')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('Learning Rate')
    ax4.set_title('Learning Rate Schedule Comparison')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    plt.suptitle('Model Learning Curves Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(output_dir, 'models_comparison_learning_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {save_path}")
    
    plt.show()
    return fig
