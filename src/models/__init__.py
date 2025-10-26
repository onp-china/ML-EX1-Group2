"""
模型架构定义模块
"""

from .simple_compare_cnn import ResNetCompareNet, BasicBlock, Bottleneck
from .fpn_architecture_v2 import FPNCompareNetV2

__all__ = [
    'ResNetCompareNet',
    'BasicBlock',
    'Bottleneck',
    'FPNCompareNetV2'
]

