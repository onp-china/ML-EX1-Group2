#!/usr/bin/env python3
"""
FPN架构V2：简化版多尺度特征融合
参考resnet_fusion的5个融合头设计，减少参数量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FPNCompareNetV2(nn.Module):
    """
    FPN多尺度特征融合数字比较网络V2（简化版）
    ==========================================
    
    功能：基于ResNet特征提取的简化多尺度融合网络
    特点：
    1. ResNet特征提取 + 简单多尺度变换
    2. 5个融合头（与resnet_fusion一致）
    3. 参数量优化：7.1M → 4.8M
    4. 保持多尺度特征表达能力
    
    预期性能：准确率88%+（相比原FPN的87.06%）
    """
    
    def __init__(self, feat_dim=256, layers=[2, 2, 2], width_mult=1.25):
        super(FPNCompareNetV2, self).__init__()
        
        # ResNet特征提取塔
        self.tower = ResNetTower(out_dim=feat_dim, layers=layers, width_mult=width_mult)
        
        # 简化的多尺度特征变换
        self.scale_transform = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
            nn.Sigmoid()
        )
        
        # 5个融合头（与resnet_fusion一致）
        self.diff_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feat_dim // 2, feat_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feat_dim // 4, 1)
        )
        
        self.concat_head = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feat_dim // 2, 1)
        )
        
        self.product_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feat_dim // 2, feat_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feat_dim // 4, 1)
        )
        
        self.cosine_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feat_dim // 2, feat_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feat_dim // 4, 1)
        )
        
        self.distance_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feat_dim // 2, feat_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feat_dim // 4, 1)
        )
        
        # 融合权重（5个融合头）
        self.fusion_weights = nn.Parameter(torch.ones(5) / 5)
        
    def forward(self, img1, img2):
        # 特征提取
        feat1 = self.tower(img1)  # [B, feat_dim, 1, 1]
        feat2 = self.tower(img2)  # [B, feat_dim, 1, 1]
        
        # 展平特征
        global_feat1 = feat1.flatten(1)  # [B, feat_dim]
        global_feat2 = feat2.flatten(1)  # [B, feat_dim]
        
        # 简化的多尺度特征变换
        # 原始特征
        feat1_orig = global_feat1
        feat2_orig = global_feat2
        
        # 变换特征（非线性变换）
        feat1_trans = self.scale_transform(global_feat1)
        feat2_trans = self.scale_transform(global_feat2)
        
        # 注意力机制
        attention_weights = self.attention(torch.cat([feat1_orig, feat2_orig], dim=1))
        feat1_attended = feat1_orig * attention_weights
        feat2_attended = feat2_orig * attention_weights
        
        # 5种融合方式
        diff_feat = torch.abs(feat1_attended - feat2_attended)
        concat_feat = torch.cat([feat1_attended, feat2_attended], dim=1)
        product_feat = feat1_attended * feat2_attended
        
        # 余弦相似度
        cosine_sim = F.cosine_similarity(feat1_attended, feat2_attended, dim=1).unsqueeze(1)
        cosine_feat = cosine_sim.expand(-1, feat1_attended.size(1)) * feat1_attended
        
        # 欧几里得距离
        distance = torch.norm(feat1_attended - feat2_attended, p=2, dim=1).unsqueeze(1)
        distance_feat = distance.expand(-1, feat1_attended.size(1)) * feat1_attended
        
        # 计算各种融合的logits
        diff_logits = self.diff_head(diff_feat)
        concat_logits = self.concat_head(concat_feat)
        product_logits = self.product_head(product_feat)
        cosine_logits = self.cosine_head(cosine_feat)
        distance_logits = self.distance_head(distance_feat)
        
        # 加权融合
        logits = (self.fusion_weights[0] * diff_logits + 
                 self.fusion_weights[1] * concat_logits + 
                 self.fusion_weights[2] * product_logits +
                 self.fusion_weights[3] * cosine_logits +
                 self.fusion_weights[4] * distance_logits)
        
        return logits


# 需要导入ResNetTower和BasicBlock
class BasicBlock(nn.Module):
    """ResNet基本块"""
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.downsample is not None:
            residual = self.downsample(x)
            
        out += residual
        out = F.relu(out)
        return out


class ResNetTower(nn.Module):
    """ResNet特征提取塔"""
    def __init__(self, out_dim=256, layers=[2, 2, 2], block=BasicBlock, width_mult=1.0):
        super(ResNetTower, self).__init__()
        
        self.in_planes = int(64 * width_mult)
        
        self.conv1 = nn.Conv2d(1, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        
        self.layer1 = self._make_layer(block, int(64 * width_mult), layers[0], stride=1)
        self.layer2 = self._make_layer(block, int(128 * width_mult), layers[1], stride=2)
        self.layer3 = self._make_layer(block, int(256 * width_mult), layers[2], stride=2)
        
        # SE注意力模块
        self.se1 = SEModule(int(64 * width_mult))
        self.se2 = SEModule(int(128 * width_mult))
        self.se3 = SEModule(int(256 * width_mult))
        
        # 特征投影
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(int(256 * width_mult), out_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.se1(out)
        out = self.layer2(out)
        out = self.se2(out)
        out = self.layer3(out)
        out = self.se3(out)
        out = self.fc(out)
        return out


class SEModule(nn.Module):
    """SE注意力模块"""
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def count_params(model):
    """Count the number of parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Test the FPN model
    model = FPNCompareNetV2(feat_dim=256, layers=[2, 2, 2], width_mult=1.25)
    
    # Test with dummy input
    xa = torch.randn(2, 1, 28, 28)
    xb = torch.randn(2, 1, 28, 28)
    
    output = model(xa, xb)
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {count_params(model):,}")