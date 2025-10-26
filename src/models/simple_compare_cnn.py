"""
MNIST数字比较模型定义文件
============================

本文件包含用于比较两个MNIST数字是否相同的神经网络模型：
1. CompareNet - 基线模型：简单的CNN + 特征拼接
2. ImprovedCompareNet - 改进模型：深度CNN + 注意力机制 + 多种融合方式
3. ResNetCompareNet - ResNet模型：ResNet架构 + 注意力机制 + 多种融合方式

任务：二分类问题，判断两个28x28的MNIST数字图像是否相同
输入：两个28x28的单通道图像
输出：二分类logit（相同=1，不同=0）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """
    ResNet基本块
    ===========
    
    功能：ResNet的基本构建单元，包含残差连接
    特点：解决梯度消失问题，支持更深的网络
    
    结构：
    input -> conv1 -> bn1 -> relu -> conv2 -> bn2 -> + -> relu -> output
           |                                        |
           |---------> shortcut (if needed) ------->|
    """
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

class Bottleneck(nn.Module):
    """
    ResNet瓶颈块
    ===========
    
    功能：ResNet的瓶颈结构，减少参数量
    特点：1x1 -> 3x3 -> 1x1 的卷积序列
    
    结构：
    input -> 1x1conv -> bn1 -> relu -> 3x3conv -> bn2 -> relu -> 1x1conv -> bn3 -> + -> relu -> output
           |                                                                      |
           |-------------------------> shortcut (if needed) --------------------->|
    """
    expansion = 4
    
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        if self.downsample is not None:
            residual = self.downsample(x)
            
        out += residual
        out = F.relu(out)
        return out

class AttentionModule(nn.Module):
    """
    改进的注意力机制模块
    ==================
    
    功能：为特征向量学习注意力权重，突出重要特征
    特点：使用残差连接和LayerNorm，提高训练稳定性
    
    输入：feat_dim维的特征向量
    输出：加权后的feat_dim维特征向量
    
    改进点：
    1. 添加残差连接：解决梯度消失问题
    2. 使用LayerNorm：提高训练稳定性
    3. 使用GELU激活：更好的非线性表达能力
    """
    def __init__(self, feat_dim):
        super().__init__()
        # LayerNorm层
        self.norm1 = nn.LayerNorm(feat_dim)
        self.norm2 = nn.LayerNorm(feat_dim)
        
        # 注意力网络：降维 -> 激活 -> 升维 -> 归一化
        self.attention = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 4),  # 降维到1/4
            nn.GELU(),                           # 更好的激活函数
            nn.Linear(feat_dim // 4, feat_dim),  # 升维回原维度
            nn.Sigmoid()                         # 输出0-1的注意力权重
        )
    
    def forward(self, x):
        # 残差连接 + LayerNorm
        residual = x
        x = self.norm1(x)
        attn_weights = self.attention(x)
        x = x * attn_weights
        return self.norm2(x + residual)

class ResNetTower(nn.Module):
    """
    ResNet特征提取塔
    ===============
    
    功能：基于ResNet架构的特征提取器，支持更深的网络
    特点：使用残差连接，解决梯度消失问题，支持多层结构
    
    网络结构：
    28x28x1 -> 14x14x64 -> 7x7x128 -> 3x3x256 -> 1x1x512 -> out_dim维特征
    
    参数量：约200万参数（相比ImprovedTower的72万参数）
    
    改进点：
    1. 使用ResNet基本块，支持残差连接
    2. 更深的网络结构（多层ResNet块）
    3. 更好的特征提取能力
    4. 解决梯度消失问题
    """
    def __init__(self, out_dim=256, layers=[2, 2, 2, 2], block=BasicBlock):
        super(ResNetTower, self).__init__()
        self.in_planes = 64
        
        # 初始卷积层 - 修复过度下采样问题
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 28x28x1 -> 28x28x64
        self.bn1 = nn.BatchNorm2d(64)
        # 移除maxpool，避免过早丢失空间信息
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet层 - 调整stride以适应新的输入尺寸
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)   # 28x28x64 -> 28x28x64
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 28x28x64 -> 14x14x128
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # 14x14x128 -> 7x7x256
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # 7x7x256 -> 4x4x512
        
        # 全局平均池化和分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 1x1x512 -> 1x1x512
        # 计算最终特征维度（考虑block的expansion）
        final_dim = 512 * block.expansion
        self.fc = nn.Sequential(
            nn.Flatten(),                        # 1x1x512*expansion -> 512*expansion
            nn.Linear(final_dim, out_dim),       # 512*expansion -> out_dim
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),                     # 防止过拟合
        )
        
    def _make_layer(self, block, planes, blocks, stride=1):
        """构建ResNet层"""
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        
        return nn.Sequential(*layers)

class SEModule(nn.Module):
    """
    SE注意力模块
    ===========
    
    功能：Squeeze-and-Excitation注意力机制，学习通道间的依赖关系
    特点：通过全局平均池化和全连接层学习通道权重
    
    参数：
    - channels: 输入通道数
    - reduction: 降维比例，默认16
    """
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

class ResNetTower(nn.Module):
    """
    ResNet特征提取塔（3层版本）
    =========================
    
    功能：基于ResNet架构的特征提取器，针对MNIST优化
    特点：使用残差连接，3层结构，避免过度复杂
    
    网络结构：
    28x28x1 -> 28x28x64 -> 14x14x128 -> 7x7x256 -> out_dim维特征
    
    参数量：约100万参数（相比4层版本减少50%）
    
    改进点：
    1. 使用ResNet基本块，支持残差连接
    2. 3层结构，适合MNIST小图像
    3. 避免过度复杂，防止过拟合
    4. 解决梯度消失问题
    """
    def __init__(self, out_dim=256, layers=[2, 2, 2], block=BasicBlock, width_mult: float = 1.0):
        super(ResNetTower, self).__init__()
        self.width_mult = float(width_mult)
        c64 = int(64 * self.width_mult)
        c128 = int(128 * self.width_mult)
        c256 = int(256 * self.width_mult)
        self.in_planes = c64
        
        # 初始卷积层 - 修复过度下采样问题
        self.conv1 = nn.Conv2d(1, c64, kernel_size=3, stride=1, padding=1, bias=False)  # 28x28x1 -> 28x28xC64
        self.bn1 = nn.BatchNorm2d(c64)
        # 移除maxpool，避免过早丢失空间信息
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet层 - 3层结构，适合MNIST
        self.layer1 = self._make_layer(block, c64, layers[0], stride=1)   # 28x28xC64 -> 28x28xC64
        self.layer2 = self._make_layer(block, c128, layers[1], stride=2)  # 28x28xC64 -> 14x14xC128
        self.layer3 = self._make_layer(block, c256, layers[2], stride=2)  # 14x14xC128 -> 7x7xC256
        
        # 全局平均池化和分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 7x7x256 -> 1x1x256
        # 计算最终特征维度（考虑block的expansion）
        final_dim = c256 * block.expansion
        self.fc = nn.Sequential(
            nn.Flatten(),                        # 1x1x256*expansion -> 256*expansion
            nn.Linear(final_dim, out_dim),       # 256*expansion -> out_dim
            nn.BatchNorm1d(out_dim),             # 批归一化
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),                     # 增加dropout率
            nn.Linear(out_dim, out_dim),         # 额外全连接层
            nn.BatchNorm1d(out_dim),             # 批归一化
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),                     # 防止过拟合
        )
        
        # 添加SE模块（3层版本）
        self.se1 = SEModule(c64)
        self.se2 = SEModule(c128)
        self.se3 = SEModule(c256)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        """构建ResNet层"""
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # 初始卷积 - 保持空间分辨率
        x = F.relu(self.bn1(self.conv1(x)))  # 28x28x1 -> 28x28x64
        # 移除maxpool，避免过早丢失空间信息
        
        # ResNet层 + SE注意力（3层版本）
        x = self.layer1(x)  # 28x28x64 -> 28x28x64
        x = self.se1(x)     # 应用SE注意力
        
        x = self.layer2(x)  # 28x28x64 -> 14x14x128
        x = self.se2(x)     # 应用SE注意力
        
        x = self.layer3(x)  # 14x14x128 -> 7x7x256
        x = self.se3(x)     # 应用SE注意力
        
        # 全局平均池化和分类
        x = self.avgpool(x)  # 7x7x256 -> 1x1x256
        x = self.fc(x)       # 1x1x256 -> out_dim
        
        return x

class ImprovedTower(nn.Module):
    """
    改进的特征提取塔
    ===============
    
    功能：从28x28的MNIST图像中提取256维特征向量
    特点：更深的网络结构，更强的特征提取能力
    
    网络结构：
    28x28x1 -> 14x14x64 -> 7x7x128 -> 1x1x256 -> 256维特征
    
    参数量：约72万参数（相比基线模型的8万参数）
    
    🚀 改进建议：
    1. 使用ResNet块：添加残差连接，解决梯度消失问题
    2. 使用SE模块：Squeeze-and-Excitation注意力机制
    3. 使用更先进的backbone：EfficientNet、ResNeXt等
    4. 添加更多正则化：DropBlock、Stochastic Depth等
    5. 使用混合精度训练：提高训练效率
    """
    def __init__(self, out_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            # 第一组：提取低级特征（边缘、纹理）
            nn.Conv2d(1, 64, 3, padding=1),      # 28x28x1 -> 28x28x64
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),     # 28x28x64 -> 28x28x64
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                     # 28x28x64 -> 14x14x64
            
            # 第二组：提取中级特征（形状、结构）
            nn.Conv2d(64, 128, 3, padding=1),    # 14x14x64 -> 14x14x128
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),   # 14x14x128 -> 14x14x128
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                     # 14x14x128 -> 7x7x128
            
            # 第三组：提取高级特征（语义信息）
            nn.Conv2d(128, 256, 3, padding=1),   # 7x7x128 -> 7x7x256
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),   # 7x7x256 -> 7x7x256
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),         # 7x7x256 -> 1x1x256
        )
        self.fc = nn.Sequential(
            nn.Flatten(),                        # 1x1x256 -> 256
            nn.Linear(256, out_dim),             # 256 -> out_dim
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),                     # 防止过拟合
        )
        
        # 🚀 改进建议：添加SE模块
        # self.se = SEModule(out_dim)
        
        # 🚀 改进建议：添加ResNet块
        # self.res_blocks = nn.ModuleList([
        #     ResNetBlock(64, 64),
        #     ResNetBlock(128, 128),
        #     ResNetBlock(256, 256)
        # ])

    def forward(self, x):
        x = self.conv(x)  # 卷积特征提取
        x = self.fc(x)    # 全连接降维
        
        # 🚀 改进建议：添加SE注意力
        # x = self.se(x)
        
        return x

class Tower(nn.Module):
    """
    基线特征提取塔
    =============
    
    功能：从28x28的MNIST图像中提取128维特征向量
    特点：简单的3层卷积结构，参数量少，训练快速
    
    网络结构：
    28x28x1 -> 14x14x32 -> 7x7x64 -> 1x1x128 -> 128维特征
    
    参数量：约8万参数
    """
    def __init__(self, out_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            # 第一层：基础特征提取
            nn.Conv2d(1, 32, 3, padding=1),      # 28x28x1 -> 28x28x32
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                     # 28x28x32 -> 14x14x32
            
            # 第二层：中级特征提取
            nn.Conv2d(32, 64, 3, padding=1),     # 14x14x32 -> 14x14x64
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                     # 14x14x64 -> 7x7x64
            
            # 第三层：高级特征提取
            nn.Conv2d(64, 128, 3, padding=1),    # 7x7x64 -> 7x7x128
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),         # 7x7x128 -> 1x1x128
        )
        self.fc = nn.Sequential(
            nn.Flatten(),                        # 1x1x128 -> 128
            nn.Linear(128, out_dim),             # 128 -> out_dim
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),                     # 轻微正则化
        )

    def forward(self, x):
        x = self.conv(x)  # 卷积特征提取
        x = self.fc(x)    # 全连接降维
        return x

class ResNetCompareNet(nn.Module):
    """
    ResNet数字比较网络
    =================
    
    功能：基于ResNet架构的数字比较网络，判断两个MNIST数字是否相同
    特点：使用ResNet特征提取 + 注意力机制 + 多种特征融合方式
    
    创新点：
    1. ResNet特征提取：更深的网络，残差连接
    2. 注意力机制：突出重要特征
    3. 多种融合：差值、拼接、乘积三种方式
    4. 自适应权重：学习最优融合比例
    
    预期性能：准确率85-90%（相比ImprovedCompareNet的80.59%）
    参数量：约300万参数（相比ImprovedCompareNet的144万）
    """
    def __init__(self, feat_dim=256, layers=[2, 2, 2], block=BasicBlock, width_mult=1.0):
        super().__init__()
        # ResNet特征提取塔
        self.tower = ResNetTower(out_dim=feat_dim, layers=layers, block=block, width_mult=width_mult)
        # 注意力机制
        self.attention = AttentionModule(feat_dim)
        
        # 五种不同的特征融合方式（增强正则化）
        # 1. 差值融合：|fa - fb|，关注差异
        self.diff_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
        
        # 2. 拼接融合：[fa, fb]，保留完整信息
        self.concat_head = nn.Sequential(
            nn.Linear(feat_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        
        # 3. 乘积融合：fa * fb，关注相似性
        self.product_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
        
        # 4. 余弦相似度融合：cos(fa, fb)，关注方向相似性
        self.cosine_head = nn.Sequential(
            nn.Linear(1, 64),  # 余弦相似度是标量
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )
        
        # 5. 欧氏距离融合：||fa - fb||，关注距离相似性
        self.distance_head = nn.Sequential(
            nn.Linear(1, 64),  # 距离是标量
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )
        
        # 可学习的融合权重（初始化为均匀分布，现在有5个融合方式）
        self.fusion_weights = nn.Parameter(torch.ones(5) / 5)
    
    def forward(self, xa, xb):
        # 1. ResNet特征提取
        fa = self.tower(xa)  # 提取图像A的特征
        fb = self.tower(xb)  # 提取图像B的特征
        
        # 2. 注意力加权
        fa = self.attention(fa)  # 对特征A应用注意力
        fb = self.attention(fb)  # 对特征B应用注意力
        
        # 3. 多种特征融合方式
        diff = torch.abs(fa - fb)                    # 差值：关注差异
        concat = torch.cat([fa, fb], dim=-1)         # 拼接：保留完整信息
        product = fa * fb                            # 乘积：关注相似性
        
        # 4. 分别通过不同的分类头
        diff_logit = self.diff_head(diff)            # 差值分支的logit
        concat_logit = self.concat_head(concat)      # 拼接分支的logit
        product_logit = self.product_head(product)   # 乘积分支的logit
        
        # 5. 自适应权重融合
        weights = F.softmax(self.fusion_weights, dim=0)  # 归一化权重
        logit = weights[0] * diff_logit + weights[1] * concat_logit + weights[2] * product_logit
        
        return logit.squeeze(1)  # 移除多余的维度

class ImprovedCompareNet(nn.Module):
    """
    改进的数字比较网络
    =================
    
    功能：判断两个MNIST数字是否相同（二分类）
    特点：使用注意力机制 + 多种特征融合方式 + 自适应权重融合
    
    创新点：
    1. 注意力机制：突出重要特征
    2. 多种融合：差值、拼接、乘积三种方式
    3. 自适应权重：学习最优融合比例
    
    性能：准确率80.59%，F1分数80.59%（相比基线模型57.11%）
    参数量：144万参数（相比基线模型17.5万）
    
    🚀 改进建议：
    1. 添加更多融合方式：余弦相似度、欧几里得距离等
    2. 使用Transformer架构：自注意力机制
    3. 添加对比学习：学习更好的特征表示
    4. 使用模型集成：多个模型的预测结果融合
    5. 添加辅助任务：数字分类、旋转预测等
    """
    def __init__(self, feat_dim=256):
        super().__init__()
        # 特征提取塔
        self.tower = ImprovedTower(out_dim=feat_dim)
        # 注意力机制
        self.attention = AttentionModule(feat_dim)
        
        # 三种不同的特征融合方式
        # 1. 差值融合：|fa - fb|，关注差异
        self.diff_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        
        # 2. 拼接融合：[fa, fb]，保留完整信息
        self.concat_head = nn.Sequential(
            nn.Linear(feat_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        
        # 3. 乘积融合：fa * fb，关注相似性
        self.product_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        
        # 🚀 改进建议：添加更多融合方式
        # 4. 余弦相似度融合
        # self.cosine_head = nn.Sequential(
        #     nn.Linear(1, 64),  # 余弦相似度是标量
        #     nn.ReLU(),
        #     nn.Linear(64, 1)
        # )
        
        # 5. 欧几里得距离融合
        # self.distance_head = nn.Sequential(
        #     nn.Linear(1, 64),  # 距离是标量
        #     nn.ReLU(),
        #     nn.Linear(64, 1)
        # )
        
        # 可学习的融合权重（初始化为均匀分布）
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3)
        
        # 🚀 改进建议：动态权重学习
        # self.weight_net = nn.Sequential(
        #     nn.Linear(feat_dim * 2, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 3),
        #     nn.Softmax(dim=-1)
        # )
    
    def forward(self, xa, xb):
        # 1. 特征提取
        fa = self.tower(xa)  # 提取图像A的特征
        fb = self.tower(xb)  # 提取图像B的特征
        
        # 2. 注意力加权
        fa = self.attention(fa)  # 对特征A应用注意力
        fb = self.attention(fb)  # 对特征B应用注意力
        
        # 3. 多种特征融合方式
        diff = torch.abs(fa - fb)                    # 差值：关注差异
        concat = torch.cat([fa, fb], dim=-1)         # 拼接：保留完整信息
        product = fa * fb                            # 乘积：关注相似性
        
        # 4. 分别通过不同的分类头
        diff_logit = self.diff_head(diff)            # 差值分支的logit
        concat_logit = self.concat_head(concat)      # 拼接分支的logit
        product_logit = self.product_head(product)   # 乘积分支的logit
        
        # 5. 自适应权重融合
        weights = F.softmax(self.fusion_weights, dim=0)  # 归一化权重
        logit = weights[0] * diff_logit + weights[1] * concat_logit + weights[2] * product_logit
        
        return logit.squeeze(1)  # 移除多余的维度

class CompareNet(nn.Module):
    """
    基线数字比较网络
    ===============
    
    功能：判断两个MNIST数字是否相同（二分类）
    特点：简单的CNN + 特征拼接，作为性能基准
    
    架构：
    1. 两个相同的特征提取塔（Tower）
    2. 特征拼接：[fa, fb]
    3. 简单的分类头
    
    性能：准确率57.11%，F1分数56.33%
    参数量：17.5万参数
    """
    def __init__(self, feat_dim=128):
        super().__init__()
        # 特征提取塔
        self.tower = Tower(out_dim=feat_dim)
        # 拼接后的特征维度
        in_dim = feat_dim * 2  
        # 分类头
        self.head = nn.Sequential(
            nn.Linear(in_dim, 256),    # 256 -> 256
            nn.ReLU(inplace=True),     # 非线性激活
            nn.Dropout(0.2),           # 正则化
            nn.Linear(256, 1)          # 256 -> 1 (二分类)
        )

    def forward(self, xa, xb):
        # 1. 特征提取
        fa = self.tower(xa)  # 提取图像A的特征
        fb = self.tower(xb)  # 提取图像B的特征
        
        # 2. 特征拼接
        fuse = torch.cat([fa, fb], dim=-1)  # [fa, fb] -> 256维
        
        # 3. 分类预测
        logit = self.head(fuse).squeeze(1)  # 256 -> 1
        return logit

def count_params(model):
    """
    计算模型参数量
    =============
    
    功能：统计模型中可训练参数的总数
    用途：模型复杂度分析，性能对比
    
    返回：参数量（整数）
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
