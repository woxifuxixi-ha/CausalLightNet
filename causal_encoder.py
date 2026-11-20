"""
因果先验编码器
基于因果发现结果设计的编码器，包含可学习的因果权重
"""
import torch
import torch.nn as nn
import numpy as np


class CausalWeightedFusion(nn.Module):
    """
    因果加权融合模块
    根据因果强度初始化权重，但允许训练中调整
    """
    def __init__(self, causal_config, num_channels=8):
        """
        参数:
            causal_config: 因果配置字典
            num_channels: 特征通道数
        """
        super(CausalWeightedFusion, self).__init__()
        
        self.causal_config = causal_config
        self.num_sources = len(causal_config.get('sources', []))
        
        if self.num_sources == 0:
            print("警告: 没有因果源，使用均匀权重")
            self.use_causal_prior = False
            return
        
        self.use_causal_prior = True
        self.source_names = causal_config['sources']
        
        # 初始化因果权重（基于发现的强度）
        prior_weights = []
        for source in self.source_names:
            weight = causal_config['weights'].get(source, 1.0)
            prior_weights.append(weight)
        
        prior_weights = torch.tensor(prior_weights, dtype=torch.float32)
        prior_weights = prior_weights / (prior_weights.sum() + 1e-8)  # 归一化
        
        # 可学习的权重参数（以先验权重初始化）
        self.causal_weights = nn.Parameter(prior_weights)
        
        # 融合卷积
        total_channels = num_channels * self.num_sources
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(total_channels, num_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
        )
        
        print(f"因果加权融合初始化:")
        print(f"  因果源数量: {self.num_sources}")
        for i, (name, w) in enumerate(zip(self.source_names, prior_weights)):
            print(f"  {name}: 初始权重={w:.4f}")
    
    def forward(self, features_dict):
        """
        前向传播
        
        参数:
            features_dict: 特征字典 {'sevir_vil': tensor, 'era5_cape': tensor, ...}
        返回:
            fused: 融合后的特征 [batch, channels, H, W]
        """
        if not self.use_causal_prior:
            feature_list = list(features_dict.values())
            return torch.mean(torch.stack(feature_list, dim=0), dim=0)
        
        # Softmax归一化权重
        normalized_weights = torch.softmax(self.causal_weights, dim=0)
        
        # 加权特征
        weighted_features = []
        for i, source in enumerate(self.source_names):
            if source in features_dict:
                feat = features_dict[source]
                weight = normalized_weights[i]
                weighted_features.append(feat * weight)
        
        # 拼接所有加权特征
        if len(weighted_features) > 0:
            concat_features = torch.cat(weighted_features, dim=1)
            fused = self.fusion_conv(concat_features)
            return fused
        else:
            raise ValueError("没有有效的特征可以融合")
    
    def get_current_weights(self):
        """获取当前的因果权重（用于监控）"""
        with torch.no_grad():
            weights = torch.softmax(self.causal_weights, dim=0).cpu().numpy()
        return dict(zip(self.source_names, weights))


class CausalPriorEncoder(nn.Module):
    """
    因果先验编码器
    分别编码各个ERA5变量，然后根据因果关系融合
    """
    def __init__(self, config_dict, causal_config=None, num_channels=5):
        """
        参数:
            config_dict: 模型配置
            causal_config: 因果配置（从CausalPriorParser获取）
            num_channels: 输入通道数（2或5）
        """
        super(CausalPriorEncoder, self).__init__()
        
        self.config_dict = config_dict
        self.num_channels = num_channels
        mn = (config_dict['GridRowColNum'] // 2) // 2  # 32 for 128x128
        
        # 创建编码器列表（自动适应通道数）
        self.encoders = nn.ModuleList()
        self.channel_names = ['vil', 'ir107', 'cape', 'cin', 't2m'][:num_channels]
        
        for i in range(num_channels):
            encoder = nn.Sequential(
                nn.Conv2d(1, 2, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(2, 4, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
            )
            self.encoders.append(encoder)
        
        # 因果融合或简单融合
        if causal_config is not None and len(causal_config.get('sources', [])) > 0:
            self.causal_fusion = CausalWeightedFusion(causal_config, num_channels=8)
            self.use_causal_fusion = True
        else:
            self.use_causal_fusion = False
            self.simple_fusion = nn.Sequential(
                nn.Conv2d(8 * num_channels, 8, kernel_size=1),
                nn.ReLU(),
                nn.LayerNorm([8, mn, mn], elementwise_affine=True)
            )
        
        print(f"因果先验编码器初始化:")
        print(f"  输入通道数: {num_channels}")
        print(f"  使用因果融合: {self.use_causal_fusion}")
    
    def forward(self, wrf):
        """
        前向传播
        
        参数:
            wrf: [batch, num_channels, H, W]
        返回:
            encoded: [batch, 8, H//4, W//4]
        """
        # 编码每个通道
        encoded_features = []
        features_dict = {}
        
        for i in range(self.num_channels):
            channel = wrf[:, i:i+1, :, :]
            enc = self.encoders[i](channel)
            encoded_features.append(enc)
            
            # 构建特征字典（用于因果融合）
            if i < len(self.channel_names):
                key_map = {
                    'vil': 'sevir_vil',
                    'ir107': 'sevir_ir107',
                    'cape': 'era5_cape',
                    'cin': 'era5_cin',
                    't2m': 'era5_t2m'
                }
                key = key_map.get(self.channel_names[i], f'feature_{i}')
                features_dict[key] = enc
        
        # 融合
        if self.use_causal_fusion:
            encoded = self.causal_fusion(features_dict)
        else:
            concat = torch.cat(encoded_features, dim=1)
            encoded = self.simple_fusion(concat)
        
        return encoded