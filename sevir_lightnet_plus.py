"""
集成因果先验的 SEVIR LightNet+ 模型
基于因果发现结果，增强闪电预测能力
"""
import torch
import torch.nn as nn
import sys
import os

# 添加 layers 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'layers'))

from layers.ConvLSTM import ConvLSTM2D
from layers.transformer_decoder import TransformerDecoder as Transformer
from causal_encoder import CausalPriorEncoder

class ModulatedDeformConvTM(nn.Module):
    """Ordinary Conv2d fallback retained from the original runnable implementation.

    The t argument is accepted for API/checkpoint compatibility, but is unused.
    This release does not implement deformable or time-modulated convolution.
    """
    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)

    def forward(self, x, t):
        return self.conv(x)


class WRFInfo(nn.Module):
    """WRF信息处理模块（双向ConvLSTM）"""
    def __init__(self, channels, config_dict):
        super(WRFInfo, self).__init__()
        assert channels % 2 == 0
        self.config_dict = config_dict
        self.channels = channels
        
        self.wrf_encoder_convLSTM2D_for = ConvLSTM2D(
            channels, channels//2, kernel_size=5, 
            img_rowcol=(config_dict['GridRowColNum']//2)//2
        )
        self.wrf_encoder_convLSTM2D_rev = ConvLSTM2D(
            channels, channels//2, kernel_size=5, 
            img_rowcol=(config_dict['GridRowColNum']//2)//2
        )

    def forward(self, wrf):
        batch_size = wrf.shape[1]
        device = wrf.device
        
        wrf_h_alltime_for = torch.zeros(
            [self.config_dict['ForecastHourNum'], batch_size, self.channels // 2,
             wrf.shape[3], wrf.shape[4]], dtype=torch.float32, device=device
        )
        wrf_h_alltime_rev = torch.zeros(
            [self.config_dict['ForecastHourNum'], batch_size, self.channels // 2,
             wrf.shape[3], wrf.shape[4]], dtype=torch.float32, device=device
        )
        
        wrf_h = torch.zeros(
            [batch_size, self.channels//2, wrf.shape[3], wrf.shape[4]], 
            dtype=torch.float32, device=device
        )
        wrf_c = torch.zeros(
            [batch_size, self.channels//2, wrf.shape[3], wrf.shape[4]], 
            dtype=torch.float32, device=device
        )
        
        for i in range(self.config_dict['ForecastHourNum']):
            wrf_h_alltime_for[i] = wrf_h
            wrf_h, wrf_c = self.wrf_encoder_convLSTM2D_for(wrf[i], wrf_h, wrf_c)

        wrf_h = torch.zeros(
            [batch_size, self.channels//2, wrf.shape[3], wrf.shape[4]], 
            dtype=torch.float32, device=device
        )
        wrf_c = torch.zeros(
            [batch_size, self.channels//2, wrf.shape[3], wrf.shape[4]], 
            dtype=torch.float32, device=device
        )
        
        for i in range(self.config_dict['ForecastHourNum']-1, -1, -1):
            wrf_h_alltime_rev[i] = wrf_h
            wrf_h, wrf_c = self.wrf_encoder_convLSTM2D_rev(wrf[i], wrf_h, wrf_c)

        wrf_h_alltime = torch.cat([wrf_h_alltime_for, wrf_h_alltime_rev], dim=2)
        
        return wrf_h_alltime, None


class CausalLightNetPlus(nn.Module):
    """
    集成因果先验的LightNet+模型
    
    关键改进:
    1. 使用CausalPriorEncoder编码WRF数据（包含ERA5因果变量）
    2. 因果权重基于发现的关系初始化，但可训练
    3. 保持原有的ConvLSTM和Transformer架构
    """
    def __init__(self, obs_tra_frames, obs_channels, wrf_tra_frames, wrf_channels, 
                 config_dict, causal_config=None):
        super(CausalLightNetPlus, self).__init__()
        
        self.config_dict = config_dict
        self.obs_tra_frames = obs_tra_frames
        self.wrf_tra_frames = wrf_tra_frames
        self.wrf_channels = wrf_channels
        
        mn = (config_dict['GridRowColNum'] // 2) // 2
        
        print(f"\n初始化CausalLightNetPlus:")
        print(f"  WRF通道数: {wrf_channels}")
        print(f"  使用因果先验: {causal_config is not None}")
        
        # 观测编码器（历史闪电）
        self.obs_encoder_module = nn.Sequential(
            nn.Conv2d(obs_channels, 4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(4, 4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(4, 8, kernel_size=5, stride=1, padding=2),
            nn.LayerNorm([8, mn, mn], elementwise_affine=True)
        )
        
        # 编码器ConvLSTM
        self.encoder_ConvLSTM = ConvLSTM2D(8, 8, kernel_size=5, img_rowcol=mn)
        
        self.encoder_h = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=1, stride=1),
            nn.ReLU(),
        )
        self.encoder_c = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=1, stride=1),
            nn.ReLU(),
        )
        
        # 【关键】使用因果先验编码器
        self.wrf_encoder_module = CausalPriorEncoder(
            config_dict=config_dict, 
            causal_config=causal_config,
            num_channels=wrf_channels
        )
        
        # 解码器ConvLSTM
        self.decoder_ConvLSTM = ConvLSTM2D(8, 8, kernel_size=5, img_rowcol=mn)
        
        # 时间调制可变形卷积
        self.dcn_tm = ModulatedDeformConvTM(8, 8, kernel_size=1, stride=1, padding=0)
        
        # 解码器（上采样到原始分辨率）
        self.decoder_module = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=1, stride=1)
        )
        
        # 融合卷积
        self.conv_fusion_h = nn.Conv2d(16, 8, kernel_size=5, stride=1, padding=2, groups=2)
        
        # Transformer
        self.transformer = Transformer(channels=8, layers=1, nhead=1, is_posemb=True)
        
        # WRF信息处理
        self.wrf_info = WRFInfo(channels=8, config_dict=config_dict)

    def forward(self, wrf, obs):
        """
        前向传播
        
        参数:
            wrf: [batch, frames, H, W, channels] - 包含VIL+IR107+ERA5变量
            obs: [batch, frames, H, W, 1] - 历史闪电观测
        
        返回:
            pre_frames: [batch, frames, H, W, 1] - 预测的闪电
        """
        # 转换维度
        obs = obs.permute(1, 0, 4, 2, 3).contiguous()  # [frames, batch, 1, H, W]
        wrf = wrf.permute(1, 0, 4, 2, 3).contiguous()  # [frames, batch, channels, H, W]

        batch_size = obs.shape[1]
        device = wrf.device
        
        # 初始化预测
        pre_frames = torch.zeros(
            [self.wrf_tra_frames, batch_size, 1, wrf.shape[3], wrf.shape[4]], 
            device=device
        )

        # 初始化隐藏状态
        mn = (self.config_dict['GridRowColNum']//2)//2
        h = torch.zeros([batch_size, 8, mn, mn], dtype=torch.float32, device=device)
        c = torch.zeros([batch_size, 8, mn, mn], dtype=torch.float32, device=device)
        
        # 1. 【使用因果编码器】编码WRF数据
        wrf_encoder = []
        for t in range(self.wrf_tra_frames):
            wrf_encoder.append(self.wrf_encoder_module(wrf[t]))
        wrf_encoder = torch.stack(wrf_encoder, dim=0)
        
        # 2. 提取WRF时序信息
        wrf_info_h, _ = self.wrf_info(wrf_encoder)

        # 3. 观测编码（历史闪电）
        for t in range(self.obs_tra_frames):
            obs_encoder = self.obs_encoder_module(obs[t])
            h, c = self.encoder_ConvLSTM(obs_encoder, h, c)
        
        h = self.encoder_h(h)
        c = self.encoder_c(c)
        
        # 4. 解码预测
        for t in range(self.wrf_tra_frames):
            his_enc = self.conv_fusion_h(torch.cat([wrf_info_h[t], h], dim=1))
            wrf_tf = self.transformer(his_enc, wrf_encoder[t])
            h, c = self.decoder_ConvLSTM(wrf_tf, h, c)
            pre = self.dcn_tm(h, t + 1)
            pre = self.decoder_module(pre)
            pre_frames[t] = pre
        
        # 转换回原始维度
        pre_frames = pre_frames.permute(1, 0, 3, 4, 2).contiguous()
        
        return pre_frames
    
    def get_causal_weights(self):
        """获取当前的因果权重（用于监控训练）"""
        if hasattr(self.wrf_encoder_module, 'causal_fusion'):
            if hasattr(self.wrf_encoder_module.causal_fusion, 'causal_weights'):
                with torch.no_grad():
                    weights = torch.softmax(
                        self.wrf_encoder_module.causal_fusion.causal_weights, 
                        dim=0
                    ).cpu().numpy()
                    sources = self.wrf_encoder_module.causal_fusion.source_names
                    return dict(zip(sources, weights))
        return {}