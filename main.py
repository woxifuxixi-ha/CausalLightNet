"""
集成因果先验与改进Focal Loss的主训练脚本（Logits偏置版）
核心改进：
1. 空间权重作为Logits偏置，而非直接乘到损失上（避免双重加权）
2. 大幅降低spatial_lambda（0.3 -> 0.1），使其仅作为微弱提示
3. 优化空间权重归一化，确保数值稳定性
4. 改进训练稳定性
"""
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'layers'))

from config_sevir import read_config
from sevir_generator import ERA5SevirCausalGenerator, load_sevir_file_list
from sevir_lightnet_plus import CausalLightNetPlus
from causal_prior_parser import CausalPriorParser
from scores import Model_eval
from vis import generate_all_visualizations, visualize_dataset_sample

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **kwargs):
        return iterable


def resolve_ablation_flags(mode):
    mode = (mode or 'full').lower()
    if mode == 'full':
        return True, True, True
    if mode == 'no_causal':
        return False, True, True
    if mode == 'no_spatial':
        return True, False, True
    if mode == 'no_focal':
        return True, False, False
    raise ValueError(f'Unknown ablation mode: {mode}')


# ================== 改进1: 动态Focal Loss ==================
class DynamicFocalLoss(nn.Module):
    """
    动态Focal Loss - gamma参数随训练进程自适应调整
    保持与原版一致的计算方式
    """
    def __init__(self, alpha=0.25, gamma_min=0.5, gamma_max=3.0, 
                 total_epochs=50, reduction='mean'):
        super(DynamicFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.total_epochs = total_epochs
        self.reduction = reduction
        self.current_gamma = gamma_min
    
    def update_gamma(self, current_epoch):
        """根据当前epoch更新gamma值（平滑过渡）"""
        progress = current_epoch / self.total_epochs
        # 使用平滑的调度策略
        self.current_gamma = self.gamma_min + (self.gamma_max - self.gamma_min) * progress
    
    def forward(self, inputs, targets):
        """
        inputs: logits (任意形状)
        targets: 0/1 标签 (与inputs相同形状)
        """
        # 计算BCE loss
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        
        # Focal Loss
        F_loss = self.alpha * (1 - pt) ** self.current_gamma * BCE_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss
    
    def get_current_gamma(self):
        return self.current_gamma


# ================== 改进2: 对数空间权重 ==================
class LogSpatialWeightMap:
    """
    对数空间权重图生成器
    基于训练集中闪电出现的长期频率统计
    """
    def __init__(self, grid_size=(384, 384), alpha=10.0, beta=1.0, device='cuda'):
        self.grid_size = grid_size
        self.alpha = alpha
        self.beta = beta
        self.device = device
        
        self.frequency_map = np.zeros(grid_size, dtype=np.float32)
        self.sample_count = 0
        self.weight_map = None
        self.normalized_weight_map = None  # 新增：归一化后的权重图
        self.is_computed = False
    
    def update_frequency(self, lightning_labels):
        """更新频率统计图"""
        if isinstance(lightning_labels, torch.Tensor):
            labels = lightning_labels.cpu().numpy()
        else:
            labels = lightning_labels
        
        # 处理形状 [B, T, H, W, C] -> [B, T, H, W]
        if labels.ndim == 5:
            labels = labels[..., 0]
        
        # 统计每个位置的闪电出现次数
        batch_sum = np.sum(labels > 0, axis=(0, 1))
        
        # 确保形状匹配
        if batch_sum.shape != self.frequency_map.shape:
            h_target, w_target = self.frequency_map.shape
            h_current, w_current = batch_sum.shape
            
            if h_current != h_target or w_current != w_target:
                # 裁剪或填充
                min_h = min(h_target, h_current)
                min_w = min(w_target, w_current)
                temp_map = np.zeros((h_target, w_target), dtype=np.float32)
                temp_map[:min_h, :min_w] = batch_sum[:min_h, :min_w]
                batch_sum = temp_map
        
        self.frequency_map += batch_sum
        self.sample_count += labels.shape[0] * labels.shape[1]
    
    def compute_weight_map(self):
        """根据统计的频率计算权重图"""
        if self.sample_count > 0:
            # 归一化频率
            H = self.frequency_map / self.sample_count
        else:
            H = self.frequency_map
        
        # 对数空间权重: W = 1 + β * log(1 + α * H)
        W_space = 1.0 + self.beta * np.log1p(self.alpha * H)
        
        self.weight_map = torch.from_numpy(W_space).float().to(self.device)
        
        # 归一化权重图：减去均值，除以标准差，使其均值为0，标准差为1
        # 这样作为偏置时不会有系统性的偏移
        mean_weight = self.weight_map.mean()
        std_weight = self.weight_map.std()
        
        if std_weight > 1e-6:  # 避免除零
            self.normalized_weight_map = (self.weight_map - mean_weight) / std_weight
        else:
            self.normalized_weight_map = self.weight_map - mean_weight
        
        self.is_computed = True
        
        return self.weight_map
    
    def get_weight_map(self, normalized=True):
        """
        获取当前权重图（如果未计算则返回None）
        
        Args:
            normalized: 是否返回归一化后的权重图
        """
        if not self.is_computed:
            return None
        
        if normalized:
            return self.normalized_weight_map
        else:
            return self.weight_map
    
    def save_statistics(self, save_path):
        """保存频率统计和权重图"""
        save_dict = {
            'frequency_map': self.frequency_map,
            'sample_count': self.sample_count,
            'alpha': self.alpha,
            'beta': self.beta
        }
        if self.weight_map is not None:
            save_dict['weight_map'] = self.weight_map.cpu().numpy()
        if self.normalized_weight_map is not None:
            save_dict['normalized_weight_map'] = self.normalized_weight_map.cpu().numpy()
        
        np.savez(save_path, **save_dict)
    
    def load_statistics(self, load_path):
        """加载已有的统计信息"""
        data = np.load(load_path)
        self.frequency_map = data['frequency_map']
        self.sample_count = int(data['sample_count'])
        self.alpha = float(data['alpha'])
        self.beta = float(data['beta'])
        
        if 'weight_map' in data and data['weight_map'] is not None:
            self.weight_map = torch.from_numpy(data['weight_map']).float().to(self.device)
            self.is_computed = True
        
        if 'normalized_weight_map' in data and data['normalized_weight_map'] is not None:
            self.normalized_weight_map = torch.from_numpy(data['normalized_weight_map']).float().to(self.device)


# ================== 改进3: Logits偏置版混合损失函数 ==================
class ImprovedFocalLossLogitsBias(nn.Module):
    """
    改进的Focal Loss - Logits偏置版
    
    核心改进：
    1. 将空间权重作为Logits的偏置，而非直接乘到损失上
    2. 避免了"双重加权"问题（Focal权重 × 空间权重）
    3. 空间先验仅影响预测倾向，不直接放大损失
    
    数学原理：
    - 原版（乘法）：Loss = FocalLoss(logits) × spatial_weight  [过于剧烈]
    - 新版（偏置）：Loss = FocalLoss(logits + spatial_bias)     [更温和]
    
    其中 spatial_bias = (normalized_spatial_weight) × lambda
    """
    def __init__(self, alpha=0.25, gamma_min=0.5, gamma_max=3.0, 
                 total_epochs=50, spatial_weight_map=None, 
                 spatial_weight_lambda=0.1,  # 默认降低到0.1
                 reduction='mean'):
        super(ImprovedFocalLossLogitsBias, self).__init__()
        self.dynamic_focal = DynamicFocalLoss(
            alpha, gamma_min, gamma_max, total_epochs, reduction='none'
        )
        self.spatial_weight_map = spatial_weight_map
        self.spatial_weight_lambda = spatial_weight_lambda
        self.reduction = reduction
        
        print(f"\n[ImprovedFocalLoss] 初始化 - Logits偏置模式")
        print(f"  - spatial_lambda: {spatial_weight_lambda:.4f} (微弱提示)")
        print(f"  - 空间权重将作为Logits偏置，而非损失乘数")
    
    def forward(self, inputs, targets):
        """
        inputs: logits [B, T, H, W, 1]
        targets: 0/1 标签 [B, T, H, W, 1]
        
        核心逻辑：
        1. 将归一化的空间权重加到logits上（作为偏置）
        2. 然后计算Focal Loss
        3. 避免了损失的直接乘法
        """
        original_shape = inputs.shape
        B, T, H, W = original_shape[0], original_shape[1], original_shape[2], original_shape[3]
        
        # 获取归一化的空间权重图
        weight_map = None
        if self.spatial_weight_map is not None:
            weight_map = self.spatial_weight_map.get_weight_map(normalized=True)
        
        # 应用空间偏置到Logits
        if weight_map is not None:
            # 扩展权重图维度 [H, W] -> [1, 1, H, W, 1]
            spatial_weights_norm = weight_map.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            
            # 确保维度匹配
            if spatial_weights_norm.shape[2:4] != (H, W):
                # 如果尺寸不匹配，进行插值调整
                spatial_weights_norm = F.interpolate(
                    spatial_weights_norm.squeeze(-1),  # [1, 1, H_w, W_w]
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                ).unsqueeze(-1)  # [1, 1, H, W, 1]
            
            # 将空间权重作为偏置加到logits上
            # 归一化后的权重均值为0，标准差为1
            # 乘以lambda后作为小幅度的偏置
            spatial_bias = spatial_weights_norm * self.spatial_weight_lambda
            inputs_biased = inputs + spatial_bias
        else:
            inputs_biased = inputs
        
        # Flatten操作（与原版一致）
        inputs_flat = torch.flatten(inputs_biased)
        targets_flat = torch.flatten(targets)
        
        # 计算Focal Loss（在偏置后的logits上）
        focal_loss = self.dynamic_focal(inputs_flat, targets_flat)
        
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss
    
    def update_epoch(self, current_epoch):
        """更新epoch相关参数"""
        self.dynamic_focal.update_gamma(current_epoch)
    
    def get_current_gamma(self):
        """获取当前gamma值"""
        return self.dynamic_focal.get_current_gamma()


def selectModel(config_dict, causal_config=None, use_causal=True):
    """选择模型"""
    wrf_channels = 5 if use_causal else 2
    
    model = CausalLightNetPlus(
        obs_tra_frames=config_dict['TruthHistoryHourNum'], 
        obs_channels=1,
        wrf_tra_frames=config_dict['ForecastHourNum'],
        wrf_channels=wrf_channels,
        config_dict=config_dict,
        causal_config=causal_config if use_causal else None
    ).to(config_dict['Device'])
    
    return model


def DoTrain_CausalImprovedLogitsBias(config_dict, 
                                     causal_report_path=None,
                                     use_dynamic_gamma=True,
                                     gamma_min=0.5, 
                                     gamma_max=2.5,
                                     use_spatial_weight=True,
                                     spatial_alpha=5.0,
                                     spatial_beta=0.5,
                                     spatial_lambda=0.1,  # 降低到0.1（原0.3）
                                     compute_spatial_weight_epochs=5,
                                     use_causal=True,
                                     use_improved_focal=True,
                                     experiment_name='full',
                                     enable_target_case_vis=False,
                                     target_case_index=2183,
                                     target_event_id='S749332',
                                     target_t0=26,
                                     show_progress=True):
    """
    集成因果先验与改进Focal Loss的训练函数（Logits偏置版）
    """
    
    # 创建目录
    os.makedirs(config_dict['ModelFileDir'], exist_ok=True)
    os.makedirs(config_dict['RecordFileDir'], exist_ok=True)
    os.makedirs(config_dict['VisResultFileDir'], exist_ok=True)

    if 'cuda' in str(config_dict.get('Device', '')).lower() and not torch.cuda.is_available():
        config_dict['Device'] = 'cpu'

    if not use_improved_focal:
        use_dynamic_gamma = False
        use_spatial_weight = False
    use_dynamic_gamma = bool(use_dynamic_gamma and use_improved_focal)

    print("=" * 70)
    print("LightNet+ 训练 - 因果先验 + 改进Focal Loss (Logits偏置版)")
    print(f"实验模式: {experiment_name}")
    print("=" * 70)
    
    # ========== 解析因果先验 ==========
    causal_config = None
    use_causal = bool(use_causal)
    
    if use_causal and causal_report_path and os.path.exists(causal_report_path):
        try:
            parser = CausalPriorParser(causal_report_path)
            causal_config = parser.get_lightning_causal_config()
            use_causal = len(causal_config.get('sources', [])) > 0
            
            print(f"\n✓ 成功加载因果先验:")
            print(f"  因果源数量: {len(causal_config['sources'])}")
            for source, weight in causal_config['weights'].items():
                print(f"  {source}: 权重={weight:.4f}")
        except Exception as e:
            print(f"\n⚠ 警告: 因果先验加载失败: {e}")
            print("  将使用标准模型（无因果先验）")
    else:
        if use_causal:
            print(f"\n⚠ 未提供因果报告或文件不存在")
            print("  将使用标准模型（无因果先验）")
        else:
            print(f"\n⚠ 当前实验模式关闭了因果先验")
    
    print(f"\n模型配置:")
    print(f"  网格大小: {config_dict['GridRowColNum']}×{config_dict['GridRowColNum']}")
    print(f"  预测帧数: {config_dict['ForecastHourNum']}")
    print(f"  历史帧数: {config_dict['TruthHistoryHourNum']}")
    print(f"  批次大小: {config_dict['Batchsize']}")
    print(f"  学习率: {config_dict['LearningRate']}")
    print(f"  训练轮数: {config_dict['EpochNum']}")
    print(f"  使用因果先验: {use_causal}")
    print(f"  使用改进Focal Loss: {use_improved_focal}")
    
    if use_improved_focal:
        print(f"\n改进Focal Loss配置（Logits偏置版）:")
        print(f"  - 动态Gamma: {'开启' if use_dynamic_gamma else '关闭'}")
        if use_dynamic_gamma:
            print(f"    * gamma范围: {gamma_min} -> {gamma_max}")
        print(f"  - 空间权重模式: {'Logits偏置（温和）' if use_spatial_weight else '关闭'}")
        if use_spatial_weight:
            print(f"    * alpha(敏感度): {spatial_alpha}")
            print(f"    * beta(权重大小): {spatial_beta}")
            print(f"    * lambda(偏置强度): {spatial_lambda} ⭐微弱提示")
            print(f"    * 统计epoch数: {compute_spatial_weight_epochs}")
            print(f"    * 逻辑: 空间权重作为Logits偏置，非损失乘数")
    else:
        print(f"\n改进Focal Loss已关闭，使用 BCEWithLogitsLoss 基线")
    print("=" * 70)

    # ========== 加载数据 ==========
    data_dir = config_dict['WRFFileDir']
    print(f"\n从目录加载数据: {data_dir}")
    
    try:
        train_files = load_sevir_file_list(data_dir, split='train', train_ratio=0.7, val_ratio=0.15)
        val_files = load_sevir_file_list(data_dir, split='val', train_ratio=0.7, val_ratio=0.15)
        test_files = load_sevir_file_list(data_dir, split='test', train_ratio=0.7, val_ratio=0.15)
    except Exception as e:
        print(f"✗ 加载数据文件失败: {e}")
        return

    print("\n创建数据加载器...")
    train_data = ERA5SevirCausalGenerator(
        train_files, config_dict, mode='train',
        lght_threshold=0.01, skip_empty=True,
        use_causal_prior=use_causal, debug=False
    )
    
    if len(train_data) == 0:
        print("✗ 训练集为空！")
        return
    
    train_loader = DataLoader(
        dataset=train_data, batch_size=config_dict['Batchsize'], 
        shuffle=True, num_workers=4, pin_memory=True
    )
    
    val_data = ERA5SevirCausalGenerator(
        val_files, config_dict, mode='val',
        lght_threshold=0.01, skip_empty=True,
        use_causal_prior=use_causal, debug=False
    )
    val_loader = DataLoader(
        dataset=val_data, batch_size=config_dict['Batchsize'], 
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    test_data = ERA5SevirCausalGenerator(
        test_files, config_dict, mode='test',
        lght_threshold=0.01, skip_empty=True,
        use_causal_prior=use_causal, debug=False
    )
    test_loader = DataLoader(
        dataset=test_data, batch_size=config_dict['Batchsize'], 
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    print(f"✓ 数据加载完成:")
    print(f"  训练样本: {len(train_data)}")
    print(f"  验证样本: {len(val_data)}")
    print(f"  测试样本: {len(test_data)}")

    # ========== 初始化空间权重图 ==========
    spatial_weight_map = None
    if use_spatial_weight:
        print("\n检测数据网格尺寸...")
        try:
            sample_X, sample_y = next(iter(train_loader))
            _, sample_obs = sample_X
            actual_grid_size = sample_obs.shape[2:4]
            print(f"✓ 检测到实际网格尺寸: {actual_grid_size}")
        except Exception as e:
            print(f"⚠ 无法自动检测网格尺寸，使用配置值: {e}")
            actual_grid_size = (config_dict['GridRowColNum'], config_dict['GridRowColNum'])
        
        spatial_weight_map = LogSpatialWeightMap(
            grid_size=actual_grid_size,
            alpha=spatial_alpha,
            beta=spatial_beta,
            device=config_dict['Device']
        )
        print(f"✓ 初始化对数空间权重图 (尺寸: {actual_grid_size})")

    # ========== 创建模型 ==========
    print("\n初始化模型...")
    model = selectModel(config_dict, causal_config, use_causal)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ 模型参数量: {n_params:,}")
    with open(os.path.join(config_dict['RecordFileDir'], 'record.txt'), 'a', encoding='utf-8') as f:
        f.write(f"model params: {n_params:,}\r\n")
    
    if use_causal:
        initial_weights = model.get_causal_weights()
        if initial_weights:
            print(f"\n初始因果权重:")
            for source, weight in initial_weights.items():
                print(f"  {source}: {weight:.4f}")

    # ========== 创建损失函数 ==========
    if use_improved_focal:
        criterion = ImprovedFocalLossLogitsBias(
            alpha=0.25,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            total_epochs=config_dict['EpochNum'],
            spatial_weight_map=spatial_weight_map if use_spatial_weight else None,
            spatial_weight_lambda=spatial_lambda,
            reduction='mean'
        )
        print(f"\n✓ 损失函数: 改进Focal Loss (Logits偏置版)")
    else:
        pos_weight_tensor = torch.tensor(2.0).to(config_dict['Device'])
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        print(f"\n✓ 损失函数: BCEWithLogitsLoss (pos_weight=2.0)")

    def compute_loss(pred, target):
        if use_improved_focal:
            return criterion(pred, target)
        return criterion(torch.flatten(pred), torch.flatten(target))

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config_dict['LearningRate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    # 评估器
    model_eval_valdata = Model_eval(config_dict, is_save_model=True)
    model_eval_testdata = Model_eval(config_dict, is_save_model=False)

    print('\n' + "=" * 70)
    print("开始训练!")
    print("=" * 70 + '\n')
    
    # ========== 训练循环 ==========
    for epoch in range(config_dict['EpochNum']):
        # 更新损失函数的epoch参数
        if hasattr(criterion, 'update_epoch'):
            criterion.update_epoch(epoch)
        current_gamma = criterion.get_current_gamma() if hasattr(criterion, 'get_current_gamma') else None
        
        model.train()
        epoch_loss = 0.0
        total_positive = 0
        total_samples = 0
        
        gamma_text = f"{current_gamma:.3f}" if current_gamma is not None else "N/A"
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{config_dict['EpochNum']-1} | Gamma: {gamma_text}")
        if use_causal:
            print("因果先验: 启用")
        if use_spatial_weight and spatial_weight_map is not None and spatial_weight_map.is_computed:
            print("空间权重: 已激活 (Logits偏置模式)")
        print(f"{'='*70}")
        
        # 在前N个epoch统计空间频率
        is_collecting_spatial_stats = (use_spatial_weight and 
                                       epoch < compute_spatial_weight_epochs)
        
        train_iterator = tqdm(
            train_loader,
            desc=f"TRAIN epoch {epoch}",
            total=len(train_loader),
            leave=False
        ) if show_progress else train_loader

        for i, (X, y) in enumerate(train_iterator):
            wrf, obs = X
            label = y
            
            # 统计空间频率
            if is_collecting_spatial_stats:
                try:
                    spatial_weight_map.update_frequency(label)
                except Exception as e:
                    if i == 0:
                        print(f"⚠ 空间频率更新警告: {e}")
            
            total_positive += label.sum().item()
            total_samples += label.numel()
            
            wrf = wrf.to(config_dict['Device'])
            obs = obs.to(config_dict['Device'])
            label = label.to(config_dict['Device'])

            # 前向传播
            pre_frames = model(wrf, obs)
            
            # 计算损失（使用Logits偏置版损失函数）
            loss = compute_loss(pre_frames, label)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()

            if show_progress and hasattr(train_iterator, 'set_postfix'):
                postfix = {
                    'loss': f"{loss.item():.5f}",
                    'avg': f"{epoch_loss / (i + 1):.5f}",
                }
                if current_gamma is not None:
                    postfix['gamma'] = f"{current_gamma:.3f}"
                train_iterator.set_postfix(postfix)

            if (i + 1) % 10 == 0 or (i + 1) == len(train_loader):
                avg_loss = epoch_loss / (i + 1)
                pos_ratio = total_positive / total_samples * 100
                
                status = "[收集空间统计]" if is_collecting_spatial_stats else ""
                print(f'训练 INFO {status}: epoch:{epoch} ({i+1}/{len(train_loader)}) '
                      f'loss:{loss.item():.5f} avg_loss:{avg_loss:.5f} '
                        f'pos_ratio:{pos_ratio:.3f}% gamma:{gamma_text}')
        
        # ========== 计算空间权重图 ==========
        if use_spatial_weight and epoch == compute_spatial_weight_epochs - 1:
            print(f"\n{'='*70}")
            print(f"计算对数空间权重图 (基于前{compute_spatial_weight_epochs}个epoch的统计)")
            print(f"{'='*70}")
            try:
                weight_map = spatial_weight_map.compute_weight_map()
                
                stats_path = os.path.join(config_dict['RecordFileDir'], 
                                         'spatial_weight_statistics.npz')
                spatial_weight_map.save_statistics(stats_path)
                
                # 获取原始和归一化权重图的统计信息
                raw_weight = spatial_weight_map.get_weight_map(normalized=False)
                norm_weight = spatial_weight_map.get_weight_map(normalized=True)
                
                print(f"✓ 空间权重图已计算并保存")
                print(f"  原始权重范围: [{raw_weight.min():.3f}, {raw_weight.max():.3f}]")
                print(f"  原始权重均值: {raw_weight.mean():.3f}")
                print(f"  归一化权重范围: [{norm_weight.min():.3f}, {norm_weight.max():.3f}]")
                print(f"  归一化权重均值: {norm_weight.mean():.3f} (应接近0)")
                print(f"  归一化权重标准差: {norm_weight.std():.3f} (应接近1)")
                print(f"  ⭐ 将作为Logits偏置使用，强度λ={spatial_lambda:.3f}")
            except Exception as e:
                print(f"✗ 空间权重图计算失败: {e}")

        # ========== 显示因果权重变化 ==========
        if use_causal and (epoch + 1) % 5 == 0:
            current_weights = model.get_causal_weights()
            if current_weights:
                print(f"\n因果权重更新 (Epoch {epoch}):")
                for source, weight in current_weights.items():
                    initial_w = causal_config['weights'].get(source, 0)
                    change = weight - initial_w
                    print(f"  {source}: {weight:.4f} (变化: {change:+.4f})")

        # ========== 验证 ==========
        print(f"\n{'='*70}")
        print(f"验证 - Epoch {epoch}")
        print(f"{'='*70}")
        
        model.eval()
        with torch.no_grad():
            save_cases = (epoch % 10 == 0) or (epoch == config_dict['EpochNum'] - 1)
            
            print("验证集评估中...")
            val_metrics = model_eval_valdata.eval(
                val_loader, model, epoch, save_cases=save_cases, stage='VAL', criterion=compute_loss, show_progress=show_progress
            )
            print("✓ 验证集评估完成")
            
            print("测试集评估中...")
            test_metrics = model_eval_testdata.eval(
                test_loader, model, epoch, save_cases=False, stage='TEST', criterion=compute_loss, show_progress=show_progress
            )
            print("✓ 测试集评估完成")
        
        print(f"\nVal Metrics:")
        print(f"  ETS: {val_metrics['ETS']:.5f}")
        print(f"  POD: {val_metrics['POD']:.5f}")
        print(f"  FAR: {val_metrics['FAR']:.5f}")
        print(f"  Precision: {1-val_metrics['FAR']:.5f}")
        print(f"  AUC: {val_metrics['AUC']:.5f}")
        print(f"  SSIM: {val_metrics['SSIM']:.5f}")
        
        print(f"\nTest metrics summary printed above.\n")
        
        scheduler.step(val_metrics['ETS'])
        
        # ========== 保存检查点 ==========
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(
                config_dict['ModelFileDir'], 
                f'causal_improved_logits_bias_epoch_{epoch}.pkl'
            )
            
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss / len(train_loader),
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'use_causal': use_causal,
                'causal_config': causal_config,
                'current_gamma': current_gamma,
                'spatial_weight_config': {
                    'alpha': spatial_alpha,
                    'beta': spatial_beta,
                    'lambda': spatial_lambda,
                    'mode': 'logits_bias'
                }
            }
            
            if use_causal:
                save_dict['current_causal_weights'] = model.get_causal_weights()
            
            torch.save(save_dict, checkpoint_path)
            print(f"✓ 检查点已保存: {checkpoint_path}")

    print("\n" + "=" * 70)
    print("训练完成!")
    print("=" * 70)
    
    # ========== 保存最终因果权重 ==========
    if use_causal:
        final_weights = model.get_causal_weights()
        print(f"\n最终因果权重:")
        for source, weight in final_weights.items():
            initial_w = causal_config['weights'].get(source, 0)
            change = weight - initial_w
            print(f"  {source}: {weight:.4f} (总变化: {change:+.4f})")
        
        weights_path = os.path.join(config_dict['ModelFileDir'], 'final_causal_weights.txt')
        with open(weights_path, 'w') as f:
            f.write("最终因果权重\n")
            f.write("="*50 + "\n")
            for source, weight in final_weights.items():
                initial_w = causal_config['weights'].get(source, 0)
                change = weight - initial_w
                f.write(f"{source}: {weight:.4f} (初始: {initial_w:.4f}, 变化: {change:+.4f})\n")
        print(f"\n✓ 因果权重已保存到: {weights_path}")
    
    # ========== 生成可视化 ==========
    print("\n生成可视化图表...")
    try:
        generate_all_visualizations(config_dict)
        print("✓ 可视化生成完成")
    except Exception as e:
        print(f"✗ 可视化生成失败: {e}")

    if enable_target_case_vis:
        print("\n生成指定事件案例可视化...")
        try:
            case_dir = os.path.join(config_dict['VisResultFileDir'], 'case_visualizations')
            visualize_dataset_sample(
                model,
                test_data,
                target_case_index,
                config_dict,
                case_dir,
                event_id=target_event_id,
                t0=target_t0,
                case_tag=experiment_name,
            )
            print("✓ 指定事件案例可视化完成")
        except Exception as e:
            print(f"✗ 指定事件案例可视化失败: {e}")


if __name__ == "__main__":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='SEVIR LightNet+ experiment runner')
    parser.add_argument('--mode', default='full', choices=['full', 'no_causal', 'no_spatial', 'no_focal'],
                        help='full / no_causal / no_spatial / no_focal')
    parser.add_argument('--config', default='config_train_sevir', help='config file path')
    parser.add_argument('--data-dir', default=None, help='override data directory')
    parser.add_argument('--output-dir', default=None, help='override output directory')
    parser.add_argument('--device', default=None, help='override device string, e.g. cuda:0 or cpu')
    parser.add_argument('--causal-report', default='causal_report_strict.txt', help='causal report path')
    parser.add_argument('--case-index', type=int, default=None, help='optional target sample index for visualization')
    parser.add_argument('--case-event', default='', help='target event id for visualization title')
    parser.add_argument('--case-t0', type=int, default=0, help='target event T0 for visualization title')
    parser.add_argument('--no-case-vis', action='store_true', help='disable target case visualization')
    parser.add_argument('--no-progress', action='store_true', help='disable tqdm progress bars')
    args = parser.parse_args()

    use_causal, use_spatial_weight, use_improved_focal = resolve_ablation_flags(args.mode)

    config_dict = read_config(args.config)
    if args.data_dir:
        config_dict['WRFFileDir'] = args.data_dir
        config_dict['TruthFileDir'] = args.data_dir
        config_dict['TruthFileDirGrid'] = args.data_dir

    if args.device:
        config_dict['Device'] = args.device
    elif 'cuda' in str(config_dict.get('Device', '')).lower() and not torch.cuda.is_available():
        config_dict['Device'] = 'cpu'

    base_save_dir = args.output_dir or os.path.join('results', 'sevir_experiments', args.mode)
    config_dict['ModelFileDir'] = os.path.join(base_save_dir, 'models')
    config_dict['RecordFileDir'] = os.path.join(base_save_dir, 'records')
    config_dict['VisResultFileDir'] = os.path.join(base_save_dir, 'visualizations')

    print("\n" + "=" * 70)
    print("SEVIR LightNet+ experiment runner")
    print("=" * 70)
    print(f"mode: {args.mode}")
    print(f"save dir: {base_save_dir}")
    print(f"data dir: {config_dict['WRFFileDir']}")
    print(f"device: {config_dict['Device']}")
    print(f"causal report: {args.causal_report}")
    print("=" * 70)

    try:
        DoTrain_CausalImprovedLogitsBias(
            config_dict,
            causal_report_path=args.causal_report,
            use_dynamic_gamma=True,
            gamma_min=0.5,
            gamma_max=2.5,
            use_spatial_weight=use_spatial_weight,
            spatial_alpha=5.0,
            spatial_beta=0.5,
            spatial_lambda=0.1,
            compute_spatial_weight_epochs=5,
            use_causal=use_causal,
            use_improved_focal=use_improved_focal,
            experiment_name=args.mode,
            enable_target_case_vis=(not args.no_case_vis and args.mode == 'full' and args.case_index is not None),
            target_case_index=args.case_index,
            target_event_id=args.case_event,
            target_t0=args.case_t0,
            show_progress=not args.no_progress,
        )
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
    except Exception as e:
        print(f"\n\n✗ 训练出错: {e}")
        import traceback
        traceback.print_exc()