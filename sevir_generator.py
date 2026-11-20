"""
增强因果信息的ERA5-SEVIR数据加载器 (修复完整版)
修复内容:
1. 添加 __len__ 方法，解决 DataLoader 报错
2. 增加 NaN 值处理，防止 AUC/ETS 计算失败
3. 适配 era5_var_frame_XX 格式的数据结构
"""
import numpy as np
import h5py
import os
from torch.utils.data import Dataset as py_Dataset
from pathlib import Path
import scipy.ndimage

# ----------------------------------------------------------------------
# 1. 将 load_sevir_file_list 放在最前面，方便其他模块导入
# ----------------------------------------------------------------------
def load_sevir_file_list(data_dir, split='train', train_ratio=0.7, val_ratio=0.15):
    """
    加载并划分数据集文件列表
    """
    if not os.path.exists(data_dir):
        raise ValueError(f"目录不存在: {data_dir}")

    # 获取所有h5文件并排序
    all_files = sorted([str(p) for p in Path(data_dir).glob("*.h5")])
    n_total = len(all_files)
    
    if n_total == 0:
        # 备用方案：尝试 os.listdir
        all_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.h5')])
        n_total = len(all_files)
        if n_total == 0:
            raise ValueError(f"在目录 {data_dir} 中未找到 .h5 文件")
    
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    if split == 'train':
        return all_files[:n_train]
    elif split == 'val':
        return all_files[n_train:n_train+n_val]
    elif split == 'test':
        return all_files[n_train+n_val:]
    else:
        raise ValueError(f"未知的划分模式: {split}")


class ERA5SevirCausalGenerator(py_Dataset):
    """
    增强版数据加载器 - 包含因果先验的ERA5变量
    自动处理分帧存储的ERA5数据 (frame_XX)
    """
    
    def __init__(self, event_files, config_dict, mode='train', 
                 lght_threshold=0.01, 
                 skip_empty=False,
                 use_causal_prior=True,
                 debug=False):
        
        self.event_files = event_files
        self.config_dict = config_dict
        self.mode = mode
        self.debug = debug
        self.skip_empty = skip_empty
        self.lght_threshold = lght_threshold
        self.use_causal_prior = use_causal_prior
        
        self.grid_size = config_dict['GridRowColNum']
        self.forecast_hours = config_dict['ForecastHourNum']
        self.history_hours = config_dict['TruthHistoryHourNum']
        
        # SEVIR变量 (通常是3D整体存储)
        self.lght_key = 'sevir_lght'
        self.vil_key = 'sevir_vil'
        self.ir107_key = 'sevir_ir107'
        
        # ERA5因果变量 (基础名称)
        self.cape_base = 'era5_cape'
        self.cin_base = 'era5_cin'
        self.t2m_base = 'era5_t2m'
        
        print(f"[ERA5SevirCausal-{mode}] 配置:")
        print(f"  网格大小: {self.grid_size}×{self.grid_size}")
        print(f"  预测帧数: {self.forecast_hours}")
        print(f"  历史帧数: {self.history_hours}")
        print(f"  使用因果先验: {self.use_causal_prior}")
        
        self.samples = self._build_sample_index()
        
        # 计算输入通道数
        self.wrf_channels = 2  # VIL + IR107
        if self.use_causal_prior:
            self.wrf_channels += 3  # + CAPE + CIN + T2M
        
        print(f"[ERA5SevirCausal-{mode}] 初始化完成:")
        print(f"  样本数: {len(self.samples)}")
        print(f"  WRF通道数: {self.wrf_channels}")

    # ==================================================================
    # 【修复 1】添加 __len__ 方法，解决 'object has no len()' 错误
    # ==================================================================
    def __len__(self):
        return len(self.samples)
    
    def _build_sample_index(self):
        """构建样本索引"""
        samples = []
        skipped_files = 0
        skipped_empty = 0
        skipped_missing_era5 = 0
        
        for file_path in self.event_files:
            if not os.path.exists(file_path):
                skipped_files += 1
                continue
            
            try:
                with h5py.File(file_path, 'r') as f:
                    # 检查必需的SEVIR变量
                    if self.lght_key not in f or self.vil_key not in f:
                        skipped_files += 1
                        continue
                    
                    # 检查ERA5变量 (检查第0帧是否存在)
                    if self.use_causal_prior:
                        # 只要 frame_00 存在，我们就认为该变量存在
                        if f"{self.cape_base}_frame_00" not in f:
                            skipped_missing_era5 += 1
                            continue
                    
                    n_frames = f[self.vil_key].shape[2]
                    window_size = self.history_hours + self.forecast_hours
                    
                    if n_frames < window_size:
                        skipped_files += 1
                        continue
                    
                    # 检查是否有闪电
                    if self.skip_empty:
                        lght_data = np.array(f[self.lght_key])
                        if lght_data.max() == 0:
                            skipped_empty += 1
                            continue
                    
                    stride = self.forecast_hours
                    for start_idx in range(0, n_frames - window_size + 1, stride):
                        # 检查预测部分是否有闪电
                        if self.skip_empty:
                            forecast_start = start_idx + self.history_hours
                            forecast_end = start_idx + window_size
                            lght_forecast = lght_data[:, :, forecast_start:forecast_end]
                            if lght_forecast.max() == 0:
                                skipped_empty += 1
                                continue
                        
                        samples.append({
                            'file_path': file_path,
                            'start_idx': start_idx,
                            'window_size': window_size
                        })
            
            except Exception as e:
                if self.debug:
                    print(f"  警告: {os.path.basename(file_path)}: {e}")
                skipped_files += 1
                continue
        
        if skipped_files > 0:
            print(f"  跳过无效文件: {skipped_files}")
        if skipped_empty > 0:
            print(f"  跳过无闪电样本: {skipped_empty}")
        if skipped_missing_era5 > 0:
            print(f"  跳过缺失ERA5的文件: {skipped_missing_era5}")
        
        return samples

    def _load_era5_var(self, f, base_name, start_frame, end_frame):
        """
        读取分散的帧 (key_frame_00, key_frame_01...) 并拼接成 3D 数组
        """
        frames = []
        # 遍历需要的时间段
        for i in range(start_frame, end_frame):
            key = f"{base_name}_frame_{i:02d}"  # 构造 key，例如 era5_cape_frame_05
            
            if key in f:
                frames.append(f[key][:])
            else:
                # 容错处理：如果缺帧，用全0或上一帧填充
                # 这里简单抛出异常，如果数据质量较差，可以改为 np.zeros_like(frames[-1])
                raise ValueError(f"文件缺失关键帧: {key}")
        
        # 将 list of [H, W] 堆叠成 [H, W, T]
        return np.stack(frames, axis=2)

    def __getitem__(self, index):
        sample_info = self.samples[index]
        
        try:
            with h5py.File(sample_info['file_path'], 'r') as f:
                start_idx = sample_info['start_idx']
                history_end = start_idx + self.history_hours
                forecast_end = history_end + self.forecast_hours
                
                # 读取SEVIR数据 (原生3D格式，直接读取切片)
                lght_data = np.array(f[self.lght_key][:, :, start_idx:forecast_end])
                vil_data = np.array(f[self.vil_key][:, :, history_end:forecast_end])
                ir107_data = np.array(f[self.ir107_key][:, :, history_end:forecast_end])
                
                # 读取ERA5因果变量 (使用拼接函数)
                if self.use_causal_prior:
                    cape_data = self._load_era5_var(f, self.cape_base, history_end, forecast_end)
                    cin_data = self._load_era5_var(f, self.cin_base, history_end, forecast_end)
                    t2m_data = self._load_era5_var(f, self.t2m_base, history_end, forecast_end)
                
                # 调整尺寸 (Resize)
                lght_data = self._resize_data(lght_data, self.grid_size)
                vil_data = self._resize_data(vil_data, self.grid_size)
                ir107_data = self._resize_data(ir107_data, self.grid_size)
                
                if self.use_causal_prior:
                    cape_data = self._resize_data(cape_data, self.grid_size)
                    cin_data = self._resize_data(cin_data, self.grid_size)
                    t2m_data = self._resize_data(t2m_data, self.grid_size)
                
                # 转换维度: [H, W, T] -> [T, H, W]
                lght_data = np.transpose(lght_data, (2, 0, 1))
                vil_data = np.transpose(vil_data, (2, 0, 1))
                ir107_data = np.transpose(ir107_data, (2, 0, 1))
                
                if self.use_causal_prior:
                    cape_data = np.transpose(cape_data, (2, 0, 1))
                    cin_data = np.transpose(cin_data, (2, 0, 1))
                    t2m_data = np.transpose(t2m_data, (2, 0, 1))
                
                # 数据预处理 (Normalization)
                lght_data = self._preprocess_lght(lght_data)
                vil_data = self._preprocess_vil(vil_data)
                ir107_data = self._preprocess_ir107(ir107_data)
                
                if self.use_causal_prior:
                    cape_data = self._preprocess_cape(cape_data)
                    cin_data = self._preprocess_cin(cin_data)
                    t2m_data = self._preprocess_t2m(t2m_data)
                
                # 构建输入
                history_batch = lght_data[:self.history_hours, :, :, np.newaxis]
                
                # WRF输入：VIL + IR107 + ERA5因果变量
                if self.use_causal_prior:
                    wrf_batch = np.stack([
                        vil_data, ir107_data, 
                        cape_data, cin_data, t2m_data
                    ], axis=-1)
                else:
                    wrf_batch = np.stack([vil_data, ir107_data], axis=-1)
                
                # 标签
                label_data = lght_data[self.history_hours:, :, :]
                label_data = (label_data > self.lght_threshold).astype(np.float32)
                label_batch = label_data.reshape(self.forecast_hours, -1, 1)
                
                return [wrf_batch.astype(np.float32), 
                        history_batch.astype(np.float32)], label_batch.astype(np.float32)
        
        except Exception as e:
            if self.debug:
                print(f"  错误: 样本 {index}: {e}")
            return self._get_zero_sample()
    
    def _resize_data(self, data, target_size):
        """调整数据大小"""
        if data.shape[0] == target_size and data.shape[1] == target_size:
            return data
        zoom_factors = [target_size / data.shape[0], target_size / data.shape[1], 1]
        return scipy.ndimage.zoom(data, zoom_factors, order=1)
    
    def _preprocess_lght(self, data): return np.clip(data, 0, 1)
    
    def _preprocess_vil(self, data):
        data = np.clip(data, 0, 254)
        return data / 254.0
    
    def _preprocess_ir107(self, data):
        data = data.copy()
        mask = data < -10000
        if mask.any():
            valid_data = data[~mask]
            fill_value = np.median(valid_data) if len(valid_data) > 0 else -40
            data[mask] = fill_value
        data = np.clip(data, -80, 40)
        return (data + 80) / 120.0
    
    # ==================================================================
    # 【修复 2】在预处理中增加 NaN 清洗，防止 AUC/ETS 计算出 NaN
    # ==================================================================
    def _preprocess_cape(self, data): 
        # 填充 NaN 为 0
        data = np.nan_to_num(data, nan=0.0)
        return np.clip(data, 0, 5000) / 5000.0
    
    def _preprocess_cin(self, data): 
        # 填充 NaN 为 0
        data = np.nan_to_num(data, nan=0.0)
        return (np.clip(data, -500, 0) + 500) / 500.0
    
    def _preprocess_t2m(self, data): 
        # 填充 NaN 为 0 (或者平均温度)
        data = np.nan_to_num(data, nan=0.0)
        return (np.clip(data, 233, 323) - 233) / 90.0
    
    def _get_zero_sample(self):
        m, n = self.grid_size, self.grid_size
        history_batch = np.zeros((self.history_hours, m, n, 1), dtype=np.float32)
        wrf_batch = np.zeros((self.forecast_hours, m, n, self.wrf_channels), dtype=np.float32)
        label_batch = np.zeros((self.forecast_hours, m * n, 1), dtype=np.float32)
        return [wrf_batch, history_batch], label_batch

# ----------------------------------------------------------------------
# 3. 简单的测试逻辑
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("="*70)
    print("测试增强因果信息的数据加载器 (修复版)")
    print("="*70)
    # 此处可以添加简单的实例化测试代码
    # dataset = ERA5SevirCausalGenerator(files, config...)
    # print(len(dataset))