"""
增强版评价指标模块
新增功能：
1. AUC (Area Under Curve)
2. SSIM (Structural Similarity Index)
3. 时序CSI统计
4. 时空误差分析
5. 案例保存用于可视化
"""
import torch
import datetime
import os
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from skimage.metrics import structural_similarity as ssim
import json
import pickle

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **kwargs):
        return iterable


def _safe_metric_divide(numerator, denominator, eps=1e-10):
    numerator = float(numerator)
    denominator = float(denominator)
    return numerator / (denominator + eps)


def _calc_metrics_from_counts(tp, fp, fn, tn, eps=1e-10):
    tp = float(tp)
    fp = float(fp)
    fn = float(fn)
    tn = float(tn)
    r = ((tp + fp) * (tp + fn)) / (tp + fp + fn + tn + eps)
    pod = _safe_metric_divide(tp, tp + fn, eps)
    far = _safe_metric_divide(fp, tp + fp, eps)
    csi = _safe_metric_divide(tp, tp + fp + fn, eps)
    ets = _safe_metric_divide(tp - r, tp + fp + fn - r, eps)
    f1 = _safe_metric_divide(2.0 * tp, 2.0 * tp + fp + fn, eps)
    return {
        'POD': pod,
        'FAR': far,
        'CSI': csi,
        'ETS': ets,
        'F1': f1,
        'LOU': csi,
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'TN': tn,
    }


def _format_metric_block(metrics):
    return (
        f"loss={metrics['loss']:.4f}  ETS={metrics['ETS']:.4f}  CSI={metrics['CSI']:.4f}  "
        f"FAR={metrics['FAR']:.4f}  POD={metrics['POD']:.4f}  AUC={metrics['AUC']:.4f}  "
        f"F1={metrics['F1']:.4f}  LOU={metrics['LOU']:.4f}  SSIM={metrics['SSIM']:.4f}"
    )


def _format_test_summary(metrics):
    lines = ["========== TEST ==========" , _format_metric_block(metrics), ""]
    lines.append("mean over lead times:")
    mean_block = metrics.get('mean_over_lead_times', {})
    lines.append(
        f"ETS={mean_block.get('ETS', 0.0):.4f}  CSI={mean_block.get('CSI', 0.0):.4f}  "
        f"FAR={mean_block.get('FAR', 0.0):.4f}  POD={mean_block.get('POD', 0.0):.4f}  "
        f"F1={mean_block.get('F1', 0.0):.4f}  LOU={mean_block.get('LOU', 0.0):.4f}"
    )
    lines.append("")
    lines.append("per lead time:")
    for item in metrics.get('per_lead_times', []):
        lines.append(
            f"t{item['lead_time']}: ETS={item['ETS']:.4f}  CSI={item['CSI']:.4f}  "
            f"FAR={item['FAR']:.4f}  POD={item['POD']:.4f}  F1={item['F1']:.4f}  LOU={item['LOU']:.4f}"
        )
    return "\n".join(lines)


class Cal_params_epoch(object):
    """计算每个epoch的评价参数"""
    
    def __init__(self):
        # n1->TP  n2->FP  n3->FN  n4->TN
        self.n1 = 0
        self.n2 = 0
        self.n3 = 0
        self.n4 = 0
        self.n1sum = 0
        self.n2sum = 0
        self.n3sum = 0
        self.n4sum = 0
        self.eps = 1e-10
        
        # 新增：按时间步统计
        self.timestep_metrics = []  # 每个时间步的指标
        
        # 新增：用于SSIM和AUC计算的原始数据
        self.all_predictions = []  # 所有预测的概率值
        self.all_labels = []  # 所有标签

    def _transform_sum(self, y_true, y_pred):
        """转换为求和格式"""
        y_true = y_true.permute(1, 0, 2, 3).cpu().contiguous()
        y_pred = y_pred.permute(1, 0, 2, 3, 4).cpu().contiguous()
        y_pred = torch.round(torch.sigmoid(y_pred))
        frames = y_true.shape[0]
        sum_true = torch.zeros(y_true[0].shape)
        sum_pred = torch.zeros(y_pred[0].shape)
        for i in range(frames):
            sum_true += y_true[i]
            sum_pred += y_pred[i]
        sum_true = torch.flatten(sum_true)
        sum_pred = torch.flatten(sum_pred)
        return sum_true, sum_pred

    def _transform(self, y_true, y_pred):
        """转换为二值化格式"""
        y_true = y_true.cpu()
        y_pred = y_pred.cpu()
        y_true = torch.flatten(y_true)
        y_pred = torch.flatten(torch.sigmoid(y_pred))
        y_pred = torch.round(y_pred)
        return y_true, y_pred

    def _POD_(self, n1, n3):
        """Probability of Detection"""
        return torch.true_divide(n1, n1 + n3 + self.eps)

    def _FAR_(self, n1, n2):
        """False Alarm Ratio"""
        return torch.true_divide(n2, n1 + n2 + self.eps)

    def _TS_(self, n1, n2, n3):
        """Threat Score"""
        return torch.true_divide(n1, n1 + n2 + n3 + self.eps)

    def _ETS_(self, n1, n2, n3, r):
        """Equitable Threat Score"""
        return torch.true_divide(n1 - r, n1 + n2 + n3 - r + self.eps)
    
    def _CSI_(self, n1, n2, n3):
        """Critical Success Index (same as TS)"""
        return self._TS_(n1, n2, n3)

    def cal_batch(self, y_true, y_pred):
        """计算单个batch的指标"""
        y_true, y_pred = self._transform(y_true, y_pred)
        n1 = torch.sum((y_pred > 0) & (y_true > 0))
        n2 = torch.sum((y_pred > 0) & (y_true < 1))
        n3 = torch.sum((y_pred < 1) & (y_true > 0))
        n4 = torch.sum((y_pred < 1) & (y_true < 1))
        r = torch.true_divide((n1 + n2) * (n1 + n3), n1 + n2 + n3 + n4)
        pod = self._POD_(n1, n3)
        far = self._FAR_(n1, n2)
        ts = self._TS_(n1, n2, n3)
        ets = self._ETS_(n1, n2, n3, r)
        self.n1 += n1
        self.n2 += n2
        self.n3 += n3
        self.n4 += n4
        return pod, far, ts, ets

    def cal_batch_sum(self, y_true, y_pred):
        """计算batch的求和指标"""
        y_true, y_pred = self._transform_sum(y_true, y_pred)
        n1 = torch.sum((y_pred > 0) & (y_true > 0))
        n2 = torch.sum((y_pred > 0) & (y_true < 1))
        n3 = torch.sum((y_pred < 1) & (y_true > 0))
        n4 = torch.sum((y_pred < 1) & (y_true < 1))
        r = torch.true_divide((n1 + n2) * (n1 + n3), n1 + n2 + n3 + n4)
        pod = self._POD_(n1, n3)
        far = self._FAR_(n1, n2)
        ts = self._TS_(n1, n2, n3)
        ets = self._ETS_(n1, n2, n3, r)
        self.n1sum += n1
        self.n2sum += n2
        self.n3sum += n3
        self.n4sum += n4
        return pod, far, ts, ets
    
    def cal_timestep_metrics(self, y_true, y_pred, grid_size):
        """
        新增：计算每个时间步的CSI指标
        
        Args:
            y_true: [batch, frames, H*W, 1] or [batch, frames, H, W, 1]
            y_pred: [batch, frames, H, W, 1] (logits)
            grid_size: int, 网格大小 H (=W)
        
        Returns:
            csi_per_time: list of CSI values for each timestep
        """
        y_true = y_true.cpu()
        y_pred = torch.sigmoid(y_pred).cpu()
        y_pred_binary = torch.round(y_pred)
        
        batch_size, n_frames = y_true.shape[0], y_true.shape[1]
        
        # 检查y_true的维度并重塑
        if len(y_true.shape) == 4:
            # [batch, frames, H*W, 1] -> [batch, frames, H, W, 1]
            y_true = y_true.reshape(batch_size, n_frames, grid_size, grid_size, 1)
        
        csi_per_time = []
        
        for t in range(n_frames):
            # 展平当前时间步的所有样本
            true_t = torch.flatten(y_true[:, t, :, :, :])
            pred_t = torch.flatten(y_pred_binary[:, t, :, :, :])
            
            # 计算混淆矩阵
            n1 = torch.sum((pred_t > 0) & (true_t > 0)).item()
            n2 = torch.sum((pred_t > 0) & (true_t < 1)).item()
            n3 = torch.sum((pred_t < 1) & (true_t > 0)).item()
            
            # 计算CSI
            csi = n1 / (n1 + n2 + n3 + self.eps)
            csi_per_time.append(csi)
        
        return csi_per_time
    
    def collect_for_auc_ssim(self, y_true, y_pred, grid_size):
        """
        新增：收集原始数据用于AUC和SSIM计算
        
        Args:
            y_true: [batch, frames, H*W, 1] or [batch, frames, H, W, 1]
            y_pred: [batch, frames, H, W, 1] (logits)
            grid_size: int, 网格大小
        """
        # 保存预测概率（用于AUC）
        y_pred_prob = torch.sigmoid(y_pred).cpu().numpy()
        y_true_np = y_true.cpu().numpy()
        
        # 检查y_true的维度并重塑
        if len(y_true_np.shape) == 4:
            # [batch, frames, H*W, 1] -> [batch, frames, H, W, 1]
            batch_size, n_frames = y_true_np.shape[0], y_true_np.shape[1]
            y_true_np = y_true_np.reshape(batch_size, n_frames, grid_size, grid_size, 1)
        
        self.all_predictions.append(y_pred_prob)
        self.all_labels.append(y_true_np)

    def cal_epoch(self):
        """计算整个epoch的指标"""
        r = torch.true_divide((self.n1 + self.n2) * (self.n1 + self.n3), 
                             self.n1 + self.n2 + self.n3 + self.n4)
        pod = self._POD_(self.n1, self.n3)
        far = self._FAR_(self.n1, self.n2)
        ts = self._TS_(self.n1, self.n2, self.n3)
        ets = self._ETS_(self.n1, self.n2, self.n3, r)
        return pod, far, ts, ets

    def cal_epoch_sum(self):
        """计算整个epoch的求和指标"""
        r = torch.true_divide((self.n1sum + self.n2sum) * (self.n1sum + self.n3sum), 
                             self.n1sum + self.n2sum + self.n3sum + self.n4sum)
        pod = self._POD_(self.n1sum, self.n3sum)
        far = self._FAR_(self.n1sum, self.n2sum)
        ts = self._TS_(self.n1sum, self.n2sum, self.n3sum)
        ets = self._ETS_(self.n1sum, self.n2sum, self.n3sum, r)
        return pod, far, ts, ets
    
    def calculate_auc(self):
        """
        新增：计算AUC (Area Under ROC Curve)
        
        Returns:
            auc_score: float
        """
        if len(self.all_predictions) == 0:
            return 0.0
        
        try:
            # 合并所有batch的数据
            all_pred = np.concatenate(self.all_predictions, axis=0)  # [N, frames, H, W, 1]
            all_true = np.concatenate(self.all_labels, axis=0)
            
            # 展平
            all_pred_flat = all_pred.flatten()
            all_true_flat = all_true.flatten()
            
            # 检查是否有两个类别
            if len(np.unique(all_true_flat)) < 2:
                print("  Warning: Only one class present, AUC is not defined")
                return 0.0
            
            # 计算AUC
            auc_score = roc_auc_score(all_true_flat, all_pred_flat)
            return auc_score
        
        except Exception as e:
            print(f"  Warning: Failed to calculate AUC: {e}")
            return 0.0
    
    def calculate_ssim(self):
        """
        新增：计算平均SSIM (Structural Similarity Index)
        
        Returns:
            mean_ssim: float
        """
        if len(self.all_predictions) == 0:
            return 0.0
        
        try:
            all_pred = np.concatenate(self.all_predictions, axis=0)
            all_true = np.concatenate(self.all_labels, axis=0)
            
            # 二值化预测
            all_pred_binary = (all_pred > 0.5).astype(np.float32)
            
            # 计算每个样本每个时间步的SSIM
            ssim_scores = []
            n_samples, n_frames, H, W, _ = all_pred.shape
            
            for i in range(min(n_samples, 100)):  # 限制计算数量以节省时间
                for t in range(n_frames):
                    pred_frame = all_pred_binary[i, t, :, :, 0]
                    true_frame = all_true[i, t, :, :, 0]
                    
                    # 计算SSIM（需要归一化到相同范围）
                    ssim_val = ssim(true_frame, pred_frame, 
                                   data_range=1.0)
                    ssim_scores.append(ssim_val)
            
            mean_ssim = np.mean(ssim_scores) if ssim_scores else 0.0
            return mean_ssim
        
        except Exception as e:
            print(f"  Warning: Failed to calculate SSIM: {e}")
            return 0.0


class Model_eval(object):
    """模型评估类 - 增强版"""
    
    def __init__(self, config_dict, is_save_model):
        self.config_dict = config_dict
        self.is_save_model = is_save_model
        
        # 原有的最优指标
        self.maxPOD = -0.5
        self.maxPOD_epoch = -1
        self.minFAR = 1.1
        self.minFAR_epoch = -1
        self.maxETS = -0.5
        self.maxETS_epoch = -1
        
        # 新增：最优指标
        self.maxAUC = -0.5
        self.maxAUC_epoch = -1
        self.maxSSIM = -0.5
        self.maxSSIM_epoch = -1
        
        # 新增：历史记录
        self.metrics_history = {
            'epoch': [],
            'POD': [],
            'FAR': [],
            'TS': [],
            'ETS': [],
            'AUC': [],
            'SSIM': [],
            'CSI_per_timestep': []  # 每个epoch的时序CSI
        }
        
        # 新增：保存可视化案例
        self.viz_save_dir = os.path.join(config_dict['VisResultFileDir'], 'cases')
        os.makedirs(self.viz_save_dir, exist_ok=True)
        
        if self.is_save_model:
            with open(os.path.join(self.config_dict['RecordFileDir'], 'record.txt'), 'a') as f:
                f.write(str(datetime.datetime.now()) + '\r\n')
                f.write(str(config_dict) + '\r\n')

    def __del__(self):
        info = ('`model name: {}`\n'
                'maxPOD: {} maxPOD_epoch: {}\n'
                'minFAR: {} minFAR_epoch: {}\n'
                'maxETS: {} maxETS_epoch: {}\n'
                'maxAUC: {} maxAUC_epoch: {}\n'
                'maxSSIM: {} maxSSIM_epoch: {}\n').format(
            self.config_dict['NetName'], 
            self.maxPOD, self.maxPOD_epoch,
            self.minFAR, self.minFAR_epoch, 
            self.maxETS, self.maxETS_epoch,
            self.maxAUC, self.maxAUC_epoch,
            self.maxSSIM, self.maxSSIM_epoch
        )
        print(info)
        
        if self.is_save_model and self.maxPOD_epoch != -1:
            with open(os.path.join(self.config_dict['RecordFileDir'], 'record.txt'), 'a') as f:
                f.write(info + '\r\n')
            
            # 保存完整的指标历史
            history_path = os.path.join(self.config_dict['RecordFileDir'], 'metrics_history.pkl')
            with open(history_path, 'wb') as f:
                pickle.dump(self.metrics_history, f)
            print(f"Metrics history saved to {history_path}")

    def eval(self, dataloader, model, epoch, save_cases=False, stage='VAL', criterion=None, show_progress=True):
        """
        评估模型 - 增强版

        Args:
            dataloader: 数据加载器
            model: 模型
            epoch: 当前epoch
            save_cases: 是否保存典型案例用于可视化
            stage: VAL / TEST / TRAIN
            criterion: 可选损失函数，用于统计评估 loss
            show_progress: 是否显示进度条

        Returns:
            metrics_dict: 包含整体、分时序和AUC/SSIM的字典
        """
        saved_cases = 0
        max_cases = 5
        grid_size = self.config_dict['GridRowColNum']
        forecast_frames = self.config_dict['ForecastHourNum']
        eps = 1e-10

        overall_counts = {'tp': 0.0, 'fp': 0.0, 'fn': 0.0, 'tn': 0.0}
        lead_counts = [
            {'tp': 0.0, 'fp': 0.0, 'fn': 0.0, 'tn': 0.0}
            for _ in range(forecast_frames)
        ]
        batch_losses = []
        pred_batches = []
        true_batches = []

        iterator = tqdm(
            dataloader,
            desc=f"{stage.upper()} epoch {epoch}",
            total=len(dataloader),
            leave=False
        ) if show_progress else dataloader

        for batch_idx, (X, y) in enumerate(iterator):
            wrf, obs = X
            label = y

            wrf = wrf.to(self.config_dict['Device'])
            obs = obs.to(self.config_dict['Device'])
            label = label.to(self.config_dict['Device'])

            pre_frames = model(wrf, obs)

            if criterion is not None:
                try:
                    batch_loss = criterion(pre_frames, label)
                except Exception:
                    batch_loss = criterion(torch.flatten(pre_frames), torch.flatten(label))
            else:
                batch_loss = torch.nn.functional.binary_cross_entropy_with_logits(pre_frames, label)
            batch_losses.append(float(batch_loss.detach().cpu().item()))

            label_cpu = label.detach().cpu().float()
            pred_prob_cpu = torch.sigmoid(pre_frames.detach().cpu().float())

            if label_cpu.dim() == 4:
                label_cpu = label_cpu.reshape(label_cpu.shape[0], label_cpu.shape[1], grid_size, grid_size, 1)
            if pred_prob_cpu.dim() == 4:
                pred_prob_cpu = pred_prob_cpu.reshape(pred_prob_cpu.shape[0], pred_prob_cpu.shape[1], grid_size, grid_size, 1)

            pred_batches.append(pred_prob_cpu.numpy())
            true_batches.append(label_cpu.numpy())

            for lead_idx in range(forecast_frames):
                true_t = label_cpu[:, lead_idx, :, :, 0].reshape(-1).numpy() > 0.5
                pred_t = pred_prob_cpu[:, lead_idx, :, :, 0].reshape(-1).numpy() >= 0.5

                tp = float(np.sum(pred_t & true_t))
                fp = float(np.sum(pred_t & (~true_t)))
                fn = float(np.sum((~pred_t) & true_t))
                tn = float(np.sum((~pred_t) & (~true_t)))

                lead_counts[lead_idx]['tp'] += tp
                lead_counts[lead_idx]['fp'] += fp
                lead_counts[lead_idx]['fn'] += fn
                lead_counts[lead_idx]['tn'] += tn

                overall_counts['tp'] += tp
                overall_counts['fp'] += fp
                overall_counts['fn'] += fn
                overall_counts['tn'] += tn

            if show_progress and hasattr(iterator, 'set_postfix') and batch_losses:
                current_overall = _calc_metrics_from_counts(
                    overall_counts['tp'], overall_counts['fp'], overall_counts['fn'], overall_counts['tn'], eps
                )
                iterator.set_postfix({
                    'loss': f"{batch_losses[-1]:.4f}",
                    'ETS': f"{current_overall['ETS']:.4f}"
                })

            if save_cases and saved_cases < max_cases and self.is_save_model:
                if label.sum() > 0:
                    try:
                        self.save_case_for_visualization(
                            wrf, obs, pre_frames, label, epoch, batch_idx
                        )
                        saved_cases += 1
                    except Exception as e:
                        print(f"Warning: Failed to save case: {e}")

        mean_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        overall_metrics = _calc_metrics_from_counts(
            overall_counts['tp'], overall_counts['fp'], overall_counts['fn'], overall_counts['tn'], eps
        )

        per_lead_metrics = []
        for lead_idx, counts in enumerate(lead_counts, start=1):
            per_lead_metrics.append({
                'lead_time': lead_idx,
                **_calc_metrics_from_counts(counts['tp'], counts['fp'], counts['fn'], counts['tn'], eps)
            })

        mean_over_leads = {}
        for key in ['ETS', 'CSI', 'FAR', 'POD', 'F1', 'LOU']:
            values = [item[key] for item in per_lead_metrics] if per_lead_metrics else [0.0]
            mean_over_leads[key] = float(np.mean(values)) if values else 0.0

        try:
            pred_concat = np.concatenate(pred_batches, axis=0) if pred_batches else np.array([])
            true_concat = np.concatenate(true_batches, axis=0) if true_batches else np.array([])

            if pred_concat.size > 0:
                pred_flat = pred_concat.reshape(-1)
                true_flat = true_concat.reshape(-1)

                if len(np.unique(true_flat)) >= 2:
                    auc_score = float(roc_auc_score(true_flat, pred_flat))
                else:
                    auc_score = 0.0

                ssim_scores = []
                n_samples = min(pred_concat.shape[0], 200)
                n_frames = pred_concat.shape[1]
                for sample_idx in range(n_samples):
                    for lead_idx in range(n_frames):
                        pred_frame = (pred_concat[sample_idx, lead_idx, :, :, 0] >= 0.5).astype(np.float32)
                        true_frame = true_concat[sample_idx, lead_idx, :, :, 0].astype(np.float32)
                        try:
                            ssim_scores.append(ssim(true_frame, pred_frame, data_range=1.0))
                        except Exception:
                            continue
                ssim_score = float(np.mean(ssim_scores)) if ssim_scores else 0.0
            else:
                auc_score = 0.0
                ssim_score = 0.0
        except Exception as e:
            print(f"  Warning: Failed to calculate AUC/SSIM: {e}")
            auc_score = 0.0
            ssim_score = 0.0

        metrics_dict = {
            'loss': mean_loss,
            'POD': float(overall_metrics['POD']),
            'FAR': float(overall_metrics['FAR']),
            'CSI': float(overall_metrics['CSI']),
            'TS': float(overall_metrics['CSI']),
            'ETS': float(overall_metrics['ETS']),
            'AUC': float(auc_score),
            'F1': float(overall_metrics['F1']),
            'LOU': float(overall_metrics['LOU']),
            'SSIM': float(ssim_score),
            'mean_over_lead_times': mean_over_leads,
            'per_lead_times': per_lead_metrics,
            'CSI_per_timestep': [item['CSI'] for item in per_lead_metrics],
        }

        if stage.upper() == 'TEST':
            print(_format_test_summary(metrics_dict))
        else:
            print(
                f"`{self.config_dict['NetName']}` {stage.upper()} epoch:{epoch} "
                f"loss={metrics_dict['loss']:.5f} ETS={metrics_dict['ETS']:.5f} "
                f"CSI={metrics_dict['CSI']:.5f} FAR={metrics_dict['FAR']:.5f} "
                f"POD={metrics_dict['POD']:.5f} AUC={metrics_dict['AUC']:.5f} "
                f"F1={metrics_dict['F1']:.5f} LOU={metrics_dict['LOU']:.5f} SSIM={metrics_dict['SSIM']:.5f}"
            )

        record_text = _format_test_summary(metrics_dict) if stage.upper() == 'TEST' else str(metrics_dict)
        with open(os.path.join(self.config_dict['RecordFileDir'], 'record.txt'), 'a', encoding='utf-8') as f:
            f.write(record_text + '\r\n')

        if self.is_save_model:
            self.metrics_history['epoch'].append(epoch)
            self.metrics_history['POD'].append(float(metrics_dict['POD']))
            self.metrics_history['FAR'].append(float(metrics_dict['FAR']))
            self.metrics_history['TS'].append(float(metrics_dict['CSI']))
            self.metrics_history['ETS'].append(float(metrics_dict['ETS']))
            self.metrics_history['AUC'].append(float(metrics_dict['AUC']))
            self.metrics_history['SSIM'].append(float(metrics_dict['SSIM']))
            self.metrics_history['CSI_per_timestep'].append(metrics_dict['CSI_per_timestep'])

            if metrics_dict['POD'] > self.maxPOD:
                self.maxPOD = metrics_dict['POD']
                self.maxPOD_epoch = epoch
                self.save_model(model, 'model_maxPOD', epoch)

            if 1e-6 < metrics_dict['FAR'] < self.minFAR:
                self.minFAR = metrics_dict['FAR']
                self.minFAR_epoch = epoch
                self.save_model(model, 'model_minFAR', epoch)

            if metrics_dict['ETS'] > self.maxETS:
                self.maxETS = metrics_dict['ETS']
                self.maxETS_epoch = epoch
                self.save_model(model, 'model_maxETS', epoch)

            if auc_score > self.maxAUC:
                self.maxAUC = auc_score
                self.maxAUC_epoch = epoch
                self.save_model(model, 'model_maxAUC', epoch)

            if ssim_score > self.maxSSIM:
                self.maxSSIM = ssim_score
                self.maxSSIM_epoch = epoch
                self.save_model(model, 'model_maxSSIM', epoch)

            self.save_model(model, 'new_model', epoch)

        return metrics_dict

    def save_model(self, model, name, epoch):
        """保存模型"""
        torch.save(
            model.state_dict(), 
            os.path.join(self.config_dict['ModelFileDir'], 
                        '{}_{}.pkl'.format(self.config_dict['NetName'], name))
        )
        info = 'save model file: {} successfully! (epoch={})'.format(name, epoch)
        print(info)
    
    def save_case_for_visualization(self, wrf, obs, pred, label, epoch, batch_idx, case_tag=None, event_id=None, sample_idx=None, t0=None):
        """
        新增：保存案例用于可视化
        
        Args:
            wrf: [batch, frames, H, W, 2]
            obs: [batch, frames, H, W, 1]
            pred: [batch, frames, H, W, 1] (logits)
            label: [batch, frames, H*W, 1] - 需要重塑
            epoch: 当前epoch
            batch_idx: batch索引
        """
        try:
            # 取第一个样本
            wrf_np = wrf[0].cpu().numpy()  # [frames, H, W, 2]
            obs_np = obs[0].cpu().numpy()  # [frames, H, W, 1]
            pred_np = torch.sigmoid(pred[0]).cpu().numpy()  # [frames, H, W, 1]
            
            # 重塑label
            H, W = pred_np.shape[1], pred_np.shape[2]
            label_np = label[0].cpu().numpy()  # [frames, H*W, 1]
            label_np = label_np.reshape(-1, H, W, 1)  # [frames, H, W, 1]
            
            # 保存为.npz文件
            save_path = os.path.join(
                self.viz_save_dir, 
                f'case_epoch{epoch}_batch{batch_idx}.npz'
            )
            
            np.savez_compressed(
                save_path,
                wrf=wrf_np,
                obs=obs_np,
                pred=pred_np,
                label=label_np,
                epoch=epoch,
                batch_idx=batch_idx,
                case_tag=case_tag if case_tag is not None else '',
                event_id=event_id if event_id is not None else '',
                sample_idx=-1 if sample_idx is None else int(sample_idx),
                t0=-1 if t0 is None else int(t0)
            )
            
            print(f"  Case saved: {save_path}")
        
        except Exception as e:
            print(f"  Warning: Failed to save case: {e}")


if __name__ == "__main__":
    print("Enhanced scores module with AUC, SSIM, and temporal CSI tracking")