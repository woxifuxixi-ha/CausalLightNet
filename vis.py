"""
可视化模块 - 修复版
修复：CSI应该是随epoch变化，而不是随时间步变化
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path
import pickle
import os

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


def plot_csi_vs_epoch(metrics_history, save_path):
    """
    【修复】绘制CSI随epoch变化的曲线
    这才是我们想要的图！
    """
    epochs = metrics_history.get('epoch', [])
    csi_history = metrics_history.get('CSI_per_timestep', [])
    
    if not epochs or not csi_history:
        print("Warning: No CSI history data available")
        return
    
    # 计算每个epoch的平均CSI（对所有时间步求平均）
    mean_csi_per_epoch = []
    for csi_timesteps in csi_history:
        if csi_timesteps and len(csi_timesteps) > 0:
            mean_csi_per_epoch.append(np.mean(csi_timesteps))
        else:
            mean_csi_per_epoch.append(0.0)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 绘制CSI随epoch变化
    ax.plot(epochs, mean_csi_per_epoch, marker='o', linewidth=2.5, 
            markersize=8, color='#2E86AB', label='Mean CSI', zorder=3)
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--', zorder=1)
    
    # 标签和标题
    ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel('CSI (Critical Success Index)', fontsize=13, fontweight='bold')
    ax.set_title('CSI Evolution During Training', fontsize=15, fontweight='bold', pad=20)
    
    # 设置y轴范围
    ax.set_ylim([0, max(1.0, max(mean_csi_per_epoch) * 1.1 if mean_csi_per_epoch else 1.0)])
    
    # 标记最佳epoch
    if mean_csi_per_epoch and max(mean_csi_per_epoch) > 0:
        best_idx = np.argmax(mean_csi_per_epoch)
        best_csi = mean_csi_per_epoch[best_idx]
        best_epoch = epochs[best_idx]
        
        ax.plot(best_epoch, best_csi, 'r*', markersize=20, 
                label=f'Best CSI: {best_csi:.4f} @Epoch {best_epoch}', zorder=4)
        
        # 添加水平线显示最佳值
        ax.axhline(y=best_csi, color='r', linestyle='--', 
                   linewidth=1.5, alpha=0.4, zorder=2)
        
        # 添加文本注释
        ax.annotate(f'Peak: {best_csi:.4f}',
                   xy=(best_epoch, best_csi),
                   xytext=(best_epoch, best_csi + 0.05),
                   fontsize=10,
                   ha='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3),
                   zorder=5)
    
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ CSI vs Epoch plot saved: {save_path}")
    if mean_csi_per_epoch:
        print(f"  CSI range: {min(mean_csi_per_epoch):.4f} - {max(mean_csi_per_epoch):.4f}")


def plot_csi_over_time(csi_data, save_path, title="CSI Over Time Steps"):
    """
    【保留】绘制CSI随时间步变化曲线
    这个用于展示某个epoch内不同预测时间的CSI
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    timesteps = np.arange(1, len(csi_data) + 1)
    
    ax.plot(timesteps, csi_data, marker='o', linewidth=2, markersize=8,
            color='#2E86AB', label='CSI')
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlabel('Time Step (Hours)', fontsize=12, fontweight='bold')
    ax.set_ylabel('CSI', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim([0, 1])
    
    mean_csi = np.mean(csi_data)
    ax.axhline(y=mean_csi, color='r', linestyle='--', linewidth=2,
               label=f'Mean CSI: {mean_csi:.4f}')
    
    ax.legend(loc='best', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ CSI time series plot saved: {save_path}")


def plot_all_metrics_vs_epoch(metrics_history, save_path):
    """
    【修复】绘制所有指标随epoch变化
    包括正确计算的CSI
    """
    epochs = metrics_history.get('epoch', [])
    
    if not epochs:
        print("Warning: No epoch data")
        return
    
    # 计算平均CSI（每个epoch对所有时间步的平均）
    csi_history = metrics_history.get('CSI_per_timestep', [])
    mean_csi = []
    for csi_timesteps in csi_history:
        if csi_timesteps and len(csi_timesteps) > 0:
            mean_csi.append(np.mean(csi_timesteps))
        else:
            mean_csi.append(0.0)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # 定义要绘制的指标
    metrics_to_plot = [
        ('ETS', 'Equitable Threat Score', '#F18F01'),
        ('POD', 'Probability of Detection', '#2E86AB'),
        ('FAR', 'False Alarm Ratio', '#A23B72'),
        ('AUC', 'Area Under Curve', '#C73E1D'),
        ('SSIM', 'Structural Similarity', '#6A994E'),
        ('CSI', 'Critical Success Index', '#BC4B51'),
    ]
    
    for idx, (metric_key, metric_name, color) in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        # 特殊处理CSI（使用计算的平均值）
        if metric_key == 'CSI':
            values = mean_csi
        elif metric_key in metrics_history:
            values = metrics_history[metric_key]
        else:
            values = []
        
        if values and len(values) > 0:
            # 绘制曲线
            ax.plot(epochs, values, marker='o', linewidth=2, markersize=5,
                   color=color, label=metric_name)
            
            # 标记最优点
            if metric_key == 'FAR':
                best_idx = np.argmin(values)
                best_val = min(values)
            else:
                best_idx = np.argmax(values)
                best_val = max(values)
            
            ax.plot(epochs[best_idx], best_val, 'r*', markersize=15,
                   label=f'Best: {best_val:.4f} @Epoch {epochs[best_idx]}')
            
            ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
            ax.set_ylabel(metric_name, fontsize=11, fontweight='bold')
            ax.set_title(f'{metric_name} vs Epoch', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='best', fontsize=9)
            
            # 设置y轴范围
            if metric_key != 'FAR':
                ax.set_ylim([0, 1])
        else:
            ax.text(0.5, 0.5, f'No {metric_name} data', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=12)
            ax.axis('off')
    
    plt.suptitle('Training Metrics Evolution', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ All metrics vs epoch plot saved: {save_path}")


def plot_csi_evolution(metrics_history, save_path):
    """
    【保留但补充说明】绘制训练过程中CSI随epoch和时间步的演化热图
    """
    csi_history = metrics_history.get('CSI_per_timestep', [])
    
    if not csi_history or len(csi_history) == 0:
        print("Warning: No CSI history data available")
        return
    
    csi_array = np.array(csi_history)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图：热图展示CSI演化
    im = ax1.imshow(csi_array.T, aspect='auto', cmap='RdYlGn', 
                    vmin=0, vmax=1, origin='lower')
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Time Step (Hours)', fontsize=12, fontweight='bold')
    ax1.set_title('CSI Heatmap: Epoch vs Time Step', fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('CSI', fontsize=11)
    
    # 右图：选择几个epoch的CSI曲线
    epochs_to_plot = [0, len(csi_array)//4, len(csi_array)//2, 
                      3*len(csi_array)//4, len(csi_array)-1]
    epochs_to_plot = [e for e in epochs_to_plot if e < len(csi_array)]
    
    for epoch_idx in epochs_to_plot:
        timesteps = np.arange(1, len(csi_array[epoch_idx]) + 1)
        ax2.plot(timesteps, csi_array[epoch_idx], marker='o', 
                label=f'Epoch {epoch_idx}', linewidth=2, markersize=6)
    
    ax2.set_xlabel('Time Step (Hours)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('CSI', fontsize=12, fontweight='bold')
    ax2.set_title('CSI Curves at Different Epochs', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='best', fontsize=10)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ CSI evolution heatmap saved: {save_path}")


def plot_spatiotemporal_error_heatmap(pred_data, label_data, save_path, 
                                     title="Spatial-Temporal Error Heatmap"):
    """绘制时空误差热图"""
    error = np.abs(pred_data - label_data)
    spatial_mean_error = error.mean(axis=(1, 2))
    temporal_mean_error = error.mean(axis=0)
    
    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1, 1])
    
    ax1 = fig.add_subplot(gs[0])
    error_reshaped = error.reshape(error.shape[0], -1)
    
    im1 = ax1.imshow(error_reshaped.T, aspect='auto', cmap='hot', 
                     interpolation='nearest')
    ax1.set_xlabel('Time Step (Hours)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Spatial Location Index', fontsize=12, fontweight='bold')
    ax1.set_title('Spatiotemporal Error Distribution', fontsize=13, fontweight='bold')
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Absolute Error', fontsize=10)
    
    ax2 = fig.add_subplot(gs[1])
    timesteps = np.arange(1, len(spatial_mean_error) + 1)
    ax2.plot(timesteps, spatial_mean_error, marker='o', linewidth=2, 
             markersize=6, color='#D62828')
    ax2.set_xlabel('Time Step', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean Spatial Error', fontsize=12, fontweight='bold')
    ax2.set_title('Error Over Time', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    ax3 = fig.add_subplot(gs[2])
    im3 = ax3.imshow(temporal_mean_error, cmap='hot', interpolation='nearest')
    ax3.set_title('Mean Temporal Error\n(Spatial Distribution)', 
                  fontsize=13, fontweight='bold')
    ax3.axis('off')
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046)
    cbar3.set_label('Error', fontsize=10)
    
    plt.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Spatiotemporal error heatmap saved: {save_path}")


def visualize_case_study(case_file, save_dir, timesteps_to_show=None):
    """可视化典型案例"""
    data = np.load(case_file)
    obs = data['obs'].squeeze(-1)
    pred = data['pred'].squeeze(-1)
    label = data['label'].squeeze(-1)
    wrf = data['wrf']
    epoch = data['epoch']
    batch_idx = data['batch_idx']
    
    n_frames = pred.shape[0]
    if timesteps_to_show is None:
        timesteps_to_show = np.linspace(0, n_frames-1, 5, dtype=int).tolist()
    
    n_cols = len(timesteps_to_show)
    fig, axes = plt.subplots(4, n_cols, figsize=(4*n_cols, 14))
    
    for col_idx, t in enumerate(timesteps_to_show):
        if t == 0:
            ax = axes[0, col_idx]
            im = ax.imshow(obs[min(t, obs.shape[0]-1)], cmap='YlOrRd', vmin=0, vmax=1)
            ax.set_title(f'History Obs\n(t={t}h)', fontsize=11, fontweight='bold')
            ax.axis('off')
            if col_idx == n_cols - 1:
                plt.colorbar(im, ax=ax, fraction=0.046)
        else:
            axes[0, col_idx].axis('off')
        
        ax = axes[1, col_idx]
        pred_binary = (pred[t] > 0.5).astype(float)
        im = ax.imshow(pred_binary, cmap='YlOrRd', vmin=0, vmax=1)
        ax.set_title(f'Prediction\n(t+{t+1}h)', fontsize=11, fontweight='bold')
        ax.axis('off')
        if col_idx == n_cols - 1:
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        ax = axes[2, col_idx]
        im = ax.imshow(label[t], cmap='YlOrRd', vmin=0, vmax=1)
        ax.set_title(f'Ground Truth\n(t+{t+1}h)', fontsize=11, fontweight='bold')
        ax.axis('off')
        if col_idx == n_cols - 1:
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        ax = axes[3, col_idx]
        error = np.abs(pred_binary - label[t])
        im = ax.imshow(error, cmap='Reds', vmin=0, vmax=1)
        ax.set_title(f'Absolute Error\n(t+{t+1}h)', fontsize=11, fontweight='bold')
        ax.axis('off')
        if col_idx == n_cols - 1:
            plt.colorbar(im, ax=ax, fraction=0.046)
    
    row_labels = ['Historical\nObservation', 'Predicted\nLightning', 
                  'Ground Truth\nLightning', 'Prediction\nError']
    for i, label in enumerate(row_labels):
        axes[i, 0].text(-0.3, 0.5, label, transform=axes[i, 0].transAxes,
                       fontsize=12, fontweight='bold', va='center', ha='right',
                       rotation=90)
    
    plt.suptitle(f'Case Study - Epoch {epoch}, Batch {batch_idx}', 
                fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'case_epoch{epoch}_batch{batch_idx}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Case study visualization saved: {save_path}")
    return save_path


def visualize_target_forecast_case(true_frames, pred_frames, save_path, title,
                                   threshold=0.5, max_frames=6):
    """绘制指定事件的预测案例：第一行真实值，第二行预测值。"""
    true_frames = np.asarray(true_frames)
    pred_frames = np.asarray(pred_frames)

    if true_frames.ndim == 4 and true_frames.shape[-1] == 1:
        true_frames = true_frames[..., 0]
    if pred_frames.ndim == 4 and pred_frames.shape[-1] == 1:
        pred_frames = pred_frames[..., 0]

    n_frames = min(true_frames.shape[0], pred_frames.shape[0], max_frames)
    fig, axes = plt.subplots(2, n_frames, figsize=(3.2 * n_frames, 6.2))
    if n_frames == 1:
        axes = np.array(axes).reshape(2, 1)

    for frame_idx in range(n_frames):
        true_ax = axes[0, frame_idx]
        pred_ax = axes[1, frame_idx]

        true_img = true_frames[frame_idx]
        pred_img = (pred_frames[frame_idx] >= threshold).astype(np.float32)

        true_im = true_ax.imshow(true_img, cmap='YlOrRd', vmin=0, vmax=1)
        pred_im = pred_ax.imshow(pred_img, cmap='YlOrRd', vmin=0, vmax=1)

        true_ax.set_title(f'Truth t{frame_idx + 1}', fontsize=11, fontweight='bold')
        pred_ax.set_title(f'Pred t{frame_idx + 1}', fontsize=11, fontweight='bold')
        true_ax.axis('off')
        pred_ax.axis('off')

        if frame_idx == n_frames - 1:
            plt.colorbar(true_im, ax=true_ax, fraction=0.046, pad=0.02)
            plt.colorbar(pred_im, ax=pred_ax, fraction=0.046, pad=0.02)

    axes[0, 0].text(-0.25, 0.5, 'Ground Truth', transform=axes[0, 0].transAxes,
                    fontsize=12, fontweight='bold', va='center', ha='right', rotation=90)
    axes[1, 0].text(-0.25, 0.5, 'Prediction', transform=axes[1, 0].transAxes,
                    fontsize=12, fontweight='bold', va='center', ha='right', rotation=90)

    plt.suptitle(title, fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Target case visualization saved: {save_path}")
    return save_path


def visualize_dataset_sample(model, dataset, sample_idx, config_dict, save_dir,
                             event_id='S749332', t0=26, case_tag='target_case',
                             threshold=0.5):
    """从数据集抽取指定样本，跑模型预测并输出 2 行 6 帧可视化。"""
    if sample_idx < 0 or sample_idx >= len(dataset):
        raise IndexError(f"sample_idx {sample_idx} out of range for dataset length {len(dataset)}")

    device = torch.device(config_dict.get('Device', 'cpu'))
    model = model.to(device)
    model.eval()

    (wrf_np, obs_np), label_np = dataset[sample_idx]
    wrf = torch.from_numpy(wrf_np).unsqueeze(0).to(device)
    obs = torch.from_numpy(obs_np).unsqueeze(0).to(device)
    label = np.asarray(label_np)

    with torch.no_grad():
        pred_logits = model(wrf, obs)
        pred_prob = torch.sigmoid(pred_logits).squeeze(0).detach().cpu().numpy()

    grid_size = int(config_dict['GridRowColNum'])
    if label.ndim == 3 and label.shape[1] == grid_size * grid_size:
        label = label.reshape(label.shape[0], grid_size, grid_size, 1)
    if label.ndim == 4 and label.shape[-1] == 1:
        label = label[..., 0]

    if pred_prob.ndim == 4 and pred_prob.shape[-1] == 1:
        pred_prob = pred_prob[..., 0]

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(
        save_dir,
        f'{case_tag}_event_{event_id}_t0_{t0}_idx_{sample_idx}.png'
    )
    title = f'Event {event_id} | T0={t0} | idx={sample_idx}'
    return visualize_target_forecast_case(
        label,
        pred_prob,
        save_path,
        title,
        threshold=threshold,
        max_frames=min(6, pred_prob.shape[0])
    )


def generate_all_visualizations(config_dict, metrics_history_file=None, 
                                cases_dir=None):
    """
    【修复】生成所有可视化图表
    主要修复：正确绘制CSI随epoch变化
    """
    print("\n" + "="*60)
    print("Generating All Visualizations (FIXED)")
    print("="*60)
    
    viz_dir = config_dict.get('VisResultFileDir', './visualization')
    os.makedirs(viz_dir, exist_ok=True)
    
    if metrics_history_file is None:
        metrics_history_file = os.path.join(
            config_dict.get('RecordFileDir', './records'),
            'metrics_history.pkl'
        )
    
    if os.path.exists(metrics_history_file):
        print(f"\nLoading metrics history from: {metrics_history_file}")
        with open(metrics_history_file, 'rb') as f:
            metrics_history = pickle.load(f)
        
        # 1. 【重点】绘制CSI随epoch变化
        print("\n1. Plotting CSI vs Epoch (FIXED - This is what you want!)...")
        plot_csi_vs_epoch(
            metrics_history,
            os.path.join(viz_dir, 'csi_vs_epoch.png')
        )
        
        # 2. 绘制所有指标随epoch变化
        print("\n2. Plotting all metrics vs Epoch...")
        plot_all_metrics_vs_epoch(
            metrics_history,
            os.path.join(viz_dir, 'all_metrics_vs_epoch.png')
        )
        
        # 3. 【补充】绘制CSI演化热图
        print("\n3. Plotting CSI evolution heatmap...")
        plot_csi_evolution(
            metrics_history,
            os.path.join(viz_dir, 'csi_evolution_heatmap.png')
        )
        
        # 4. 【补充】绘制最后一个epoch的CSI随时间步变化
        if 'CSI_per_timestep' in metrics_history and len(metrics_history['CSI_per_timestep']) > 0:
            print("\n4. Plotting CSI over time steps (final epoch, for reference)...")
            final_csi = metrics_history['CSI_per_timestep'][-1]
            if len(final_csi) > 0:
                plot_csi_over_time(
                    final_csi,
                    os.path.join(viz_dir, 'csi_timesteps_final_epoch.png'),
                    title=f"CSI Over Time Steps (Final Epoch {metrics_history['epoch'][-1]})"
                )
    else:
        print(f"\nWarning: Metrics history file not found: {metrics_history_file}")
    
    # 5. 可视化案例
    if cases_dir is None:
        cases_dir = os.path.join(viz_dir, 'cases')
    
    if os.path.exists(cases_dir):
        print(f"\n5. Visualizing case studies from: {cases_dir}")
        case_files = sorted(Path(cases_dir).glob('case_*.npz'))
        
        if len(case_files) > 0:
            print(f"   Found {len(case_files)} case files")
            n_cases_to_viz = min(5, len(case_files))
            for i, case_file in enumerate(case_files[-n_cases_to_viz:]):
                print(f"   Visualizing case {i+1}/{n_cases_to_viz}: {case_file.name}")
                try:
                    visualize_case_study(
                        str(case_file),
                        os.path.join(viz_dir, 'case_visualizations')
                    )
                except Exception as e:
                    print(f"   Error visualizing {case_file.name}: {e}")
    
    print("\n" + "="*60)
    print(f"✓ All visualizations saved to: {viz_dir}")
    print("  Key files:")
    print(f"    - csi_vs_epoch.png (主要图表)")
    print(f"    - all_metrics_vs_epoch.png")
    print(f"    - csi_evolution_heatmap.png")
    print("="*60 + "\n")


if __name__ == "__main__":
    print("Fixed visualization module - CSI vs Epoch correctly implemented")