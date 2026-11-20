"""Paired six-lead evaluation of existing XR checkpoints; no training imports."""
import argparse
import csv
import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
MODES = ('full', 'no_causal', 'no_spatial', 'no_focal')
FOLDERS = dict(zip(MODES, ('FULL', 'no_causal', 'no_spatial', 'no_focal')))


def load_checkpoint(path):
    # Prefer restricted loading where the installed PyTorch supports allowlisting.
    allowed = [np.core.multiarray.scalar, np.dtype,
               type(np.dtype('float32')), type(np.dtype('float64'))]
    safe_globals = getattr(torch.serialization, 'safe_globals', None)
    if safe_globals is not None:
        with safe_globals(allowed):
            return torch.load(path, map_location='cpu', weights_only=True)
    # Legacy PyTorch cannot allowlist the NumPy metadata in our epoch files.
    # This branch is for the project's own trusted training checkpoints only.
    kwargs = {'map_location': 'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        kwargs['weights_only'] = False
    return torch.load(path, **kwargs)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def build_models(args, config, device):
    from causal_prior_parser import CausalPriorParser
    from sevir_lightnet_plus import CausalLightNetPlus
    prior = CausalPriorParser(str(args.causal_report)).get_lightning_causal_config()
    if not prior.get('sources'):
        raise ValueError('Causal report contains no lightning sources')
    models, provenance = {}, {}
    for mode in MODES:
        folder = args.weights_root / FOLDERS[mode] / 'models'
        path = folder / 'LightNet_plus_model_maxETS.pkl'
        if not path.is_file():
            raise FileNotFoundError(f'Missing validation-maxETS checkpoint: {path}')
        saved = load_checkpoint(path)
        state = saved.get('model_state_dict', saved)
        causal = mode != 'no_causal'
        # Epoch metadata validates source ordering, which state_dict alone cannot encode.
        epoch_paths = sorted(folder.glob('causal_improved_logits_bias_epoch_*.pkl'),
                             key=lambda p: int(p.stem.rsplit('_', 1)[1]))
        if not epoch_paths:
            raise FileNotFoundError(f'No epoch checkpoint to verify causal metadata in {folder}')
        metadata = load_checkpoint(epoch_paths[-1])
        if metadata.get('use_causal') != causal:
            raise ValueError(f'{mode}: checkpoint causal setting does not match mode')
        causal_config = metadata.get('causal_config') if causal else None
        if causal and causal_config.get('sources') != prior['sources']:
            raise ValueError(f'{mode}: causal source order differs from the supplied report')
        model = CausalLightNetPlus(
            obs_tra_frames=12, obs_channels=1, wrf_tra_frames=6,
            wrf_channels=5 if causal else 2, config_dict=config,
            causal_config=causal_config)
        model.load_state_dict(state, strict=True)
        model.eval().to(device)
        models[mode] = model
        provenance[mode] = {
            'checkpoint': str(path.resolve()), 'sha256': sha256(path),
            'metadata_checkpoint': str(epoch_paths[-1].resolve()),
            'causal_config': causal_config,
            'dcn_class': type(model.dcn_tm).__module__ + '.' + type(model.dcn_tm).__name__,
        }
        print(f'{mode}: strict checkpoint load OK', flush=True)
    return models, provenance


def contingency_counts(probability, truth):
    if probability.shape != truth.shape or probability.ndim != 5:
        raise ValueError(f'Expected matching [B,T,H,W,1], got {probability.shape}, {truth.shape}')
    if not torch.isfinite(probability).all() or not torch.isfinite(truth).all():
        raise ValueError('Non-finite predictions or labels')
    pred, target = probability >= 0.5, truth > 0.5
    axes = (0, 2, 3, 4)
    return torch.stack([(pred & target).sum(axes), (pred & ~target).sum(axes),
                        (~pred & target).sum(axes), (~pred & ~target).sum(axes)], 1).cpu().numpy()


def metric_rows(mode, counts, samples):
    rows = []
    for i, values in enumerate(counts):
        tp, fp, fn, tn = map(int, values)
        total = tp + fp + fn + tn
        random_hits = (tp + fp) * (tp + fn) / (total + 1e-10)
        rows.append(dict(model=mode, lead_min=5 * (i + 1), samples=samples,
                         TP=tp, FP=fp, FN=fn, TN=tn,
                         ETS=(tp - random_hits) / (tp + fp + fn - random_hits + 1e-10),
                         POD=tp / (tp + fn + 1e-10), FAR=fp / (tp + fp + 1e-10)))
    return rows


def plot_rows(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    labels = ('Full', 'Without causal inputs/fusion', 'Without spatial prior', 'Weighted BCE')
    colors = ('#242424', '#547AA5', '#C28E52', '#9A7197')
    for ax, metric, panel in zip(axes, ('ETS', 'POD', 'FAR'), 'abc'):
        for mode, label, color, marker in zip(MODES, labels, colors, ('o', 's', '^', 'D')):
            selected = sorted((r for r in rows if r['model'] == mode), key=lambda r: float(r['lead_min']))
            if len(selected) != 6 or [int(r['lead_min']) for r in selected] != [5, 10, 15, 20, 25, 30]:
                raise ValueError(f'Expected six distinct leads for {mode}')
            ax.plot([float(r['lead_min']) for r in selected], [float(r[metric]) for r in selected],
                    label=label, color=color, marker=marker, ms=3.5, lw=1.3)
        ax.set(xlabel='Lead time (min)', ylabel=metric, xticks=[5, 10, 15, 20, 25, 30])
        if metric != 'ETS':
            ax.set_ylim(0, 1)
        ax.text(-0.18, 1.05, panel, transform=ax.transAxes, weight='bold', fontsize=10)
        ax.grid(axis='y', alpha=0.18, linewidth=0.5)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.005), fontsize=7)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.32, wspace=0.38)
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(output / f'leadtime_ablation.{extension}', dpi=600)
    plt.close(fig)


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--weights-root', type=Path, default=ROOT / 'XR')
    parser.add_argument('--causal-report', type=Path, default=ROOT / 'causal_report_strict.txt')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/leadtime_ablation')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--check-weights-only', action='store_true')
    parser.add_argument('--plot-csv', type=Path, help='Redraw existing evaluation CSV without inference')
    parser.add_argument('--save-predictions', action='store_true', help='Save per-batch probabilities and labels (large)')
    parser.add_argument('--include-empty-targets', action='store_true',
                        help='Keep zero-lightning files/windows for shared density analysis')
    args = parser.parse_args()
    if args.plot_csv:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with args.plot_csv.open(newline='', encoding='utf-8') as stream:
            plot_rows(list(csv.DictReader(stream)), args.output_dir)
        return
    if not args.check_weights_only and args.data_dir is None:
        parser.error('--data-dir is required for evaluation')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    config = dict(GridRowColNum=128, ForecastHourNum=6, TruthHistoryHourNum=12, Device=str(device))
    models, provenance = build_models(args, config, device)
    if args.check_weights_only:
        return
    from sevir_generator import ERA5SevirCausalGenerator, load_sevir_file_list
    # Use one five-channel dataset for every model. no_causal receives its first two channels.
    # Fail on reads instead of silently accepting the legacy loader's zero fallback.
    files = load_sevir_file_list(str(args.data_dir), split='test', train_ratio=0.7, val_ratio=0.15)
    dataset = StrictDataset(files, config, mode='test', lght_threshold=0.01,
                            skip_empty=not args.include_empty_targets, use_causal_prior=True)
    if not len(dataset):
        raise ValueError('No eligible common test windows')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError('Evaluation output directory must be empty; choose a new --output-dir')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, sample in enumerate(dataset.samples):
        manifest.append(dict(sample_idx=index, event_file=Path(sample['file_path']).name,
                             event_id=Path(sample['file_path']).stem, start_idx=sample['start_idx'],
                             forecast_start_idx=sample['start_idx'] + 12))
    with (args.output_dir / 'samples.json').open('w', encoding='utf-8') as stream:
        json.dump(manifest, stream, indent=2)
    metadata = dict(checkpoints=provenance, seed=args.seed, threshold=0.5, comparison='>=',
                    include_empty_targets=args.include_empty_targets,
                    lead_minutes=[5, 10, 15, 20, 25, 30], frame_minutes=5,
                    frame_minutes_basis='Requested 5-minute cadence; source timestamps not verified',
                    config=config, test_files=[str(Path(p).resolve()) for p in files],
                    samples=len(dataset), common_sample_policy='Causal-eligible test windows for all models',
                    event_id_basis='H5 filename stem; event_file is the pairing identity',
                    aggregation='Pool pixel contingency counts across all samples separately per lead',
                    torch_version=str(torch.__version__), numpy_version=np.__version__)
    with (args.output_dir / 'run.json').open('w', encoding='utf-8') as stream:
        json.dump(metadata, stream, indent=2)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    counts = {mode: np.zeros((6, 4), dtype=np.int64) for mode in MODES}
    offset = 0
    with torch.inference_mode():
        for batch_index, ((wrf, obs), label) in enumerate(loader):
            wrf, obs = wrf.to(device), obs.to(device)
            if not torch.isfinite(wrf).all() or not torch.isfinite(obs).all():
                raise ValueError(f'Non-finite input in batch {batch_index}')
            truth = label.to(device).reshape(-1, 6, 128, 128, 1)
            arrays = dict(label=truth.cpu().numpy(), sample_idx=np.arange(offset, offset + len(label)),
                          lead_minutes=np.arange(5, 31, 5))
            for mode, model in models.items():
                inputs = wrf[..., :2] if mode == 'no_causal' else wrf
                probability = torch.sigmoid(model(inputs, obs))
                counts[mode] += contingency_counts(probability, truth)
                if args.save_predictions:
                    arrays[mode + '_probability'] = probability.cpu().numpy()
            if args.save_predictions:
                np.savez_compressed(args.output_dir / f'predictions_batch{batch_index:05d}.npz', **arrays)
            offset += len(label)
            if batch_index % 20 == 0:
                print(f'Evaluated {offset}/{len(dataset)} common samples', flush=True)
    rows = [row for mode in MODES for row in metric_rows(mode, counts[mode], offset)]
    with (args.output_dir / 'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_rows(rows, args.output_dir)
    print(f'Completed: {args.output_dir.resolve()}', flush=True)


# Module-level subclass supports Windows DataLoader spawn without changing the old loader.
from sevir_generator import ERA5SevirCausalGenerator


class StrictDataset(ERA5SevirCausalGenerator):
    def _get_zero_sample(self):
        raise RuntimeError('Dataset read failed; refusing a silently substituted zero sample')


if __name__ == '__main__':
    main()
