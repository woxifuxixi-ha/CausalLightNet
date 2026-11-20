"""Density stratification using the paired probabilities saved by experiment 1."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

MODES = ('full', 'no_causal', 'no_spatial', 'no_focal')
GROUPS = ('zero', 'low', 'medium', 'high')


def assign_groups(densities, cuts=None):
    positive = densities[densities > 0]
    if cuts is None:
        cuts = np.quantile(positive, [1 / 3, 2 / 3]).tolist() if len(positive) else [0., 0.]
    lower, upper = cuts
    # Equal densities are never split by sample order; ties can leave a group empty.
    groups = np.where(densities == 0, 'zero', np.where(
        densities <= lower, 'low', np.where(densities <= upper, 'medium', 'high')))
    return groups, list(cuts)


def summarize(counts, n, false_alarm_samples, zero_group=False):
    tp, fp, fn, tn = map(int, counts)
    total = tp + fp + fn + tn
    hits_random = (tp + fp) * (tp + fn) / total if total else 0.
    denominator = tp + fp + fn - hits_random
    return dict(samples=n, TP=tp, FP=fp, FN=fn, TN=tn,
                ETS=(tp - hits_random) / denominator if denominator and not zero_group else None,
                POD=tp / (tp + fn) if tp + fn else None,
                FAR=fp / (tp + fp) if tp + fp else None,
                false_positive_rate=fp / (fp + tn) if fp + tn else None,
                predicted_positive_rate=(tp + fp) / total if total else None,
                samples_with_false_alarms=false_alarm_samples,
                sample_false_alarm_rate=false_alarm_samples / n if n else None)


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(prediction_dir, output_dir, cuts=None, boundary_provenance=None):
    run = json.loads((prediction_dir / 'run.json').read_text(encoding='utf-8'))
    manifest = json.loads((prediction_dir / 'samples.json').read_text(encoding='utf-8'))
    if run.get('threshold') != 0.5 or run.get('comparison') != '>=':
        raise ValueError('Expected experiment 1 probability >= 0.5')
    if not (prediction_dir / 'metrics.csv').is_file():
        raise ValueError('Experiment 1 has not completed (metrics.csv is missing)')
    files = sorted(prediction_dir.glob('predictions_batch*.npz'))
    if not files:
        raise FileNotFoundError('No probabilities: rerun experiment 1 with --save-predictions')
    ids, densities, counts, alarms = [], [], [], []
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            label = data['label']
            if label.ndim != 5 or label.shape[1] != 6 or label.shape[-1] != 1:
                raise ValueError(f'{path}: expected [B,6,H,W,1] labels')
            if not np.isin(label, [0, 1]).all():
                raise ValueError(f'{path}: labels must be binary')
            if not np.array_equal(data['lead_minutes'], [5, 10, 15, 20, 25, 30]):
                raise ValueError(f'{path}: unexpected leads')
            sample_ids = data['sample_idx']
            if sample_ids.shape != (len(label),) or sample_ids.dtype.kind not in 'iu':
                raise ValueError(f'{path}: invalid sample indices')
            ids.extend(sample_ids.tolist())
            target = label > 0.5
            densities.extend(target.mean(axis=(1, 2, 3, 4)).tolist())
            batch_counts, batch_alarms = [], []
            for mode in MODES:
                probability = data[mode + '_probability']
                if (probability.shape != label.shape or not np.isfinite(probability).all()
                        or np.any((probability < 0) | (probability > 1))):
                    raise ValueError(f'{path}: invalid {mode} probabilities')
                pred = probability >= 0.5
                batch_counts.append(np.stack([
                    (pred & target).sum(axis=(2, 3, 4)),
                    (pred & ~target).sum(axis=(2, 3, 4)),
                    (~pred & target).sum(axis=(2, 3, 4)),
                    (~pred & ~target).sum(axis=(2, 3, 4))], axis=-1))
                batch_alarms.append((pred & ~target).any(axis=(2, 3, 4)))
            counts.append(np.stack(batch_counts, axis=1))  # B,model,lead,count
            alarms.append(np.stack(batch_alarms, axis=1))
    expected = [item['sample_idx'] for item in manifest]
    if len(expected) != run['samples'] or len(set(ids)) != len(ids) or sorted(ids) != sorted(expected):
        raise ValueError('Incomplete or duplicated cached predictions; sample manifest mismatch')
    densities = np.asarray(densities)
    groups, boundaries = assign_groups(densities, cuts)
    counts, alarms = np.concatenate(counts), np.concatenate(alarms)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError('Choose an empty output directory')
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, per_lead = [], []
    for group in GROUPS:
        selected = groups == group
        n = int(selected.sum())
        for model_index, mode in enumerate(MODES):
            c, a = counts[selected, model_index], alarms[selected, model_index]
            rows.append(dict(model=mode, density_group=group,
                             **summarize(c.sum(axis=(0, 1)), n, int(a.any(axis=1).sum()), group == 'zero')))
            for lead in range(6):
                per_lead.append(dict(model=mode, density_group=group, lead_min=5 * (lead + 1),
                                     **summarize(c[:, lead].sum(axis=0), n, int(a[:, lead].sum()), group == 'zero')))
    write_csv(output_dir / 'density_metrics.csv', rows)
    write_csv(output_dir / 'density_per_lead.csv', per_lead)
    by_id = {item['sample_idx']: item for item in manifest}
    write_csv(output_dir / 'density_samples.csv', [
        dict(by_id[index], density=float(density), density_group=str(group))
        for index, density, group in zip(ids, densities, groups)])
    settings = dict(source_run=str((prediction_dir / 'run.json').resolve()),
                    density_definition='Positive target cells / (6 * H * W); six-frame window mean',
                    boundary_method=(boundary_provenance['boundary_method'] if boundary_provenance
                                     else 'explicit' if cuts is not None
                                     else 'nonzero test-window terciles (linear quantiles)'),
                    boundaries=boundaries, boundary_provenance=boundary_provenance,
                    intervals='zero: d=0; low: 0<d<=q1; medium: q1<d<=q2; high: d>q2',
                    group_counts={g: int((groups == g).sum()) for g in GROUPS},
                    include_empty_targets=run.get('include_empty_targets', False),
                    undefined_values='Empty CSV field; zero-group ETS and POD not applicable; FAR undefined if no predicted positives',
                    source_checkpoints=run.get('checkpoints'), prediction_files=[p.name for p in files])
    (output_dir / 'density_run.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
    print(json.dumps(settings['group_counts']))
    if not run.get('include_empty_targets', False):
        print('NOTE: source evaluation filtered empty targets; zero-group results do not represent all zero-lightning windows.')
    print(f'Completed: {output_dir.resolve()}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--density-cuts', type=float, nargs=2, metavar=('LOW', 'HIGH'),
                        help='Optional fixed fractions instead of nonzero-target terciles')
    parser.add_argument('--boundaries-json', type=Path, help='Training-derived boundaries from fit_density_boundaries.py')
    args = parser.parse_args()
    if args.density_cuts is not None and not (0 < args.density_cuts[0] < args.density_cuts[1] < 1):
        parser.error('Require 0 < LOW < HIGH < 1')
    provenance = None
    if args.boundaries_json:
        if args.density_cuts is not None:
            parser.error('Choose --boundaries-json OR --density-cuts')
        provenance = json.loads(args.boundaries_json.read_text(encoding='utf-8'))
        if provenance.get('source_split') != 'train':
            parser.error('Boundary file must have source_split=train')
        args.density_cuts = provenance.get('boundaries')
        if (not isinstance(args.density_cuts, list) or len(args.density_cuts) != 2 or
                not all(np.isfinite(args.density_cuts)) or
                not 0 < args.density_cuts[0] <= args.density_cuts[1] <= 1):
            parser.error('Invalid training boundaries')
        if provenance.get('config') != dict(GridRowColNum=128, ForecastHourNum=6, TruthHistoryHourNum=12):
            parser.error('Boundary data dimensions must match the six-frame protocol')
    evaluate(args.prediction_dir, args.output_dir, args.density_cuts, provenance)


if __name__ == '__main__':
    main()
