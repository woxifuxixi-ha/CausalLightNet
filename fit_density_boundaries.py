"""Fit nonzero target-density terciles using the training split only."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from evaluate_leadtime_ablation import StrictDataset
from sevir_generator import load_sevir_file_list


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    files = load_sevir_file_list(str(args.data_dir), split='train', train_ratio=0.7, val_ratio=0.15)
    config = dict(GridRowColNum=128, ForecastHourNum=6, TruthHistoryHourNum=12)
    dataset = StrictDataset(files, config, mode='train', lght_threshold=0.01,
                            skip_empty=True, use_causal_prior=True)
    densities = []
    for index in range(len(dataset)):
        _, label = dataset[index]
        if not np.isfinite(label).all():
            raise ValueError(f'Nonfinite label at {index}')
        densities.append(float((label > 0.5).mean()))
        if index % 500 == 0:
            print(f'Training targets processed: {index + 1}/{len(dataset)}', flush=True)
    positive = np.asarray([d for d in densities if d > 0])
    if not len(positive):
        raise ValueError('No positive training targets for density boundaries')
    result = dict(source_split='train', boundary_method='nonzero training-window terciles (linear quantiles)',
                  boundaries=np.quantile(positive, [1/3, 2/3]).tolist(), config=config,
                  density_definition='Positive target cells / (6 * H * W); six-frame window mean',
                  samples=len(densities), nonzero_samples=len(positive),
                  train_files=[str(Path(p).resolve()) for p in files],
                  sample_policy='Causal-eligible training windows; skip_empty=True; label >0.01')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'Boundaries: {result["boundaries"]}; saved to {args.output}')


if __name__ == '__main__':
    main()
