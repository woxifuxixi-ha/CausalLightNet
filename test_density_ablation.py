import csv
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from evaluate_density_ablation import assign_groups, summarize, evaluate, MODES


class DensityTests(unittest.TestCase):
    def test_ties_are_not_arbitrarily_split(self):
        groups, cuts = assign_groups(np.array([0., 0.2, 0.2, 0.2]))
        self.assertEqual(groups.tolist(), ['zero', 'low', 'low', 'low'])
        self.assertEqual(cuts, [0.2, 0.2])

    def test_zero_group_false_alarms(self):
        row = summarize([0, 2, 0, 8], 2, 1, True)
        self.assertIsNone(row['ETS'])
        self.assertIsNone(row['POD'])
        self.assertEqual(row['FAR'], 1)
        self.assertEqual(row['false_positive_rate'], 0.2)
        self.assertEqual(row['sample_false_alarm_rate'], 0.5)
        self.assertIsNone(summarize([0, 0, 0, 10], 1, 0, True)['FAR'])

    def test_shared_cache_end_to_end_and_missing_sample(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / 'source'
            source.mkdir()
            (source / 'run.json').write_text(json.dumps(dict(samples=4, threshold=0.5,
                                                           comparison='>=', include_empty_targets=True)))
            (source / 'samples.json').write_text(json.dumps([dict(sample_idx=i) for i in range(4)]))
            (source / 'metrics.csv').write_text('completed test fixture')
            label = np.zeros((4, 6, 2, 2, 1), dtype=np.float32)
            for i in range(4):
                label[i].reshape(-1)[:6*i] = 1
            arrays = dict(label=label, sample_idx=np.arange(4), lead_minutes=np.arange(5, 31, 5))
            arrays.update({mode + '_probability': label.copy() for mode in MODES})
            arrays['full_probability'][0, 0, 0, 0, 0] = 0.5
            np.savez(source / 'predictions_batch00000.npz', **arrays)
            out = Path(temp) / 'out'
            evaluate(source, out, [0.3, 0.6])
            with (out / 'density_metrics.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 16)
            self.assertTrue(all(row['samples'] == '1' for row in rows))
            self.assertEqual(rows[0]['FP'], '1')
            self.assertEqual(rows[0]['POD'], '')
            arrays['sample_idx'] = np.array([0, 1, 2, 2])
            np.savez(source / 'predictions_batch00000.npz', **arrays)
            with self.assertRaises(ValueError):
                evaluate(source, Path(temp) / 'bad')


if __name__ == '__main__':
    unittest.main()
