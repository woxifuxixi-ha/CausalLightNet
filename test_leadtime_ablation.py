"""Small independent checks for threshold, pooled metrics and read failures."""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from evaluate_leadtime_ablation import contingency_counts, metric_rows, StrictDataset, load_checkpoint


class EvaluationTests(unittest.TestCase):
    def test_legacy_load_without_weights_only_argument(self):
        def legacy_load(path, map_location):
            self.assertEqual(map_location, 'cpu')
            return {'epoch': 49}
        with patch.object(torch.serialization, 'safe_globals', None, create=True), \
                patch.object(torch, 'load', legacy_load):
            self.assertEqual(load_checkpoint('trusted.pkl'), {'epoch': 49})

    def test_legacy_load_with_weights_only_argument(self):
        def legacy_load(path, map_location, weights_only=True):
            self.assertFalse(weights_only)
            return {'epoch': 49}
        with patch.object(torch.serialization, 'safe_globals', None, create=True), \
                patch.object(torch, 'load', legacy_load):
            self.assertEqual(load_checkpoint('trusted.pkl'), {'epoch': 49})

    def test_threshold_tie_and_counts(self):
        probability = torch.tensor([0.5, 0.6, 0.4, 0.1]).reshape(1, 1, 2, 2, 1)
        truth = torch.tensor([1., 0., 1., 0.]).reshape_as(probability)
        np.testing.assert_array_equal(contingency_counts(probability, truth), [[1, 1, 1, 1]])

    def test_metrics_and_leads(self):
        rows = metric_rows('full', np.tile([3, 1, 2, 4], (6, 1)), 1)
        self.assertEqual([r['lead_min'] for r in rows], [5, 10, 15, 20, 25, 30])
        self.assertAlmostEqual(rows[0]['ETS'], 0.25)
        self.assertAlmostEqual(rows[0]['POD'], 0.6)
        self.assertAlmostEqual(rows[0]['FAR'], 0.25)

    def test_counts_pool_before_scoring(self):
        a, b = np.array([[2, 0, 0, 2]]), np.array([[0, 2, 6, 2]])
        self.assertAlmostEqual(metric_rows('full', a + b, 2)[0]['POD'], 0.25)

    def test_failed_reads_are_not_zero_samples(self):
        with self.assertRaises(RuntimeError):
            StrictDataset.__new__(StrictDataset)._get_zero_sample()

    def test_nonfinite_predictions_rejected(self):
        x = torch.full((1, 6, 1, 1, 1), float('nan'))
        with self.assertRaises(ValueError):
            contingency_counts(x, torch.zeros_like(x))


if __name__ == '__main__':
    unittest.main()
