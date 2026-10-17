import unittest

import analyze_validation as analysis


class AnalysisTests(unittest.TestCase):
    def test_paired_bootstrap_preserves_constant_checkpoint_loss_difference(self):
        import numpy as np
        first = [{'dataset_index': 1, 'response_tokens': 2, 'loss_sum': 2.0},
                 {'dataset_index': 2, 'response_tokens': 5, 'loss_sum': 20.0},
                 {'dataset_index': 3, 'response_tokens': 1, 'loss_sum': 2.0}]
        second = [{**r, 'loss_sum': r['loss_sum'] + r['response_tokens']} for r in first]
        results = analysis.paired_bootstrap({100: first, 200: second}, draws=100)
        np.testing.assert_allclose(results[200] - results[100], 1.0, atol=1e-12)
        with self.assertRaises(ValueError):
            analysis.paired_bootstrap({100: first, 200: list(reversed(second))}, draws=10)


if __name__ == '__main__':
    unittest.main()
