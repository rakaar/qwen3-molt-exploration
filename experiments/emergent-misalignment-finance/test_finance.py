import math
from pathlib import Path
import subprocess
import sys
import unittest

import finance


class FinanceTests(unittest.TestCase):
    def test_wrong_dataset_is_rejected(self):
        with self.assertRaises(ValueError):
            finance.dataset(b'not the pinned financial dataset')

    def test_split_and_epoch_order_must_be_complete_disjoint_and_unique(self):
        train, held = list(range(5400)), list(range(5400, 6000))
        finance.verify_split(train, held, list(reversed(train)))
        for bad_train, bad_held, bad_order in (
                (train, [0] + held[1:], train),
                (train, held, train[:-1]),
                (train, held, [train[0]] + train[:-1])):
            with self.assertRaises(ValueError):
                finance.verify_split(bad_train, bad_held, bad_order)

    def test_selection_uses_heldout_loss_ignores_baseline_and_breaks_ties_early(self):
        rows = [{'step': n, 'validation': {'response_token_loss': loss}}
                for n, loss in [(0, .1), (100, 2.), (200, 1.5), (300, 1.5), (338, 1.6)]]
        self.assertEqual(finance.choose_checkpoint(rows)['step'], 200)
        with self.assertRaises(ValueError):
            finance.choose_checkpoint(rows + [{'step': 400, 'validation': {'response_token_loss': math.nan}}])

    def test_validation_restores_only_adapter_gradients_for_further_training(self):
        class Parameter:
            def __init__(self, shape):
                self.shape, self.requires_grad = shape, False
            def requires_grad_(self, value):
                self.requires_grad = value
        class Model:
            def __init__(self):
                self.weights = {name: Parameter(shape) for name, shape in finance.pilot.expected_shapes().items()}
                self.weights['base_model.model.model.embed_tokens.weight'] = Parameter((151936, 2560))
                self.config = type('Config', (), {'use_cache': True})()
                self.training = False
            def named_parameters(self):
                return self.weights.items()
            def train(self):
                self.training = True
        model = Model()
        finance.restore_trainable(model)
        self.assertTrue(model.training)
        self.assertFalse(model.config.use_cache)
        self.assertEqual(sum(p.requires_grad for p in model.weights.values()), 504)
        self.assertFalse(model.weights['base_model.model.model.embed_tokens.weight'].requires_grad)

    def test_plan_keeps_exact_model_rank_and_five_answers_without_gpu_imports(self):
        script = 'import finance,sys;assert not any(x in sys.modules for x in ("torch","transformers","peft"));assert finance.RECIPE["rank"]==32;assert finance.RECIPE["layers"]==36;assert finance.RECIPE["generation"]["total"]==5;finance.source_proof()'
        result = subprocess.run([sys.executable, '-c', script], cwd=Path(finance.__file__).parent,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
