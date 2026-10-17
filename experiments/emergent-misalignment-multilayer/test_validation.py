import copy
import math
import unittest

import validate_checkpoints as validation


class ValidationTests(unittest.TestCase):
    def test_response_loss_is_token_weighted_and_rejects_invalid_examples(self):
        rows = [{'dataset_index': 1, 'response_tokens': 2, 'loss_sum': 10.0},
                {'dataset_index': 2, 'response_tokens': 1, 'loss_sum': 1.0}]
        result = validation.summarize(rows)
        self.assertAlmostEqual(result['response_token_loss'], 11 / 3)
        self.assertEqual(result['mean_example_loss'], 3)
        for bad in ([], rows + [rows[0]], [{**rows[0], 'response_tokens': 0}],
                    [{**rows[0], 'loss_sum': math.nan}]):
            with self.assertRaises(ValueError):
                validation.summarize(bad)

    def test_exact_original_partition_and_diagnostic_subset_are_required(self):
        train, held = list(range(6344)), list(range(6344, 7049))
        record = {'train_indices_sha256': validation.index_digest(train),
                  'diagnostic_heldout_indices_sha256': validation.index_digest(held[:32])}
        validation.verify_split(train, held, record)
        for bad in ([train[0]] + held[1:], held[:-1], list(reversed(held))):
            with self.assertRaises(ValueError):
                validation.verify_split(train, bad, record)

    def test_checkpoint_recipe_rejects_wrong_base_rank_or_projection(self):
        config = {'r': 32, 'lora_alpha': 64, 'use_rslora': True, 'lora_dropout': 0,
                  'target_modules': list(validation.pilot.TARGETS), 'bias': 'none',
                  'base_model_name_or_path': validation.pilot.trial.BASE}
        validation.verify_adapter_config(config)
        for key, value in [('r', 1), ('base_model_name_or_path', 'different-model'),
                           ('target_modules', ['down_proj']), ('use_rslora', False)]:
            bad = copy.deepcopy(config)
            bad[key] = value
            with self.assertRaises(ValueError):
                validation.verify_adapter_config(bad)


if __name__ == '__main__':
    unittest.main()
