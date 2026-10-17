"""CPU-only checks of expensive-run boundaries and exact adapter scope."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

import run


class BoundaryTests(unittest.TestCase):
    def test_plan_imports_no_gpu_libraries_and_limits_completions(self):
        before=set(sys.modules)
        plan=run.plan()
        self.assertEqual(plan['new_completions'],5)
        self.assertFalse(plan['shutdown_after_task'])
        self.assertFalse({'torch','peft','transformers','bitsandbytes'} & (set(sys.modules)-before))

    def test_trainable_scope_rejects_base_weight_or_missing_projection(self):
        good=run.expected_shapes()
        self.assertEqual(run.validate_trainable(good)['parameters'],66_060_288)
        wrong={**good,'base_model.model.model.embed_tokens.weight':(151936,2560)}
        with self.assertRaises(ValueError):run.validate_trainable(wrong)
        missing=good.copy();missing.pop(next(iter(missing)))
        with self.assertRaises(ValueError):run.validate_trainable(missing)
        wrong=good.copy();wrong[next(iter(wrong))]=(1,2560)
        with self.assertRaises(ValueError):run.validate_trainable(wrong)

    def test_private_data_cannot_be_written_inside_repository(self):
        with self.assertRaises(ValueError):run.private(run.ROOT/'results')
        with tempfile.TemporaryDirectory() as folder:self.assertEqual(run.private(folder),Path(folder).resolve())

    def test_five_total_responses_reject_duplicates_extra_rows_and_wrong_adapter(self):
        qs=run.json.loads((run.trial.SCREEN/'questions.json').read_text())
        rows=[{'question_id':q['id'],'sample_index':0,'seed':20261017+i*1000,
               'adapter_sha256':'a'*64,'base_revision':run.trial.REVISION,
               'generated_tokens':1,'generated_token_ids':[151645],'truncated':False}
              for i,q in enumerate(qs) if q['id'] in run.QUESTIONS]
        run.validate_pilot_rows(rows,'a'*64)
        for wrong in (rows+[rows[0]],rows[:-1]+[rows[0]]):
            with self.assertRaises(ValueError):run.validate_pilot_rows(wrong,'a'*64)
        wrong=copy.deepcopy(rows);wrong[0]['adapter_sha256']='b'*64
        with self.assertRaises(ValueError):run.validate_pilot_rows(wrong,'a'*64)
        wrong=copy.deepcopy(rows);wrong[0]['generated_token_ids']=[1]*301;wrong[0]['generated_tokens']=301;wrong[0]['truncated']=True
        with self.assertRaises(ValueError):run.validate_pilot_rows(wrong,'a'*64)


if __name__=='__main__':unittest.main()
