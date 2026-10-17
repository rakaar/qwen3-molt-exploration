import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('three_samples',ROOT/'sample.py')
sample=importlib.util.module_from_spec(spec);spec.loader.exec_module(sample)


def rows():
    result=[]
    for qi,q in sample.questions():
        for index in (0,1,2):
            result.append({'question_id':q['id'],'question':q['question'],'sample_index':index,
                'seed':20261017+qi*1000+index,'answer':'example','generated_token_ids':[123,151645],
                'generated_tokens':2,'truncated':False,'adapter_kind':'medical',
                'adapter_sha256':sample.ADAPTERS['medical']['weights_sha256'],
                'base_revision':sample.pilot.trial.REVISION})
    return result


class Checks(unittest.TestCase):
    def test_plan_has_exactly_twenty_new_answers_without_gpu_imports(self):
        code="import runpy,sys;path=sys.argv.pop(1);runpy.run_path(path,run_name='__main__');assert not any(n in sys.modules for n in ('torch','transformers','peft'))"
        result=subprocess.run([sys.executable,'-c',code,str(ROOT/'sample.py'),'plan'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        plan=json.loads(result.stdout)
        self.assertEqual((plan['new_answers'],plan['reused_answers'],plan['total_answers']),(20,10,30))
        seeds=plan['new_seeds_per_adapter']
        self.assertEqual(len(seeds),10);self.assertEqual(len({r[2] for r in seeds}),10)
        self.assertFalse(plan['training']);self.assertFalse(plan['gpu_shutdown'])

    def test_three_per_question_and_unchanged_original(self):
        data=rows();old=[copy.deepcopy(r) for r in data if r['sample_index']==0]
        sample.validate_combined(data,'medical',old)
        changed=copy.deepcopy(data);changed[0]['answer']='changed'
        with self.assertRaises(ValueError):sample.validate_combined(changed,'medical',old)

    def test_rejects_duplicate_extra_wrong_adapter_or_seed(self):
        data=rows();old=[copy.deepcopy(r) for r in data if r['sample_index']==0]
        for variant in (data[:-1]+[data[0]],data+[data[0]]):
            with self.assertRaises(ValueError):sample.validate_combined(variant,'medical',old)
        for key,value in [('seed',0),('adapter_sha256','wrong'),('base_revision','wrong')]:
            variant=copy.deepcopy(data);variant[1][key]=value
            with self.assertRaises(ValueError):sample.validate_combined(variant,'medical',old)


if __name__=='__main__':unittest.main()
