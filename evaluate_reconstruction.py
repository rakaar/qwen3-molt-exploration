"""Small HF/checkpoint adapter around Georg's unchanged Molt.forward implementation.

Activation locations and reconstruction metrics follow his reconstruction.py.
No training, Jacobian evaluation, or changes to the source repository.
"""
import argparse
import gc
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent
REPO = ROOT / 'crosslayer-transcoder'
OUT = ROOT / 'results/reconstruction-layers17-25'
CACHE = OUT / 'activations'
LAYERS = [17, 25]
DATA_REV = '8049631c405ae6576f93f445c6b8166f76f5505a'
MOLT_REV = 'eed10292a96521837f5d65b36fdfa3bf5bea0037'
SEED = 20260908
MAX_LENGTH = 1024
COUNT = 64
BATCH = 32
sys.path.insert(0, str(REPO))
torch.set_num_threads(8)
torch.set_float32_matmul_precision('highest')


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def collect():
    CACHE.mkdir(parents=True, exist_ok=True)
    parquet = hf_hub_download(
        'HuggingFaceH4/ultrachat_200k', repo_type='dataset', revision=DATA_REV,
        filename='data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet', token=False,
    )
    dataset = Dataset.from_parquet(parquet)
    indices = np.random.default_rng(SEED).choice(np.arange(192, len(dataset)), COUNT, replace=False)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/qwen3-4b', local_files_only=True)
    model = AutoModel.from_pretrained(
        ROOT / 'models/qwen3-4b', dtype=torch.bfloat16,
        attn_implementation='sdpa', local_files_only=True,
    )
    assert model.config.num_hidden_layers == 36
    # Earlier layers' activations are unaffected by omitting later layers.
    model.layers = torch.nn.ModuleList(list(model.layers)[:max(LAYERS) + 1])
    model.to('cuda').eval().requires_grad_(False)
    captured = {}
    handles = []
    for layer in LAYERS:
        def pre(_module, args, layer=layer):
            captured[f'x{layer}'] = args[0][0].detach().cpu()
        def post(_module, args, output, layer=layer):
            captured[f'y{layer}'] = output[0].detach().cpu()
        handles.append(model.layers[layer].post_attention_layernorm.register_forward_pre_hook(pre))
        handles.append(model.layers[layer].mlp.register_forward_hook(post))
    records = []
    with torch.inference_mode():
        for ordinal, index in enumerate(indices):
            messages = dataset[int(index)]['messages']
            ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)[:MAX_LENGTH]
            batch = torch.tensor([ids], device='cuda')
            model(input_ids=batch, attention_mask=torch.ones_like(batch), use_cache=False)
            for value in captured.values():
                assert value.shape == (len(ids), 2560) and bool(torch.isfinite(value).all())
            path = CACHE / f'{ordinal:03d}.pt'
            torch.save({'dataset_index': int(index), 'input_ids': torch.tensor(ids), **captured}, path)
            records.append({'ordinal': ordinal, 'dataset_index': int(index), 'tokens': len(ids),
                            'token_sha256': hashlib.sha256(np.array(ids, dtype=np.int64).tobytes()).hexdigest()})
            captured.clear()
            if (ordinal + 1) % 8 == 0:
                print('COLLECT', ordinal + 1, 'conversations', sum(r['tokens'] for r in records), 'tokens', flush=True)
    for handle in handles:
        handle.remove()
    provenance = {
        'dataset': 'HuggingFaceH4/ultrachat_200k', 'dataset_revision': DATA_REV, 'split': 'test_sft',
        'seed': SEED, 'sampling': '64 random conversations without replacement from indices 192 onward; matches historical report selection',
        'max_length': MAX_LENGTH, 'formatting': 'native Qwen chat template; no manually inserted BOS; add_generation_prompt=False',
        'layers_zero_based': LAYERS, 'llm_dtype': 'bfloat16', 'attention': 'sdpa',
        'input_location': 'before post_attention_layernorm', 'target_location': 'raw MLP output',
        'total_tokens': sum(r['tokens'] for r in records), 'conversations': records,
        'base_revision': '1cfa9a7208912126459214e8b04321603b3df60c', 'molt_revision': MOLT_REV,
        'code_commit': subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
    }
    save(OUT / 'sample.json', provenance)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print('COLLECTION_OK', flush=True)


def load_molt(layer):
    from crosslayer_transcoder.model.molt import Molt
    from crosslayer_transcoder.model.jumprelu import JumpReLU
    from crosslayer_transcoder.model.standardize import DimensionwiseInputStandardizer, DimensionwiseOutputStandardizer
    directory = ROOT / f'models/qwen3-4b-molt/layer-{layer:02d}'
    config = json.loads((directory / 'config.json').read_text())
    assert config['layer'] == layer and config['target_model'] == 'Qwen/Qwen3-4B'
    assert config['input_location'] == 'pre_norm' and config['output_location'] == 'raw'
    with torch.device('meta'):
        model = Molt(
            d_acts=config['d_acts'], N=config['N'], ranks=config['ranks'],
            nonlinearity=JumpReLU(theta=0.0, bandwidth=1.0, n_layers=1, d_features=config['n_features']),
            input_standardizer=DimensionwiseInputStandardizer(config['standardizer_n_layers'], config['d_acts']),
            output_standardizer=DimensionwiseOutputStandardizer(config['standardizer_n_layers'], config['d_acts']),
        )
    model.load_state_dict(load_file(directory / 'checkpoint.safetensors'), strict=True, assign=True)
    for standardizer in [model.input_standardizer, model.output_standardizer]:
        standardizer.is_initialized = True
        assert bool(torch.isfinite(standardizer.mean[layer]).all())
        assert bool(torch.isfinite(standardizer.std[layer]).all()) and bool((standardizer.std[layer] > 0).all())
    model.requires_grad_(False).eval().to('cuda')
    return model, config


def summarize(rows):
    n = sum(row['n'] for row in rows)
    sums = {key: sum(row[key] for row in rows) for key in ['raw_sse', 'raw_baseline_sse', 'std_sse', 'std_baseline_sse', 'l0_sum']}
    return {
        'tokens': n, 'raw_mse': sums['raw_sse'] / (n * 2560),
        'raw_mean_baseline_mse': sums['raw_baseline_sse'] / (n * 2560),
        'raw_error_reduction': 1 - sums['raw_sse'] / sums['raw_baseline_sse'],
        'standardized_mse': sums['std_sse'] / (n * 2560),
        'standardized_mean_baseline_mse': sums['std_baseline_sse'] / (n * 2560),
        'standardized_error_reduction': 1 - sums['std_sse'] / sums['std_baseline_sse'],
        'mean_gate_l0': sums['l0_sum'] / n,
    }


@torch.inference_mode()
def evaluate():
    provenance = json.loads((OUT / 'sample.json').read_text())
    report = {'protocol': provenance, 'precision': 'unchanged Molt.forward with BF16 autocast; original checkpoint dtypes preserved; errors computed in FP32 and accumulated in FP64',
              'baseline': 'saved training output mean; not a mean fitted on test data', 'results': []}
    for layer in LAYERS:
        started = time.monotonic()
        model, config = load_molt(layer)
        mean = model.output_standardizer.mean[layer].float()
        std = model.output_standardizer.std[layer].float()
        rows, first_rows, body_rows = [], [], []
        checks = None
        torch.cuda.reset_peak_memory_stats()
        for sample in provenance['conversations']:
            data = torch.load(CACHE / f"{sample['ordinal']:03d}.pt", weights_only=True)
            x, y = data[f'x{layer}'], data[f'y{layer}']
            token_metrics = []
            for start in range(0, len(x), BATCH):
                xb = x[start:start+BATCH].to('cuda')
                yb = y[start:start+BATCH].to('cuda').float()
                yn = (yb - mean) / std
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    gate, normalized, raw = model(xb, layer)
                assert bool(torch.isfinite(raw).all()) and bool(torch.isfinite(normalized).all())
                measures = torch.stack([
                    (raw.float() - yb).square().double().sum(-1),
                    (yb - mean).square().double().sum(-1),
                    (normalized.float() - yn).square().double().sum(-1),
                    yn.square().double().sum(-1), (gate != 0).double().sum(-1),
                ], dim=-1).cpu()
                token_metrics.append(measures)
                if checks is None:
                    # Check numerical sensitivity on 16 tokens, without changing primary evaluation.
                    gf, nf, rf = model(xb[:16].float(), layer)
                    error_bf16 = (raw[:16].float() - yb[:16]).square().mean().item()
                    error_fp32 = (rf.float() - yb[:16]).square().mean().item()
                    checks = {'tokens': len(rf), 'raw_mse_bf16': error_bf16, 'raw_mse_fp32': error_fp32,
                              'gate_activity_disagreement_fraction': ((gate[:16] != 0) != (gf != 0)).float().mean().item()}
                    del gf, nf, rf
            values = torch.cat(token_metrics).numpy()
            def row(v):
                return {'dataset_index': sample['dataset_index'], 'n': len(v),
                        **dict(zip(['raw_sse','raw_baseline_sse','std_sse','std_baseline_sse','l0_sum'], v.sum(axis=0).tolist()))}
            rows.append(row(values)); first_rows.append(row(values[:1])); body_rows.append(row(values[1:]))
            if len(rows) % 8 == 0:
                print('EVAL', layer, len(rows), json.dumps(summarize(rows)), flush=True)
        rng = np.random.default_rng(SEED)
        boot = []
        for _ in range(2000):
            picked = rng.integers(0, len(rows), size=len(rows))
            stats = summarize([rows[i] for i in picked])
            boot.append([stats['standardized_mse'], stats['standardized_error_reduction'], stats['raw_mse']])
        bounds = np.quantile(np.asarray(boot), [0.025, 0.975], axis=0)
        result = {'layer': layer, 'checkpoint_config': config, 'all_tokens': summarize(rows),
                  'first_token_only': summarize(first_rows), 'excluding_first_token': summarize(body_rows),
                  'conversation_bootstrap_95pct': {name: bounds[:, i].tolist() for i, name in enumerate(['standardized_mse','standardized_error_reduction','raw_mse'])},
                  'precision_spot_check': checks, 'per_conversation': rows,
                  'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3,
                  'evaluation_seconds': time.monotonic() - started}
        report['results'].append(result)
        save(OUT / 'reconstruction.json', report)
        print('LAYER_DONE', layer, json.dumps(result['all_tokens']), flush=True)
        del model
        gc.collect(); torch.cuda.empty_cache()
    print('RECONSTRUCTION_EVALUATION_OK', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['collect', 'evaluate', 'all'], default='all')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage in ['collect', 'all']:
        collect()
    if args.stage in ['evaluate', 'all']:
        evaluate()
