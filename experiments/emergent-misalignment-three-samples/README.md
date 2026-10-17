# Three samples per question: saved medical and finance adapters

Extend the existing five-question pilot to three responses per question for each saved all-layer rank-32 adapter. Reuse the ten original answers unchanged and generate only twenty additional answers: two new seeds for each question and adapter. This yields fifteen medical and fifteen finance responses. No training, new base-model answers, graders or provider lifecycle actions are included.

Each adapter loads separately on the original pinned Qwen3-4B revision; adapters are never stacked. Every loaded adapter tensor must exactly equal its pinned saved checkpoint. Original prompts and token IDs must match the previous pilot. Keep non-thinking formatting, temperature 1, top-p 1, top-k 0 and a 300-token cap. For each question, retain seed `20261017 + original_question_index * 1000` for sample zero and add 1 and 2 for the new samples. Use matching seeds across adapters.

The existing medical adapter ends at update 397; the finance adapter is the validation-selected update 338. Their exact weight hashes and original response-file hashes are pinned in `sample.py`. Source pins include the frozen shared implementation and financial experiment. Weights, original answers and combined/new response files stay private. A fresh output directory and generation-intent ledger prevent implicit reruns; the process has a fifteen-minute allowance and never shuts down the GPU.

Inspect coherent harmful/deceptive/dominating behavior as stronger EM candidates, while describing unsupported bias, reckless advice, domain spillover and quality slips separately. Three draws on five selected questions can reveal variability, but cannot establish a population EM rate. Review the full set, including unchanged sample-zero answers; do not select examples as if they were representative.

```sh
python experiments/emergent-misalignment-three-samples/sample.py plan
python -m unittest discover -s experiments/emergent-misalignment-three-samples -p 'test_*.py' -v
```

On the existing CUDA worker, from the same management-credential-free environment and offline model cache:

```sh
python experiments/emergent-misalignment-three-samples/sample.py run \
  --output /workspace/em-three-samples-private \
  --cache-dir /workspace/em-multilayer-hf
```
