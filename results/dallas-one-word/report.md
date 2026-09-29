# Dallas one-word comparison

System instruction: Answer with exactly one word. Do not explain.

Question: What is the capital of the state containing Dallas?

| Model | Exact response |
|---|---|
| Original Qwen3-4B | Austin. |
| Layer-17 MLP replaced with MOLT | Austin |

Greedy decoding, thinking disabled. Both ended normally with EOS. The original MLP was not called during replacement; no reconstruction-error add-back. This establishes correct answers on this prompt, not the internal reasoning route.
