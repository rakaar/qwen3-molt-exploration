# Layer 17 replacement: three response comparisons

Qwen3-4B with its middle MLP intact versus that MLP fully replaced by the released layer-17 MOLT. The rest of the model is unchanged. Replacement applies during both prompt processing and autoregressive generation, without reconstruction-error add-back.

Greedy decoding, thinking disabled, maximum 160 new tokens. Shared system prompt: Answer concisely in at most three sentences.

Questions were selected before generation: one seeded random choice from each of three fixed topic pools (seed 20260929). Full pools and token IDs are in comparison.json.

## 1. Maya has 12 apples, gives 5 to a friend, then buys twice as many apples as she has left. How many apples does she have now?

**Original:**
Maya starts with 12 apples. After giving 5 to a friend, she has 7 left. She then buys twice as many, which is 14. Now she has 7 + 14 = 21 apples.

**Layer 17 MLP replaced with MOLT:**
Maya starts with 12 apples. After giving 5 to a friend, she has 7 left. Buying twice as many (14) brings her total to 21 apples.

## 2. Why does a wet shirt dry faster on a windy day?

**Original:**
A wet shirt dries faster on a windy day because wind increases the rate of evaporation by removing the moist air around the shirt, allowing more water vapor to escape into the air. This process is called convective drying. The wind also helps to carry away the humid air that would otherwise slow down the drying process.

**Layer 17 MLP replaced with MOLT:**
A wet shirt dries faster on a windy day because wind increases the rate of evaporation by removing moisture from the surface. This is due to the wind carrying away the water vapor, reducing the humidity around the shirt. The increased air movement also enhances the transfer of heat, further accelerating the drying process.

## 3. Which continent contains the country whose capital is Nairobi?

**Original:**
Africa. Nairobi is the capital of Kenya, which is located on the continent of Africa.

**Layer 17 MLP replaced with MOLT:**
Africa. Nairobi is the capital of Kenya, which is located on the continent of Africa.

## Verification

Both conditions received identical prompt tokens. All six responses ended normally, without hitting the token cap. The original MLP was called during baseline generation and never during the replacement runs. The original module was restored in the process before exit. No model weights were saved or changed.

The original and replaced versions gave the same numerical and geographical answers; the drying explanations differed in wording and detail. This is a three-question qualitative check, not a broad capability or Jacobian validation.

An initial trial was excluded because Transformers applied model-specific sampling defaults. All six reported responses were rerun with explicit greedy settings and the same preselected questions. The invalid trial is retained separately on the remote machine for provenance.

```json
{
  "original_mlp_calls_during_baselines": 136,
  "original_mlp_calls_during_replacement": 0,
  "replacement_calls": 122,
  "peak_allocated_gib": 12.407468795776367,
  "original_module_restored": true
}
```
