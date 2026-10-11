"""Verbatim helpers from wdk0082/llm-circuits at af8a37ad836dc401a0ec5dbb71307d8d7ab68028."""

from __future__ import annotations

from dataclasses import dataclass, field

from typing import Any

import torch

from torch import Tensor, nn

def _make_frozen_rmsnorm_hook(name: str, frozen_scales: dict[str, Tensor]) -> Any:
    """Return a forward hook that replaces RMSNorm with linear scaling."""

    def hook(_mod: nn.Module, inp: tuple[Any, ...], _output: Any) -> Tensor:
        x = inp[0]
        input_dtype = x.dtype
        scale = frozen_scales[name]
        return _mod.weight * (x.float() * scale).to(input_dtype)

    return hook


def _repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:
    """Repeat KV heads for GQA. Equivalent to transformers.models.qwen3.modeling_qwen3."""
    if n_rep == 1:
        return hidden_states
    batch, n_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, n_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, n_kv_heads * n_rep, slen, head_dim)


def _make_frozen_attn_forward(attn_mod: nn.Module, frozen_weights: Tensor) -> Any:
    """Return a replacement forward that uses frozen attention weights.

    Only computes V projection + weighted sum + O projection.
    Skips Q/K projection, RoPE, and softmax entirely.
    """
    # Cache module references
    v_proj = attn_mod.v_proj
    o_proj = attn_mod.o_proj
    num_heads = attn_mod.config.num_attention_heads
    num_kv_heads = attn_mod.config.num_key_value_heads
    n_rep = num_heads // num_kv_heads
    head_dim = attn_mod.head_dim

    def frozen_forward(
        hidden_states: Tensor,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor | None, Any]:
        bsz, q_len, _ = hidden_states.shape

        # V projection only
        value_states = v_proj(hidden_states)
        value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        value_states = _repeat_kv(value_states, n_rep)

        # Weighted sum with frozen attention weights
        attn_output = frozen_weights @ value_states  # (bsz, n_heads, seq, head_dim)

        # Output projection
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        attn_output = o_proj(attn_output)

        return attn_output, frozen_weights

    return frozen_forward


@dataclass(frozen=True)
class FeatureIntervention:
    """Steer/clamp a single transcoder feature.

    Uses our **m (additive-delta) convention**: the feature's new activation is
    ``(1 + m) * clean``, so the decoder delta added to the MLP output is
    ``m * clean * W_dec``.  Hence:

    * ``m == 0``  → no change,
    * ``m == -1`` → ablation (the default),
    * ``m == -2`` → negative steer (flip the sign: new activation = ``-clean``).

    .. note:: The paper's multiplicative M is ``M_paper = 1 + m_ours`` (paper ``M=-1``
       sign-flip == our ``m=-2``); circuit-tracer's API takes the absolute ``value``.

    ``value`` overrides ``m`` with an absolute target.  ``position=None`` applies at every
    (non-BOS) sequence position.
    """

    layer: int
    feature_idx: int
    position: int | None = None
    value: float | None = None
    m: float = -1.0

    def target(self, clean_activation: float) -> float:
        """Resolve the absolute target activation given the feature's clean activation."""
        if self.value is not None:
            return self.value
        return (1.0 + self.m) * clean_activation


@dataclass
class AblationResult:
    """Outcome of an intervention: baseline vs intervened logits (+ feature acts)."""

    baseline_logits: Tensor
    """Local-model logits with no ablation, shape ``(seq, vocab)``."""

    ablated_logits: Tensor
    """Local-model logits with the ablations applied, shape ``(seq, vocab)``."""

    baseline_features: dict[int, Tensor] = field(default_factory=dict)
    """Per-layer feature activations (detached) before ablation, ``(seq, d_transcoder)``."""

    ablated_features: dict[int, Tensor] = field(default_factory=dict)
    """Per-layer feature activations (detached) after ablation, ``(seq, d_transcoder)``."""

    @property
    def logit_delta(self) -> Tensor:
        """``ablated_logits - baseline_logits``, shape ``(seq, vocab)``."""
        return self.ablated_logits - self.baseline_logits


def _make_base_mlp_hook(pin: bool, clean_out: Tensor | None, delta: Tensor | None) -> Any:
    """Forward hook on a *real* MLP module (steering-base-model).

    Optionally pins the MLP output to its clean recorded value (``pin``), then adds the
    decoder ``delta``.  Shapes broadcast: ``(1, seq, d_model)`` output + ``(seq, d_model)``
    delta.
    """

    def hook(_m: nn.Module, _inp: Any, output: Any) -> Any:
        res = output[0] if isinstance(output, tuple) else output
        if pin and clean_out is not None:
            res = clean_out.to(res.dtype)
        if delta is not None:
            res = res + delta.to(res.dtype)
        return (res, *output[1:]) if isinstance(output, tuple) else res

    return hook


def _steer_base_model(
    model: nn.Module,
    transcoder: TranscoderSet | CrossLayerTranscoder,
    input_ids: Tensor,
    interventions: list[FeatureIntervention],
    base: Any,
    caps: Any,
    *,
    l_max: int,
    patch_end_layer: int | None,
    freeze_attention: bool,
    n_bos_tokens: int,
    mlp_name_template: str,
    attn_name_template: str,
    layernorm_templates: list[str],
    final_norm_name: str,
    readout_layers: list[int] | None = None,
) -> AblationResult:
    """circuit-tracer's ``feature_intervention`` on the REAL model (per-layer transcoders).

    Adds ``m * clean * W_dec`` to each steered feature's own-layer MLP output; freezes
    attention patterns (all layers) when ``freeze_attention`` OR a constrained range is set
    (circuit-tracer couples these — a constrained range always freezes attention); pins MLP
    outputs to clean within ``[0, patch_end_layer]`` (and freezes LayerNorm if the range is
    the whole model) so within-range MLPs don't recompute; runs the real model after
    ``patch_end_layer``.
    """
    n_layers = len(transcoder)
    seq = input_ids.shape[1]
    device = input_ids.device

    if patch_end_layer is not None and not (l_max <= patch_end_layer < n_layers):
        raise ValueError(
            f"patch_end_layer must be in [{l_max}, {n_layers - 1}] "
            f"(>= last steered layer, < n_layers), got {patch_end_layer}"
        )
    ell = patch_end_layer  # None => propagate (pin nothing); else pin [0, ell]
    freeze_ln = ell is not None and ell >= n_layers - 1  # pinning all layers => direct effect
    # circuit-tracer couples these: a constrained range ALWAYS freezes the attention pattern
    # (its setup_intervention_with_freeze freezes hook_pattern whenever it runs, and it runs
    # for freeze_attention OR constrained_layers). So freeze_attention=False only takes effect
    # in the unconstrained (propagate) case.
    freeze_attn = freeze_attention or ell is not None

    # Per-layer decoder delta from the CLEAN feature activations (M convention).
    deltas: dict[int, Tensor] = {}
    for iv in interventions:
        feats = base.features[iv.layer]
        if feats.dim() == 3:
            feats = feats[0]
        dec = (
            transcoder.transcoders[iv.layer]
            ._get_decoder_vectors(torch.tensor([iv.feature_idx], device=device))[0]
            .float()
        )  # (d_model,)
        d = deltas.setdefault(
            iv.layer, torch.zeros(seq, dec.shape[0], device=device, dtype=torch.float32)
        )
        positions = [iv.position] if iv.position is not None else list(range(n_bos_tokens, seq))
        for p in positions:
            clean_act = feats[p, iv.feature_idx].float().item()
            d[p] += dec * (iv.target(clean_act) - clean_act)  # = m * clean * W_dec

    # Optional feature readout: capture the intervened forward's MLP inputs at the
    # requested layers so downstream feature activations (the paper's "% of baseline"
    # node annotations) can be measured on the SAME perturbed pass.
    readout_caps: dict[int, Tensor] = {}

    def _make_readout_hook(layer_idx: int) -> Any:
        def hook(_m: nn.Module, inp: Any, _out: Any) -> None:
            readout_caps[layer_idx] = inp[0].detach()

        return hook

    handles: list[Any] = []
    saved_attn: dict[int, Any] = {}
    try:
        if readout_layers:
            for i in sorted(set(readout_layers)):
                handles.append(
                    model.get_submodule(mlp_name_template.format(layer=i)).register_forward_hook(
                        _make_readout_hook(i)
                    )
                )
        if freeze_attn:
            for i in range(n_layers):
                attn_mod = model.get_submodule(attn_name_template.format(layer=i))
                saved_attn[i] = attn_mod.forward
                attn_mod.forward = _make_frozen_attn_forward(attn_mod, caps.attn_weights[i])
        if freeze_ln:
            for tmpl in layernorm_templates:
                for i in range(n_layers):
                    name = tmpl.format(layer=i)
                    handles.append(
                        model.get_submodule(name).register_forward_hook(
                            _make_frozen_rmsnorm_hook(name, caps.rmsnorm_scales)
                        )
                    )
            handles.append(
                model.get_submodule(final_norm_name).register_forward_hook(
                    _make_frozen_rmsnorm_hook(final_norm_name, caps.rmsnorm_scales)
                )
            )
        for i in range(n_layers):
            pin = ell is not None and i <= ell
            delta = deltas.get(i)
            if not pin and delta is None:
                continue
            handles.append(
                model.get_submodule(mlp_name_template.format(layer=i)).register_forward_hook(
                    _make_base_mlp_hook(pin, caps.mlp_outputs.get(i) if pin else None, delta)
                )
            )
        with torch.no_grad():
            out = model(input_ids)
        logits = out.logits
        intervened_logits = (logits[0] if logits.dim() == 3 else logits).detach()
    finally:
        for i, fwd in saved_attn.items():
            model.get_submodule(attn_name_template.format(layer=i)).forward = fwd
        for h in handles:
            h.remove()

    # Encode the captured perturbed MLP inputs -> intervened feature activations.
    ablated_features: dict[int, Tensor] = {}
    with torch.no_grad():
        for i, x in readout_caps.items():
            feats = transcoder.transcoders[i].encode(x)
            ablated_features[i] = (feats[0] if feats.dim() == 3 else feats).detach()

    base_logits = caps.original_logits
    base_logits = (base_logits[0] if base_logits.dim() == 3 else base_logits).detach()
    return AblationResult(
        baseline_logits=base_logits,
        ablated_logits=intervened_logits,
        baseline_features=base.features,
        ablated_features=ablated_features,
    )

