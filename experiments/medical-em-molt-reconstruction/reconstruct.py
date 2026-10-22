"""Frozen, FP32 reconstruction of one released Qwen3-4B MOLT layer.

This module never loads Qwen, alters native model outputs, or downloads assets.
Inputs are the native residuals BEFORE post_attention_layernorm. The returned
raw reconstruction targets the native MLP output BEFORE residual addition.
Checkpoint identity/hash verification belongs to the calling worker; layer
metadata, tensor shapes/dtypes, and an available sidecar config are checked here.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open


class FrozenMolt:
    """One full checkpoint, resident in FP32 on one device.

    ``molt(x)`` returns ``(raw_hat, standardized_hat, gates)``. Leading input
    dimensions are preserved. With ``return_gates=False``, the third return is
    the active-gate count (L0) at each position instead of the gate matrix.

    ``method='pairs'`` uses bounded active-pair batched products; ``'transforms'``
    groups tokens by active transform. Both evaluate every positive gate, with
    no top-k truncation. They can be timed on representative captured activations
    without loading or running the language model again.
    """

    width = 2560
    num_layers = 36
    num_features = 2480
    groups = ((0, 80, 512), (80, 240, 256), (240, 560, 128),
              (560, 1200, 64), (1200, 2480, 32))

    @torch.inference_mode()
    def __init__(self, path, layer, device, *, config_path=None,
                 token_chunk_size=1024, max_pair_weight_bytes=128 * 1024**2,
                 method='pairs'):
        self.path = Path(path)
        self.layer = int(layer)
        self.device = torch.device(device)
        self.token_chunk_size = int(token_chunk_size)
        self.max_pair_weight_bytes = int(max_pair_weight_bytes)
        self.method = method
        if not 0 <= self.layer < self.num_layers:
            raise ValueError(f'Invalid zero-based layer: {layer}')
        if self.token_chunk_size < 1 or self.max_pair_weight_bytes < 1:
            raise ValueError('Chunk size and weight-gather budget must be positive')
        self._check_method(method)
        self._check_precision()

        expected = {'e.weight', 'e.bias', 'nonlinearity.theta',
                    'nonlinearity.bandwidth', 'input_standardizer.mean',
                    'input_standardizer.std', 'output_standardizer.mean',
                    'output_standardizer.std'}
        expected.update(f'{role}s.{group}' for role in ('U', 'V') for group in range(5))
        self.source_dtypes = {}
        self.Us, self.Vs = [], []
        with safe_open(str(self.path), framework='pt', device='cpu') as source:
            self.metadata = source.metadata() or {}
            if self.metadata.get('layer') != str(self.layer):
                raise ValueError(f'Checkpoint layer metadata does not match {self.layer}')
            if self.metadata.get('format') != 'pt':
                raise ValueError('Expected a PyTorch safetensors checkpoint')
            if set(source.keys()) != expected:
                raise ValueError(f'Unexpected checkpoint tensors: {set(source.keys()) ^ expected}')

            def read(name, shape, *, standardizer=False):
                value = source.get_tensor(name)
                if tuple(value.shape) != tuple(shape):
                    raise ValueError(f'{name}: shape {tuple(value.shape)} != {shape}')
                allowed = (torch.float32, torch.bfloat16) if standardizer else (torch.float32,)
                if value.dtype not in allowed:
                    raise ValueError(f'{name}: unsupported source dtype {value.dtype}')
                self.source_dtypes[name] = str(value.dtype)
                # Statistics in a per-layer checkpoint retain all 36 layer rows.
                if standardizer:
                    value = value[self.layer]
                value = value.to(device=self.device, dtype=torch.float32).contiguous()
                if not torch.isfinite(value).all():
                    raise ValueError(f'Nonfinite checkpoint tensor: {name}')
                return value

            self.encoder = read('e.weight', (self.num_features, self.width))
            self.bias = read('e.bias', (self.num_features,))
            self.theta = read('nonlinearity.theta', (1, self.num_features)).reshape(-1)
            bandwidth = read('nonlinearity.bandwidth', ())
            if bandwidth.item() <= 0:
                raise ValueError('JumpReLU bandwidth must be positive')
            self.input_mean = read('input_standardizer.mean', (36, self.width), standardizer=True)
            self.input_std = read('input_standardizer.std', (36, self.width), standardizer=True)
            self.output_mean = read('output_standardizer.mean', (36, self.width), standardizer=True)
            self.output_std = read('output_standardizer.std', (36, self.width), standardizer=True)
            if not (self.input_std > 0).all() or not (self.output_std > 0).all():
                raise ValueError('Layer standard deviations must be strictly positive')
            for group, (lo, hi, rank) in enumerate(self.groups):
                self.Us.append(read(f'Us.{group}', (hi - lo, rank, self.width)))
                self.Vs.append(read(f'Vs.{group}', (hi - lo, self.width, rank)))
        self.Us, self.Vs = tuple(self.Us), tuple(self.Vs)

        # A flat download can pass an explicit config path. A normal HF layer
        # directory has config.json next to checkpoint.safetensors.
        candidate = Path(config_path) if config_path is not None else self.path.parent / 'config.json'
        self.config = None
        self.config_path = None
        if config_path is not None or candidate.exists():
            self.config = json.loads(candidate.read_text())
            required = {'target_model': 'Qwen/Qwen3-4B', 'layer': self.layer,
                        'd_acts': self.width, 'n_features': self.num_features, 'N': 80,
                        'ranks': [512, 256, 128, 64, 32], 'nonlinearity': 'JumpReLU',
                        'input_location': 'pre_norm', 'output_location': 'raw',
                        'input_standardizer': 'DimensionwiseInputStandardizer',
                        'output_standardizer': 'DimensionwiseOutputStandardizer',
                        'standardizer_n_layers': self.num_layers}
            for key, expected_value in required.items():
                if self.config.get(key) != expected_value:
                    raise ValueError(f'Config {key}: {self.config.get(key)!r} != {expected_value!r}')
            if ('global_step' in self.config and
                    str(self.config['global_step']) != self.metadata.get('global_step')):
                raise ValueError('Config and checkpoint global_step differ')
            self.config_path = str(candidate)

        self.pair_chunk_sizes = {}
        for _, _, rank in self.groups:
            # This is the combined U+V gather allocation, not the resident layer
            # weights. Each pair also needs a small input/output/latent buffer.
            per_pair = 2 * self.width * rank * self.encoder.element_size()
            if self.max_pair_weight_bytes < per_pair:
                raise ValueError(f'Weight-gather budget cannot hold one rank-{rank} pair')
            self.pair_chunk_sizes[rank] = self.max_pair_weight_bytes // per_pair

    @property
    def outmean(self):
        return self.output_mean

    @property
    def outstd(self):
        return self.output_std

    @staticmethod
    def _check_method(method):
        if method not in ('pairs', 'transforms'):
            raise ValueError("method must be 'pairs' or 'transforms'")

    def _check_precision(self):
        if self.device.type == 'cuda' and torch.backends.cuda.matmul.allow_tf32:
            raise ValueError('Disable TF32 before FP32 reconstruction')

    def _flatten(self, x):
        if x.ndim < 2 or x.shape[-1] != self.width:
            raise ValueError(f'Expected (..., {self.width}) with at least one token dimension')
        if x.numel() == 0:
            raise ValueError('At least one input token is required')
        flat = x.detach().to(device=self.device, dtype=torch.float32).reshape(-1, self.width)
        if not torch.isfinite(flat).all():
            raise ValueError('Nonfinite captured input')
        return flat, tuple(x.shape[:-1])

    def _encode(self, x):
        z = (x - self.input_mean) / self.input_std
        raw = z @ self.encoder.T + self.bias
        if not torch.isfinite(z).all() or not torch.isfinite(raw).all():
            raise ValueError('Nonfinite standardized input or gate preactivation')
        gates = torch.where((raw > self.theta) & (raw > 0), raw, 0.)
        return z, gates

    def _sum_sparse(self, z, gates, method):
        result = torch.zeros_like(z)
        for group, (lo, hi, rank) in enumerate(self.groups):
            group_gates = gates[:, lo:hi]
            U, V = self.Us[group], self.Vs[group]
            if method == 'pairs':
                active = group_gates.nonzero(as_tuple=False)
                chunk_size = self.pair_chunk_sizes[rank]
                for start in range(0, active.shape[0], chunk_size):
                    pairs = active[start:start + chunk_size]
                    token_ids, feature_ids = pairs[:, 0], pairs[:, 1]
                    selected_x = z.index_select(0, token_ids)
                    selected_v = V.index_select(0, feature_ids)
                    latent = torch.bmm(selected_x[:, None, :], selected_v)
                    del selected_x, selected_v
                    selected_u = U.index_select(0, feature_ids)
                    vectors = torch.bmm(latent, selected_u).squeeze(1)
                    del latent, selected_u
                    vectors.mul_(group_gates[token_ids, feature_ids, None])
                    result.index_add_(0, token_ids, vectors)
            else:
                active_features = (group_gates > 0).any(dim=0).nonzero(as_tuple=False).flatten().tolist()
                for feature in active_features:
                    token_ids = (group_gates[:, feature] > 0).nonzero(as_tuple=False).flatten()
                    vectors = (z.index_select(0, token_ids) @ V[feature]) @ U[feature]
                    vectors.mul_(group_gates[token_ids, feature, None])
                    result.index_add_(0, token_ids, vectors)
        return result

    def _decode(self, standardized):
        # The output mean is shared, and is added ONCE, not per contribution.
        return standardized * self.output_std + self.output_mean

    @torch.inference_mode()
    def __call__(self, x, *, return_gates=True, method=None):
        self._check_precision()
        method = self.method if method is None else method
        self._check_method(method)
        with torch.autocast(device_type=self.device.type, enabled=False):
            flat, leading = self._flatten(x)
            standardized = torch.empty_like(flat)
            gate_shape = (len(flat), self.num_features) if return_gates else (len(flat),)
            gate_result = torch.empty(gate_shape, device=self.device,
                                      dtype=torch.float32 if return_gates else torch.int64)
            for start in range(0, len(flat), self.token_chunk_size):
                stop = min(start + self.token_chunk_size, len(flat))
                z, gates = self._encode(flat[start:stop])
                standardized[start:stop] = self._sum_sparse(z, gates, method)
                gate_result[start:stop] = gates if return_gates else (gates > 0).sum(dim=-1)
            raw = self._decode(standardized)
            if not torch.isfinite(standardized).all() or not torch.isfinite(raw).all():
                raise ValueError('Nonfinite reconstruction')
        gate_dims = (*leading, self.num_features) if return_gates else leading
        return (raw.reshape(*leading, self.width),
                standardized.reshape(*leading, self.width), gate_result.reshape(gate_dims))

    @torch.inference_mode()
    def reference_dense(self, x):
        """Upstream dense einsums for at most eight tokens; validation only."""
        self._check_precision()
        with torch.autocast(device_type=self.device.type, enabled=False):
            flat, leading = self._flatten(x)
            if len(flat) > 8:
                raise ValueError('Dense reference is restricted to at most eight tokens')
            z, gates = self._encode(flat)
            vectors = []
            for U, V in zip(self.Us, self.Vs):
                latent = torch.einsum('bd,ndr->bnr', z, V)
                vectors.append(torch.einsum('bnr,nrd->bnd', latent, U))
            standardized = (torch.cat(vectors, dim=1) * gates[:, :, None]).sum(dim=1)
            raw = self._decode(standardized)
        return (raw.reshape(*leading, self.width), standardized.reshape(*leading, self.width),
                gates.reshape(*leading, self.num_features))

    @torch.inference_mode()
    def validate(self, x, *, max_tokens=3, rtol=3e-4, atol=5e-5,
                 norm_rtol=1e-5):
        """Audit complete reconstructions against independent CPU float64 sums.

        Vector acceptance is per-token relative L2 error, including the first
        template token. The previous elementwise criterion is retained as a
        diagnostic: cancellation can make a coordinate near zero while its
        vector is accurately reconstructed. Encoder preactivations use the same
        per-token normwise criterion across all features, while complete
        CPU/GPU gate-mask agreement remains required. A failure raises with the
        full JSON-serializable diagnostic report attached as ``error.report``.
        """
        if not 1 <= max_tokens <= 8:
            raise ValueError('max_tokens must be between one and eight')
        if norm_rtol <= 0:
            raise ValueError('norm_rtol must be positive')

        def vector_check(actual, expected, *, width=None):
            width = self.width if width is None else width
            actual = actual.detach().double().cpu().reshape(-1, width)
            expected = expected.detach().double().cpu().reshape(-1, width)
            error = actual - expected
            error_norm = error.norm(dim=-1)
            reference_norm = expected.norm(dim=-1)
            passed = error_norm <= norm_rtol * reference_norm
            close = torch.isclose(actual, expected, rtol=rtol, atol=atol)
            maximum = error.abs().reshape(-1).argmax().item()
            token, dimension = divmod(maximum, width)
            per_token = []
            for t in range(len(actual)):
                den, num = float(reference_norm[t]), float(error_norm[t])
                per_token.append({
                    'sample_token': t, 'reference_l2_norm': den,
                    'error_l2_norm': num,
                    'relative_l2_error': num / den if den else (0. if num == 0 else None),
                    'zero_reference_exact': num == 0 if den == 0 else None,
                    'max_absolute_error': float(error[t].abs().max()),
                    'normwise_pass': bool(passed[t]),
                })
            return {
                'normwise_pass': bool(passed.all()), 'per_token': per_token,
                'elementwise_diagnostic': {
                    'rtol': rtol, 'atol': atol, 'pass': bool(close.all()),
                    'failing_elements': int((~close).sum()),
                    'max_absolute_error': float(error.abs().max()),
                    'worst_absolute_coordinate': [token, dimension],
                    'actual_at_worst_coordinate': float(actual[token, dimension]),
                    'reference_at_worst_coordinate': float(expected[token, dimension]),
                },
            }

        flat, _ = self._flatten(x)
        positions = torch.linspace(0, len(flat) - 1, min(max_tokens, len(flat)),
                                   device=self.device).round().long().unique()
        sample = flat.index_select(0, positions)
        dense = self.reference_dense(sample)
        implementations = {'dense': dense}
        implementations.update({method: self(sample, method=method)
                                for method in ('pairs', 'transforms')})

        with torch.autocast(device_type=self.device.type, enabled=False):
            z, gates = self._encode(sample)
            gpu_raw = (z @ self.encoder.T + self.bias).double().cpu()
            gpu_gates = gates.double().cpu()
            native64 = sample.double().cpu()
            z64 = ((native64 - self.input_mean.double().cpu()) /
                   self.input_std.double().cpu())
            outstd64 = self.output_std.double().cpu()
            outmean64 = self.output_mean.double().cpu()
            # Recompute every gate independently from the native input, not
            # from a GPU-standardized vector or a GPU-selected active mask.
            encoder64 = self.encoder.double().cpu()
            cpu_raw = z64 @ encoder64.T + self.bias.double().cpu()
            del encoder64
            theta64 = self.theta.double().cpu()
            boundary = torch.maximum(theta64, torch.zeros_like(theta64))
            cpu_mask = (cpu_raw > theta64) & (cpu_raw > 0)
            gpu_mask = gpu_gates > 0
            cpu_gates = torch.where(cpu_mask, cpu_raw, 0.)
            raw_close = torch.isclose(gpu_raw, cpu_raw, rtol=rtol, atol=atol)
            raw_error = (gpu_raw - cpu_raw).abs()
            margin = (cpu_raw - boundary).abs()
            encoder_vector = vector_check(gpu_raw, cpu_raw, width=self.num_features)
            scalar_failures = []
            for token, feature in (~raw_close).nonzero(as_tuple=False).tolist():
                scalar_failures.append({
                    'sample_token': token, 'source_token': int(positions[token]),
                    'transform': feature,
                    'raw_cpu_float64': float(cpu_raw[token, feature]),
                    'raw_gpu_float32': float(gpu_raw[token, feature]),
                    'threshold': float(theta64[feature]),
                    'effective_boundary': float(boundary[feature]),
                    'cpu_margin_to_boundary': float(margin[token, feature]),
                    'observed_raw_absolute_error': float(raw_error[token, feature]),
                    'scalar_tolerance': float(atol + rtol * cpu_raw[token, feature].abs()),
                    'cpu_active': bool(cpu_mask[token, feature]),
                    'gpu_active': bool(gpu_mask[token, feature]),
                })
            gate_mismatches = []
            for token, feature in (cpu_mask != gpu_mask).nonzero(as_tuple=False).tolist():
                gate_mismatches.append({
                    'sample_token': token, 'transform': feature,
                    'raw_cpu_float64': float(cpu_raw[token, feature]),
                    'raw_gpu_float32': float(gpu_raw[token, feature]),
                    'threshold': float(theta64[feature]),
                    'effective_boundary': float(boundary[feature]),
                    'cpu_margin_to_boundary': float(margin[token, feature]),
                    'observed_raw_absolute_error': float(raw_error[token, feature]),
                    'cpu_active': bool(cpu_mask[token, feature]),
                    'gpu_active': bool(gpu_mask[token, feature]),
                    'within_scalar_tolerance': bool(raw_close[token, feature]),
                })
            # Report closest boundaries even when masks agree, so a passing
            # audit does not conceal features poised to cross the discontinuity.
            closest = margin.reshape(-1).argsort()[:min(12, margin.numel())]
            near_boundaries = []
            for flat_index in closest.tolist():
                token, feature = divmod(flat_index, self.num_features)
                near_boundaries.append({
                    'sample_token': token, 'transform': feature,
                    'cpu_margin_to_boundary': float(margin[token, feature]),
                    'raw_cpu_float64': float(cpu_raw[token, feature]),
                    'raw_gpu_float32': float(gpu_raw[token, feature]),
                    'effective_boundary': float(boundary[feature]),
                    'observed_raw_absolute_error': float(raw_error[token, feature]),
                    'cpu_active': bool(cpu_mask[token, feature]),
                    'gpu_active': bool(gpu_mask[token, feature]),
                })

            factors64 = {}

            def cpu_factors(feature):
                if feature not in factors64:
                    group = next(g for g, (lo, hi, _) in enumerate(self.groups) if lo <= feature < hi)
                    local = feature - self.groups[group][0]
                    factors64[feature] = (self.Vs[group][local].double().cpu(),
                                          self.Us[group][local].double().cpu())
                return factors64[feature]

            cpu_sum = torch.zeros_like(z64)
            shared_gate_sum = torch.zeros_like(z64)
            # Union is deliberate: differences in either active mask must be
            # measured rather than silently restricting the CPU reference.
            for token, feature in (cpu_mask | gpu_mask).nonzero(as_tuple=False).tolist():
                V64, U64 = cpu_factors(feature)
                low64 = (z64[token] @ V64) @ U64
                cpu_sum[token] += cpu_gates[token, feature] * low64
                shared_gate_sum[token] += gpu_gates[token, feature] * low64
            cpu_hat = cpu_sum * outstd64 + outmean64
            shared_gate_hat = shared_gate_sum * outstd64 + outmean64

            comparisons = {}
            for name, actual in implementations.items():
                comparisons[name] = {
                    'raw_vs_full_cpu_float64': vector_check(actual[0], cpu_hat),
                    'standardized_vs_full_cpu_float64': vector_check(actual[1], cpu_sum),
                    # Secondary diagnostic isolates decoder arithmetic from
                    # GPU encoder/mask differences. It cannot override a fail
                    # against the fully independent reference above.
                    'raw_vs_cpu_float64_with_gpu_gates': vector_check(actual[0], shared_gate_hat),
                    'standardized_vs_cpu_float64_with_gpu_gates': vector_check(actual[1], shared_gate_sum),
                    'gpu_gates_exactly_match_dense': bool(torch.equal(actual[2], dense[2])),
                }
                if name != 'dense':
                    comparisons[name]['raw_vs_dense'] = vector_check(actual[0], dense[0])
                    comparisons[name]['standardized_vs_dense'] = vector_check(actual[1], dense[1])

            audits = []
            for group, (lo, hi, _rank) in enumerate(self.groups):
                active = gpu_mask[:, lo:hi].nonzero(as_tuple=False)
                token, local_feature = active[0].tolist() if len(active) else (0, 0)
                feature = lo + local_feature
                V64, U64 = cpu_factors(feature)
                low64 = (z64[token] @ V64) @ U64
                U, V = self.Us[group][local_feature], self.Vs[group][local_feature]
                low32 = (z[token] @ V) @ U
                expected_contribution = cpu_gates[token, feature] * low64 * outstd64
                actual_contribution = gates[token, feature] * low32 * self.output_std
                audits.append({
                    'transform': feature, 'sample_token': token,
                    'active': bool(gpu_mask[token, feature]),
                    'cpu_active': bool(cpu_mask[token, feature]),
                    'raw_cpu_float64': float(cpu_raw[token, feature]),
                    'raw_gpu_float32': float(gpu_raw[token, feature]),
                    'encoder_scalar_pass': bool(raw_close[token, feature]),
                    'threshold': float(theta64[feature]),
                    'cpu_margin_to_boundary': float(margin[token, feature]),
                    'threshold_rounding_ambiguous': bool(cpu_mask[token, feature] != gpu_mask[token, feature]),
                    'ungated_uv': vector_check(low32, low64),
                    'raw_contribution': vector_check(actual_contribution, expected_contribution),
                })
            zero_checks = {}
            for method in ('pairs', 'transforms'):
                zeros = self._sum_sparse(z, torch.zeros_like(gates), method)
                zero_checks[method] = bool(torch.equal(zeros, torch.zeros_like(z)) and
                    torch.equal(self._decode(zeros), self.output_mean.expand_as(zeros)))

        passed = (encoder_vector['normwise_pass'] and not gate_mismatches and all(zero_checks.values()) and
                  all(c['gpu_gates_exactly_match_dense'] for c in comparisons.values()) and
                  all(v['normwise_pass'] for c in comparisons.values() for v in c.values()
                      if isinstance(v, dict) and 'normwise_pass' in v) and
                  all(a['ungated_uv']['normwise_pass'] and
                      a['raw_contribution']['normwise_pass'] for a in audits if a['active']))
        report = {
            'validation_passed': passed, 'layer': self.layer,
            'criteria_revision': 'cpu-full-active-sum-normwise-v3',
            'criteria_rationale': (
                'FP32 matrix products and reduction orders can differ at near-zero coordinates due to '
                'cancellation. Accept only per-token relative L2 error <= norm_rtol against an independent '
                'CPU float64 full reconstruction in both raw and standardized coordinates; zero reference '
                'vectors require exact zero error. Apply the same per-token relative L2 criterion to the '
                'complete encoder preactivation vector across all features, retaining every scalar failure '
                'and its gate-boundary margin as diagnostics. All CPU/GPU gate masks must agree exactly. '
                'Individual UV and contribution checks are required for active audited features; arbitrary '
                'inactive audit features are diagnostic only. A shared-GPU-gate CPU reference is diagnostic '
                'only and cannot override independent-reference failure.'),
            'sample_positions': positions.cpu().tolist(), 'dense_reference_tokens': len(sample),
            'norm_rtol': norm_rtol, 'elementwise_diagnostic_rtol': rtol,
            'elementwise_diagnostic_atol': atol, 'encoder_scalar_rtol': rtol,
            'encoder_scalar_atol': atol,
            'full_cpu_float64_active_counts': cpu_mask.sum(-1).tolist(),
            'gpu_active_counts': gpu_mask.sum(-1).tolist(),
            'encoder_scalar_all_pass': bool(raw_close.all()),
            'encoder_scalar_failing_cells': int((~raw_close).sum()),
            'encoder_scalar_failure_details': scalar_failures,
            'encoder_preactivation_vs_cpu_float64': encoder_vector,
            'encoder_max_absolute_error': float(raw_error.max()),
            'all_cpu_gate_masks_agree': not gate_mismatches,
            'gate_mask_mismatches': gate_mismatches, 'closest_gate_boundaries': near_boundaries,
            'full_reference_comparisons': comparisons, 'cpu_float64_contributions': audits,
            'zero_active_reconstruction_equals_output_mean': all(zero_checks.values()),
            'zero_active_checks_by_method': zero_checks,
            'sidecar_config_validated': self.config_path is not None,
            'pair_weight_budget_bytes': self.max_pair_weight_bytes,
            'pair_chunk_sizes_by_rank': self.pair_chunk_sizes,
        }
        if not passed:
            error = AssertionError('MOLT reconstruction validation failed: ' +
                                   json.dumps(report, sort_keys=True, allow_nan=False))
            error.report = report
            raise error
        return report
