"""SSR-Merge (Subspace Signal Routing) statistics and router solve.

Wei et al., "SSR-Merge: Subspace Signal Routing for Training-Free LoRA Merging in
Diffusion Models", ICML 2026 — https://arxiv.org/abs/2606.10617

Per shared layer, with K linear adapters (LoRA or LoKR) and unified input basis A_comb (R x in):

    Z_k    = A_comb X_k                 (X_k: layer inputs while adapter k is active)
    T_k    = Delta_k(X_k) - Delta_kept(X_k)
    G      = sum_k Z_k Z_k^T            P = sum_k T_k Z_k^T
    B_out  = P (G/N + lambda I)^-1

For plain LoRA merges without kept residuals, P = B_comb Q so B_out = B_comb Q (G/N + lambda I)^-1,
which is mathematically identical to the paper's router R. Generalizing to P allows exact calibration
and output-space reconstruction for full-rank or factorized LoKR adapters (and mixed LoRA + LoKR).
"""

import logging

import torch

import comfy.model_management
import comfy.utils
from comfy.patcher_extension import PatcherInjection
from comfy.weight_adapter.lora import LoRAAdapter


class LoRAFactor:
    is_lokr = False

    def __init__(self, down, up, patch, strength):
        self.down = down  # (r, in)
        self.up = up      # (out, r), alpha and strength already folded in
        self.patch = patch
        self.strength = strength
        self.out_dim = up.shape[0]
        self.dtype = up.dtype

    def forward(self, x2d):
        z = x2d @ self.down.to(x2d.device, torch.float32).T
        return z @ self.up.to(x2d.device, torch.float32).T

    def input_basis(self, max_rank):
        return self.down.float()

    def passthrough_entries(self, key):
        return [(key, self.patch, self.strength)]


class LoKRFactor:
    is_lokr = True

    def __init__(self, w1, w2, scale, patch, strength, dtype):
        self.w1 = w1          # (p1, q1)
        self.w2 = w2          # (p2, q2)
        self.scale = scale    # strength * (alpha / dim)
        self.patch = patch
        self.strength = strength
        self.out_dim = w1.shape[0] * w2.shape[0]
        self.dtype = dtype

    def forward(self, x2d):
        w1 = self.w1.to(x2d.device, torch.float32)
        w2 = self.w2.to(x2d.device, torch.float32)
        # (w1 ⊗ w2) x without materializing the full (p1*p2, q1*q2) matrix
        hb = x2d.reshape(-1, w1.shape[1], w2.shape[1]) @ w2.T   # (N, q1, p2)
        hc = (hb.transpose(1, 2) @ w1.T).transpose(1, 2)        # (N, p1, p2)
        return hc.reshape(-1, self.out_dim) * self.scale

    def input_basis(self, max_rank):
        # Kronecker SVD: right singular vectors of (w1 ⊗ w2) are (v1_i ⊗ v2_j) with sigma = s1_i * s2_j
        _, s1, vh1 = torch.linalg.svd(self.w1.float(), full_matrices=False)
        _, s2, vh2 = torch.linalg.svd(self.w2.float(), full_matrices=False)
        sigmas = (s1[:, None] * s2[None, :]).reshape(-1)
        r = min(max_rank, sigmas.numel())
        top = torch.topk(sigmas, r).indices
        i1 = top // s2.shape[0]
        i2 = top % s2.shape[0]
        return (vh1[i1, :, None] * vh2[i2, None, :]).reshape(r, -1)

    def passthrough_entries(self, key):
        return [(key, self.patch, self.strength)]


class SlicedFactor:
    """Combines dim-0 output slices of one adapter on a fused Linear when another adapter targets the full weight."""

    def __init__(self, slices, out_dim):
        self.slices = slices  # [(slice_key, start, size, factor)]
        self.out_dim = out_dim
        self.dtype = slices[0][3].dtype
        self.is_lokr = all(f.is_lokr for _, _, _, f in slices)

    def forward(self, x2d):
        out = torch.zeros(x2d.shape[0], self.out_dim, device=x2d.device, dtype=torch.float32)
        for _, start, size, factor in self.slices:
            out[:, start:start + size] += factor.forward(x2d)
        return out

    def input_basis(self, max_rank):
        return torch.cat([factor.input_basis(max_rank) for _, _, _, factor in self.slices], dim=0)

    def passthrough_entries(self, key):
        return [(slice_key, factor.patch, factor.strength) for slice_key, _, _, factor in self.slices]


def _unify_fused_and_sliced(adapters_list):
    """If one adapter targets a fused weight key and another targets dim-0 slices of it, combine the slices."""
    bare_out_dim = {}
    for d in adapters_list:
        for k, f in d.items():
            if isinstance(k, str):
                bare_out_dim[k] = f.out_dim
    if not bare_out_dim:
        return adapters_list

    unified = []
    for d in adapters_list:
        grouped = {}
        out = {}
        for k, f in d.items():
            if isinstance(k, tuple) and k[0] in bare_out_dim and k[1][0] == 0:
                grouped.setdefault(k[0], []).append((k, k[1][1], k[1][2], f))
            else:
                out[k] = f
        for weight_key, slices in grouped.items():
            slices.sort(key=lambda s: s[1])
            out[weight_key] = SlicedFactor(slices, bare_out_dim[weight_key])
        unified.append(out)
    return unified


class LayerTap:
    """Forward hook on one Linear (or a slice of a fused Linear): applies the active adapter and accumulates SSR statistics."""

    def __init__(self, merger, key, adapter_indices, adapters, lokr_rank, keep_lokr_residual):
        self.merger = merger
        self.key = key
        self.offset = key[1] if isinstance(key, tuple) else None
        self.local = {k: j for j, k in enumerate(adapter_indices)}
        self.adapter_indices = adapter_indices
        self.adapters = adapters
        self.shared = len(adapters) >= 2
        self.kept_indices = [j for j, a in enumerate(adapters) if a.is_lokr] if (self.shared and keep_lokr_residual) else []
        self.dtype = adapters[0].dtype
        self.count = 0

        if self.shared:
            basis = torch.cat([a.input_basis(lokr_rank) for a in adapters], dim=0)  # (R, in)
            # Normalize rows so ridge lambda_reg acts uniformly across LoRA (Kaiming) and LoKR (SVD unit-norm) blocks
            norms = basis.norm(dim=1, keepdim=True).clamp_min(1e-12)
            self.A = basis / norms
            rank = self.A.shape[0]
            self.G = torch.zeros(rank, rank, dtype=torch.float64)
            self.P = torch.zeros(adapters[0].out_dim, rank, dtype=torch.float32)
            self.ref = 0.0
            self.target_ref = 0.0
            self.err_sum = 0.0
            self.err_avg = 0.0

    def _apply_delta(self, y, delta):
        if self.offset is not None and self.offset[0] == 0:
            _, start, size = self.offset
            y_out = y.clone()
            y_out[..., start:start + size] += delta.reshape(*y.shape[:-1], size).to(y.dtype)
            return y_out
        return y + delta.reshape(y.shape).to(y.dtype)

    def __call__(self, module, args, y):
        j = self.local.get(self.merger.active)
        if j is None:
            return None
        x = args[0]
        if self.offset is not None and self.offset[0] == 1:
            _, start, size = self.offset
            x2d = x[..., start:start + size].reshape(-1, size).float()
        else:
            x2d = x.reshape(-1, x.shape[-1]).float()

        if not self.shared:
            delta = self.adapters[j].forward(x2d)
            return self._apply_delta(y, delta)

        deltas = [a.forward(x2d) for a in self.adapters]
        delta = deltas[j]
        if self.kept_indices:
            kept_delta = sum(deltas[m] for m in self.kept_indices)
            target = delta - kept_delta
        else:
            target = delta

        z = x2d @ self.A.to(x.device, torch.float32).T  # (N, R)
        # float32 rounding here makes G indefinite for large merged ranks, which the small ridge cannot absorb
        zd = z.to(self.merger.fp64_device, torch.float64)
        self.G += (zd.T @ zd).cpu()
        self.P += (target.T @ z).cpu()
        self.count += z.shape[0]

        self.ref += (delta * delta).sum(dtype=torch.float64).item()
        self.target_ref += (target * target).sum(dtype=torch.float64).item()
        sum_delta = sum(deltas)
        diff_sum = sum_delta - delta
        diff_avg = sum_delta / len(deltas) - delta
        self.err_sum += (diff_sum * diff_sum).sum(dtype=torch.float64).item()
        self.err_avg += (diff_avg * diff_avg).sum(dtype=torch.float64).item()

        return self._apply_delta(y, delta)

    def solve(self, lambda_reg, device):
        """Returns (merged_lora_patch_or_None, kept_or_passthrough_entries, stats_or_None)."""
        if not self.shared or self.count == 0:
            passthrough = []
            for j, a in enumerate(self.adapters):
                for entry_key, patch, strength in a.passthrough_entries(self.key):
                    passthrough.append((self.adapter_indices[j], entry_key, patch, strength))
            return None, passthrough, None

        G_d = (self.G / self.count).to(device)
        P = self.P.to(device, torch.float64) / self.count
        eye_d = torch.eye(G_d.shape[0], device=device, dtype=torch.float64)
        B_d = torch.linalg.solve(G_d + lambda_reg * eye_d, P.T).T  # P (G + lambda I)^-1

        err_ssr = (torch.sum((B_d @ G_d) * B_d) - 2.0 * torch.sum(B_d * P)).item() + self.target_ref / self.count
        ref = self.ref / self.count
        errors = {
            "ssr": err_ssr,
            "average": self.err_avg / self.count,
            "sum": self.err_sum / self.count,
        }

        A_out = self.A.to("cpu", self.dtype)
        B_out = B_d.to("cpu", self.dtype)
        merged_patch = LoRAAdapter(set(), (B_out, A_out, None, None, None, None))
        kept = []
        for m in self.kept_indices:
            for entry_key, patch, strength in self.adapters[m].passthrough_entries(self.key):
                kept.append((self.adapter_indices[m], entry_key, patch, strength))
        return merged_patch, kept, (errors, ref)


class SSRMerger:
    def __init__(self, adapters_list, lokr_rank=64, keep_lokr_residual=False):
        """adapters_list: one {weight key: LoRAFactor | LoKRFactor} dict per adapter."""
        self.active = None  # index of the adapter applied during calibration
        device = comfy.model_management.get_torch_device()
        self.fp64_device = device if comfy.model_management.supports_fp64(device) else torch.device("cpu")
        adapters_list = _unify_fused_and_sliced(adapters_list)
        self.taps = {}
        all_keys = sorted(
            {k for adapters in adapters_list for k in adapters},
            key=lambda k: (k[0], k[1]) if isinstance(k, tuple) else (k, ()),
        )
        for key in all_keys:
            ks = [k for k, adapters in enumerate(adapters_list) if key in adapters]
            self.taps[key] = LayerTap(
                self,
                key,
                ks,
                [adapters_list[k][key] for k in ks],
                lokr_rank=lokr_rank,
                keep_lokr_residual=keep_lokr_residual,
            )

    def injections(self):
        handles = []

        def inject(patcher):
            for key, tap in self.taps.items():
                weight_key = key[0] if isinstance(key, tuple) else key
                module = comfy.utils.get_attr(patcher.model, weight_key[:-len(".weight")])
                handles.append(module.register_forward_hook(tap))

        def eject(patcher):
            for h in handles:
                h.remove()
            handles.clear()

        return [PatcherInjection(inject=inject, eject=eject)]

    def solve(self, lambda_reg):
        """Returns (merged_loras: {key: LoRAAdapter}, passthrough: [(adapter_idx, key, patch, strength)])."""
        merged, passthrough, undersampled = {}, [], 0
        err_total, ref_total = {}, 0.0
        for key, tap in self.taps.items():
            merged_patch, kept, stats = tap.solve(lambda_reg, self.fp64_device)
            if merged_patch is not None:
                merged[key] = merged_patch
            passthrough.extend(kept)
            if stats is None:
                continue
            errors, ref = stats
            for name, e in errors.items():
                err_total[name] = err_total.get(name, 0.0) + e
            ref_total += ref
            if tap.count < tap.A.shape[0]:
                undersampled += 1
        if err_total:
            logging.info("SSR merge relative reconstruction error: " + ", ".join(f"{n}={e / max(ref_total, 1e-30):.4f}" for n, e in err_total.items()))
        if undersampled:
            logging.warning(f"SSR merge: {undersampled} layers saw fewer calibration tokens than the merged rank; add or lengthen calibration prompts.")
        return merged, passthrough
