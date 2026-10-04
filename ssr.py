"""SSR-Merge (Subspace Signal Routing) statistics and router solve.

Wei et al., "SSR-Merge: Subspace Signal Routing for Training-Free LoRA Merging in
Diffusion Models", ICML 2026 — https://arxiv.org/abs/2606.10617

Per layer, with K LoRAs (A_k, B_k):

    A_comb = [A_1; ...; A_K]            B_comb = [B_1 ... B_K]
    Z_k    = A_comb X_k                 (X_k: layer inputs while LoRA k is active)
    G      = sum_k Z_k Z_k^T            Q = sum_k E_k (A_k X_k) Z_k^T
    R      = Q (G/N + lambda I)^-1      B_merged = B_comb R

G and Q (two R x R float64 matrices per layer, R = sum of ranks) live in system RAM.
They also give the exact reconstruction error of any router, used for the log line.
"""

import logging

import torch

import comfy.model_management
import comfy.utils
from comfy.patcher_extension import PatcherInjection


class LayerTap:
    """Forward hook on one Linear: applies the active LoRA and accumulates its SSR statistics."""

    def __init__(self, merger, loras, As, Bs):
        self.merger = merger
        self.local = {k: j for j, k in enumerate(loras)}
        self.A = torch.cat(As, dim=0)  # (R, in)
        self.Bs = Bs  # [(out, r_k)]
        self.slices, start = [], 0
        for a in As:
            self.slices.append(slice(start, start + a.shape[0]))
            start += a.shape[0]
        self.G = torch.zeros(start, start, dtype=torch.float64)
        self.Q = torch.zeros(start, start, dtype=torch.float64)
        self.count = 0

    def __call__(self, module, args, y):
        j = self.local.get(self.merger.active)
        if j is None:
            return None
        sl = self.slices[j]
        x = args[0]
        z = x.reshape(-1, x.shape[-1]).float() @ self.A.to(x.device, torch.float32).T  # (N, R)
        self.G += (z.T @ z).cpu().double()
        self.Q[sl] += (z[:, sl].T @ z).cpu().double()
        self.count += z.shape[0]
        delta = z[:, sl] @ self.Bs[j].to(x.device, torch.float32).T
        return y + delta.reshape(y.shape).to(y.dtype)

    def solve(self, lambda_reg, device):
        """Returns (A_merged, B_merged) and, for layers shared by several LoRAs, the reconstruction errors."""
        B = torch.cat(self.Bs, dim=1)
        if len(self.slices) < 2 or self.count == 0:
            return (self.A, B), None
        eye = torch.eye(self.G.shape[0], dtype=torch.float64)
        G, Q = self.G / self.count, self.Q / self.count
        R = torch.linalg.solve(G + lambda_reg * eye, Q.T).T  # Q G^-1, G symmetric
        Bd = B.to(device, torch.float32)
        B_merged = (Bd @ R.to(device, torch.float32)).to("cpu", B.dtype)

        # sum_k ||B_comb R Z_k - B_k A_k X_k||^2 expands to terms in G, Q and the diagonal blocks of Q.
        BtB = (Bd.T @ Bd).cpu().double()
        ref = sum(torch.sum(BtB[sl, sl] * Q[sl, sl]).item() for sl in self.slices)
        errors = {}
        for name, router in (("ssr", R), ("average", eye / len(self.slices)), ("sum", eye)):
            errors[name] = torch.sum(BtB * (router @ G @ router.T)).item() - 2 * torch.sum(BtB * (router @ Q.T)).item() + ref
        return (self.A, B_merged), (errors, ref)


class SSRMerger:
    def __init__(self, loras):
        """loras: one {weight key: (A, B)} dict per LoRA, with alpha and strength folded into B."""
        self.active = None  # index of the LoRA applied during calibration
        self.taps = {}
        for key in sorted({k for lora in loras for k in lora}):
            ks = [k for k, lora in enumerate(loras) if key in lora]
            self.taps[key] = LayerTap(self, ks, [loras[k][key][0] for k in ks], [loras[k][key][1] for k in ks])

    def injections(self):
        handles = []

        def inject(patcher):
            for key, tap in self.taps.items():
                module = comfy.utils.get_attr(patcher.model, key[:-len(".weight")])
                handles.append(module.register_forward_hook(tap))

        def eject(patcher):
            for h in handles:
                h.remove()
            handles.clear()

        return [PatcherInjection(inject=inject, eject=eject)]

    def solve(self, lambda_reg):
        """Returns {weight key: (A_merged, B_merged)}."""
        device = comfy.model_management.get_torch_device()
        merged, undersampled = {}, 0
        err_total, ref_total = {}, 0.0
        for key, tap in self.taps.items():
            merged[key], stats = tap.solve(lambda_reg, device)
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
        return merged
