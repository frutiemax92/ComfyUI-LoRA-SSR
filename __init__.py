import torch

import comfy.lora
import comfy.lora_convert
import comfy.model_management
import comfy.sample
import comfy.utils
import folder_paths
from comfy.weight_adapter import WeightAdapterBase
from comfy.weight_adapter.lokr import LoKrAdapter
from comfy.weight_adapter.lora import LoRAAdapter
from comfy_api.latest import ComfyExtension, io

from .ssr import LoKRFactor, LoRAFactor, SSRMerger

WEB_DIRECTORY = "./js"


def _is_valid_linear_key(key):
    if isinstance(key, str):
        return True
    if isinstance(key, tuple) and len(key) == 2 and isinstance(key[0], str):
        offset = key[1]
        return isinstance(offset, tuple) and len(offset) == 3 and offset[0] in (0, 1)
    return False


def load_lora_factors(model, lora_name, strength):
    """Returns ({weight key: LoRAFactor | LoKRFactor} with alpha and strength folded in, other patches of the file)."""
    path = folder_paths.get_full_path_or_raise("loras", lora_name)
    sd = comfy.lora_convert.convert_lora(comfy.utils.load_torch_file(path, safe_load=True))
    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    factors, other = {}, {}
    for key, patch in comfy.lora.load_lora(sd, key_map).items():
        if not isinstance(patch, WeightAdapterBase):
            other[key] = patch
            continue
        if not _is_valid_linear_key(key):
            raise ValueError(f"SSR merging supports Linear weight keys (and slice offsets) only; {lora_name} has a {patch.name} patch on {key}.")
        if isinstance(patch, LoRAAdapter):
            up, down, alpha, mid, dora_scale, reshape = patch.weights
            if mid is not None or dora_scale is not None or reshape is not None or down.ndim != 2:
                raise ValueError(f"SSR merging supports plain Linear LoRA layers only; {lora_name} has a LoCon/DoRA layer on {key}.")
            scale = strength * (alpha / down.shape[0] if alpha is not None else 1.0)
            factors[key] = LoRAFactor(down, up * scale, patch, strength)
        elif isinstance(patch, LoKrAdapter):
            w1, w2, alpha, w1_a, w1_b, w2_a, w2_b, t2, dora_scale = patch.weights
            if t2 is not None or dora_scale is not None:
                raise ValueError(f"SSR merging supports plain Linear LoKR layers only; {lora_name} has a Tucker/DoRA LoKR layer on {key}.")
            if any(t is not None and t.ndim != 2 for t in (w1, w2, w1_a, w1_b, w2_a, w2_b)):
                raise ValueError(f"SSR merging supports Linear LoKR layers only; {lora_name} has a Conv LoKR layer on {key}.")
            dim = None
            if w1 is None:
                dim = w1_b.shape[0]
                w1 = (w1_a.float() @ w1_b.float()).to(w1_a.dtype)
            if w2 is None:
                dim = w2_b.shape[0]
                w2 = (w2_a.float() @ w2_b.float()).to(w2_a.dtype)
            scale = strength * (alpha / dim if alpha is not None and dim is not None else 1.0)
            factors[key] = LoKRFactor(w1, w2, scale, patch, strength, w2.dtype)
        else:
            raise ValueError(f"SSR merging supports LoRA and LoKR layers only; {lora_name} has a {patch.name} patch on {key}.")
    return factors, other


class SSRMergeLoRAs(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SSRMergeLoRAs",
            display_name="SSR - Merge LoRAs",
            category="loaders",
            description="Merges LoRA and LoKR adapters into the model with SSR-Merge (Subspace Signal Routing). Each adapter needs calibration conditioning typical of how it is used, including its trigger words.",
            inputs=[
                io.Model.Input("model", tooltip="The unpatched model."),
                io.Float.Input("strength", default=1.0, min=-10.0, max=10.0, step=0.01, tooltip="Strength of the merged adapter."),
                io.Float.Input("lambda_reg", default=1e-4, min=0.0, max=1.0, step=1e-5, round=1e-6, advanced=True, tooltip="Ridge term on the SSR correlation matrix."),
                io.Int.Input("lokr_rank", default=64, min=8, max=512, step=8, advanced=True, tooltip="Maximum subspace rank extracted per LoKR layer via Kronecker SVD."),
                io.Boolean.Input("keep_lokr_residual", default=False, advanced=True, tooltip="Keep full-rank LoKR adapters intact and solve a routed LoRA correction to cancel interference. Best for LoKR + LoRA merges."),
                io.Int.Input("calibration_steps", default=1, min=1, max=100, advanced=True, tooltip="Sampler steps per calibration pass; 1 is a single pass on pure noise, as in the paper."),
                io.Int.Input("width", default=1024, min=16, max=16384, step=16, advanced=True),
                io.Int.Input("height", default=1024, min=16, max=16384, step=16, advanced=True),
                io.Int.Input("seed", default=42, min=0, max=0xffffffffffffffff, control_after_generate=False, advanced=True),
            ],
            outputs=[io.Model.Output()],
            # lora_<n> ({"lora", "strength"}) and conditioning_<n> inputs are created by js/ssr_merge_loras.js
            accept_all_inputs=True,
        )

    @classmethod
    def execute(cls, model, strength, lambda_reg, lokr_rank, keep_lokr_residual, calibration_steps, width, height, seed, **kwargs) -> io.NodeOutput:
        rows = []
        for name, value in kwargs.items():
            if name.startswith("lora_") and isinstance(value, dict) and value.get("lora"):
                n = name[len("lora_"):]
                rows.append((int(n), value["lora"], float(value["strength"]), kwargs.get(f"conditioning_{n}")))
        rows.sort()
        if len(rows) < 2:
            raise ValueError("SSR merging needs at least 2 LoRAs.")
        for _, lora_name, _, conditioning in rows:
            if not conditioning:
                raise ValueError(f"Connect calibration conditioning for {lora_name}.")

        loaded = [load_lora_factors(model, lora_name, lora_strength) for _, lora_name, lora_strength, _ in rows]
        merger = SSRMerger([factors for factors, _ in loaded], lokr_rank=lokr_rank, keep_lokr_residual=keep_lokr_residual)

        calib = model.clone()
        calib.set_injections("ssr_merge", merger.injections())
        latent = torch.zeros([1, 4, height // 8, width // 8], device=comfy.model_management.intermediate_device(), dtype=comfy.model_management.intermediate_dtype())
        latent = comfy.sample.fix_empty_latent_channels(calib, latent, 8)
        noise = comfy.sample.prepare_noise(latent, seed)
        pbar = comfy.utils.ProgressBar(sum(len(row[3]) for row in rows) * calibration_steps)
        try:
            for k, (_, _, _, conditioning) in enumerate(rows):
                merger.active = k
                for cond in conditioning:
                    comfy.sample.sample(calib, noise, calibration_steps, 1.0, "euler", "simple", [cond], [cond], latent,
                                        callback=lambda *args: pbar.update(1), disable_pbar=True, seed=seed)
        finally:
            merger.active = None
            calib.eject_model()

        merged, passthrough = merger.solve(lambda_reg)
        out = model.clone()
        if merged:
            out.add_patches(merged, strength)
        by_adapter = {}
        for idx, key, patch, patch_strength in passthrough:
            by_adapter.setdefault((idx, patch_strength), {})[key] = patch
        for (_, patch_strength), patches in by_adapter.items():
            out.add_patches(patches, strength * patch_strength)
        for (_, _, lora_strength, _), (_, other) in zip(rows, loaded):
            if other:
                out.add_patches(other, strength * lora_strength)
        return io.NodeOutput(out)


class SSRExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [SSRMergeLoRAs]


async def comfy_entrypoint() -> SSRExtension:
    return SSRExtension()
