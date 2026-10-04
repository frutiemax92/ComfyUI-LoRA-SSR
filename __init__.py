import torch

import comfy.lora
import comfy.lora_convert
import comfy.model_management
import comfy.sample
import comfy.utils
import folder_paths
from comfy.weight_adapter import WeightAdapterBase
from comfy.weight_adapter.lora import LoRAAdapter
from comfy_api.latest import ComfyExtension, io

from .ssr import SSRMerger

WEB_DIRECTORY = "./js"


def load_lora_factors(model, lora_name, strength):
    """Returns ({weight key: (A, B)} with alpha and strength folded into B, other patches of the file)."""
    path = folder_paths.get_full_path_or_raise("loras", lora_name)
    sd = comfy.lora_convert.convert_lora(comfy.utils.load_torch_file(path, safe_load=True))
    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    factors, other = {}, {}
    for key, patch in comfy.lora.load_lora(sd, key_map).items():
        if not isinstance(patch, WeightAdapterBase):
            other[key] = patch
            continue
        if not isinstance(patch, LoRAAdapter) or not isinstance(key, str):
            raise ValueError(f"SSR merging supports plain LoRA layers only; {lora_name} has a {patch.name} patch on {key}.")
        up, down, alpha, mid, dora_scale, reshape = patch.weights
        if mid is not None or dora_scale is not None or reshape is not None or down.ndim != 2:
            raise ValueError(f"SSR merging supports plain Linear LoRA layers only; {lora_name} has a LoCon/DoRA layer on {key}.")
        scale = strength * (alpha / down.shape[0] if alpha is not None else 1.0)
        factors[key] = (down, up * scale)
    return factors, other


class SSRMergeLoRAs(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SSRMergeLoRAs",
            display_name="SSR - Merge LoRAs",
            category="loaders",
            description="Merges LoRAs into the model with SSR-Merge (Subspace Signal Routing). Each LoRA needs calibration conditioning typical of how it is used, including its trigger words.",
            inputs=[
                io.Model.Input("model", tooltip="The unpatched model."),
                io.Float.Input("strength", default=1.0, min=-10.0, max=10.0, step=0.01, tooltip="Strength of the merged LoRA."),
                io.Float.Input("lambda_reg", default=1e-4, min=0.0, max=1.0, step=1e-5, round=1e-6, advanced=True, tooltip="Ridge term on the SSR correlation matrix."),
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
    def execute(cls, model, strength, lambda_reg, calibration_steps, width, height, seed, **kwargs) -> io.NodeOutput:
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
        merger = SSRMerger([factors for factors, _ in loaded])

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

        out = model.clone()
        out.add_patches({key: LoRAAdapter(set(), (B, A, None, None, None, None)) for key, (A, B) in merger.solve(lambda_reg).items()}, strength)
        for (_, _, lora_strength, _), (_, other) in zip(rows, loaded):
            out.add_patches(other, strength * lora_strength)
        return io.NodeOutput(out)


class SSRExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [SSRMergeLoRAs]


async def comfy_entrypoint() -> SSRExtension:
    return SSRExtension()
