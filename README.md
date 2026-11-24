# ComfyUI-LoRA-SSR

A ComfyUI node that merges several LoRAs into a model with
[SSR-Merge](https://arxiv.org/abs/2606.10617) (Subspace Signal Routing, ICML 2026).
Ported from [LoRAMergingTool](https://github.com/frutiemax92/LoRAMergingTool); tested with Krea 2.

## SSR - Merge LoRAs

- **model**: the unpatched model. The output is that model with the merged LoRA applied.
- **LoRA rows**: use **+ Add LoRA** to add a row and **✕** to remove one. Click the name to pick a
  LoRA. Drag the bar to set its strength (0 to 2), or double-click the bar to type a value.
  At least two LoRAs are needed.
- **calibration conditioning**: under each LoRA, connect the prompts that are typical for that
  LoRA, including its trigger words. Use *Conditioning (Combine)* to pass several prompts.
  SSR tells the LoRAs apart by what each one sees during calibration. If every LoRA gets the
  same prompt, the result is a plain average.
- **strength**: strength of the merged LoRA.
- **lambda_reg**, **lokr_rank**, **keep_lokr_residual**, **calibration_steps**, **width**, **height**, **seed**:
  calibration and merge settings. `lokr_rank` caps the per-layer Kronecker-SVD subspace rank extracted from LoKR
  layers (default 64). Enabling `keep_lokr_residual` keeps full-rank LoKR adapters intact in the merged model and
  solves a routed LoRA correction to cancel interference (recommended for LoKR + LoRA merges).

For each prompt, the node runs one calibration pass with that adapter active and records what
each Linear adapter layer sees. A progress bar shows these passes. It then solves a routed merge
for each shared layer (unshared layers pass through untouched). ComfyUI caches the merged model,
so the merge only runs again when an input changes.

The log shows the reconstruction error of the SSR merge next to plain averaging and plain
summing, which lets you check that routing helps.

Linear LoRA and Linear LoKR layers (both full-matrix and factorized) are supported, including mixed
LoRA + LoKR merges. LoCon, DoRA, LoHa, and Conv/Tucker LoKR layers fail with an error.

## Memory

The base model is not quantized. Calibration uses ComfyUI's normal model loading and
offloading, so a 12 GB GPU runs Krea 2 fp8 at 1024×1024. The adapter factors and per-layer
statistics (`G` as R×R float64, `P` as out×R float32, where R is the sum of adapter subspace ranks)
stay in system RAM.
