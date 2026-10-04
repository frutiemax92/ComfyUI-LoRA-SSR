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
- **lambda_reg**, **calibration_steps**, **width**, **height**, **seed**: calibration
  settings. The defaults are the ones from `merge.py` in LoRAMergingTool.

For each prompt, the node runs one calibration pass with that LoRA active and records what
each LoRA layer sees. A progress bar shows these passes. It then solves a router for each
layer. ComfyUI caches the merged model, so the merge only runs again when an input changes.

The log shows the reconstruction error of the SSR router next to plain averaging and plain
summing, which lets you check that routing helps.

Only plain LoRA layers on Linear weights are supported. LoCon, DoRA, LoHa and LoKr fail with
an error.

## Memory

The base model is not quantized. Calibration uses ComfyUI's normal model loading and
offloading, so a 12 GB GPU runs Krea 2 fp8 at 1024×1024. The LoRA factors and the per-layer
statistics (R×R float64 matrices, where R is the sum of the LoRA ranks) stay in system RAM.
