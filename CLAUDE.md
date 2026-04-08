---
name: K-EXAONE quantization pipeline
description: 5-step pipeline to produce W4A16 AWQ quantized K-EXAONE-236B-A23B; steps 1-3 complete
type: project
---

Goal: produce a W4A16 AWQ quantized model of `LGAI-EXAONE/K-EXAONE-236B-A23B` (only FP8 and GGUF exist on HuggingFace).

**Why:** No W4A16 AWQ checkpoint exists yet; user is building one for deployment/evaluation.

**Pipeline steps:**
- Step 1: vLLM inference — COMPLETE
- Step 2: lm-eval baseline evaluation — COMPLETE
- Step 3: MoE-aware W4A16 AWQ quantization (llm-compressor) — COMPLETE (code written, not yet run)
- Step 4: lm-eval evaluation of quantized model — PENDING
- Step 5: Performance enhancement — PENDING

**Key files created (all in `/workspace/llm-compressor/`):**
- `src/llmcompressor/modeling/k_exaone_moe.py` — CalibrationExaoneMoeSparseMoEBlock (permanent MoE wrapper)
- `src/llmcompressor/modeling/__init__.py` — added import of CalibrationExaoneMoeSparseMoEBlock
- `examples/quantizing_moe/k_exaone_example.py` — AWQ W4A16 quantization script for K-EXAONE-236B-A23B
- `examples/quantization_w4a16/exaone4_example.py` — GPTQ W4A16 for dense EXAONE-4.0-32B
- `src/llmcompressor/modifiers/awq/mappings.py` — added `_exaone_moe_mappings` + `ExaoneMoeForCausalLM` registry entry

**Installed package also patched** at `/usr/local/lib/python3.12/dist-packages/llmcompressor/` (same files).

**How to apply:** Run `examples/quantizing_moe/k_exaone_example.py` with 1+ B200 GPU and ≥512 GB CPU RAM; use `device_map="cpu"` for single-GPU runs.
