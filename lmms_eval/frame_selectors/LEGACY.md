# lmms-eval frame selectors — legacy status

**As of 2026-04-23**, all logic in this directory is **legacy / HF-path only**.
The active implementations now live in the vLLM submodule so training
and serving share exactly one code path.

| Legacy file (here) | Replacement (vLLM-side) | Notes |
|---|---|---|
| `patch_selector.py` | `third_party/vllm/vllm/multimodal/frame_selectors/codec_patch_mask.py` | Per-patch MV+residual scoring (OneVision-Encoder arXiv 2602.08683). Ported verbatim; edit the vLLM copy first. |
| `ipb_selector.py` | `third_party/vllm/vllm/multimodal/frame_selectors/ipb_selector*.py` | Per-frame IPB selection. The vLLM variants (`ipb_selector_vXX.py`) cover v2–v26 and beyond. |

## When each path is live

- **vLLM path** (primary): `scripts/modified_vllm/eval_qwen3vl_backend_budget.sh`
  launches the OpenAI-compatible vLLM server with `ipb_selection_opencv` /
  `ipb_selection_torchcodec` backends. Codec mask is enabled via
  `--codec_mask_mode token`. No code in this directory runs.
- **HF path** (legacy, still supported for ablations): `--model qwen3_vl`
  simple model reads `LMMS_CODEC_PATCHIFY` and calls `patch_selector.py`
  directly. Do **not** also enable server-side codec mask in this mode —
  the two will double-mask.

## What not to do

- **Do not add new codec features here.** Put them in the vLLM-side module.
- **Do not call the legacy functions from new vLLM-path code.** If you find
  a call site in a selector `ipb_selector_vXX.py`, file a follow-up to
  migrate rather than import across the submodule boundary.
- **Do not delete this directory** until the HF path (qwen3_vl simple) is
  retired; downstream experiments still depend on it.

## Import-time deprecation warnings

`patch_selector.py` emits `DeprecationWarning` on import. In a clean
run, this is a quiet hint that you're on the legacy path. Under
`-Werror::DeprecationWarning` it becomes a hard error — useful for CI
that should no longer touch this tree.
