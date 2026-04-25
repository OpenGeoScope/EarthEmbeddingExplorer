# AGENTS.md — EarthEmbeddingExplorer

## Scope & routing

- No sub-module AGENTS.md files exist; all agent guidance lives here.
- `MajorTOM/` and `models/FarSLIP/` are vendored third-party forks. Do not reformat, lint, or refactor them unless you are intentionally patching the fork.

## Non-discoverable environment & config behavior

- **`DOWNLOAD_ENDPOINT`** controls remote download routing (`modelscope.cn` | `modelscope.ai` | `huggingface`). `modelscope.ai` additionally sets `MODELSCOPE_DOMAIN=www.modelscope.ai` at runtime in both `models/load_config.py` and each model wrapper.
- `configs/config_local.yaml` silently takes **precedence** over `configs/config.yaml` if it exists (`load_config()` checks it first).
- **`ckpt_path` values in `configs/config.yaml` are dead placeholders** (`"Download_From_$DOWNLOAD_ENPOINT"` — note the typo). Actual checkpoint downloads are hard-coded per model in `models/*_model.py` based on `DOWNLOAD_ENDPOINT`. Editing the config `ckpt_path` alone will not change download behavior.
- **ModelScope.ai mirrors use the `VoyagerX/` namespace**, not the original HF/MS repo IDs:
  - SigLIP → `VoyagerX/ViT-SO400M-14-SigLIP-384`
  - SatCLIP → `VoyagerX/SatCLIP-ViT16-L40`
  - DINOv2 → `VoyagerX/dinov2-large`
  - FarSLIP → `VoyagerX/FarSLIP`
  These mappings are hard-coded in each model wrapper and are not derived from config.

## Landmines

- **FarSLIP must use its vendored `models/FarSLIP/open_clip` fork.** The checkpoint format and `create_model_and_transforms` signature differ from the public `open_clip_torch` package. Swapping the import will break model loading.
- **SatCLIP image search requires a 12-band numpy array** (shape `[H, W, 12]`). Plain RGB uploads are rejected. The only in-app source for valid multiband data is the "Download Image by Geolocation" button, which populates `multiband_state` passed through the Gradio event chain.
- **`data_utils.py` silently rewrites HuggingFace parquet URLs to ModelScope.** `_prepare_row_dict` replaces `https://huggingface.co/.../resolve/main/...` with `https://modelscope.cn/.../resolve/master/...`. True HuggingFace URLs require bypassing this helper.
- **`requirements.txt` duplicates `gradio`** (unpinned line + `gradio==5.49.1`). The pinned version wins, but adding further constraints can create pip resolver conflicts.
