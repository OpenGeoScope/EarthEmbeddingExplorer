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

## OLMoEarth integration notes

- **Package:** `olmoearth-pretrain-minimal` (PyPI), not the full `olmoearth_pretrain` (training deps).
- **Band reordering required:** MajorTOM `[B01..B12]` → OLMoEarth `[B02,B03,B04,B08,B05,B06,B07,B8A,B11,B12,B01,B09]`. See `models/olmoearth_model.py:REORDER_INDICES`.
- **Mask shape constraint:** `encoder(..., mask=...)` last dim must equal `num_bandsets` (3 for `sentinel2_l2a`), not 12 or 1. The wrapper collapses the 12-band mask to 3 bandsets before passing to the encoder.
- **Capabilities:** Image-only. No text or location encoding. Appears only in Image Search and Mixed Search (image modality) dropdowns.
- **Model sizes:** Nano (1.4M params, ~2GB VRAM, dim=128), Tiny (6.2M), Base (89M), Large (308M). Default in config is `nano`.
- **Checkpoint loading:** Uses `olmoearth_pretrain_minimal.load_model_from_id()` (HuggingFace). No ModelScope mirror currently configured.
- **Embedding generation:** `generate_embeddings.py --model_name olmoearth --split Core-S2RGB-249k` works; output path is controlled by `configs/config.yaml:olmoearth.embedding_path`.
