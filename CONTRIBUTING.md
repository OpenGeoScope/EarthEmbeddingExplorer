# Contributing to EarthEmbeddingExplorer

## Welcome! 🌍

Thank you for your interest in contributing to EarthEmbeddingExplorer! This is an open-source tool for **cross-modal retrieval of global satellite imagery** using natural language, images, or geographic coordinates.

We warmly welcome contributions that make the project more useful for geoscience research, education, and exploration: new embedding models, new datasets, retrieval performance improvements, UI enhancements, bug fixes, and documentation improvements.

**Quick links:** [GitHub](https://github.com/VoyagerXvoyagerx/EarthEmbeddingExplorer) · [ModelScope Demo](https://modelscope.cn/studios/Major-TOM/EarthEmbeddingExplorer/) · [HuggingFace Demo](https://huggingface.co/spaces/ML4Sustain/EarthExplorer) · [Tutorial Paper](https://arxiv.org/abs/2603.29441)

---

## Project Architecture Overview

Before you start, here is a quick map of the codebase:

```
EarthEmbeddingExplorer/
├── app.py                      # Gradio web app entry point
├── core/
│   ├── model_manager.py        # Loads all 6 models (SigLIP, FarSLIP, SatCLIP, DINOv2, Clay, OlmoEarth)
│   ├── search_engine.py        # Text / image / location / mixed search logic
│   ├── filters.py              # Post-search time & geo filters
│   └── exporters.py            # Download results as ZIP
├── ui/
│   ├── callbacks.py            # Gradio UI callbacks (map click, reset, etc.)
│   └── utils.py                # UI helpers
├── models/
│   ├── siglip_model.py         # SigLIP wrapper
│   ├── farslip_model.py        # FarSLIP wrapper
│   ├── satclip_model.py        # SatCLIP wrapper (multi-spectral)
│   ├── dinov2_model.py         # DINOv2 wrapper
│   ├── clay_model.py           # Clay v1.5 wrapper (multi-spectral)
│   ├── olmoearth_model.py      # OlmoEarth wrapper (multi-spectral)
│   └── load_config.py          # Config & remote-path resolver (hf:// / ms://)
├── data_utils.py               # Parquet HTTP-Range download, image processing
├── visualize.py                # Map plotting, gallery formatting
├── generate_embeddings.py      # CLI script to generate embedding GeoParquets
├── MajorTOM/
│   └── embedder/
│       ├── MajorTOM_Embedder.py   # Fragments images & runs model forward pass
│       └── models/                # Embedder-specific model adapters
└── configs/
    └── config.yaml             # Model checkpoints & embedding dataset paths
```

**Key design principles:**
- **Unified model interface:** Every model in `models/` exposes `encode_text()`, `encode_image()`, `encode_location()`, and `search()`.
- **Model-agnostic multi-spectral handling:** Models declare their expected Sentinel-2 bands via `self.bands` and `self.requires_multiband`. The search engine and UI callbacks use these attributes to automatically extract and reorder bands from the generic 12-band MajorTOM format via `reorder_multiband()` in `data_utils.py`—no hard-coded model names.
- **Local-first, remote-fallback:** `models/load_config.py` resolves `hf://` and `ms://` URLs automatically.
- **On-demand imagery:** The app never downloads the full dataset; it fetches individual rows via HTTP Range requests using `parquet_url` + `parquet_row` stored in each embedding.

---

## How to Contribute

### 1. Fork & Clone

```bash
git clone https://github.com/YOUR_USERNAME/EarthEmbeddingExplorer.git
cd EarthEmbeddingExplorer
```

### 2. Create a Branch

```bash
git checkout -b feat/your-feature-name
```

### 3. Commit Message Format

We follow [Conventional Commits](https://www.conventionalcommits.org/).

**Format:**
```
<type>(<scope>): <subject>
```

**Types:**
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation only
- `style:` Code style (formatting, whitespace)
- `refactor:` Code change that neither fixes a bug nor adds a feature
- `perf:` Performance improvement
- `test:` Adding or updating tests
- `chore:` Build, tooling, or maintenance

**Examples:**
```bash
feat(models): add RemoteCLIP embedding model
fix(app): correct similarity threshold slider behavior
docs(readme): update dataset description
refactor(search_engine): simplify score fusion in mixed search
```

### 4. Pull Request Title Format

PR titles follow the same convention:

```
<type>(<scope>): <description>
```

- Scope must be **lowercase**.
- Keep the description short and descriptive.

**Examples:**
```
feat(models): add DINOv2 for visual similarity search
fix(app): handle empty query input gracefully
docs(tutorial): add Chinese translation
```

### 5. Code Style and Quality

We use **[Ruff](https://github.com/astral-sh/ruff)** for linting and formatting.

```bash
# Install
pip install ruff

# Check
ruff check .
ruff format . --check

# Auto-fix
ruff check --fix .
ruff format .
```

**Note:** `pyproject.toml` excludes `MajorTOM/` and `models/` from Ruff because they contain third-party forks. Please do **not** re-format files inside those directories unless you are deliberately modifying them.

**Rules enabled:** `E`, `F`, `I`, `W`, `N`, `UP`, `RUF`, `B`

---

## Deployment Guide (for Contributors)

If you want to deploy your fork to ModelScope Studio for live testing:

### Prerequisites: README.md YAML Metadata

ModelScope Studio requires a YAML metadata header at the top of `README.md`. If this header is missing or empty, the studio will fail to initialize with a "YAML metadata missing or empty" error.

Add the following YAML frontmatter to the very first line of `README.md` (before the `# Title`):

```markdown
---
domain:
- multi-modal
tags:
- remote-sensing
- satellite-imagery
- cross-modal-retrieval
- embedding
deployspec:
  entry_file: app.py
license: MIT
---
```

| Field | Description |
| :--- | :--- |
| `domain` | Studio domain: `cv`, `nlp`, `audio`, `multi-modal`, or `AutoML` |
| `tags` | Custom tags for discovery and filtering |
| `datasets` | (Optional) Associated datasets |
| `models` | (Optional) Associated models |
| `deployspec.entry_file` | Entry point file. For Gradio/Streamlit SDK, default is `app.py` |
| `license` | Open-source license, e.g. `MIT`, `Apache-2.0` |

> **Note:** If you are opening a PR to the upstream repo, please include the YAML header in your `README.md` changes so the maintainers' studio can also pick it up.

---

### 1. Option A: Duplicate a new studio (first time)

1. **(Optional)** Apply to join [xGPU-Explorers](https://modelscope.cn/organization/xGPU-Explorers) for free GPU access.
2. Click **Duplicate** on the [project page](https://modelscope.cn/studios/Major-TOM/EarthEmbeddingExplorer/).
3. Configure resources and set `DOWNLOAD_ENDPOINT`:
   - `modelscope.cn` — mainland China (fastest)
   - `modelscope.ai` — international users
4. Publish your studio.

### 1. Option B: Update an existing studio

If you already have a studio (e.g. `YOUR_USERNAME/EarthEmbeddingExplorer`) and just want to push new code:

1. Skip the **Duplicate** step above.
2. Proceed directly to **Step 2** below.

---

### 2. Push code to ModelScope Studio

1. Fork the GitHub repo and push your branch:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/EarthEmbeddingExplorer.git
   git push origin your-branch
   ```

2. In ModelScope Studio, click **Download Studio** to get the Git URL with your access token.

   ![download](images/download_studio.png)

3. Add ModelScope as an upstream remote. **Choose the correct domain for your account:**

   ```bash
   # For mainland China users (modelscope.cn)
   git remote add modelscope https://oauth2:YOUR_TOKEN@modelscope.cn/studios/YOUR_USERNAME/EarthEmbeddingExplorer.git

   # For international users (modelscope.ai)
   git remote add modelscope https://oauth2:YOUR_TOKEN@www.modelscope.ai/studios/YOUR_USERNAME/EarthEmbeddingExplorer.git
   ```

4. Push your branch to the studio's `master` branch:

   ```bash
   # Normal push (if histories are compatible)
   git push modelscope your-branch:master

   # Force push (if the studio's master has a different history — common for existing studios)
   git push modelscope your-branch:master --force
   ```

   > **Why `--force`?** ModelScope Studio initializes its own `master` branch with system commits. If your branch is based on `upstream/main`, the histories will diverge. You must use `--force` to overwrite the studio's master. Alternatively, you can create a merge branch based on the studio's master and merge your changes in, then push without `--force`.

5. Go to settings and restart or deep reboot the studio.

6. Verify the deployed studio works, then open a PR on GitHub.

---

### Troubleshooting

| Symptom | Cause | Fix |
| :--- | :--- | :--- |
| `non-fast-forward` error on push | Studio master has a different history from your branch | Use `git push modelscope your-branch:master --force`, or merge studio's master into your branch first |
| `YAML metadata missing or empty` | `README.md` lacks the YAML frontmatter | Add the YAML header described in **Prerequisites** above |
| `curl 18 transfer closed` during clone | Network timeout while studio clones your repo | Add environment variables in studio settings: `GIT_HTTP_LOW_SPEED_TIME=300`, `GIT_HTTP_LOW_SPEED_LIMIT=1000`, then deep reboot |
| `ModuleNotFoundError` after restart | Missing dependency in `requirements.txt` | Add the package to `requirements.txt`, commit, push, and deep reboot |
| Models fail to load | `DOWNLOAD_ENDPOINT` mismatch or missing checkpoints | Check studio environment variables; ensure `DOWNLOAD_ENDPOINT` matches your region (`modelscope.cn` vs `modelscope.ai`) |

---

## Contribution Areas

### Adding a New Embedding Model

We welcome new vision-language or vision-only models that improve retrieval quality or support new modalities (e.g., temporal, multi-spectral).

**Required interface:** Every model lives in `models/<name>_model.py` and must implement:

| Method | Purpose |
| :--- | :--- |
| `__init__(ckpt_path, embedding_path, device)` | Load config, set paths, lazy-load weights in `load_model()` |
| `load_model()` | Download weights if needed (respect `DOWNLOAD_ENDPOINT`), initialize inference model |
| `encode_text(text)` | Return a text embedding `torch.Tensor` (or `None` if unsupported) |
| `encode_image(image)` | Return an image embedding `torch.Tensor`. Accepts `PIL.Image`, `torch.Tensor`, or `np.ndarray`. |
| `encode_location(lat, lon)` | Return a location embedding `torch.Tensor` (or `None` if unsupported) |
| `search(query_embedding, top_percent)` | Compute cosine similarity against `self.image_embeddings`, return `(probs, filtered_indices, top_indices)` |

**Multi-spectral models** (e.g., SatCLIP, Clay, OlmoEarth) must additionally declare:

| Attribute | Purpose |
| :--- | :--- |
| `self.requires_multiband = True` | Signals the search engine that this model needs 12-band Sentinel-2 input instead of RGB. |
| `self.bands = ['B02', 'B03', ...]` | Lists the band names (in order) the model expects. The generic `data_utils.reorder_multiband()` will automatically extract these from the 12-band MajorTOM format. |
| `self.size = (H, W)` | Spatial resolution expected by the model's encoder (used for auto-resize if needed). |

**Vendored third-party code:** If your model wraps a forked or vendored repository (like `models/Clay/` or `models/FarSLIP/`), place it under `models/<Vendor>/` and ensure:
- The directory contains a valid `__init__.py` so Python treats it as a package.
- Your wrapper injects the vendor root into `sys.path` before importing absolute paths inside the vendored code (see `models/clay_model.py` for the pattern).
- Dependencies are added to `requirements.txt`.

**Registration checklist:**
1. Add the model class to `models/__init__.py`.
2. Add an entry to `core/model_manager.py` in `_load_all_models()`.
3. Add an entry to `generate_embeddings.py` in `MODEL_MAP`.
4. Add a config block to `configs/config.yaml` with `ckpt_path`, `model_name`, `tokenizer_path` (if needed), and `embedding_path`.
5. Add the model name to the Image Search dropdown in `app.py`.
6. Update `requirements.txt` if new dependencies are needed.
7. Update `README.md` and `doc.md` with the model description.
8. Add a test case in `tests/test_image_search.py` (or create a new test file).

**Weight hosting:** Upload your model weights to HuggingFace or ModelScope so the `DOWNLOAD_ENDPOINT` mechanism works for both China (`modelscope.cn`) and international users (`modelscope.ai` / `huggingface`).

---

### Adding a New Dataset

We use **MajorTOM Core-S2L2A** (Sentinel-2 Level 2A, ~23 TB) as the source imagery. To add a new source dataset or a new sampling strategy:

**Dataset requirements:**
- Global or regional coverage with georeferenced imagery.
- Clear licensing (open data preferred).
- Accessible via HTTP or cloud storage (Parquet shards preferred).

**Metadata requirements:** The metadata parquet **must** contain the following columns so that the app can fetch raw images on demand:

| Column | Type | Description |
| :--- | :--- | :--- |
| `product_id` | string | Unique scene identifier |
| `grid_cell` | string | MajorTOM hierarchical grid code |
| `timestamp` | string | Acquisition time (e.g., `20221115T161819`) |
| `centre_lat` | float | Center latitude (WGS84) |
| `centre_lon` | float | Center longitude (WGS84) |
| `parquet_url` | string | Full URL to the Parquet shard containing the raw image |
| `parquet_row` | int64 | Global row index inside that shard |

> **Tip:** If your source imagery is stored locally (e.g., `images_249k/part_00001.parquet`), you can keep relative `parquet_url` paths for local use, but the final embedding datasets uploaded to ModelScope/HuggingFace should use absolute URLs (e.g., `https://huggingface.co/datasets/Major-TOM/Core-S2L2A/resolve/main/images/part_00001.parquet`).

**Implementation:**
- Add dataset loading logic in `data_utils.py` if the format differs from MajorTOM.
- If your dataset provides a different band order than MajorTOM's `['B01', 'B02', ..., 'B12']`, update `data_utils.MULTIBAND_COLUMNS` and `reorder_multiband()` so downstream models can still auto-map bands via `model.bands`.
- Update `doc.md` with source, resolution, and preprocessing steps.

---

### Embedding Generation Pipeline

The `generate_embeddings.py` script turns raw MajorTOM Parquet shards into embedding GeoParquets.

**How it works:**
1. Reads `metadata.parquet` (must contain `parquet_url` and `parquet_row`).
2. Reads image bands from `images/part_*.parquet` row groups.
3. For each image, looks up metadata by `grid_cell` + `product_id`.
4. Runs `MajorTOM_Embedder.forward()` (tiling) or `_embed_single_fragment()` (pre-cropped).
5. Outputs a GeoParquet with one row per fragment, including `embedding`, `geometry`, and metadata.

**To generate embeddings for a new model:**

```bash
python generate_embeddings.py \
    --model_name <your_model> \
    --meta_path /path/to/metadata_249k.parquet \
    --parquet_input /path/to/images_249k/ \
    --output_path /path/to/output/<MODEL>_crop_384x384.parquet \
    --fragment_size 384
```

**Output schema requirements:** The GeoParquet must contain these columns to be compatible with the app:

```
unique_id, embedding, timestamp, product_id, grid_cell,
grid_row_u, grid_col_r, geometry, centre_lat, centre_lon,
utm_footprint, utm_crs, pixel_bbox, parquet_row, parquet_url
```

If you modify `MajorTOM_Embedder.py` or `generate_embeddings.py`, please run a small test (e.g., `--max_row_groups 1`) and verify the output schema with `pd.read_parquet(output).columns`.

---

### Enhancing the Web App

The web app is built with **Gradio** and split into modular layers:

| File | Responsibility |
| :--- | :--- |
| `app.py` | Gradio layout, event wiring, launch logic |
| `ui/callbacks.py` | Map click handlers, reset buttons, initial plot setup |
| `ui/utils.py` | UI helpers (e.g., formatting status messages) |
| `core/model_manager.py` | Model lifecycle (load, cache, retrieve) |
| `core/search_engine.py` | All search modes: `search_text`, `search_image`, `search_location`, `search_mixed` |
| `core/filters.py` | Post-search time-range and geo-bounding-box filters |
| `core/exporters.py` | ZIP export of results (thumbnail / RGB / multiband) |
| `data_utils.py` | HTTP-Range download from Parquet shards, image normalization |
| `visualize.py` | Plotly map traces, matplotlib top-K overview, gallery formatting |

**Contribution ideas:**
- **New query modalities:** bounding-box drawing on the map, time-series queries, multi-image queries.
- **UI/UX:** Better layout, responsive design, clearer error messages.
- **Visualization:** Side-by-side model comparison, temporal animations, score histograms.
- **Export:** GeoJSON/KML export, CSV metadata download.

If you modify `app.py`, test locally with `python app.py` before pushing.

---

### Testing

We use `pytest` for testing. Tests live in `tests/` and are tracked in git.

**Run all image-search tests:**
```bash
python tests/test_image_search.py --model all
```

**Run a single model:**
```bash
python tests/test_image_search.py --model Clay --lat -3 --lon -63
```

**What the test does:**
1. Loads the model and its embeddings.
2. Finds the nearest `product_id` to the query lat/lon.
3. Downloads the image (multiband for multi-spectral models, thumbnail for RGB models).
4. Reorders bands via `reorder_multiband()` if needed.
5. Encodes the image and runs a top-5 search.
6. Asserts the embedding is non-empty and results are sorted.

When adding a new model, please add it to `tests/test_image_search.py` (or create a new test file) and verify the full pipeline end-to-end before opening a PR.

---

### Improving Retrieval Performance

- **Similarity Search framework integration:** We plan to support FAISS/Milvus for approximate nearest-neighbor search. Implementing IVF or HNSW indexes for our embedding datasets is a high-priority item.
- **Similarity metrics:** Experiment with cosine, Euclidean, or learned fusion strategies.
- **Benchmarking:** Add scripts to benchmark retrieval speed and accuracy across models.

Please include before/after benchmarks in your PR description.

---

### Documentation and Tutorials

- Document real-world use cases with screenshots or notebooks.

---

### Bug Fixes and Refactoring

- Small fixes, clearer error messages, and edge-case handling are always welcome.
- For large refactors, open an issue first to align on approach.
- Update `requirements.txt` versions only with clear justification.

---


## Do's and Don'ts

### ✅ DO

- Start with small, focused changes.
- Discuss large or design-sensitive changes in an issue first.
- Write or update tests where applicable.
- Update documentation for user-facing changes.
- Use conventional commit messages and PR titles.
- Be respectful and constructive.
- Cite relevant papers or datasets when adding new models.

### ❌ DON'T

- Don't open very large PRs without prior discussion.
- Don't ignore Ruff failures.
- Don't mix unrelated changes in one PR.
- Don't break existing APIs or pipelines without migration notes.
- Don't add heavy or optional dependencies to the core install without discussion.
- Don't redistribute datasets or models without checking their licenses.

---

## Roadmap

We welcome contributions aligned with our roadmap:

- [x] Support DINOv2 embedding model and embedding datasets.
- [x] Increase geographic coverage to ~1.2% of Earth's land surface (~249k samples).
- [x] Add mixed search (text + image + location fusion).
- [x] Support Clay v1.5 multi-spectral embedding model.
- [x] Refactor multi-spectral handling to be model-agnostic (`requires_multiband` + `reorder_multiband`).
- [ ] **Support FAISS for faster similarity search.** (High priority)
- [ ] Add more embedding models (e.g., new remote-sensing CLIP variants, Prithvi, DOFA).
- [ ] Improve UI/UX and add new visualization features.
- [ ] Support larger datasets or full MajorTOM coverage.
- [ ] What features do you want? Leave an issue!

---

## Getting Help

- **GitHub Issues:** [https://github.com/VoyagerXvoyagerx/EarthEmbeddingExplorer/issues](https://github.com/VoyagerXvoyagerx/EarthEmbeddingExplorer/issues)
- **ModelScope Feedback:** [ModelScope Studio Discussions](https://modelscope.ai/studios/Major-TOM/EarthEmbeddingExplorer/feedback)

---

## Citation

If you use EarthEmbeddingExplorer in your research, please cite:

```bibtex
@inproceedings{
  zheng2026earthembeddingexplorer,
  title={EarthEmbeddingExplorer: A Web Application for Cross-Modal Retrieval of Global Satellite Images},
  author={Yijie Zheng and Weijie Wu and Bingyue Wu and Long Zhao and Guoqing Li and Mikolaj Czerkawski and Konstantin Klemmer},
  booktitle={4th ICLR Workshop on Machine Learning for Remote Sensing (Tutorial Track)},
  year={2026},
  url={https://openreview.net/forum?id=LSsEenJVqD}
}
```

---

Thank you for contributing to EarthEmbeddingExplorer. Your work helps make it a better tool for exploring and understanding our planet. 🌍
