# Contributing to EarthEmbeddingExplorer

## Welcome! 🌍

Thank you for your interest in contributing to EarthEmbeddingExplorer! EarthEmbeddingExplorer is an open-source tool that lets you search satellite imagery using **natural language**, **images**, or **geographic locations**. It enables cross-modal retrieval of global satellite images and visualizes them on an interactive map.

We warmly welcome contributions that help make EarthEmbeddingExplorer more useful for geoscience research, education, and exploration: whether you add new embedding models, improve retrieval performance, enhance the UI, fix bugs, or improve documentation.

**Quick links:** [GitHub](https://github.com/ML4Sustain/EarthEmbeddingExplorer) · [ModelScope Demo](https://modelscope.cn/studios/Major-TOM/EarthEmbeddingExplorer/) · [HuggingFace Demo](https://huggingface.co/spaces/ML4Sustain/EarthExplorer) · [Tutorial](https://huggingface.co/spaces/ML4Sustain/EarthExplorer/blob/main/Tutorial.md) · [License](LICENSE)

---

## How to Contribute

To keep collaboration smooth and maintain quality, please follow these guidelines.

### 1. Check Existing Issues and Roadmap

Before starting:

- **Check [Open Issues](https://github.com/ML4Sustain/EarthEmbeddingExplorer/issues)** and our [Roadmap](#roadmap--contribution-priorities) below.
- **If a related issue exists** and is open or unassigned: comment to say you want to work on it to avoid duplicate effort.
- **If no related issue exists**: open a new issue describing your proposal. The maintainers will respond and can help align with the project direction.

### 2. Commit Message Format

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification for clear history and tooling.

**Format:**
```
<type>(<scope>): <subject>
```

**Types:**
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation only
- `style:` Code style (whitespace, formatting, etc.)
- `refactor:` Code change that neither fixes a bug nor adds a feature
- `perf:` Performance improvement
- `test:` Adding or updating tests
- `chore:` Build, tooling, or maintenance

**Examples:**
```bash
feat(models): add FarSLIP embedding model
fix(app): correct similarity threshold slider behavior
docs(readme): update dataset description
refactor(data_utils): simplify parquet shard loading
test(visualize): add tests for embedding normalization
```

### 3. Pull Request Title Format

PR titles should follow the same convention:

**Format:** ` <type>(<scope>): <description> `

- Use one of: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`, `perf`, `style`, `build`, `revert`.
- **Scope must be lowercase** (letters, numbers, hyphens, underscores only).
- Keep the description short and descriptive.

**Examples:**
```
feat(models): add DINOv2 for visual similarity search
fix(app): handle empty query input gracefully
docs(tutorial): add Chinese translation
refactor(data_utils): optimize HTTP range request logic
```

### 4. Code Style and Quality

We use **[Ruff](https://github.com/astral-sh/ruff)** to enforce consistent code style and catch common errors automatically.

#### Setup Ruff

```bash
# Install Ruff
pip install ruff

# Run code style checks
ruff check .

# Auto-fix fixable issues
ruff check --fix .

# Format code
ruff format .
```

#### Configuration

Ruff is configured in `pyproject.toml` with the following rules enabled:

| Rule | Description |
|------|-------------|
| `E` | pycodestyle errors |
| `F` | pyflakes (unused imports/variables) |
| `I` | isort (import sorting) |
| `W` | pycodestyle warnings |
| `N` | pep8-naming (naming conventions) |
| `UP` | pyupgrade (modern syntax) |
| `RUF` | Ruff-specific rules |
| `B` | flake8-bugbear (common bug patterns) |

#### Required local gate (must pass before push/PR):

```bash
# Install dependencies
pip install -r requirements.txt

# Run Ruff checks
ruff check .
ruff format . --check

# Run tests (if applicable)
pytest
```

- **If pre-commit modifies files:** Commit those changes, then rerun until it passes cleanly.
- **CI policy:** Pull requests with failing Ruff checks are not merge-ready.
- **Documentation:** Update docs and README when you add or change user-facing behavior.

---

## Types of Contributions

EarthEmbeddingExplorer is designed to be **extensible**: you can add new embedding models, datasets, visualization features, and more. Below are the main contribution areas we care about.

---

### Adding New Embedding Models

EarthEmbeddingExplorer currently supports multiple embedding models, including **SigLIP**, **DINOv2**, **FarSLIP**, and **SatCLIP**. We welcome new models that can enhance retrieval quality or support new modalities.

Contributed models should have the following characteristics:

1. (Required) Compatible with the existing embedding pipeline (`data_utils.py` and `visualize.py`). The model should be able to encode images, text, or locations into embeddings that can be compared via similarity metrics.
2. (Required) Open-source weights and inference code available (e.g., on HuggingFace or GitHub).
3. (Recommended) Pre-trained on remote sensing or geospatial data, or fine-tuned for specific Earth observation tasks.

If you wish to submit a Pull Request, please ensure that the following conditions are met:

1. (Required) Include model loading and encoding logic in `data_utils.py` or a separate module, and provide a test script or notebook demonstrating the model works end-to-end.
2. (Required) Add the model to the model list in the documentation (`doc.md` and `README.md`). If the model's configuration or usage is significantly different, add a dedicated section explaining how to use it.
3. (Recommended) Provide example retrieval results (screenshots or saved outputs) showing the model's performance on representative queries (e.g., "glacier", "city with coastline", or specific coordinates).

---

### Adding New Datasets

We currently use **MajorTOM Core-S2L2A** (Sentinel-2 Level 2A) from ESA. Contributions that add or support new datasets are welcome.

- **Dataset requirements:**
  - Global or regional coverage with georeferenced imagery.
  - Clear licensing (open data preferred, e.g., Creative Commons or public domain).
  - Accessible via HTTP or cloud storage (e.g., ModelScope Datasets).
- **Implementation:**
  - Add dataset loading logic in `data_utils.py`.
  - Update the metadata index and parquet shard handling if applicable.
  - Document the dataset source, resolution, and any preprocessing steps in `doc.md`.

If you contribute a new dataset, please ensure it integrates with the existing pipeline and does not break current functionality.

---

### Improving Retrieval Performance

Contributions that make similarity search faster or more accurate are highly valued.

- **FAISS integration:** We plan to support FAISS for faster approximate nearest neighbor search. Implementing FAISS indexing (IVF, HNSW, etc.) for our embedding datasets is a priority.
- **Similarity metrics:** Experiment with or add new similarity metrics (e.g., cosine similarity, Euclidean distance, cross-modal fusion strategies).
- **Caching and optimization:** Improve caching of embeddings, optimize batch processing, or reduce memory usage.
- **Benchmarking:** Add scripts or notebooks to benchmark retrieval speed and accuracy across models and datasets.

If you work on performance improvements, please include before/after benchmarks in your PR description.

---

### Enhancing the Web App (Gradio)

The interactive demo runs on Gradio (`app.py`). Contributions that improve the user experience are welcome.

- **UI/UX improvements:** Better layout, clearer visualizations, more intuitive controls, or responsive design.
- **New query modalities:** Support for new input types (e.g., bounding box drawing on the map, time-series queries, or multi-image queries).
- **Visualization enhancements:** Add new ways to visualize results (e.g., side-by-side comparison, temporal animations, or interactive charts).
- **Bug fixes:** Fix UI glitches, improve error messages, or handle edge cases gracefully.

If you modify `app.py`, please test locally and ensure the app launches correctly with `python app.py`.

---

### Visualization and Analysis Tools

The `visualize.py` module handles result display and map rendering.

- **Map features:** Add new basemap layers, heatmaps, clustering of results, or export options (e.g., GeoJSON, KML).
- **Result analysis:** Tools to analyze retrieval results statistically (e.g., geographic distribution, score histograms, or diversity metrics).
- **Export functionality:** Allow users to download top-K results with metadata (coordinates, scores, image URLs).

---

### Documentation and Tutorials

- **Tutorial improvements:** Fix errors, add examples, or clarify steps in `doc.md` and `doc_zh.md`.
- **Code documentation:** Add docstrings, type hints, or inline comments to clarify complex logic.
- **Translation:** Improve or add translations for Chinese and other language versions of the docs.
- **Examples and use cases:** Document real-world use cases (e.g., "finding glacier regions", "mapping urban expansion") with screenshots or notebooks.

---

### Bug Fixes and Refactoring

- **Bug fixes:** Small fixes, clearer error messages, and handling edge cases are valuable.
- **Refactoring:** Code cleanup that keeps behavior the same is welcome. For large refactors, please open an issue first to align on approach.
- **Dependency updates:** Update `requirements.txt` versions or add optional dependencies with clear justification.

---

### Other Contributions

- **Deployment guides:** Instructions for deploying EarthEmbeddingExplorer on other platforms (e.g., HuggingFace Spaces, ModelScope, local GPU/CPU, Docker).
- **Examples and workflows:** Tutorials or example notebooks (e.g., "analyze land cover change over time", "compare models on the same query").
- **Any other useful things!**

---

## Roadmap & Contribution Priorities

We welcome contributions aligned with our roadmap:

- [x] Support DINOv2 Embedding model and embedding datasets.
- [x] Increase the geographical coverage (sample rate) to 1.2% of the Earth's land surface.
- [ ] **Support FAISS for faster similarity search.** (High priority)
- [ ] Add more embedding models (e.g., new remote sensing CLIP variants).
- [ ] Improve UI/UX and add new visualization features.
- [ ] Support larger datasets or full MajorTOM coverage.
- [ ] What features do you want? Leave an issue [here](https://modelscope.ai/studios/Major-TOM/EarthEmbeddingExplorer/feedback)!

If you're interested in working on a roadmap item, please comment on the corresponding issue or open a new one.

---

## Do's and Don'ts

### ✅ DO

- Start with small, focused changes.
- Discuss large or design-sensitive changes in an issue first.
- Write or update tests where applicable.
- Update documentation for user-facing changes.
- Use conventional commit messages and PR titles.
- Be respectful and constructive (we follow a welcoming Code of Conduct).
- Cite relevant papers or datasets in the docs when adding new models or datasets.

### ❌ DON'T

- Don't open very large PRs without prior discussion.
- Don't ignore CI or pre-commit failures.
- Don't mix unrelated changes in one PR.
- Don't break existing APIs or pipelines without a good reason and clear migration notes.
- Don't add heavy or optional dependencies to the core install without discussing in an issue.
- Don't redistribute datasets or models without checking their licenses.

---

## Getting Help

- **Issues and features:** [ModelScope Issues](https://modelscope.ai/studios/Major-TOM/EarthEmbeddingExplorer/feedback)

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
