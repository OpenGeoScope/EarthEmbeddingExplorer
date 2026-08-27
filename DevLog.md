# Development Log

This log condenses the Git history into weekly user-facing features, bug fixes, and major performance improvements; weeks containing only merge, formatting, metadata, or documentation changes are omitted.

## 2026-W35 (Aug 24-30)

- **Feature:** Migrated OlmoEarth retrieval to OlmoEarth-v1_2-Base with real acquisition timestamps and 3D temporal encoding.
- **Fix:** Preserved the true spatial extent of resized OlmoEarth inputs by passing the effective 30 m encoder resolution.
- **Fix:** Aligned Core-S2L2A-249k source URLs, row-group indices, and online band resampling so downloaded queries match indexed pixels.
- **Fix:** Added an international ModelScope mirror for OlmoEarth-v1_2-Base where the official AllenAI repository is unavailable.
- **Fix:** Declared EPSG:4326 explicitly in generated embedding GeoParquet files.
- **Performance:** Added deterministic multi-GPU sharding, sorted metadata lookup, batched decoding, and incremental progress accounting for embedding generation.
- **Fix:** Preserved downloaded multiband and timestamp state until the user replaces the query via upload, webcam, clipboard, or an example image.

## 2026-W33 (Aug 10-16)

- **Fix:** Made Qwen3-VL honor the configured inference device.
- **Performance:** Kept mixed-search similarity computation on the embedding device.

## 2026-W32 (Aug 3-9)

- **Feature:** Added TIPSv2 and Qwen3-VL-Embedding-2B for image, text, and native mixed-query retrieval.
- **Feature:** Added endpoint-aware model and embedding downloads for ModelScope China and international deployments.
- **Feature:** Added startup warm-up stages, clearer model display names, and detailed search status reporting.
- **Fix:** Added Gradio 6 compatibility for styling, hidden-tab examples, and example re-rendering.
- **Fix:** Prevented vendored Clay from changing the process-wide cuDNN API configuration.

## 2026-W23 (Jun 1-7)

- **Feature:** Added command-line and environment-based selection of which models to load at application startup.
- **Performance:** Added batched single-fragment inference to the embedding generation pipeline.
- **Fix:** Corrected top-percentage ranking so gallery candidates follow the full similarity-sorted result set.
- **Fix:** Switched DINOv2 embedding generation to the intended large checkpoint.
- **Fix:** Corrected download endpoint placeholders and normalized regional model configuration handling.

## 2026-W18 (Apr 27-May 3)

- **Feature:** Integrated OlmoEarth for 12-band Sentinel-2 embedding generation and image retrieval.
- **Fix:** Aligned OlmoEarth with the model-agnostic multispectral interface.
- **Fix:** Restored nearest-neighbor multispectral resizing to prevent embedding drift.
- **Fix:** Corrected regional ModelScope repository identifiers for OlmoEarth.

## 2026-W17 (Apr 20-26)

- **Feature:** Added source `parquet_url` and `parquet_row` fields so retrieved samples can download their original imagery.
- **Feature:** Added model-agnostic multispectral download, band reordering, preprocessing, and search handling.
- **Feature:** Integrated Clay v1.5 for multispectral embedding generation and image retrieval.
- **Feature:** Added image-search integration tests across all registered models.
- **Fix:** Corrected Clay vendored import resolution in environments without a separately installed `claymodel` package.
- **Fix:** Corrected model download endpoint handling and regional repository identifiers across wrappers.
- **Fix:** Corrected full-dataset map rendering, map-click inputs, and high-resolution map coordinate mapping.
- **Fix:** Restored top-five image display for mixed search and passed model metadata through result export.
- **Fix:** Added reliable ZIP naming and source-image download metadata to exported search results.

## 2026-W16 (Apr 13-19)

- **Feature:** Added a reusable command-line pipeline for generating MajorTOM embedding datasets and GeoParquet metadata.
- **Feature:** Added adaptive single-fragment processing for pre-cropped imagery and equal-size fragment handling.
- **Feature:** Added local-first path resolution with remote ModelScope and Hugging Face fallback configuration.
- **Feature:** Added configurable Sentinel-2 true-color normalization without coupling display preprocessing to model inputs.
- **Fix:** Unified model loading and simplified local and remote configuration handling.

## 2026-W15 (Apr 6-12)

- **Feature:** Split application logic into reusable core, UI callback, exporter, filtering, and model-management modules.
- **Fix:** Corrected SatCLIP multiband image search and UI callback integration.
- **Fix:** Corrected full-dataset map rendering and map-click callback inputs after the application refactor.

## 2026-W13 (Mar 23-29)

- **Feature:** Released the initial camera-ready EarthEmbeddingExplorer application for global satellite embedding search.
- **Feature:** Enabled the FarSLIP retrieval model in the default model-loading workflow.
- **Feature:** Added ModelScope.ai deployment support.
- **Fix:** Corrected DINOv2 checkpoint-path validation.
- **Fix:** Corrected SatCLIP image download mode for multispectral queries.
