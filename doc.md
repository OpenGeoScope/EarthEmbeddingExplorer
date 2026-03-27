# EarthEmbeddingExplorer

<div style="display: flex; gap: 0.2em; align-items: center; justify-content: center;">
    <a href="https://modelscope.cn/studios/Major-TOM/EarthEmbeddingExplorer/"><img src="https://img.shields.io/badge/Open in ModelScope.cn-xGPU-624aff"></a>
    <a href="https://modelscope.ai/studios/Major-TOM/EarthEmbeddingExplorer/"><img src="https://img.shields.io/badge/Open in ModelScope.ai-xGPU-624aff"></a>
    <a href="https://modelscope.cn/datasets/VoyagerX/EarthEmbeddings"><img src="https://img.shields.io/badge/👾 MS-Dataset-624aff"></a>
    <a href="https://huggingface.co/datasets/ML4RS-Anonymous/EarthEmbeddings/tree/main"><img src="https://img.shields.io/badge/🤗 HF-Dataset-FFD21E"></a>
    <a href="https://huggingface.co/spaces/ML4Sustain/EarthExplorer/blob/main/Tutorial.md"> <img src="https://img.shields.io/badge/Document-📖-007bff"> </a>
    <a href="https://modelscope.cn/studios/VoyagerX/EarthExplorer/file/view/master/Tutorial_zh.md?status=1"> <img src="https://img.shields.io/badge/中文文档-📖-007bff"> </a>
    <a href="https://openreview.net/forum?id=LSsEenJVqD"> <img src="https://img.shields.io/badge/Tutorial-@ICLR26📖-007bff"> </a>
</div>

## Background

### What is this project about?
EarthEmbeddingExplorer is a tool that lets you search satellite imagery using **natural language**, **images**, or **geographic locations**. In simple terms, you can enter prompts like “a satellite image of a glacier” or “a satellite image of a city with a coastline”, and the system will find places on Earth that match your description and visualize them on a map.

EarthEmbeddingExplorer enables users to explore the Earth in multiple ways without leaving their desk, and it can be useful for many geoscience tasks. For example, geologists can quickly locate glacier regions; biologists can rapidly map forest cover; and architects can study urban patterns across different parts of the world.

## How does it work? (Core ideas)

### Satellite imagery dataset
We use **MajorTOM** (Major TOM: Expandable Datasets for Earth Observation) released by the European Space Agency (ESA) [1]. Specifically, we use the [Core-S2L2A](https://modelscope.cn/datasets/Major-TOM/Core-S2L2A) subset.

| Dataset | Imagery source | Number of samples | Sensor type |
| :--- | :--- | :--- | :--- |
| MajorTOM-Core-S2L2A | Sentinel-2 Level 2A | 2,245,886 | Multispectral |

MajorTOM Core-S2L2A provides global Sentinel-2 multispectral imagery (10 m resolution). We convert the RGB bands into embeddings using CLIP-like models (e.g., SigLIP), which saves substantial time because we do not need to preprocess raw imagery ourselves. In addition, embeddings (vectors) are much smaller than raw imagery, and they are significantly faster to search.

To keep EarthEmbeddingExplorer responsive, we build a smaller but representative version of the dataset.

The original tiles in Core-S2L2A are large (1068×1068 pixels), but most AI models expect smaller inputs (384×384 or 224×224 pixels).
1. **Cropping**: for simplicity, from each original tile we only take the **center** 384×384 (or 224×224) crop to generate an embedding.
2. **Uniform sampling**: using MajorTOM’s grid coding system, we sample **1%** of the data (about 22,000 images). This preserves global coverage while keeping search fast.

<div align="center">
  <img src="images/samples.png" width="50%" />
  <br>
  <em>Figure 1: Geographic distribution of our sampled satellite image embeddings.</em>
</div>

### Retrieval models
The core of image retrieval includes **CLIP (Contrastive Language-Image Pre-training)** [2] and **DINOv2 (self-supervised vision transformers)** [7]. We use CLIP's improved variants such as **SigLIP (Sigmoid Language-Image Pre-training)** [3], **FarSLIP (Fine-grained Aligned Remote Sensing Language Image Pretraining)** [4], and **SatCLIP (Satellite Location-Image Pretraining)** [5], along with **DINOv2** for pure visual similarity search [7].


An analogy: when teaching a child, you show a picture of a glacier and say “glacier”. After seeing many examples, the child learns to associate the visual concept with the word.

CLIP-like models learn in a similar way, but at much larger scale.
- An image encoder turns an **image** into an **embedding** (a vector of numbers).
- A text (or location) encoder turns **text** (or **latitude/longitude**) into an embedding.

The key property is: if an image matches a text description (or location), their embeddings will be close; otherwise they will be far apart.

<div align="center">
  <img src="images/CLIP.png" width="40%" />
  <br>
  <em>Figure 2: How CLIP-like models connect images and text.</em>
</div>

DINOv2, on the other hand, is a self-supervised vision model that learns rich visual representations without requiring paired text data. It excels at capturing visual patterns and can be used for image-to-image similarity search.

The four models we use differ in their encoders and training data:

| Model | Encoder type | Training data |
| :--- | :--- | :--- |
| SigLIP | image encoder + text encoder | natural image–text pairs from the web |
| DINOv2 | image encoder only | web-scale natural images (self-supervised) |
| FarSLIP | image encoder + text encoder | satellite image–text pairs |
| SatCLIP | image encoder + location encoder | satellite image–location pairs |

<div align="center">
  <img src="images/embedding.png" width="30%" />
  <br>
  <em>Figure 3: Converting satellite images into embedding vectors.</em>
</div>

In EarthEmbeddingExplorer:
1. We precompute embeddings for ~250k globally distributed satellite images using SigLIP, DINOv2, FarSLIP, and SatCLIP.
2. When you provide a query (text like "a satellite image of glacier", an image, or a location such as (-89, 120)), we encode the query into an embedding using the corresponding encoder.
3. We compare the query embedding with all image embeddings, visualize similarities on a map, and show the top-5 most similar images.

## System architecture

<div align="center">
  <img src="images/framework_en.png" width="70%" />
  <br>
  <em>Figure 4: EarthEmbeddingExplorer system architecture on ModelScope.</em>
</div>

We deploy EarthEmbeddingExplorer on ModelScope: the models, embedding datasets, and raw imagery datasets are all hosted on the platform. The app runs on [xGPU](https://www.modelscope.cn/brand/view/xGPU), allowing flexible access to GPU resources and faster retrieval.

### How is the raw imagery stored?

MajorTOM Core-S2L2A is large (about 23 TB), so we do not download the full dataset. Instead, the raw imagery is stored as **Parquet shards**:

- **Shard storage**: the dataset is split into many remote Parquet files (shards), each containing a subset of the samples.
- **Columnar storage**: different fields/bands (e.g., B04/B03/B02, thumbnail) are stored as separate columns; we only read what we need.
- **Metadata index**: we maintain a small index table mapping `product_id → (parquet_url, parquet_row)` so the system can locate “which shard and which position” contains a given image.

With this design, when a user only needs a small number of images from the retrieval results, the system can use **HTTP Range requests** to download only a small byte range from a Parquet file (the target row/row group and the requested columns), rather than downloading the full 23 TB dataset—enabling near real-time retrieval of raw images.

### What happens when you use the app? 

1. **Enter a query**: you can enter text, upload an image, or input a latitude/longitude. You can also click on the map to use the clicked location as a query.
2. **Compute similarity**: the app encodes your query into an embedding vector and computes similarity scores against all satellite image embeddings.
3. **Show results**: the system filters out low-similarity results and shows the highest-scoring locations (and scores) on the map. You can adjust the threshold using a slider.
4. **Download raw images on demand**: for the top-5 most similar images, the system looks up their `parquet_url` and row position via the metadata index, then uses HTTP Range to fetch only the required data (RGB bands) and displays the images quickly in the UI.

## Examples
<div align="center">
  <img src="images/Text_Search.jpg" width="99%" />
  <br>
  <em>Figure 5: Search by text.</em>
</div>
<br>

<div align="center">
  <img src="images/Image_Search_Amazon.jpg" width="99%" />
  <br>
  <em>Figure 6: Search by image.</em>
</div>
<br>

<div align="center">
  <img src="images/Location_Search_Amazon.jpg" width="99%" />
  <br>
  <em>Figure 7: Search by location.</em>
</div>

## Limitations

While EarthEmbeddingExplorer has strong potential, it also has limitations. SigLIP is primarily trained on “natural images” from the internet (people, pets, cars, everyday objects) rather than satellite imagery. This domain gap can make it harder for the model to understand certain scientific terms or distinctive geographic patterns that are uncommon in typical web photos.

FarSLIP may perform poorly on non-remote-sensing concepts described in text, such as queries like “an image of face”.

## Acknowledgements

We thank the following open-source projects and datasets that made EarthEmbeddingExplorer possible:

**Models:**
- [SigLIP](https://huggingface.co/timm/ViT-SO400M-14-SigLIP-384) - Vision Transformer model for image-text alignment
- [FarSLIP](https://github.com/NJU-LHRS/FarSLIP) - Fine-grained satellite image-text pretraining model
- [SatCLIP](https://github.com/microsoft/satclip) - Satellite location-image pretraining model
- [DINOv2](https://huggingface.co/facebook/dinov2-large) - Self-supervised vision transformer

**Datasets:**
- [MajorTOM](https://github.com/ESA-PhiLab/MajorTOM) - Expandable datasets for Earth observation by ESA

We are grateful to the research communities and organizations that developed and shared these resources.

## Contributors
- [Yijie Zheng](https://voyagerxvoyagerx.github.io/)
- [Weijie Wu](https://github.com/go-bananas-wwj)
- [Bingyue Wu](https://brynn-wu.github.io/Brynn-Wu)
- [Mikolaj Czerkawski](https://mikonvergence.github.io/)
- [Konstantin Klemmer](https://konstantinklemmer.github.io/)

## Roadmap
- [x] Support DINOv2 Embedding model and embedding datasets.
- [x] Increase the geographical coverage (sample rate) to 1.2% of of the Earth's land surface.
- [ ] Support FAISS for faster similarity search.
- [ ] What features do you want? Leave an issue [here](https://huggingface.co/spaces/ML4Sustain/EarthExplorer/discussions)!

We warmly welcome new contributors!

## Citation
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

## References

[1] Francis, A., & Czerkawski, M. (2024). Major TOM: Expandable Datasets for Earth Observation. IGARSS 2024.

[2] Radford, A., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. ICML 2021.

[3] Zhai, X., et al. (2023). Sigmoid Loss for Language-Image Pre-Training. ICCV 2023.

[4] Li, Z., et al. (2025). FarSLIP: Discovering Effective CLIP Adaptation for Fine-Grained Remote Sensing Understanding. arXiv 2025.

[5] Klemmer, K. et al. (2025). SatCLIP: Global, General-Purpose Location Embeddings with Satellite Imagery. AAAI 2025.

[6] Czerkawski, M., Kluczek, M., & Bojanowski, J. S. (2024). Global and Dense Embeddings of Earth: Major TOM Floating in the Latent Space. arXiv preprint arXiv:2412.05600.

[7] Oquab, M., et al. (2023). DINOv2: Learning Robust Visual Features without Supervision. arXiv preprint arXiv:2304.07193.

[8] Zheng, et al. (2026). EarthEmbeddingExplorer: A Web Application for Cross-Modal Retrieval of Global Satellite Images. 4th ICLR Workshop on ML4RS (Tutorial Track)
