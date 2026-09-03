from pathlib import Path

from PIL import Image

from data_utils import load_multispectral_geotiff

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_EXAMPLES = [
    {
        "preview": "./examples/multispectral_27_43_preview.png",
        "tif": "./examples/multispectral_27_43.tif",
        "label": "(27, 43)\nMultispectral",
    },
    {
        "preview": "./examples/multispectral_minus4_minus63_preview.png",
        "tif": "./examples/multispectral_minus4_minus63.tif",
        "label": "(-4, -63)\nMultispectral",
    },
    {
        "preview": "./examples/multispectral_28_2_85_7_preview.png",
        "tif": "./examples/multispectral_28_2_85_7.tif",
        "label": "(28.2, 85.7)\nMultispectral",
    },
    {
        "preview": "./examples/multispectral_41_85_preview.png",
        "tif": "./examples/multispectral_41_85.tif",
        "label": "(41, 85)\nMultispectral",
    },
    {
        "preview": "./examples/circular_field_aerial.png",
        "tif": None,
        "label": "AIGC\nRGB",
    },
    {
        "preview": "./examples/circular_farmland_stick_fig.png",
        "tif": None,
        "label": "AIGC\nRGB",
    },
    {
        "preview": "./examples/river_stick_fig.png",
        "tif": None,
        "label": "AIGC\nRGB",
    },
]

IMAGE_EXAMPLE_GALLERY = [(item["preview"], item["label"]) for item in IMAGE_EXAMPLES]
AIGC_IMAGE_EXAMPLE_FILES = [[item["preview"]] for item in IMAGE_EXAMPLES if item["tif"] is None]


def select_image_example(index):
    """Return preview and model state for the selected image example."""
    item = IMAGE_EXAMPLES[int(index)]
    preview_path = _PROJECT_ROOT / item["preview"].removeprefix("./")
    with Image.open(preview_path) as image:
        preview = image.convert("RGB").copy()

    if item["tif"] is None:
        return preview, None, None, "Selected AIGC RGB example."

    tif_path = _PROJECT_ROOT / item["tif"].removeprefix("./")
    multiband, metadata = load_multispectral_geotiff(tif_path)
    return (
        preview,
        multiband,
        metadata,
        (f"Selected {item['label'].replace(chr(10), ' ')} example. Acquisition: {metadata.get('product_datetime')}."),
    )
