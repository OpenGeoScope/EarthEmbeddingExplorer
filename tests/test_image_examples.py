import numpy as np
import pytest
import rasterio

from models.clay_model import ClayModel
from models.olmoearth_model import OlmoEarthModel
from ui.image_examples import IMAGE_EXAMPLES, select_image_example

EXPECTED_TIMESTAMPS = [
    "20200124T074211",
    "20190812T143759",
    "20210221T044819",
    "20190409T050659",
]


def test_image_example_order_and_labels():
    assert [item["label"] for item in IMAGE_EXAMPLES] == [
        "(27, 43)\nMultispectral",
        "(-4, -63)\nMultispectral",
        "(28.2, 85.7)\nMultispectral",
        "(41, 85)\nMultispectral",
        "AIGC\nRGB",
        "AIGC\nRGB",
        "AIGC\nRGB",
    ]
    assert [item["preview"].rsplit("/", 1)[-1] for item in IMAGE_EXAMPLES[4:]] == [
        "circular_field_aerial.png",
        "circular_farmland_stick_fig.png",
        "river_stick_fig.png",
    ]


@pytest.mark.parametrize("index", range(4))
def test_multispectral_examples_populate_pixels_and_metadata(index):
    preview, multiband, metadata, status = select_image_example(index)

    assert preview.size == (384, 384)
    assert multiband.shape == (384, 384, 12)
    assert multiband.dtype == np.uint16
    assert metadata["timestamp"] == EXPECTED_TIMESTAMPS[index]
    assert metadata["product_datetime"] == EXPECTED_TIMESTAMPS[index]
    assert metadata["clay_time_input_source"] == "tiff_tag"
    assert metadata["clay_latlon_input_source"] == "tiff_bounds"
    assert np.count_nonzero(metadata["clay_time_input"]) > 0
    assert np.count_nonzero(metadata["clay_latlon_input"]) > 0
    latlon, time = ClayModel._metadata_inputs(metadata, batch_size=1, device="cpu")
    assert np.count_nonzero(latlon.numpy()) > 0
    assert np.count_nonzero(time.numpy()) > 0
    parsed = OlmoEarthModel._parse_timestamp(metadata["timestamp"])
    timestamp = EXPECTED_TIMESTAMPS[index]
    assert parsed == (int(timestamp[6:8]), int(timestamp[4:6]) - 1, int(timestamp[:4]))
    assert "Multispectral" in status

    with rasterio.open(IMAGE_EXAMPLES[index]["tif"]) as dataset:
        assert dataset.count == 12
        assert dataset.crs is not None
        assert list(dataset.descriptions) == metadata["band_names"]


@pytest.mark.parametrize("index", range(4, 7))
def test_aigc_rgb_examples_clear_multispectral_state(index):
    preview, multiband, metadata, status = select_image_example(index)

    assert preview.width > 0 and preview.height > 0
    assert multiband is None
    assert metadata is None
    assert status == "Selected AIGC RGB example."


def test_first_multispectral_example_uses_expected_circular_farmland_product():
    _preview, _multiband, metadata, _status = select_image_example(0)

    assert metadata["product_id"] == "S2A_MSIL2A_20200124T074211_N9999_R092_T38RLR_20230127T183552"
