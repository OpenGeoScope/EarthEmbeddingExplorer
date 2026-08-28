"""Shared spatiotemporal metadata handling for Clay v1.5."""

from __future__ import annotations

import math
from datetime import date, datetime

import numpy as np
from pyproj import CRS, Transformer

MISSING_SOURCE = "missing_zero_fallback"

_TIME_TAG_KEYS = (
    "TIFFTAG_DATETIME",
    "ACQUISITIONDATETIME",
    "ACQUISITION_DATE",
    "DATETIME",
    "DATE_TIME",
    "SENSING_TIME",
)


def parse_timestamp(value):
    """Parse common Major-TOM and GeoTIFF timestamps, returning ``None`` on failure."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    if isinstance(value, np.datetime64):
        if np.isnat(value):
            return None
        value = np.datetime_as_string(value, unit="us")
    if isinstance(value, float) and math.isnan(value):
        return None

    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "none"}:
        return None

    normalized = text.removesuffix("Z").replace(" UTC", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        pass

    for fmt in (
        "%Y%m%dT%H%M%S",
        "%Y:%m:%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def encode_clay_time(value):
    """Return Clay's ISO-week/hour cyclical encoding or ``None``."""
    parsed = parse_timestamp(value)
    if parsed is None:
        return None
    week_angle = parsed.isocalendar().week * 2 * math.pi / 52
    hour_angle = parsed.hour * 2 * math.pi / 24
    return np.asarray(
        [math.sin(week_angle), math.cos(week_angle), math.sin(hour_angle), math.cos(hour_angle)],
        dtype=np.float32,
    )


def encode_clay_latlon(lat, lon):
    """Return Clay's latitude/longitude cyclical encoding or ``None``."""
    try:
        lat = float(lat)
        lon = float(lon)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(lat) or not math.isfinite(lon) or not -90 <= lat <= 90 or not -180 <= lon <= 180:
        return None

    lat_angle = math.radians(lat)
    lon_angle = math.radians(lon)
    return np.asarray(
        [math.sin(lat_angle), math.cos(lat_angle), math.sin(lon_angle), math.cos(lon_angle)],
        dtype=np.float32,
    )


def timestamp_from_tiff_tags(tags):
    """Return the first parseable timestamp stored in GeoTIFF tags."""
    if not tags:
        return None
    normalized = {str(key).upper(): value for key, value in tags.items()}
    for key in _TIME_TAG_KEYS:
        value = normalized.get(key)
        if parse_timestamp(value) is not None:
            return value
    return None


def wgs84_centroid(bounds, crs):
    """Transform a raster-bounds midpoint into ``(latitude, longitude)``."""
    if bounds is None or crs is None:
        return None
    try:
        left, bottom, right, top = bounds
        x = (float(left) + float(right)) / 2
        y = (float(bottom) + float(top)) / 2
        transformer = Transformer.from_crs(CRS.from_user_input(crs), CRS.from_epsg(4326), always_xy=True)
        lon, lat = transformer.transform(x, y)
    except (TypeError, ValueError):
        return None
    if encode_clay_latlon(lat, lon) is None:
        return None
    return float(lat), float(lon)


def resolve_clay_metadata(time_candidates=(), latlon_candidates=()):
    """Resolve Clay inputs by precedence and record their provenance."""
    time_input = None
    time_value = None
    time_source = MISSING_SOURCE
    for value, source in time_candidates:
        encoded = encode_clay_time(value)
        if encoded is not None:
            time_input = encoded
            time_value = value
            time_source = source
            break

    latlon_input = None
    latitude = None
    longitude = None
    latlon_source = MISSING_SOURCE
    for lat, lon, source in latlon_candidates:
        encoded = encode_clay_latlon(lat, lon)
        if encoded is not None:
            latlon_input = encoded
            latitude = float(lat)
            longitude = float(lon)
            latlon_source = source
            break

    missing_fields = []
    if time_input is None:
        time_input = np.zeros(4, dtype=np.float32)
        missing_fields.append("time")
    if latlon_input is None:
        latlon_input = np.zeros(4, dtype=np.float32)
        missing_fields.extend(["latitude", "longitude"])

    return {
        "timestamp": time_value,
        "latitude": latitude,
        "longitude": longitude,
        "clay_time_input": time_input,
        "clay_latlon_input": latlon_input,
        "clay_time_input_source": time_source,
        "clay_latlon_input_source": latlon_source,
        "clay_missing_metadata": missing_fields,
    }


def clay_metadata_status(metadata):
    """Build a concise user-facing status line for resolved Clay metadata."""
    missing = metadata.get("clay_missing_metadata", []) if metadata else ["time", "latitude", "longitude"]
    if not missing:
        return "Clay metadata input: time + latitude/longitude."
    labels = ", ".join(missing)
    return f"Clay metadata missing: {labels}; zero encoding used for missing input."
