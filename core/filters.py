"""Filter options and application for search results."""

import numpy as np
import pandas as pd


def build_filter_options(
    enable_time=False,
    start_date="2016-01-01",
    end_date="2024-12-31",
    enable_geo=False,
    lat_min=-90,
    lat_max=90,
    lon_min=-180,
    lon_max=180,
):
    """Pack UI filter controls into a single dict for search functions."""
    return {
        "time": {"enabled": enable_time, "start": start_date, "end": end_date},
        "geo": {"enabled": enable_geo, "lat_min": lat_min, "lat_max": lat_max, "lon_min": lon_min, "lon_max": lon_max},
        # Future: "polygon": {"enabled": False, "coordinates": []}
    }


def apply_filters(df_embed, probs, filtered_indices, top_indices, filter_options):
    """Apply post-search filters (time range, geo bounding box, etc.) to retrieval results.

    Returns:
        tuple: (new_filtered_indices, new_top_indices, df_for_geo, probs_for_geo, warnings)
            ``warnings`` is a list of human-readable strings describing filter
            options that were adjusted, invalid, or could not be applied, so
            callers can surface them in the status bar instead of failing silently.
    """
    warnings = []
    if not filter_options:
        return filtered_indices, top_indices, df_embed, probs, warnings

    global_mask = np.ones(len(df_embed), dtype=bool)

    # --- Time filter ---
    time_opts = filter_options.get("time", {})
    if time_opts.get("enabled"):
        try:
            timestamps = pd.to_datetime(df_embed["timestamp"])
            start = pd.to_datetime(time_opts["start"])
            end = pd.to_datetime(time_opts["end"])
            if start > end:
                warnings.append(
                    f"Time filter start date ({start.date()}) is after end date ({end.date()}); no records can match."
                )
            end = end + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            time_mask = ((timestamps >= start) & (timestamps <= end)).values
            global_mask &= time_mask
            print(f"Time filter: {time_mask.sum()}/{len(time_mask)} records in range [{start}, {end}]")
        except Exception as e:
            warnings.append(
                f"Failed to parse time filter dates (start={time_opts.get('start')!r}, end={time_opts.get('end')!r}); "
                "time filter not applied."
            )
            print(f"Time filter parse error, skipping: {e}")

    # --- Geo bounding box filter ---
    geo_opts = filter_options.get("geo", {})
    if geo_opts.get("enabled"):
        try:
            lats = pd.to_numeric(df_embed["centre_lat"], errors="coerce").values
            lons = pd.to_numeric(df_embed["centre_lon"], errors="coerce").values
            g_lat_min, g_lat_max = float(geo_opts.get("lat_min", -90)), float(geo_opts.get("lat_max", 90))
            g_lon_min, g_lon_max = float(geo_opts.get("lon_min", -180)), float(geo_opts.get("lon_max", 180))

            if g_lat_min > g_lat_max:
                g_lat_min, g_lat_max = g_lat_max, g_lat_min
                warnings.append(f"Geo filter lat_min > lat_max; swapped to [{g_lat_min}, {g_lat_max}].")

            lat_mask = (lats >= g_lat_min) & (lats <= g_lat_max)
            if g_lon_min > g_lon_max:
                # Box crosses the ±180° antimeridian: match the union of the two ranges.
                lon_mask = (lons >= g_lon_min) | (lons <= g_lon_max)
                warnings.append(
                    f"Geo filter lon_min ({g_lon_min}) > lon_max ({g_lon_max}); interpreted as crossing the "
                    f"±180° antimeridian ([{g_lon_min}, 180] ∪ [-180, {g_lon_max}])."
                )
            else:
                lon_mask = (lons >= g_lon_min) & (lons <= g_lon_max)
            geo_mask = lat_mask & lon_mask
            global_mask &= geo_mask
            print(
                f"Geo filter: {geo_mask.sum()}/{len(geo_mask)} records in box [{g_lat_min},{g_lat_max}] x [{g_lon_min},{g_lon_max}]"
            )
        except Exception as e:
            warnings.append(f"Failed to parse geo filter bounds; geo filter not applied. ({e})")
            print(f"Geo filter error, skipping: {e}")

    # --- Polygon filter (future) ---
    # poly_opts = filter_options.get("polygon", {})
    # if poly_opts.get("enabled"):
    #     from shapely.geometry import Point, Polygon
    #     poly = Polygon(poly_opts["coordinates"])
    #     global_mask &= np.array([poly.contains(Point(lo, la)) for la, lo in zip(lats, lons)])

    # If nothing was filtered out, return limited results (top_indices + filtered_indices)
    if global_mask.all():
        df_for_geo = df_embed.iloc[filtered_indices]
        probs_for_geo = probs[filtered_indices]
        return filtered_indices, top_indices, df_for_geo, probs_for_geo, warnings

    print(f"Filter applied: {global_mask.sum()}/{len(global_mask)} records pass all filters")

    # Apply global mask
    new_filtered_indices = filtered_indices[global_mask[filtered_indices]]
    new_top_indices = top_indices[global_mask[top_indices]]
    df_for_geo = df_embed.iloc[new_filtered_indices]
    probs_for_geo = probs[new_filtered_indices]

    print(f"After filter: {len(new_filtered_indices)} filtered, {len(new_top_indices)} top indices")

    return new_filtered_indices, new_top_indices, df_for_geo, probs_for_geo, warnings
