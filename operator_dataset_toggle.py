from __future__ import annotations

import base64
import json
import math
import re
import warnings
import zlib
from collections import defaultdict
from datetime import datetime, timedelta

import h3
import numpy as np
import pandas as pd


RENAME_DIRECT: dict[str, str] = {
    "vehicle_start_latitude": "start_lat",
    "vehicle_start_longitude": "start_lng",
    "vehicle_end_latitude": "end_lat",
    "vehicle_end_longitude": "end_lng",
    "start_date": "start_time",
    "end_date": "end_time",
}

VEHICLE_ID_ALIASES: list[str] = [
    "vehicle_id",
    "vehicleid",
    "vehicle",
    "scooter_id",
    "bike_id",
    "vehicle_identifier",
    "qrcode",
    "qr_code",
]

START_TIME_ALIASES: list[str] = [
    "start_time",
    "start_time_local",
    "start_date",
    "time_ride_start_local",
    "time_ride_start",
    "started_at",
    "start_datetime",
    "starttime",
    "timestamp_start",
]

END_TIME_ALIASES: list[str] = [
    "end_time",
    "end_time_local",
    "end_date",
    "time_ride_end_local",
    "time_ride_end",
    "ended_at",
    "end_datetime",
    "endtime",
    "timestamp_end",
]

START_LAT_ALIASES: list[str] = [
    "start_lat",
    "vehicle_start_latitude",
    "lat_start",
    "latstart",
    "start_latitude",
    "startlatitude",
    "pickup_lat",
    "latpickup",
]

START_LNG_ALIASES: list[str] = [
    "start_lng",
    "start_lon",
    "vehicle_start_longitude",
    "start_long",
    "lng_start",
    "lngstart",
    "start_longitude",
    "startlongitude",
    "pickup_lng",
    "lon_start",
    "long_start",
    "lngpickup",
]

END_LAT_ALIASES: list[str] = [
    "end_lat",
    "vehicle_end_latitude",
    "lat_end",
    "latend",
    "end_latitude",
    "endlatitude",
    "dropoff_lat",
    "latdropoff",
]

END_LNG_ALIASES: list[str] = [
    "end_lng",
    "end_lon",
    "vehicle_end_longitude",
    "end_long",
    "lng_end",
    "lngend",
    "end_longitude",
    "endlongitude",
    "dropoff_lng",
    "lon_end",
    "long_end",
    "lngdropoff",
]

START_H3_ALIASES: list[str] = ["start_h3", "start_loc_h3_9", "start_h3_9"]
END_H3_ALIASES: list[str] = ["end_h3", "end_loc_h3_9", "end_h3_9"]

START_LOC_ALIASES: list[str] = ["start_loc", "start_location", "start_geojson"]
END_LOC_ALIASES: list[str] = ["end_loc", "end_location", "end_geojson"]

CANONICAL_FIELD_ALIASES: dict[str, list[str]] = {
    "vehicle_id": VEHICLE_ID_ALIASES,
    "start_time": START_TIME_ALIASES,
    "end_time": END_TIME_ALIASES,
    "start_lat": START_LAT_ALIASES,
    "start_lng": START_LNG_ALIASES,
    "end_lat": END_LAT_ALIASES,
    "end_lng": END_LNG_ALIASES,
    "start_loc": START_LOC_ALIASES,
    "end_loc": END_LOC_ALIASES,
    "start_h3": START_H3_ALIASES,
    "end_h3": END_H3_ALIASES,
}

REQUIRED_BASE_FIELDS: tuple[str, ...] = ("vehicle_id", "start_time", "end_time")
REQUIRED_LATLNG_FIELDS: tuple[str, ...] = (
    "start_lat",
    "start_lng",
    "end_lat",
    "end_lng",
)
REQUIRED_GEOJSON_FIELDS: tuple[str, ...] = ("start_loc", "end_loc")


def clean_numeric_series(series: pd.Series) -> pd.Series:
    """Coerce a coordinate/numeric column to float, repairing corrupted values.

    Some operator exports contain stray whitespace injected inside values
    (e.g. ``"5 1.961097"`` instead of ``51.961097``), which makes pandas load
    the whole column as strings. Numbers never legitimately contain
    whitespace, so we strip it from string values before converting.
    """
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    stripped = series.astype(str).str.replace(r"\s+", "", regex=True)
    return pd.to_numeric(stripped, errors="coerce")


# Matches a date + time-of-day inside a timestamp string whose whitespace has
# been removed, e.g. "2025-05-0104:49:33.563UTC" -> ("2025-05-01", "04:49:33.563").
_COMPACT_TS_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2})T?(\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)"
)


def _repair_timestamp_strings(series: pd.Series) -> pd.Series:
    """Last-resort parse of timestamps corrupted by stray internal spaces.

    Handles values like ``"2025-05-01 0 4:49:33.563 UT C"``: all whitespace is
    removed and the date/time components are re-extracted by pattern, so a
    space anywhere (inside the tz suffix, the time digits, or the date) no
    longer turns the row into NaT.
    """
    compact = series.astype(str).str.replace(r"\s+", "", regex=True)
    parts = compact.str.extract(_COMPACT_TS_RE)
    repaired = parts[0].str.cat(parts[1], sep=" ")
    # format="mixed": repaired values may or may not carry fractional seconds;
    # letting pandas infer one format from the first row would NaT the rest.
    return pd.to_datetime(repaired, format="mixed", errors="coerce", utc=True)


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _extract_coords(value) -> tuple[float, float]:
    try:
        if pd.isna(value):
            return (np.nan, np.nan)
        if isinstance(value, (dict, list)):
            obj = value
        else:
            obj = json.loads(value)
        coords = None
        if isinstance(obj, dict):
            coords = obj.get("coordinates")
        elif isinstance(obj, list):
            coords = obj
        if not coords or len(coords) < 2:
            return (np.nan, np.nan)
        lng, lat = float(coords[0]), float(coords[1])
        return (lat, lng)
    except Exception:
        return (np.nan, np.nan)


def standardize_rides_dataframe(df_rides: pd.DataFrame) -> pd.DataFrame:
    df = df_rides.copy()
    df.columns = (
        pd.Index(df.columns)
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    for src, dst in RENAME_DIRECT.items():
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})

    if "start_loc" in df.columns and (
        "start_lat" not in df.columns or "start_lng" not in df.columns
    ):
        start_coords = df["start_loc"].apply(_extract_coords)
        df["start_lat"] = start_coords.apply(lambda x: x[0])
        df["start_lng"] = start_coords.apply(lambda x: x[1])

    if "end_loc" in df.columns and (
        "end_lat" not in df.columns or "end_lng" not in df.columns
    ):
        end_coords = df["end_loc"].apply(_extract_coords)
        df["end_lat"] = end_coords.apply(lambda x: x[0])
        df["end_lng"] = end_coords.apply(lambda x: x[1])

    vehicle_id = _pick_col(df, VEHICLE_ID_ALIASES)
    start_time = _pick_col(df, START_TIME_ALIASES)
    end_time = _pick_col(df, END_TIME_ALIASES)
    start_lat = _pick_col(df, START_LAT_ALIASES)
    start_lng = _pick_col(df, START_LNG_ALIASES)
    end_lat = _pick_col(df, END_LAT_ALIASES)
    end_lng = _pick_col(df, END_LNG_ALIASES)
    start_h3 = _pick_col(df, START_H3_ALIASES)
    end_h3 = _pick_col(df, END_H3_ALIASES)

    rename_map = {}
    if vehicle_id and vehicle_id != "vehicle_id":
        rename_map[vehicle_id] = "vehicle_id"
    if start_time and start_time != "start_time":
        rename_map[start_time] = "start_time"
    if end_time and end_time != "end_time":
        rename_map[end_time] = "end_time"
    if start_lat and start_lat != "start_lat":
        rename_map[start_lat] = "start_lat"
    if start_lng and start_lng != "start_lng":
        rename_map[start_lng] = "start_lng"
    if end_lat and end_lat != "end_lat":
        rename_map[end_lat] = "end_lat"
    if end_lng and end_lng != "end_lng":
        rename_map[end_lng] = "end_lng"
    if start_h3 and start_h3 != "start_h3" and (
        "start_lat" not in df.columns or "start_lng" not in df.columns
    ):
        rename_map[start_h3] = "start_h3"
    if end_h3 and end_h3 != "end_h3" and (
        "end_lat" not in df.columns or "end_lng" not in df.columns
    ):
        rename_map[end_h3] = "end_h3"

    if rename_map:
        df = df.rename(columns=rename_map)

    required = [
        "vehicle_id",
        "start_time",
        "end_time",
        "start_lat",
        "start_lng",
        "end_lat",
        "end_lng",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        warnings.warn(
            f"df_rides is missing expected columns: {missing}. Available columns: {list(df.columns)}"
        )

    # Repair + coerce coordinate columns up front. Corrupted exports (stray
    # spaces inside values) otherwise leave these columns as mixed str/float,
    # which crashes min()/max() in the auto search-area derivation.
    for col in ("start_lat", "start_lng", "end_lat", "end_lng"):
        if col in df.columns:
            df[col] = clean_numeric_series(df[col])
    return df


def _parse_time_series(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    if parsed.isna().any():
        try:
            parsed2 = pd.to_datetime(
                series.astype(str).str.replace(" UTC", "", regex=False),
                errors="coerce",
                utc=True,
            )
            parsed = parsed.fillna(parsed2)
        except Exception:
            pass

        try:
            mask = parsed.isna()
            if bool(mask.any()):
                parsed.loc[mask] = pd.to_datetime(
                    series.loc[mask].astype(str), errors="coerce", utc=True
                )
        except Exception:
            pass

        try:
            mask = parsed.isna()
            if bool(mask.any()):
                parsed.loc[mask] = _repair_timestamp_strings(series.loc[mask])
        except Exception:
            pass
    return parsed


def add_datetime_and_dwell_columns(
    df_rides: pd.DataFrame, dwell_same_loc_radius_m: float = 50
) -> tuple[pd.DataFrame, pd.Timestamp]:
    df = df_rides.copy()
    raw_start_time = df["start_time"].copy()
    raw_end_time = df["end_time"].copy()

    df_st = _parse_time_series(raw_start_time)
    df_et = _parse_time_series(raw_end_time)

    n_st_nat = int(df_st.isna().sum())
    n_et_nat = int(df_et.isna().sum())
    if n_st_nat or n_et_nat:
        def sample_bad(mask, raw_series, label):
            cols = [c for c in ["ride_id", "vehicle_id"] if c in df.columns]
            try:
                sample = (
                    df.loc[mask, cols].copy()
                    if cols
                    else df.loc[mask].iloc[0:0].copy()
                )
                sample[label] = raw_series.loc[mask].astype(str).head(5).values
                return sample.head(5).to_dict(orient="records")
            except Exception:
                return []

        samples = {
            "start_time": sample_bad(df_st.isna(), raw_start_time, "start_time_raw"),
            "end_time": sample_bad(df_et.isna(), raw_end_time, "end_time_raw"),
        }
        warnings.warn(
            f"Datetime parse failures: start_time NaT={n_st_nat:,}, end_time NaT={n_et_nat:,}. "
            f"These rows will be dropped by TIME_PERIOD filters. Samples: {samples}"
        )

    df["start_time"] = df_st.dt.tz_convert(None)
    df["end_time"] = df_et.dt.tz_convert(None)
    dataset_end_max = df["end_time"].max()

    df = df.sort_values(["vehicle_id", "start_time"]).reset_index(drop=True)
    df["next_start_time"] = df.groupby("vehicle_id")["start_time"].shift(-1)
    df["next_start_lat"] = (
        df.groupby("vehicle_id")["start_lat"].shift(-1)
        if "start_lat" in df.columns
        else np.nan
    )
    df["next_start_lng"] = (
        df.groupby("vehicle_id")["start_lng"].shift(-1)
        if "start_lng" in df.columns
        else np.nan
    )

    df["dwell_to_next_start_mins"] = (
        (df["next_start_time"] - df["end_time"]).dt.total_seconds() / 60.0
    )
    df.loc[df["dwell_to_next_start_mins"] < 0, "dwell_to_next_start_mins"] = np.nan

    radius = float(dwell_same_loc_radius_m)
    r_earth = 6_371_000.0
    lat1 = np.radians(
        pd.to_numeric(df.get("end_lat", np.nan), errors="coerce").to_numpy()
    )
    lon1 = np.radians(
        pd.to_numeric(df.get("end_lng", np.nan), errors="coerce").to_numpy()
    )
    lat2 = np.radians(
        pd.to_numeric(df.get("next_start_lat", np.nan), errors="coerce").to_numpy()
    )
    lon2 = np.radians(
        pd.to_numeric(df.get("next_start_lng", np.nan), errors="coerce").to_numpy()
    )
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    dist_m = r_earth * c
    within_radius = (dist_m <= radius) & df["next_start_time"].notna().to_numpy()
    df["dwell_same_loc_mins"] = np.where(
        within_radius, df["dwell_to_next_start_mins"], np.nan
    )
    # End -> next-start displacement (m). Large values are relocation/van-move
    # proxies used by the pre/post deployment comparison.
    df["dist_to_next_start_m"] = np.where(
        df["next_start_time"].notna().to_numpy(), dist_m, np.nan
    )
    return df, dataset_end_max


def compute_map_search_area(
    df_rides: pd.DataFrame,
    map_search_area_mode: str = "auto",
    map_search_area_manual: list[float] | tuple[float, float, float, float] | None = None,
) -> list[float]:
    if map_search_area_mode == "manual":
        if not map_search_area_manual:
            raise ValueError("MAP_SEARCH_AREA_MODE='manual' requires MAP_SEARCH_AREA_MANUAL")
        return [float(x) for x in map_search_area_manual]

    cols = []
    if "end_lat" in df_rides.columns and "end_lng" in df_rides.columns:
        cols.append(
            df_rides[["end_lat", "end_lng"]].rename(
                columns={"end_lat": "lat", "end_lng": "lng"}
            )
        )
    if "start_lat" in df_rides.columns and "start_lng" in df_rides.columns:
        cols.append(
            df_rides[["start_lat", "start_lng"]].rename(
                columns={"start_lat": "lat", "start_lng": "lng"}
            )
        )
    if not cols:
        raise ValueError("No start/end lat/lng columns available to auto-derive MAP_SEARCH_AREA")

    coords = pd.concat(cols, axis=0, ignore_index=True)
    coords["lat"] = clean_numeric_series(coords["lat"])
    coords["lng"] = clean_numeric_series(coords["lng"])
    coords = coords[coords["lat"].notna() & coords["lng"].notna()]
    if coords.empty:
        raise ValueError("No non-null coordinates available to auto-derive MAP_SEARCH_AREA")

    pad = 0.001
    return [
        float(coords["lat"].min() - pad),
        float(coords["lat"].max() + pad),
        float(coords["lng"].min() - pad),
        float(coords["lng"].max() + pad),
    ]


def h3_latlng_to_cell(lat, lng, resolution):
    if hasattr(h3, "latlng_to_cell"):
        return h3.latlng_to_cell(lat, lng, resolution)
    return h3.geo_to_h3(lat, lng, resolution)


def h3_cell_to_latlng(cell):
    if hasattr(h3, "cell_to_latlng"):
        return h3.cell_to_latlng(cell)
    return h3.h3_to_geo(cell)


def h3_cell_to_boundary(cell):
    if hasattr(h3, "cell_to_boundary"):
        return h3.cell_to_boundary(cell)
    return h3.h3_to_geo_boundary(cell)


def add_h3_columns(
    df_rides: pd.DataFrame,
    map_search_area: list[float],
    h3_resolution: int,
) -> pd.DataFrame:
    df = df_rides.copy()
    for col in ["start_lat", "start_lng", "end_lat", "end_lng"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    min_lat, max_lat, min_lng, max_lng = map_search_area

    def in_bounds(lat, lng):
        return (
            lat is not None
            and lng is not None
            and not (pd.isna(lat) or pd.isna(lng))
            and lat >= min_lat
            and lat <= max_lat
            and lng >= min_lng
            and lng <= max_lng
        )

    if "end_lat" in df.columns and "end_lng" in df.columns:
        df["end_in_bounds"] = [
            in_bounds(lat, lng)
            for lat, lng in zip(df["end_lat"].tolist(), df["end_lng"].tolist())
        ]
        df["end_h3"] = None
        end_idx = df["end_in_bounds"]
        df.loc[end_idx, "end_h3"] = [
            h3_latlng_to_cell(lat, lng, h3_resolution)
            for lat, lng in zip(df.loc[end_idx, "end_lat"], df.loc[end_idx, "end_lng"])
        ]
    else:
        warnings.warn("end_lat/end_lng not found; interval metrics will be limited.")

    if "start_lat" in df.columns and "start_lng" in df.columns:
        df["start_in_bounds"] = [
            in_bounds(lat, lng)
            for lat, lng in zip(df["start_lat"].tolist(), df["start_lng"].tolist())
        ]
        df["start_h3"] = None
        start_idx = df["start_in_bounds"]
        df.loc[start_idx, "start_h3"] = [
            h3_latlng_to_cell(lat, lng, h3_resolution)
            for lat, lng in zip(df.loc[start_idx, "start_lat"], df.loc[start_idx, "start_lng"])
        ]
    else:
        warnings.warn("start_lat/start_lng not found; net-accumulation metrics will be limited.")

    return df


def reward_function(optimiser_weights: dict[str, float]):
    return lambda a, b: (
        a * optimiser_weights.get("chargecount", 0.0)
        + b * optimiser_weights.get("fleetcoverage", 1.0)
    )


def populate_list_intervals(
    time_period: tuple[datetime, datetime], cycle_duration: int
) -> list[tuple[datetime, datetime]]:
    result = []
    interval_start = time_period[0]
    interval_end = time_period[0] + timedelta(days=cycle_duration)
    while interval_start < time_period[1]:
        result.append((interval_start, interval_end))
        interval_start = interval_end
        interval_end = interval_end + timedelta(days=cycle_duration)
    if result:
        result[-1] = (result[-1][0], time_period[1])
    return result


def get_unique_vehicles(ridelist, visited):
    visited_set = visited if isinstance(visited, set) else set(visited)
    seen = set()
    unique_list = []
    for item in ridelist:
        if item not in seen and item not in visited_set:
            seen.add(item)
            unique_list.append(item)
    return unique_list


def compute_time_period(
    df_rides: pd.DataFrame,
    map_search_area: list[float],
    dataset_end_max,
    time_period_mode: str,
    auto_start: bool,
    auto_start_p75_frac: float,
    time_period_start_fixed: datetime,
    time_period_end_fixed: datetime | None = None,
) -> tuple[tuple[datetime, datetime], datetime, datetime]:
    time_period_start = time_period_start_fixed
    try:
        dataset_end = dataset_end_max
        if dataset_end is None:
            dataset_end = pd.to_datetime(df_rides.get("end_time", pd.NaT), errors="coerce").max()

        if pd.isna(dataset_end):
            warnings.warn(
                "DATASET_END_MAX is NaT; falling back to fixed end date. "
                "(Check datetime parsing warnings above.)"
            )
            time_period_end = time_period_end_fixed or datetime(2025, 10, 30)
        else:
            time_period_end = dataset_end + timedelta(microseconds=1)

        try:
            min_lat, max_lat, min_lng, max_lng = map_search_area
            end_lat = pd.to_numeric(
                df_rides.get("end_lat", pd.Series([], dtype="float64")), errors="coerce"
            )
            end_lng = pd.to_numeric(
                df_rides.get("end_lng", pd.Series([], dtype="float64")), errors="coerce"
            )
            end_in_bounds = (
                end_lat.notna()
                & end_lng.notna()
                & (end_lat >= min_lat)
                & (end_lat <= max_lat)
                & (end_lng >= min_lng)
                & (end_lng <= max_lng)
            )
            end_time = pd.to_datetime(df_rides.get("end_time", pd.NaT), errors="coerce")
            tmp = pd.DataFrame({"end_time": end_time, "in_bounds": end_in_bounds})
            tmp = tmp[tmp["end_time"].notna() & tmp["in_bounds"]].copy()
            tmp["day"] = tmp["end_time"].dt.floor("D")
            daily = tmp.groupby("day").size().sort_index()
            nonzero = daily[daily > 0]
        except Exception:
            nonzero = pd.Series([], dtype="int64")

        if time_period_mode == "dataset":
            if len(nonzero):
                time_period_start = pd.Timestamp(nonzero.index.min()).to_pydatetime()
            else:
                time_period_start = time_period_start_fixed
                warnings.warn(
                    "TIME_PERIOD_MODE='dataset': no non-zero in-bounds days; using TIME_PERIOD_START_FIXED"
                )
        elif time_period_mode == "manual":
            time_period_start = time_period_start_fixed
            if time_period_end_fixed is not None:
                time_period_end = time_period_end_fixed
        else:
            if auto_start and len(nonzero):
                p75 = float(nonzero.quantile(0.75))
                threshold = float(auto_start_p75_frac) * p75
                cand = nonzero[nonzero >= threshold]
                if len(cand):
                    time_period_start = pd.Timestamp(cand.index.min()).to_pydatetime()
                else:
                    time_period_start = pd.Timestamp(nonzero.index.min()).to_pydatetime()
                    warnings.warn(
                        f"AUTO_START found no day >= threshold ({threshold:.2f}); "
                        f"using first non-zero day {time_period_start.date()}"
                    )
                warnings.warn(
                    f"AUTO_START enabled. P75(non-zero daily ends)={p75:.1f}, "
                    f"threshold={threshold:.1f}. Chosen TIME_PERIOD_START={time_period_start.date()}."
                )
            else:
                time_period_start = time_period_start_fixed
                warnings.warn("AUTO_START: insufficient data; using TIME_PERIOD_START_FIXED")

        time_period = (time_period_start, time_period_end)
    except Exception as exc:
        warnings.warn(
            f"Failed to set TIME_PERIOD from dataset: {exc}. Falling back to fixed dates."
        )
        time_period = (
            time_period_start_fixed,
            time_period_end_fixed or datetime(2025, 10, 30),
        )
        time_period_start, time_period_end = time_period

    return time_period, time_period_start, time_period_end


def compute_station_selection(
    df_rides: pd.DataFrame,
    list_time_intervals: list[tuple[datetime, datetime]],
    i_sigma: int,
    i_s: int,
    optimiser_weights: dict[str, float],
) -> tuple[dict[str, float], list[tuple[str, float]], list[str]]:
    dict_scores = defaultdict(int)
    score_weights = range(1, i_sigma + 1)[::-1]
    reward = reward_function(optimiser_weights)

    for interval in list_time_intervals:
        start_time, end_time = interval
        df_interval = df_rides[
            (df_rides["end_time"] >= start_time) & (df_rides["end_time"] < end_time)
        ]
        if "end_h3" in df_interval.columns and "end_in_bounds" in df_interval.columns:
            df_end = df_interval[df_interval["end_in_bounds"] & df_interval["end_h3"].notna()]
            rides_by_hex = df_end.groupby("end_h3")["vehicle_id"].apply(list).to_dict()
        else:
            rides_by_hex = {}

        unique_vehicles_count = int(df_interval["vehicle_id"].nunique())
        visited_ids = set()
        ordered_locations = []

        for _ in range(i_sigma):
            highest = {"location": None, "value": 0, "visited": []}
            for hex_id, vehicle_ids in rides_by_hex.items():
                chargecount = len(vehicle_ids)
                unique_vehicles = get_unique_vehicles(vehicle_ids, visited_ids)
                fleetcoverage = (
                    len(unique_vehicles) / unique_vehicles_count * 100.0
                    if unique_vehicles_count
                    else 0.0
                )
                value = reward(chargecount, fleetcoverage)
                if value > highest["value"]:
                    highest["location"] = hex_id
                    highest["value"] = value
                    highest["visited"] = unique_vehicles

            if highest["location"] is not None:
                ordered_locations.append(highest)
                visited_ids.update(highest["visited"])
                rides_by_hex[highest["location"]] = []

        for idx, rec in enumerate(ordered_locations):
            dict_scores[rec["location"]] += score_weights[idx]

    optimised_locations = [(hex_id, score) for hex_id, score in dict_scores.items() if score != 0]
    sorted_locations = sorted(optimised_locations, key=lambda x: -x[1])[:i_s]
    predefined_locations = [row[0] for row in sorted_locations]
    return dict(dict_scores), sorted_locations, predefined_locations


def compute_hex_metrics(
    df_rides: pd.DataFrame,
    time_period: tuple[datetime, datetime],
    predefined_locations_h3: list[str],
    dict_scores: dict[str, float],
    cycle_duration: int,
    dwell_same_loc_radius_m: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_cells = set(predefined_locations_h3)
    period_start, period_end = time_period

    df_end_tp = df_rides[
        (df_rides["end_time"] >= period_start) & (df_rides["end_time"] < period_end)
    ].copy()
    df_start_tp = df_rides[
        (df_rides["start_time"] >= period_start) & (df_rides["start_time"] < period_end)
    ].copy()

    if "end_h3" in df_end_tp.columns:
        end_mask = (
            df_end_tp["end_in_bounds"]
            if "end_in_bounds" in df_end_tp.columns
            else pd.Series(True, index=df_end_tp.index)
        )
        df_end_all = df_end_tp[end_mask & df_end_tp["end_h3"].notna()].copy()
    else:
        df_end_all = df_end_tp.iloc[0:0].copy()

    if len(df_end_all):
        g_end = df_end_all.groupby("end_h3")
        end_counts = g_end.size().rename("end_rides")
        dwell_col = (
            "dwell_same_loc_mins"
            if "dwell_same_loc_mins" in df_end_all.columns
            else "dwell_to_next_start_mins"
        )
        med_dwell = g_end[dwell_col].median().rename("median_dwell_mins")
        ge_30 = g_end[dwell_col].apply(lambda s: (s >= 30).sum()).rename("dwells_ge_30")
        ge_60 = g_end[dwell_col].apply(lambda s: (s >= 60).sum()).rename("dwells_ge_60")
        ge_120 = g_end[dwell_col].apply(lambda s: (s >= 120).sum()).rename("dwells_ge_120")
        dwell_defined = g_end[dwell_col].apply(lambda s: s.notna().sum()).rename("dwells_defined")

        df_hex_metrics_all = pd.concat(
            [end_counts, med_dwell, ge_30, ge_60, ge_120, dwell_defined], axis=1
        )
        df_hex_metrics_all["share_dwells_ge_60_pct"] = np.where(
            df_hex_metrics_all["dwells_defined"] > 0,
            (df_hex_metrics_all["dwells_ge_60"] / df_hex_metrics_all["dwells_defined"])
            * 100.0,
            np.nan,
        )

        df_daily = df_end_all[["end_h3", "end_time", dwell_col]].copy()
        df_daily = df_daily.rename(columns={dwell_col: "dwell_mins"})
        df_daily["end_day"] = df_daily["end_time"].dt.floor("D")
        df_hex_metrics_all["dwell_same_loc_radius_m"] = float(dwell_same_loc_radius_m)

        start_day = pd.Timestamp(period_start).floor("D")
        end_day_exclusive = pd.Timestamp(period_end).floor("D")
        all_days = pd.date_range(start_day, end_day_exclusive - pd.Timedelta(days=1), freq="D")
        weekday_days = [d for d in all_days if d.dayofweek < 5]
        weekend_days = [d for d in all_days if d.dayofweek >= 5]

        if len(all_days):
            daily_counts = (
                df_daily.groupby(["end_h3", "end_day"])
                .size()
                .rename("end_rides_day")
                .reset_index()
            )
            daily_pivot = (
                daily_counts.pivot(index="end_h3", columns="end_day", values="end_rides_day")
                .reindex(index=df_hex_metrics_all.index)
                .reindex(columns=all_days)
                .fillna(0)
            )
            end_daily_pivot = daily_pivot
            df_hex_metrics_all = df_hex_metrics_all.join(
                pd.DataFrame({"end_rides_day_avg": end_daily_pivot.mean(axis=1)}),
                how="left",
            )
            df_hex_metrics_all["weekday_end_day_avg"] = (
                end_daily_pivot[weekday_days].mean(axis=1) if len(weekday_days) else np.nan
            )
            df_hex_metrics_all["weekend_end_day_avg"] = (
                end_daily_pivot[weekend_days].mean(axis=1) if len(weekend_days) else np.nan
            )
        else:
            df_hex_metrics_all["end_rides_day_avg"] = np.nan

        for col in [
            "dwells_defined",
            "dwell_same_loc_radius_m",
            "weekday_end_day_avg",
            "weekend_end_day_avg",
        ]:
            if col not in df_hex_metrics_all.columns:
                df_hex_metrics_all[col] = np.nan
    else:
        df_hex_metrics_all = pd.DataFrame()

    if not df_hex_metrics_all.empty and "start_h3" in df_start_tp.columns:
        start_mask = (
            df_start_tp["start_in_bounds"]
            if "start_in_bounds" in df_start_tp.columns
            else pd.Series(True, index=df_start_tp.index)
        )
        df_start_all = df_start_tp[start_mask & df_start_tp["start_h3"].notna()].copy()
        start_counts = df_start_all.groupby("start_h3").size().rename("start_rides")
        df_hex_metrics_all = df_hex_metrics_all.join(start_counts, how="left").fillna(
            {"start_rides": 0}
        )
        df_hex_metrics_all["net_accum"] = (
            df_hex_metrics_all["end_rides"] - df_hex_metrics_all["start_rides"]
        )

        try:
            start_day = pd.Timestamp(period_start).floor("D")
            end_day_exclusive = pd.Timestamp(period_end).floor("D")
            all_days = pd.date_range(start_day, end_day_exclusive - pd.Timedelta(days=1), freq="D")
            if len(all_days):
                df_start_daily = df_start_all[["start_h3", "start_time"]].copy()
                df_start_daily["start_day"] = df_start_daily["start_time"].dt.floor("D")
                start_daily_counts = (
                    df_start_daily.groupby(["start_h3", "start_day"])
                    .size()
                    .rename("start_rides_day")
                    .reset_index()
                )
                start_daily_pivot = (
                    start_daily_counts.pivot(
                        index="start_h3", columns="start_day", values="start_rides_day"
                    )
                    .reindex(index=df_hex_metrics_all.index)
                    .reindex(columns=all_days)
                    .fillna(0)
                )

                df_end_daily_tmp = df_end_all[["end_h3", "end_time"]].copy()
                df_end_daily_tmp["end_day"] = df_end_daily_tmp["end_time"].dt.floor("D")
                end_daily_counts_tmp = (
                    df_end_daily_tmp.groupby(["end_h3", "end_day"])
                    .size()
                    .rename("end_rides_day")
                    .reset_index()
                )
                end_daily_pivot_tmp = (
                    end_daily_counts_tmp.pivot(
                        index="end_h3", columns="end_day", values="end_rides_day"
                    )
                    .reindex(index=df_hex_metrics_all.index)
                    .reindex(columns=all_days)
                    .fillna(0)
                )
                net_daily = end_daily_pivot_tmp - start_daily_pivot
                df_hex_metrics_all["net_trips_day_avg"] = net_daily.mean(axis=1)
            else:
                df_hex_metrics_all["net_trips_day_avg"] = np.nan
        except Exception:
            df_hex_metrics_all["net_trips_day_avg"] = np.nan
    elif not df_hex_metrics_all.empty:
        df_hex_metrics_all["start_rides"] = 0
        df_hex_metrics_all["net_accum"] = df_hex_metrics_all["end_rides"]
        df_hex_metrics_all["net_trips_day_avg"] = np.nan

    if not df_hex_metrics_all.empty:
        score_map = dict(dict_scores)
        df_hex_metrics_all["station_score"] = [
            float(score_map.get(hx, 0)) for hx in df_hex_metrics_all.index
        ]
        df_hex_metrics_all["is_selected"] = [
            hx in selected_cells for hx in df_hex_metrics_all.index
        ]
        rank_df = df_hex_metrics_all[["station_score", "end_rides"]].copy()
        rank_df = rank_df.sort_values(["station_score", "end_rides"], ascending=[False, False])
        df_hex_metrics_all["rank"] = (
            rank_df.reset_index().reset_index().set_index("end_h3")["index"] + 1
        )

        try:
            total_vehicles_period = int(df_end_all["vehicle_id"].nunique())
            if total_vehicles_period <= 0:
                df_hex_metrics_all["incremental_coverage_pct"] = 0.0
                df_hex_metrics_all["cumulative_coverage_pct"] = 0.0
            else:
                hex_to_vehicles = (
                    df_end_all.groupby("end_h3")["vehicle_id"]
                    .apply(lambda s: set(s.dropna().tolist()))
                    .to_dict()
                )
                inc_cov = {}
                cum_cov = {}
                covered = set()
                for hx in rank_df.index.tolist():
                    vehs = hex_to_vehicles.get(hx, set())
                    new = vehs - covered
                    covered |= vehs
                    inc_cov[hx] = (len(new) / total_vehicles_period) * 100.0
                    cum_cov[hx] = (len(covered) / total_vehicles_period) * 100.0
                df_hex_metrics_all["incremental_coverage_pct"] = [
                    float(inc_cov.get(hx, 0.0)) for hx in df_hex_metrics_all.index
                ]
                df_hex_metrics_all["cumulative_coverage_pct"] = [
                    float(cum_cov.get(hx, 0.0)) for hx in df_hex_metrics_all.index
                ]
        except Exception:
            df_hex_metrics_all["incremental_coverage_pct"] = np.nan
            df_hex_metrics_all["cumulative_coverage_pct"] = np.nan

        try:
            intervals = populate_list_intervals(time_period, cycle_duration)
            n_intervals = max(len(intervals), 1)
            sum_cov = {}
            for start_time, end_time in intervals:
                df_i = df_end_all[
                    (df_end_all["end_time"] >= start_time)
                    & (df_end_all["end_time"] < end_time)
                ]
                total_i = int(df_i["vehicle_id"].nunique())
                if total_i <= 0:
                    continue
                per_hex_i = df_i.groupby("end_h3")["vehicle_id"].nunique()
                for hx, n in per_hex_i.items():
                    sum_cov[hx] = sum_cov.get(hx, 0.0) + (float(n) / total_i) * 100.0
            df_hex_metrics_all["avg_5day_vehicle_coverage_pct"] = [
                float(sum_cov.get(hx, 0.0)) / n_intervals for hx in df_hex_metrics_all.index
            ]
        except Exception:
            df_hex_metrics_all["avg_5day_vehicle_coverage_pct"] = np.nan

        df_hex_metrics_all = df_hex_metrics_all.reset_index().rename(columns={"end_h3": "h3"})
        centres = df_hex_metrics_all["h3"].apply(
            lambda hx: h3_cell_to_latlng(hx) if pd.notna(hx) else (np.nan, np.nan)
        )
        df_hex_metrics_all["centre_lat"] = centres.apply(lambda x: x[0])
        df_hex_metrics_all["centre_lng"] = centres.apply(lambda x: x[1])

    if (
        isinstance(df_hex_metrics_all, pd.DataFrame)
        and not df_hex_metrics_all.empty
        and "is_selected" in df_hex_metrics_all.columns
    ):
        df_selected_summary = df_hex_metrics_all[df_hex_metrics_all["is_selected"]].copy()
    else:
        df_selected_summary = pd.DataFrame()

    return df_hex_metrics_all, df_selected_summary


def build_general_stats_data(
    df_rides: pd.DataFrame, time_period: tuple[datetime, datetime]
) -> tuple[dict, str]:
    period_start, period_end = time_period
    df_tp = df_rides[
        (df_rides["end_time"] >= period_start) & (df_rides["end_time"] < period_end)
    ].copy()
    if "end_in_bounds" in df_tp.columns:
        df_tp = df_tp[df_tp["end_in_bounds"]]

    total_rides = len(df_tp)
    unique_vehicles = df_tp["vehicle_id"].nunique() if "vehicle_id" in df_tp.columns else 0
    df_tp["_date"] = pd.to_datetime(df_tp["end_time"], errors="coerce").dt.normalize()
    daily = df_tp.groupby("_date").agg(_rides=("vehicle_id", "size"), _vehs=("vehicle_id", "nunique"))
    daily["_rpv"] = daily["_rides"] / daily["_vehs"].clip(lower=1)
    total_days = len(daily)
    rides_per_day = (total_rides / total_days) if total_days else 0
    ur_rides_day_veh = (rides_per_day / unique_vehicles) if unique_vehicles else 0
    avg_active_veh_day = float(daily["_vehs"].mean()) if len(daily) else 0
    avg_rides_per_vehicle_per_day = daily["_rpv"].mean() if len(daily) else 0
    base_day = pd.Timestamp(period_start).normalize()
    daily_day_idx = [int((d - base_day).days) for d in daily.index]
    daily_rides_arr = daily["_rides"].tolist()
    daily_vehs_arr = daily["_vehs"].tolist()

    veh_day_ranges = []
    if len(df_tp) and "vehicle_id" in df_tp.columns:
        veh_day_groups = df_tp.groupby("vehicle_id")["_date"].agg(["min", "max"])
        veh_day_ranges = [
            [int((row["min"] - base_day).days), int((row["max"] - base_day).days)]
            for _, row in veh_day_groups.iterrows()
        ]

    df_tp["_month"] = pd.to_datetime(df_tp["end_time"], errors="coerce").dt.to_period("M")
    monthly = (
        df_tp.groupby("_month")
        .agg(rides=("vehicle_id", "size"), vehicles=("vehicle_id", "nunique"))
        .reset_index()
    )
    monthly["_month_str"] = monthly["_month"].astype(str)
    monthly["_days"] = monthly["_month"].apply(
        lambda month: min(
            (month.end_time.date() - month.start_time.date()).days + 1,
            (
                pd.Timestamp(period_end).normalize()
                - pd.Timestamp(period_start).normalize()
            ).days,
        )
    )
    monthly["_days"] = monthly["_days"].clip(lower=1)
    monthly["_rides_per_day"] = monthly["rides"] / monthly["_days"]
    monthly_rpv = []
    monthly_avg_vehs = []
    for _, row in monthly.iterrows():
        month = row["_month"]
        mask = daily.index.to_period("M") == month
        monthly_rpv.append(daily.loc[mask, "_rpv"].mean() if mask.any() else 0)
        monthly_avg_vehs.append(float(daily.loc[mask, "_vehs"].mean()) if mask.any() else 0.0)
    monthly["_rides_per_day_per_veh"] = monthly_rpv
    monthly["_avg_active_veh_day"] = monthly_avg_vehs
    monthly["_ur_vs_mu"] = monthly["_rides_per_day"] / monthly["vehicles"].replace(0, np.nan)

    def _fmt_monthly_ur(value) -> str:
        return f"{float(value):.2f}" if pd.notna(value) else "—"

    rows_html = "".join(
        f"<tr><td style='padding:4px 8px;'>{row['_month_str']}</td>"
        f"<td style='padding:4px 8px; font-variant-numeric:tabular-nums;'>{row['rides']:,}</td>"
        f"<td style='padding:4px 8px; font-variant-numeric:tabular-nums;'>{row['vehicles']:,}</td>"
        f"<td style='padding:4px 8px; font-variant-numeric:tabular-nums;'>{row['_rides_per_day']:.1f}</td>"
        f"<td style='padding:4px 8px; font-variant-numeric:tabular-nums;'>{_fmt_monthly_ur(row['_ur_vs_mu'])}</td>"
        f"<td style='padding:4px 8px; font-variant-numeric:tabular-nums;'>{row['_avg_active_veh_day']:.0f}</td></tr>"
        for _, row in monthly.iterrows()
    )

    monthly_list = []
    for _, row in monthly.iterrows():
        monthly_list.append(
            {
                "month": str(row["_month_str"]),
                "rides": int(row["rides"]),
                "vehicles": int(row["vehicles"]),
                "days": float(row["_days"]),
                "rpv_day": float(row["_rides_per_day_per_veh"])
                if pd.notna(row["_rides_per_day_per_veh"])
                else 0.0,
                "avg_active_veh_day": float(row["_avg_active_veh_day"])
                if pd.notna(row["_avg_active_veh_day"])
                else 0.0,
                "ur_vs_monthly_unique": float(row["_ur_vs_mu"])
                if pd.notna(row["_ur_vs_mu"])
                else None,
            }
        )

    general_stats = {
        "total_rides": int(total_rides),
        "unique_vehicles": int(unique_vehicles),
        "total_days": int(total_days),
        "rides_per_day": float(rides_per_day),
        "ur_rides_day_veh": float(ur_rides_day_veh),
        "avg_active_veh_day": float(avg_active_veh_day),
        "avg_rpv_day": float(avg_rides_per_vehicle_per_day),
        "daily_idx": daily_day_idx,
        "daily_rides": daily_rides_arr,
        "daily_vehs": daily_vehs_arr,
        "veh_ranges": veh_day_ranges,
        "monthly": monthly_list,
    }
    return general_stats, rows_html


def _clean_nan(obj):
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_nan(v) for v in obj]
    return obj


def _pack(obj) -> str:
    data = json.dumps(_clean_nan(obj), ensure_ascii=False).encode("utf-8")
    return base64.b64encode(zlib.compress(data)).decode("ascii")


def build_payload(
    df_rides: pd.DataFrame,
    df_hex_metrics_all: pd.DataFrame,
    time_period: tuple[datetime, datetime],
    cycle_duration: int,
    i_sigma: int,
    i_s: int,
    optimiser_weights: dict[str, float],
    dwell_same_loc_radius_m: float,
    general_stats_data: dict,
) -> dict:
    metrics_by_hex = {}
    if isinstance(df_hex_metrics_all, pd.DataFrame) and not df_hex_metrics_all.empty:
        df_tooltip = df_hex_metrics_all
        for _, row in df_tooltip.iterrows():
            key = row.get("h3")
            if key is None or (isinstance(key, float) and np.isnan(key)):
                continue
            metrics_by_hex[str(key)] = {
                "rank": row.get("rank", np.nan),
                "station_score": row.get("station_score", np.nan),
                "is_selected": row.get("is_selected", False),
                "avg_5day_vehicle_coverage_pct": row.get(
                    "avg_5day_vehicle_coverage_pct", np.nan
                ),
                "end_rides": row.get("end_rides", np.nan),
                "start_rides": row.get("start_rides", np.nan),
                "net_accum": row.get("net_accum", np.nan),
                "median_dwell_mins": row.get("median_dwell_mins", np.nan),
                "dwells_ge_30": row.get("dwells_ge_30", np.nan),
                "dwells_ge_60": row.get("dwells_ge_60", np.nan),
                "dwells_ge_120": row.get("dwells_ge_120", np.nan),
                "share_dwells_ge_60_pct": row.get("share_dwells_ge_60_pct", np.nan),
                "end_rides_day_avg": row.get("end_rides_day_avg", np.nan),
                "net_trips_day_avg": row.get("net_trips_day_avg", np.nan),
                "weekday_end_day_avg": row.get("weekday_end_day_avg", np.nan),
                "weekend_end_day_avg": row.get("weekend_end_day_avg", np.nan),
            }

    hex_fc = {"type": "FeatureCollection", "features": []}
    period_start, period_end = time_period
    df_end_tp = df_rides[
        (df_rides["end_time"] >= period_start) & (df_rides["end_time"] < period_end)
    ].copy()
    if "end_in_bounds" in df_end_tp.columns:
        df_end_tp = df_end_tp[df_end_tp["end_in_bounds"]]
    if "end_h3" in df_end_tp.columns:
        df_end_tp = df_end_tp[df_end_tp["end_h3"].notna()].copy()
    else:
        df_end_tp = df_end_tp.iloc[0:0].copy()

    active_h3 = (
        set(df_end_tp["end_h3"].astype(str).unique().tolist())
        if len(df_end_tp) and "end_h3" in df_end_tp.columns
        else set()
    )

    df_draw = df_hex_metrics_all.copy()
    if not df_draw.empty and "h3" not in df_draw.columns:
        df_draw = df_draw.reset_index()
        if "index" in df_draw.columns and "h3" not in df_draw.columns:
            df_draw = df_draw.rename(columns={"index": "h3"})
    if not df_draw.empty and "h3" in df_draw.columns and len(active_h3):
        df_draw = df_draw[df_draw["h3"].astype(str).isin(active_h3)].copy()
    if df_draw.empty and isinstance(df_hex_metrics_all, pd.DataFrame):
        df_draw = df_hex_metrics_all.copy()
        if "h3" not in df_draw.columns:
            df_draw = df_draw.reset_index()
            if "index" in df_draw.columns and "h3" not in df_draw.columns:
                df_draw = df_draw.rename(columns={"index": "h3"})

    h3_set = set(df_draw["h3"].astype(str).tolist()) if "h3" in df_draw.columns else set()
    if len(h3_set) and len(df_end_tp):
        df_end_tp = df_end_tp[df_end_tp["end_h3"].astype(str).isin(h3_set)].copy()

    if len(df_end_tp) and "vehicle_id" in df_end_tp.columns:
        codes, _ = pd.factorize(df_end_tp["vehicle_id"].astype(str), sort=False)
        df_end_tp["_vid"] = codes.astype(int)
        veh_by_h3 = (
            df_end_tp.groupby(df_end_tp["end_h3"].astype(str))["_vid"]
            .unique()
            .apply(lambda arr: arr.tolist())
            .to_dict()
        )
    else:
        veh_by_h3 = {}

    df_all = df_rides.copy()
    try:
        if "vehicle_id" in df_all.columns:
            codes_all, _ = pd.factorize(df_all["vehicle_id"].astype(str), sort=False)
            df_all["_vid_all"] = codes_all.astype(int)
    except Exception:
        df_all["_vid_all"] = pd.Series(np.nan, index=df_all.index)

    try:
        ds_base_day = pd.Timestamp(period_start).normalize()
        if "end_time" in df_all.columns:
            tmp_et = pd.to_datetime(df_all["end_time"], errors="coerce")
            if "end_in_bounds" in df_all.columns:
                tmp_et = tmp_et[df_all["end_in_bounds"]]
            tmp_et = tmp_et.dropna()
            if len(tmp_et):
                ds_base_day = pd.Timestamp(tmp_et.min()).normalize()
    except Exception:
        ds_base_day = pd.Timestamp(period_start).normalize()

    df_end_all = df_all.copy()
    if "end_in_bounds" in df_end_all.columns:
        df_end_all = df_end_all[df_end_all["end_in_bounds"]]
    if "end_h3" in df_end_all.columns:
        df_end_all = df_end_all[df_end_all["end_h3"].notna()].copy()

    if (
        len(df_end_all)
        and "_vid_all" in df_end_all.columns
        and "end_time" in df_end_all.columns
        and "end_h3" in df_end_all.columns
    ):
        df_end_all["_day"] = (
            pd.to_datetime(df_end_all["end_time"], errors="coerce").dt.normalize()
            - ds_base_day
        ).dt.days
        df_end_all = df_end_all[df_end_all["_day"].notna()].copy()
        df_end_all["_h3s"] = df_end_all["end_h3"].astype(str)
        ev_end = (
            df_end_all.groupby(["_day", "_h3s", "_vid_all"]).size().reset_index(name="cnt")
        )
        end_events = [
            [int(day), str(h3_id), int(vid), int(cnt)]
            for day, h3_id, vid, cnt in ev_end[["_day", "_h3s", "_vid_all", "cnt"]].itertuples(
                index=False, name=None
            )
        ]
    else:
        end_events = []

    df_start_all = df_all.copy()
    if "start_in_bounds" in df_start_all.columns:
        df_start_all = df_start_all[df_start_all["start_in_bounds"]]
    if "start_h3" in df_start_all.columns:
        df_start_all = df_start_all[df_start_all["start_h3"].notna()].copy()

    if (
        len(df_start_all)
        and "_vid_all" in df_start_all.columns
        and "start_time" in df_start_all.columns
        and "start_h3" in df_start_all.columns
    ):
        df_start_all["_day"] = (
            pd.to_datetime(df_start_all["start_time"], errors="coerce").dt.normalize()
            - ds_base_day
        ).dt.days
        df_start_all = df_start_all[df_start_all["_day"].notna()].copy()
        df_start_all["_h3s"] = df_start_all["start_h3"].astype(str)
        ev_start = (
            df_start_all.groupby(["_day", "_h3s", "_vid_all"]).size().reset_index(name="cnt")
        )
        start_events = [
            [int(day), str(h3_id), int(vid), int(cnt)]
            for day, h3_id, vid, cnt in ev_start[["_day", "_h3s", "_vid_all", "cnt"]].itertuples(
                index=False, name=None
            )
        ]
    else:
        start_events = []

    od_events = []
    try:
        need_od = ["start_h3", "end_h3", "start_time", "vehicle_id"]
        if all(col in df_all.columns for col in need_od) and len(df_all):
            df_od = df_all.copy()
            if "start_in_bounds" in df_od.columns:
                df_od = df_od[df_od["start_in_bounds"]]
            if "end_in_bounds" in df_od.columns:
                df_od = df_od[df_od["end_in_bounds"]]
            df_od = df_od[df_od["start_h3"].notna() & df_od["end_h3"].notna()]
            df_od = df_od[df_od["start_time"].notna()]
            df_od["_day"] = (
                pd.to_datetime(df_od["start_time"], errors="coerce").dt.normalize() - ds_base_day
            ).dt.days
            df_od["_hour"] = pd.to_datetime(df_od["start_time"], errors="coerce").dt.hour
            df_od = df_od[df_od["_day"].notna()].copy()
            df_od["_h3o"] = df_od["start_h3"].astype(str)
            df_od["_h3d"] = df_od["end_h3"].astype(str)
            ev_od = (
                df_od.groupby(["_day", "_hour", "_h3o", "_h3d"])
                .agg(trips=("vehicle_id", "size"), uniq_veh=("vehicle_id", "nunique"))
                .reset_index()
            )
            if len(ev_od):
                h3_all = pd.unique(ev_od[["_h3o", "_h3d"]].values.ravel("K"))
                centers = {}
                for h3_id in h3_all:
                    try:
                        lat, lng = h3_cell_to_latlng(h3_id)
                        centers[h3_id] = (float(lat), float(lng))
                    except Exception:
                        centers[h3_id] = (np.nan, np.nan)

                def dist_km(h1, h2):
                    try:
                        lat1, lng1 = centers.get(h1, (np.nan, np.nan))
                        lat2, lng2 = centers.get(h2, (np.nan, np.nan))
                        if np.isnan(lat1) or np.isnan(lat2):
                            return np.nan
                        r_earth = 6371.0088
                        dlat = np.radians(lat2 - lat1)
                        dlon = np.radians(lng2 - lng1)
                        a = (
                            np.sin(dlat / 2) ** 2
                            + np.cos(np.radians(lat1))
                            * np.cos(np.radians(lat2))
                            * np.sin(dlon / 2) ** 2
                        )
                        return 2 * r_earth * np.arcsin(np.sqrt(a))
                    except Exception:
                        return np.nan

                ev_od["dist_km"] = ev_od.apply(
                    lambda row: dist_km(row["_h3o"], row["_h3d"]), axis=1
                )
                od_events = [
                    [
                        int(row["_day"]),
                        int(row["_hour"]),
                        str(row["_h3o"]),
                        str(row["_h3d"]),
                        int(row["trips"]),
                        int(row["uniq_veh"]),
                        float(row["dist_km"]) if pd.notna(row["dist_km"]) else None,
                    ]
                    for _, row in ev_od.iterrows()
                ]
    except Exception:
        od_events = []

    try:
        if len(end_events):
            dataset_days = max((row[0] for row in end_events), default=-1) + 1
        elif len(start_events):
            dataset_days = max((row[0] for row in start_events), default=-1) + 1
        else:
            dataset_days = 0
    except Exception:
        dataset_days = 0

    # mean position of the ride ends inside each hex over the whole dataset, for the ride-end centre export
    end_mean: dict = {}
    try:
        if {"end_h3", "end_lat", "end_lng"} <= set(df_rides.columns):
            _em = (
                df_rides.dropna(subset=["end_h3", "end_lat", "end_lng"])
                .groupby("end_h3")
                .agg(lat=("end_lat", "mean"), lng=("end_lng", "mean"), n=("end_lat", "size"))
            )
            end_mean = {str(k): (float(v.lat), float(v.lng), int(v.n)) for k, v in _em.iterrows()}
    except Exception:
        end_mean = {}
    for _, row in df_draw.iterrows():
        hx = row.get("h3")
        if hx is None or (isinstance(hx, float) and np.isnan(hx)):
            continue
        metrics = metrics_by_hex.get(str(hx), {})
        centre_lat, centre_lng = h3_cell_to_latlng(hx)
        boundary = h3_cell_to_boundary(hx)
        coords = [[lng, lat] for lat, lng in boundary]
        if len(coords) and coords[0] != coords[-1]:
            coords.append(coords[0])
        props = {
            "h3": str(hx),
            "center_lat": float(centre_lat),
            "center_lng": float(centre_lng),
            "end_mean_lat": end_mean.get(str(hx), (None, None, None))[0],
            "end_mean_lng": end_mean.get(str(hx), (None, None, None))[1],
            "end_n": end_mean.get(str(hx), (None, None, None))[2],
            "rank": None
            if (
                isinstance(metrics.get("rank", None), float)
                and np.isnan(metrics.get("rank", np.nan))
            )
            else metrics.get("rank", None),
            "station_score": None
            if (
                isinstance(metrics.get("station_score", None), float)
                and np.isnan(metrics.get("station_score", np.nan))
            )
            else metrics.get("station_score", None),
            "is_selected_initial": bool(metrics.get("is_selected", False)),
            "avg_5day_vehicle_coverage_pct": metrics.get(
                "avg_5day_vehicle_coverage_pct", None
            ),
            "end_rides": metrics.get("end_rides", None),
            "start_rides": metrics.get("start_rides", None),
            "net_accum": metrics.get("net_accum", None),
            "veh": veh_by_h3.get(str(hx), []),
            "end_rides_day_avg": metrics.get("end_rides_day_avg", None),
            "net_trips_day_avg": metrics.get("net_trips_day_avg", None),
            "weekday_end_day_avg": metrics.get("weekday_end_day_avg", None),
            "weekend_end_day_avg": metrics.get("weekend_end_day_avg", None),
            "median_dwell_mins": metrics.get("median_dwell_mins", None),
            "share_dwells_ge_60_pct": metrics.get("share_dwells_ge_60_pct", None),
            "dwell_same_loc_radius_m": float(dwell_same_loc_radius_m),
        }
        hex_fc["features"].append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [coords]},
                "properties": props,
            }
        )

    raw_end_events = []
    raw_start_events = []
    try:
        min_base = pd.Timestamp(ds_base_day)
        df_raw_end = df_all.copy()
        if "end_in_bounds" in df_raw_end.columns:
            df_raw_end = df_raw_end[df_raw_end["end_in_bounds"]]
        if "end_h3" in df_raw_end.columns:
            df_raw_end = df_raw_end[df_raw_end["end_h3"].notna()].copy()
        if "end_time" in df_raw_end.columns:
            df_raw_end["_end_t"] = pd.to_datetime(df_raw_end["end_time"], errors="coerce")
            df_raw_end = df_raw_end[df_raw_end["_end_t"].notna()].copy()
        if len(h3_set):
            df_raw_end = df_raw_end[df_raw_end["end_h3"].astype(str).isin(h3_set)].copy()
        if len(df_raw_end):
            df_raw_end["_end_min"] = (
                (df_raw_end["_end_t"] - min_base).dt.total_seconds() // 60
            ).astype("int")
            if "next_start_time" in df_raw_end.columns:
                df_raw_end["_next_t"] = pd.to_datetime(
                    df_raw_end["next_start_time"], errors="coerce"
                )
                df_raw_end["_next_min"] = (
                    (df_raw_end["_next_t"] - min_base).dt.total_seconds() // 60
                ).round().astype("Int64")
            else:
                df_raw_end["_next_min"] = pd.Series(pd.NA, index=df_raw_end.index)
            if "dwell_same_loc_mins" not in df_raw_end.columns:
                df_raw_end["dwell_same_loc_mins"] = pd.NA
            raw_end_events = [
                [str(h), int(m), (int(n) if pd.notna(n) else None), (float(d) if pd.notna(d) else None)]
                for h, m, n, d in df_raw_end[
                    ["end_h3", "_end_min", "_next_min", "dwell_same_loc_mins"]
                ].itertuples(index=False, name=None)
            ]

        df_raw_start = df_all.copy()
        if "start_in_bounds" in df_raw_start.columns:
            df_raw_start = df_raw_start[df_raw_start["start_in_bounds"]]
        if "start_h3" in df_raw_start.columns:
            df_raw_start = df_raw_start[df_raw_start["start_h3"].notna()].copy()
        if "start_time" in df_raw_start.columns:
            df_raw_start["_start_t"] = pd.to_datetime(df_raw_start["start_time"], errors="coerce")
            df_raw_start = df_raw_start[df_raw_start["_start_t"].notna()].copy()
        if len(h3_set):
            df_raw_start = df_raw_start[
                df_raw_start["start_h3"].astype(str).isin(h3_set)
            ].copy()
        if len(df_raw_start):
            df_raw_start["_start_min"] = (
                (df_raw_start["_start_t"] - min_base).dt.total_seconds() // 60
            ).astype("int")
            raw_start_events = [
                [str(h), int(m)]
                for h, m in df_raw_start[["start_h3", "_start_min"]].itertuples(
                    index=False, name=None
                )
            ]
    except Exception:
        raw_end_events = []
        raw_start_events = []

    neighbors_by_hex = {}
    try:
        hex_centers = []
        hex_ids = []
        for feat in hex_fc.get("features", []):
            props = feat.get("properties", {})
            hid = props.get("h3")
            clat = props.get("center_lat")
            clng = props.get("center_lng")
            if hid and clat is not None and clng is not None:
                hex_ids.append(str(hid))
                hex_centers.append((clat, clng))

        if len(hex_centers) > 1:
            nb_arr = np.array(hex_centers)
            rad = np.radians(nb_arr)
            r_earth = 6_371_000.0
            xyz = np.column_stack(
                [
                    r_earth * np.cos(rad[:, 0]) * np.cos(rad[:, 1]),
                    r_earth * np.cos(rad[:, 0]) * np.sin(rad[:, 1]),
                    r_earth * np.sin(rad[:, 0]),
                ]
            )
            from scipy.spatial import cKDTree

            tree = cKDTree(xyz)
            pairs = tree.query_pairs(r=300, output_type="ndarray")
            for i, j in pairs:
                dist = float(np.linalg.norm(xyz[i] - xyz[j]))
                hi, hj = hex_ids[i], hex_ids[j]
                neighbors_by_hex.setdefault(hi, []).append([hj, round(dist, 1)])
                neighbors_by_hex.setdefault(hj, []).append([hi, round(dist, 1)])
    except Exception as exc:
        warnings.warn(f"Neighbor precompute skipped: {exc}")

    dataset_day0_iso = pd.Timestamp(ds_base_day).strftime("%Y-%m-%d")
    default_win_start = int(
        (pd.Timestamp(period_start).normalize() - pd.Timestamp(ds_base_day).normalize()).days
    )
    default_win_end = int(
        (pd.Timestamp(period_end).normalize() - pd.Timestamp(ds_base_day).normalize()).days
    ) - 1
    if dataset_days > 0:
        default_win_start = max(0, min(default_win_start, dataset_days - 1))
        default_win_end = max(0, min(default_win_end, dataset_days - 1))
        if default_win_end < default_win_start:
            default_win_end = default_win_start

    try:
        start_day = pd.Timestamp(period_start).floor("D")
        end_day_excl = pd.Timestamp(period_end).floor("D")
        last_obs = end_day_excl - pd.Timedelta(days=1)
        edge_month_impute = bool(
            start_day.day != 1 or last_obs.day != last_obs.days_in_month
        )
    except Exception:
        edge_month_impute = False

    return {
        "general_stats": general_stats_data,
        "hex_data_b64": _pack(hex_fc),
        "end_events_b64": _pack(end_events),
        "start_events_b64": _pack(start_events),
        "od_events_b64": _pack(od_events),
        "raw_end_events_b64": _pack(raw_end_events),
        "raw_start_events_b64": _pack(raw_start_events),
        "neighbors_b64": _pack(neighbors_by_hex),
        "dataset_day0_iso": dataset_day0_iso,
        "dataset_days": int(dataset_days),
        "default_win_start": int(default_win_start),
        "default_win_end": int(default_win_end),
        "edge_month_impute": bool(edge_month_impute),
        "default_bucket_days": int(cycle_duration),
        "default_sigma": int(i_sigma),
        "default_topn": int(min(300, max(1, i_s))),
        "default_wcharge": float(optimiser_weights.get("chargecount", 0.0)),
        "default_wfleet": float(optimiser_weights.get("fleetcoverage", 1.0)),
    }


# --- Pre/post deployment comparison ---------------------------------------
#
# All comparison statistics are computed here, in Python, once. The JS
# "Compare" tab is strictly a renderer. Statistics are pooled at
# COMPARE_H3_RES (res-10 parents of the res-11 analysis hexes) because the
# post window is short: at raw res 11 the median cell has ~2 ride-ends over
# 15 days, which makes per-cell inference meaningless.

COMPARE_H3_RES = 10
COMPARE_MIN_ENDS = 30          # per-cell floor: below this a cell is "insufficient"
COMPARE_Z_COLOR = 1.96         # |z| needed before a cell is colored on the map
COMPARE_Z_CANDIDATE = 3.0      # z needed for a station-candidate row to pre-check
COMPARE_MIN_LIFT = 0.5         # +50% end-share lift for candidate pre-check
COMPARE_SAME_SEASON_MIN_DAYS = 21
COMPARE_OUTLIER_FRAC = 0.5     # day is an outlier if total ends < frac * median
COMPARE_BOOT_ITERS = 500
COMPARE_MAX_CURVE_K = 150
COMPARE_MAX_CANDIDATES = 40
COMPARE_DWELL_CAP_MINS = 1440.0
COMPARE_RELOC_THRESHOLD_M = 200.0


def h3_cell_to_parent(cell, res):
    if hasattr(h3, "cell_to_parent"):
        return h3.cell_to_parent(cell, res)
    return h3.h3_to_parent(cell, res)


def _cmp_day_frame(df: pd.DataFrame, t0, t1, bbox) -> pd.DataFrame:
    """Per-ride rows (day, cell, vehicle, dwell, dist) for in-window, in-bbox ends."""
    min_lat, max_lat, min_lng, max_lng = bbox
    m = (
        df["end_time"].notna()
        & (df["end_time"] >= t0)
        & (df["end_time"] < t1)
        & df["end_h3"].notna()
    )
    lat = pd.to_numeric(df.get("end_lat"), errors="coerce")
    lng = pd.to_numeric(df.get("end_lng"), errors="coerce")
    m = m & lat.ge(min_lat) & lat.le(max_lat) & lng.ge(min_lng) & lng.le(max_lng)
    sub = df.loc[m, ["end_time", "end_h3", "vehicle_id"]].copy()
    sub["day"] = sub["end_time"].dt.normalize()
    sub["cell"] = [
        str(h3_cell_to_parent(hx, COMPARE_H3_RES)) for hx in sub["end_h3"].tolist()
    ]
    sub["dwell"] = (
        pd.to_numeric(df.loc[m, "dwell_same_loc_mins"], errors="coerce")
        .clip(upper=COMPARE_DWELL_CAP_MINS)
        .values
        if "dwell_same_loc_mins" in df.columns
        else np.nan
    )
    sub["dist_next"] = (
        pd.to_numeric(df.loc[m, "dist_to_next_start_m"], errors="coerce").values
        if "dist_to_next_start_m" in df.columns
        else np.nan
    )
    return sub


def _cmp_matrix(sub: pd.DataFrame, days: list, cells: list) -> np.ndarray:
    """cells x days end-count matrix (zero-filled)."""
    day_ix = {d: i for i, d in enumerate(days)}
    cell_ix = {c: i for i, c in enumerate(cells)}
    mat = np.zeros((len(cells), len(days)), dtype=np.float64)
    grouped = sub.groupby(["cell", "day"]).size()
    for (c, d), n in grouped.items():
        if c in cell_ix and d in day_ix:
            mat[cell_ix[c], day_ix[d]] += n
    return mat


def _cmp_gini(counts: np.ndarray) -> float:
    v = np.sort(np.asarray(counts, dtype=np.float64))
    n = len(v)
    tot = v.sum()
    if n == 0 or tot <= 0:
        return 0.0
    cum = np.cumsum(v)
    return float((n + 1 - 2 * (cum / tot).sum()) / n)


def _cmp_outlier_days(day_totals: pd.Series) -> list:
    """Low-volume anomaly days (holidays, storms) for the sensitivity duals.

    A day flags when its total is a robust-z < -2 outlier (median/MAD based —
    catches e.g. Midsummer at ~80% of median volume) or below half the median
    (absolute floor for series too short for a stable MAD).
    """
    if not len(day_totals):
        return []
    med = float(day_totals.median())
    if med <= 0:
        return []
    mad = float((day_totals - med).abs().median())
    sd = 1.4826 * mad
    out = []
    for d, t in day_totals.items():
        rz = (t - med) / sd if sd > 0 else 0.0
        if rz < -2.0 or t < COMPARE_OUTLIER_FRAC * med:
            out.append(d)
    return out


def _cmp_weekday_weights(post_days: list) -> dict[int, float]:
    n = len(post_days)
    out: dict[int, float] = {}
    for d in post_days:
        out[d.dayofweek] = out.get(d.dayofweek, 0.0) + 1.0 / n
    return out


def _cmp_day_weights(base_days: list, wk_weights: dict[int, float]) -> np.ndarray:
    """Per-pre-day weights so the weighted pre day-mix matches the post weekday mix."""
    by_dow: dict[int, int] = {}
    for d in base_days:
        by_dow[d.dayofweek] = by_dow.get(d.dayofweek, 0) + 1
    w = np.array(
        [wk_weights.get(d.dayofweek, 0.0) / by_dow[d.dayofweek] for d in base_days],
        dtype=np.float64,
    )
    if w.sum() <= 0:
        w = np.ones(len(base_days), dtype=np.float64) / max(len(base_days), 1)
    else:
        w = w / w.sum()
    return w


def _cmp_scalar_kpis(
    mat: np.ndarray,
    day_w: np.ndarray,
    sub: pd.DataFrame,
    days: list,
    day_sel: np.ndarray | None = None,
) -> dict:
    """Scalar KPIs for one window given a cells x days matrix and day weights.

    ``day_sel`` (indices with repetition allowed) supports the day-block
    bootstrap: statistics are recomputed as if the selected days were the
    observed sample.
    """
    if day_sel is None:
        day_sel = np.arange(mat.shape[1])
    w = day_w[day_sel]
    w = w / w.sum() if w.sum() > 0 else np.ones(len(day_sel)) / max(len(day_sel), 1)
    cell_counts = mat[:, day_sel] @ w  # weighted mean daily ends per cell
    total = cell_counts.sum()
    shares = cell_counts / total if total > 0 else cell_counts

    day_keys = [days[i] for i in day_sel]
    by_day = sub.groupby("day")
    rides, actives, dwells, reloc_n, reloc_x = [], [], [], 0.0, 0.0
    for d in day_keys:
        try:
            g = by_day.get_group(d)
        except KeyError:
            continue
        rides.append(len(g))
        actives.append(g["vehicle_id"].nunique())
        dw = g["dwell"].dropna()
        if len(dw):
            dwells.append(dw.values)
        dn = g["dist_next"].dropna()
        reloc_n += len(dn)
        reloc_x += float((dn > COMPARE_RELOC_THRESHOLD_M).sum())
    all_dwells = np.concatenate(dwells) if dwells else np.array([])
    return {
        "shares": shares,
        "cell_counts": cell_counts,
        "gini": _cmp_gini(cell_counts),
        "trips_per_active": float(
            np.mean([r / a for r, a in zip(rides, actives) if a > 0])
        )
        if rides
        else 0.0,
        "median_dwell": float(np.median(all_dwells)) if len(all_dwells) else None,
        "reloc_rate": (reloc_x / reloc_n) if reloc_n > 0 else None,
    }


def build_comparison_payload(
    pre_bundle: dict,
    post_bundle: dict,
    station_count: int | None = None,
    deploy_date: str | None = None,
) -> dict:
    """Compute the full pre/post deployment comparison. JS only renders this."""
    warn: list[str] = []

    # --- footprint: bbox intersection ---
    a = [float(x) for x in pre_bundle["map_search_area"]]
    b = [float(x) for x in post_bundle["map_search_area"]]
    bbox = [max(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), min(a[3], b[3])]
    if bbox[0] >= bbox[1] or bbox[2] >= bbox[3]:
        return {
            "enabled": False,
            "reason": (
                "The pre and post datasets cover non-overlapping areas — "
                "they appear to be different cities. No comparison possible."
            ),
        }

    df_pre = pre_bundle["df_rides"]
    df_post = post_bundle["df_rides"]
    post_t0, post_t1 = post_bundle["time_period"]
    pre_t0, pre_t1 = pre_bundle["time_period"]

    # --- burn-in: drop post days on/before the deployment-completion date ---
    if deploy_date:
        try:
            dd = pd.Timestamp(deploy_date)
            if post_t0 <= dd < post_t1:
                post_t0 = (dd + pd.Timedelta(days=1)).to_pydatetime()
                warn.append(f"burn_in:{deploy_date}")
        except Exception:
            pass

    post_sub = _cmp_day_frame(df_post, post_t0, post_t1, bbox)
    pre_sub_full = _cmp_day_frame(df_pre, pre_t0, pre_t1, bbox)
    if post_sub.empty or pre_sub_full.empty:
        return {
            "enabled": False,
            "reason": "Not enough in-area rides in the pre or post window to compare.",
        }

    def _drop_partial_days(sub: pd.DataFrame) -> pd.DataFrame:
        """Remove boundary artifacts: days with <5% of the median daily volume.

        These are typically partial first/last days of an export (a handful
        of rides crossing midnight), not real low-demand days — genuine
        anomalies like holidays stay in and are handled by the outlier-day
        sensitivity instead.
        """
        totals = sub.groupby("day").size()
        med = float(totals.median()) if len(totals) else 0.0
        if med <= 0:
            return sub
        keep = {d for d, t in totals.items() if t >= 0.05 * med}
        return sub[sub["day"].isin(keep)].copy()

    post_sub = _drop_partial_days(post_sub)
    pre_sub_full = _drop_partial_days(pre_sub_full)
    if post_sub.empty or pre_sub_full.empty:
        return {
            "enabled": False,
            "reason": "Not enough in-area rides in the pre or post window to compare.",
        }

    post_days = sorted(post_sub["day"].unique())
    post_months = {d.month for d in post_days}

    # --- baseline selection: same-season subset when available ---
    pre_days_all = sorted(pre_sub_full["day"].unique())
    season_days = [d for d in pre_days_all if d.month in post_months]
    if len(season_days) >= COMPARE_SAME_SEASON_MIN_DAYS:
        baseline_mode = "same_season"
        base_days = season_days
    else:
        baseline_mode = "full_pre"
        base_days = pre_days_all
        # Only a real confound if the baseline actually spans other seasons.
        if any(d.month not in post_months for d in base_days):
            warn.append("seasonality")
    pre_sub = pre_sub_full[pre_sub_full["day"].isin(base_days)].copy()

    if len(post_days) < 30:
        warn.append("short_post_window")

    # --- outlier days (low-volume; catches holidays without a calendar) ---
    pre_day_totals = pre_sub.groupby("day").size()
    post_day_totals = post_sub.groupby("day").size()
    pre_outliers = _cmp_outlier_days(pre_day_totals)
    post_outliers = _cmp_outlier_days(post_day_totals)
    if pre_outliers or post_outliers:
        warn.append("outlier_days")

    # --- union cell set + matrices ---
    cells = sorted(set(pre_sub["cell"]).union(post_sub["cell"]))
    cell_ix = {c: i for i, c in enumerate(cells)}
    pre_mat = _cmp_matrix(pre_sub, base_days, cells)
    post_mat = _cmp_matrix(post_sub, post_days, cells)

    wk_w = _cmp_weekday_weights(post_days)
    missing_dows = {d.dayofweek for d in post_days} - {d.dayofweek for d in base_days}
    if missing_dows:
        warn.append("weekday_gap")
    pre_day_w = _cmp_day_weights(base_days, wk_w)
    post_day_w = np.ones(len(post_days), dtype=np.float64) / len(post_days)

    pre_k = _cmp_scalar_kpis(pre_mat, pre_day_w, pre_sub, base_days)
    post_k = _cmp_scalar_kpis(post_mat, post_day_w, post_sub, post_days)

    # --- pooled counts for significance (unweighted; conservative) ---
    x_pre = pre_mat.sum(axis=1)
    x_post = post_mat.sum(axis=1)
    n_pre, n_post = float(x_pre.sum()), float(x_post.sum())
    scale = n_post / n_pre if n_pre > 0 else 0.0
    expected_post = x_pre * scale

    # Per-cell significance uses day-block-bootstrap standard errors, computed
    # in the bootstrap loop further below. Parametric z's failed calibration
    # on real data: a binomial two-proportion z gave ~10% false positives on
    # a null split, and a quasi-Poisson residual collapsed under city-wide
    # dilution because daily counts have a heavier-than-Poisson,
    # size-dependent variance. Resampling whole days captures that variance
    # structure exactly. A robust background share-ratio gamma is fitted
    # first (and re-fitted inside every bootstrap replicate) to absorb
    # city-wide multiplicative drift, so "significant" means "this cell
    # moved more than the background did".
    sufficient = (x_post >= COMPARE_MIN_ENDS) | (expected_post >= COMPARE_MIN_ENDS)
    s_pre = pre_k["shares"]
    s_post = post_k["shares"]
    delta_pp = (s_post - s_pre) * 100.0
    _EPS_SHARE = 1e-9

    def _gamma_share(sp: np.ndarray, sq: np.ndarray) -> float:
        # Ratio only over cells with genuine pre mass — a near-empty
        # resample would otherwise blow the ratio up through the epsilon.
        m = sufficient & (sp > _EPS_SHARE) & (sq > _EPS_SHARE)
        if m.sum() >= 10:
            g = float(np.median(sq[m] / sp[m]))
            return float(min(max(g, 0.2), 5.0))
        return 1.0

    gamma = _gamma_share(s_pre, s_post)
    d_pt = s_post - gamma * s_pre

    # --- pre ranking at compare resolution (from the optimiser's hex table) ---
    hexm = pre_bundle["df_hex_metrics_all"]
    cell_score: dict[str, float] = {}
    cell_ends_rank: dict[str, float] = {}
    best_member_rank: dict[str, float] = {}
    if isinstance(hexm, pd.DataFrame) and not hexm.empty and "h3" in hexm.columns:
        for _, row in hexm.iterrows():
            try:
                parent = str(h3_cell_to_parent(row["h3"], COMPARE_H3_RES))
            except Exception:
                continue
            cell_score[parent] = cell_score.get(parent, 0.0) + float(
                row.get("station_score", 0.0) or 0.0
            )
            cell_ends_rank[parent] = cell_ends_rank.get(parent, 0.0) + float(
                row.get("end_rides", 0.0) or 0.0
            )
            r = row.get("rank")
            if pd.notna(r):
                best_member_rank[parent] = min(
                    best_member_rank.get(parent, float("inf")), float(r)
                )
    order = sorted(
        cells,
        key=lambda c: (-cell_score.get(c, 0.0), -cell_ends_rank.get(c, 0.0), c),
    )
    pre_rank = {c: i + 1 for i, c in enumerate(order)}

    # --- rank-capture curve (fixed pre-optimiser ordering for ALL series) ---
    # The band is the 95% day-block-bootstrap range of the PRE curve under the
    # same fixed ordering: "capture levels consistent with pre-period
    # day-to-day noise". A post curve outside it indicates a real shift.
    # (Filled from the bootstrap loop below.)
    k_max = min(COMPARE_MAX_CURVE_K, len(cells))
    order_ix = np.array([cell_ix[c] for c in order[:k_max]])
    c_pre = np.cumsum(s_pre[order_ix]) * 100.0
    c_post = np.cumsum(s_post[order_ix]) * 100.0
    band_lo = band_hi = None

    k_hat = int(station_count) if station_count else None

    post_dwell_by_cell = post_sub.groupby("cell")["dwell"].median()
    pre_dwell_by_cell = pre_sub.groupby("cell")["dwell"].median()
    lift = np.where(s_pre > 0, (s_post - s_pre) / np.maximum(s_pre, 1e-12), np.inf)

    def _cand_row(i: int) -> dict:
        c = cells[i]
        lat, lng = h3_cell_to_latlng(c)
        return {
            "cell": c,
            "lat": float(lat),
            "lng": float(lng),
            "x_pre": float(x_pre[i]),
            "x_post": float(x_post[i]),
            "s_pre_pct": float(s_pre[i] * 100.0),
            "s_post_pct": float(s_post[i] * 100.0),
            "delta_pp": float(delta_pp[i]),
            "lift_pct": float(lift[i] * 100.0) if np.isfinite(lift[i]) else None,
            "z": float(z[i]),
            "pre_rank": int(pre_rank.get(c, 0)),
            "best_member_rank": (
                int(best_member_rank[c]) if c in best_member_rank else None
            ),
            "dwell_median_pre": (
                float(pre_dwell_by_cell[c]) if c in pre_dwell_by_cell.index and pd.notna(pre_dwell_by_cell[c]) else None
            ),
            "dwell_median_post": (
                float(post_dwell_by_cell[c]) if c in post_dwell_by_cell.index and pd.notna(post_dwell_by_cell[c]) else None
            ),
            "activated": bool(activated[i]),
            "in_rank_window": bool(
                k_hat is not None and pre_rank.get(c, 10**9) <= 3 * k_hat
            ),
        }

    # --- day-block weekday-stratified bootstrap ---
    # One loop yields: CIs for scalar KPI deltas, the pre-curve noise band,
    # and the per-cell standard errors that drive the significance classes.
    rng = np.random.default_rng(20260614)

    def _strata(days_list):
        s: dict[int, list[int]] = {}
        for i, d in enumerate(days_list):
            s.setdefault(d.dayofweek, []).append(i)
        strata = list(s.values())
        # A size-1 stratum always redraws the same day and contributes zero
        # bootstrap variance — with short windows that silently deflates
        # every SE. Weekday mix is already handled by the day weights, so
        # fall back to plain day resampling when strata are thin.
        if strata and min(len(ix) for ix in strata) < 3:
            return [list(range(len(days_list)))]
        return strata

    pre_strata = _strata(base_days)
    post_strata = _strata(post_days)
    cap_k = min(k_hat or 20, k_max)

    def _capture_at_k(shares):
        return float(np.cumsum(shares[order_ix])[cap_k - 1] * 100.0) if cap_k >= 1 else 0.0

    boot: dict[str, list[float]] = {
        "gini": [], "capture": [], "reloc": [], "tpa": [], "dwell": []
    }
    pre_curves: list[np.ndarray] = []
    d_sum = np.zeros(len(cells), dtype=np.float64)
    d_sq = np.zeros(len(cells), dtype=np.float64)
    n_eff = 0
    n_degenerate = 0
    full_pre_total = pre_mat.sum()
    full_post_total = post_mat.sum()
    for it in range(COMPARE_BOOT_ITERS):
        pre_sel = np.concatenate(
            [rng.choice(ix, size=len(ix), replace=True) for ix in pre_strata]
        )
        post_sel = np.concatenate(
            [rng.choice(ix, size=len(ix), replace=True) for ix in post_strata]
        )
        # A resample landing mostly on near-empty days yields meaningless
        # shares (and once poisoned the SE accumulators) — skip it.
        if (
            pre_mat[:, pre_sel].sum() < 0.25 * full_pre_total * len(pre_sel) / max(pre_mat.shape[1], 1)
            or post_mat[:, post_sel].sum() < 0.25 * full_post_total * len(post_sel) / max(post_mat.shape[1], 1)
        ):
            n_degenerate += 1
            continue
        pk = _cmp_scalar_kpis(pre_mat, pre_day_w, pre_sub, base_days, pre_sel)
        qk = _cmp_scalar_kpis(post_mat, post_day_w, post_sub, post_days, post_sel)
        pre_curves.append(np.cumsum(pk["shares"][order_ix]) * 100.0)
        d_boot = qk["shares"] - _gamma_share(pk["shares"], qk["shares"]) * pk["shares"]
        d_sum += d_boot
        d_sq += d_boot * d_boot
        n_eff += 1
        boot["gini"].append(qk["gini"] - pk["gini"])
        boot["capture"].append(_capture_at_k(qk["shares"]) - _capture_at_k(pk["shares"]))
        if pk["reloc_rate"] is not None and qk["reloc_rate"] is not None:
            boot["reloc"].append(qk["reloc_rate"] - pk["reloc_rate"])
        boot["tpa"].append(qk["trips_per_active"] - pk["trips_per_active"])
        if pk["median_dwell"] is not None and qk["median_dwell"] is not None:
            boot["dwell"].append(qk["median_dwell"] - pk["median_dwell"])
    if n_eff < 50:
        return {
            "enabled": False,
            "reason": "Too few valid bootstrap resamples — the datasets have too few usable days.",
        }
    pre_curves_arr = np.vstack(pre_curves)
    band_lo = np.percentile(pre_curves_arr, 2.5, axis=0)
    band_hi = np.percentile(pre_curves_arr, 97.5, axis=0)

    # --- per-cell significance from bootstrap SEs ---
    d_mean = d_sum / n_eff
    d_var = np.maximum(d_sq / n_eff - d_mean**2, 0.0)
    d_se = np.sqrt(d_var)
    z = np.where(d_se > 0, d_pt / np.where(d_se > 0, d_se, 1.0), 0.0)

    # SEs are estimated from a small number of days, so use t-quantiles with
    # day-based degrees of freedom instead of fixed normal cutoffs (with a
    # months-long pre window df is large and these converge to 1.96 / 3.0).
    from scipy.stats import t as _t_dist

    df_days = max(len(base_days) + len(post_days) - 2, 3)
    z_color_thr = float(_t_dist.ppf(0.975, df_days) / 1.959964 * COMPARE_Z_COLOR)
    z_cand_thr = float(_t_dist.ppf(0.99865, df_days) / 3.0 * COMPARE_Z_CANDIDATE)

    cls = np.where(
        ~sufficient,
        "insufficient",
        np.where(
            z >= z_color_thr,
            "sig_up",
            np.where(z <= -z_color_thr, "sig_down", "ns"),
        ),
    )

    # --- station candidates (inferred when count unknown) ---
    activated = (
        (z >= z_cand_thr)
        & (lift >= COMPARE_MIN_LIFT)
        & (x_post >= COMPARE_MIN_ENDS)
    )
    # No hard pre-rank filter: real deployments skip ranks ("next best spot"),
    # so an activated cell far down the pre ranking is still a legitimate
    # candidate — the row is annotated instead (in_rank_window) and the user
    # confirms/rejects.
    cand_ix = [i for i in np.argsort(-z) if activated[i]][:COMPARE_MAX_CANDIDATES]
    declined_ix = [i for i in np.argsort(z) if cls[i] == "sig_down"][:20]
    candidates = [_cand_row(i) for i in cand_ix]
    declined = [_cand_row(i) for i in declined_ix]

    # --- guard metrics ---
    # M7 off-station retention: pre top-quintile demand cells excluding candidates.
    cand_set = {cells[i] for i in cand_ix}
    q80 = np.quantile(s_pre[s_pre > 0], 0.8) if (s_pre > 0).any() else 0.0
    guard_ix = [
        i for i, c in enumerate(cells) if s_pre[i] >= q80 and c not in cand_set
    ]
    retention = (
        float(s_post[guard_ix].sum() / max(s_pre[guard_ix].sum(), 1e-12))
        if guard_ix
        else None
    )

    # M8 start-side symmetry at candidate cells.
    def _start_share(df, t0, t1, days_used):
        min_lat, max_lat, min_lng, max_lng = bbox
        m = (
            df["start_time"].notna()
            & (df["start_time"] >= t0)
            & (df["start_time"] < t1)
            & df.get("start_h3", pd.Series(index=df.index, dtype=object)).notna()
        )
        lat = pd.to_numeric(df.get("start_lat"), errors="coerce")
        lng = pd.to_numeric(df.get("start_lng"), errors="coerce")
        m = m & lat.ge(min_lat) & lat.le(max_lat) & lng.ge(min_lng) & lng.le(max_lng)
        sub = df.loc[m, ["start_time", "start_h3"]].copy()
        sub["day"] = sub["start_time"].dt.normalize()
        sub = sub[sub["day"].isin(set(days_used))]
        sub["cell"] = [
            str(h3_cell_to_parent(hx, COMPARE_H3_RES)) for hx in sub["start_h3"]
        ]
        tot = len(sub)
        if tot == 0:
            return None
        in_cand = float(sub["cell"].isin(cand_set).sum())
        return in_cand / tot

    start_share_pre = _start_share(df_pre, pre_t0, pre_t1, base_days)
    start_share_post = _start_share(df_post, post_t0, post_t1, post_days)

    def _ci(key):
        v = boot[key]
        if len(v) < 50:
            return None
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    # --- sensitivity: point estimates excluding outlier days ---
    pre_keep = np.array(
        [i for i, d in enumerate(base_days) if d not in set(pre_outliers)]
    )
    post_keep = np.array(
        [i for i, d in enumerate(post_days) if d not in set(post_outliers)]
    )
    sens = None
    if len(pre_keep) and len(post_keep) and (
        len(pre_keep) < len(base_days) or len(post_keep) < len(post_days)
    ):
        pk = _cmp_scalar_kpis(pre_mat, pre_day_w, pre_sub, base_days, pre_keep)
        qk = _cmp_scalar_kpis(post_mat, post_day_w, post_sub, post_days, post_keep)
        sens = {
            "gini_delta": qk["gini"] - pk["gini"],
            "capture_delta": _capture_at_k(qk["shares"]) - _capture_at_k(pk["shares"]),
            "reloc_delta": (
                qk["reloc_rate"] - pk["reloc_rate"]
                if pk["reloc_rate"] is not None and qk["reloc_rate"] is not None
                else None
            ),
            "tpa_delta": qk["trips_per_active"] - pk["trips_per_active"],
        }

    # --- dwell censoring guard ---
    pre_dwell_share = float(pre_sub["dwell"].notna().mean())
    post_dwell_share = float(post_sub["dwell"].notna().mean())
    if abs(pre_dwell_share - post_dwell_share) > 0.10:
        warn.append("dwell_censoring")

    # --- cells GeoJSON (compare resolution polygons) ---
    features = []
    for i, c in enumerate(cells):
        if x_pre[i] == 0 and x_post[i] == 0:
            continue
        boundary = h3_cell_to_boundary(c)
        coords = [[lng, lat] for lat, lng in boundary]
        if coords and coords[0] != coords[-1]:
            coords.append(coords[0])
        lat_c, lng_c = h3_cell_to_latlng(c)
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [coords]},
                "properties": {
                    "cell": c,
                    "lat": float(lat_c),
                    "lng": float(lng_c),
                    "x_pre": float(x_pre[i]),
                    "x_post": float(x_post[i]),
                    "s_pre_pct": float(s_pre[i] * 100.0),
                    "s_post_pct": float(s_post[i] * 100.0),
                    "delta_pp": float(delta_pp[i]),
                    "z": float(z[i]),
                    "cls": str(cls[i]),
                    "pre_rank": int(pre_rank.get(c, 0)),
                },
            }
        )

    kpis = {
        "capture": {
            "label": f"Top-{cap_k} recommended-cell demand capture",
            "pre": _capture_at_k(s_pre),
            "post": _capture_at_k(s_post),
            "unit": "%",
            "ci": _ci("capture"),
            "better": "up",
        },
        "gini": {
            "label": "Demand concentration (Gini)",
            "pre": pre_k["gini"],
            "post": post_k["gini"],
            "unit": "",
            "ci": _ci("gini"),
            "better": "up",
        },
        "reloc": {
            "label": f"Relocation rate (next start >{int(COMPARE_RELOC_THRESHOLD_M)} m)",
            "pre": pre_k["reloc_rate"],
            "post": post_k["reloc_rate"],
            "unit": "share",
            "ci": _ci("reloc"),
            "better": "down",
        },
        "tpa": {
            "label": "Trips per active vehicle per day",
            "pre": pre_k["trips_per_active"],
            "post": post_k["trips_per_active"],
            "unit": "",
            "ci": _ci("tpa"),
            "better": None,
        },
        "dwell": {
            "label": "Median same-spot dwell (mins, capped 24 h)",
            "pre": pre_k["median_dwell"],
            "post": post_k["median_dwell"],
            "unit": "min",
            "ci": _ci("dwell"),
            "better": None,
        },
    }

    n_sufficient = int(sufficient.sum())
    n_sig = int(((cls == "sig_up") | (cls == "sig_down")).sum())

    return _clean_nan(
        {
            "enabled": True,
            "method": {
                "compare_res": COMPARE_H3_RES,
                "bbox": bbox,
                "baseline_mode": baseline_mode,
                "pre_label": pre_bundle.get("label", "Pre"),
                "post_label": post_bundle.get("label", "Post"),
                "pre_window": [str(base_days[0].date()), str(base_days[-1].date())],
                "post_window": [str(post_days[0].date()), str(post_days[-1].date())],
                "n_pre_days": len(base_days),
                "n_post_days": len(post_days),
                "pre_outlier_days": [str(d.date()) for d in pre_outliers],
                "post_outlier_days": [str(d.date()) for d in post_outliers],
                "weekday_reweighted": True,
                "z_color": COMPARE_Z_COLOR,
                "z_candidate": COMPARE_Z_CANDIDATE,
                "se_method": "weekday_stratified_day_block_bootstrap",
                "boot_iters": COMPARE_BOOT_ITERS,
                "background_ratio": round(float(gamma), 3),
                "z_color_effective": round(z_color_thr, 2),
                "z_candidate_effective": round(z_cand_thr, 2),
                "min_lift": COMPARE_MIN_LIFT,
                "min_ends": COMPARE_MIN_ENDS,
                "station_count": k_hat,
                "deploy_date": deploy_date,
                "k_used": cap_k,
                "stations_inferred": k_hat is None,
                "warnings": warn,
            },
            "denominators": {
                "pre": {
                    "rides": int(n_pre),
                    "dwells_defined": int(pre_sub["dwell"].notna().sum()),
                    "mean_daily_active": float(
                        pre_sub.groupby("day")["vehicle_id"].nunique().mean()
                    ),
                },
                "post": {
                    "rides": int(n_post),
                    "dwells_defined": int(post_sub["dwell"].notna().sum()),
                    "mean_daily_active": float(
                        post_sub.groupby("day")["vehicle_id"].nunique().mean()
                    ),
                },
            },
            "kpis": kpis,
            "sensitivity": sens,
            "guards": {
                "off_station_retention": retention,
                "start_share_pre": start_share_pre,
                "start_share_post": start_share_post,
            },
            "curve": {
                "k": list(range(1, k_max + 1)),
                "pre": [float(v) for v in c_pre],
                "post": [float(v) for v in c_post],
                "band_lo": [float(v) for v in band_lo] if band_lo is not None else None,
                "band_hi": [float(v) for v in band_hi] if band_hi is not None else None,
                "order": order[:k_max],
            },
            "candidates": candidates,
            "declined": declined,
            "cells_fc": {"type": "FeatureCollection", "features": features},
            "stats": {
                "n_cells": len(cells),
                "n_sufficient": n_sufficient,
                "n_significant": n_sig,
            },
        }
    )


def build_analysis_bundle(
    df_rides: pd.DataFrame,
    *,
    label: str,
    map_search_area_mode: str,
    map_search_area_manual: list[float] | tuple[float, float, float, float] | None,
    h3_resolution: int,
    time_period_mode: str,
    auto_start: bool,
    auto_start_p75_frac: float,
    time_period_start_fixed: datetime,
    time_period_end_fixed: datetime | None,
    cycle_duration: int,
    i_sigma: int,
    i_s: int,
    optimiser_weights: dict[str, float],
    dwell_same_loc_radius_m: float = 50,
) -> dict:
    df_processed, dataset_end_max = add_datetime_and_dwell_columns(
        df_rides, dwell_same_loc_radius_m=dwell_same_loc_radius_m
    )
    map_search_area = compute_map_search_area(
        df_processed,
        map_search_area_mode=map_search_area_mode,
        map_search_area_manual=map_search_area_manual,
    )
    df_processed = add_h3_columns(
        df_processed, map_search_area=map_search_area, h3_resolution=h3_resolution
    )
    time_period, _, _ = compute_time_period(
        df_processed,
        map_search_area=map_search_area,
        dataset_end_max=dataset_end_max,
        time_period_mode=time_period_mode,
        auto_start=auto_start,
        auto_start_p75_frac=auto_start_p75_frac,
        time_period_start_fixed=time_period_start_fixed,
        time_period_end_fixed=time_period_end_fixed,
    )
    list_time_intervals = populate_list_intervals(time_period, cycle_duration)
    dict_scores, list_sorted_optimisedlocations, predefined_locations_h3 = compute_station_selection(
        df_processed,
        list_time_intervals=list_time_intervals,
        i_sigma=i_sigma,
        i_s=i_s,
        optimiser_weights=optimiser_weights,
    )
    df_hex_metrics_all, df_selected_summary = compute_hex_metrics(
        df_processed,
        time_period=time_period,
        predefined_locations_h3=predefined_locations_h3,
        dict_scores=dict_scores,
        cycle_duration=cycle_duration,
        dwell_same_loc_radius_m=dwell_same_loc_radius_m,
    )
    general_stats_data, general_stats_rows_html = build_general_stats_data(
        df_processed, time_period=time_period
    )
    payload = build_payload(
        df_processed,
        df_hex_metrics_all=df_hex_metrics_all,
        time_period=time_period,
        cycle_duration=cycle_duration,
        i_sigma=i_sigma,
        i_s=i_s,
        optimiser_weights=optimiser_weights,
        dwell_same_loc_radius_m=dwell_same_loc_radius_m,
        general_stats_data=general_stats_data,
    )
    return {
        "label": label,
        "df_rides": df_processed,
        "dataset_end_max": dataset_end_max,
        "map_search_area": map_search_area,
        "time_period": time_period,
        "list_time_intervals": list_time_intervals,
        "dict_scores": dict_scores,
        "list_sorted_optimisedlocations": list_sorted_optimisedlocations,
        "predefined_locations_h3": predefined_locations_h3,
        "df_hex_metrics_all": df_hex_metrics_all,
        "df_selected_summary": df_selected_summary,
        "dwell_same_loc_radius_m": float(dwell_same_loc_radius_m),
        "general_stats_data": general_stats_data,
        "general_stats_rows_html": general_stats_rows_html,
        "payload": payload,
    }
