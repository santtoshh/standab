from __future__ import annotations

import base64
import json
import math
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
