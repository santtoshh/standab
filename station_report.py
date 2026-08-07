#!/usr/bin/env python3
"""Standab station-performance report generator (standalone).

This module is a *self-contained* generator: it reads one or more raw ride-export
CSVs, computes station/docking/energy metrics against a fixed set of physical rack
coordinates, and writes a single HTML file with hand-rolled inline SVG charts. The
output has **no external CSS/JS/font/CDN references** and renders offline via
``file://``.

It is deliberately independent of the Flask webapp (``app.py``) and the analysis
notebook. The only thing it borrows from the rest of the project is the corrupted
timestamp repair routine ``operator_dataset_toggle._parse_time_series`` so that
timestamps damaged by stray internal spaces are healed identically everywhere.

Pipeline
--------
1. **Load** every ``--rides`` CSV as raw strings. Column *names* are normalised via
   an alias map -- never positions, because the two Vaasa exports genuinely ship
   their coordinate columns in different orders (``start_lon,start_lat`` in the
   Jun file vs ``start_lat,start_lon`` in the Jul/Aug file).
2. **Concatenate + de-duplicate.** The two files are contiguous in time, but an
   exact duplicate row across the join would double-count a ride, so exact
   duplicates over all seven fields are dropped.
3. **Parse timestamps** with ``_parse_time_series`` (UTC-tagged, then stripped back
   to naive local wall-clock -- the exports carry local naive times, so tagging
   them UTC shifts nothing and ``tz_localize(None)`` returns the original clock).
   Rows whose start or end timestamp cannot be parsed are dropped.
4. **Drop sub-``MIN_RIDE_M`` rides** using a vectorised haversine start->end. This
   exclusion is *global*: every metric in the report is computed on the surviving
   set. Rides with missing end coordinates cannot be measured, so they are *not*
   dropped here -- they survive and are reported separately as "unassessable".
5. **Cluster racks into sites** by single-link (transitive) clustering at
   ``MERGE_RADIUS_M``. Two racks within the radius join the same site; a site's
   position is the centroid of its racks. For the 22 Vaasa racks this yields 12
   sites with rack counts [4,3,2,2,2,2,2,1,1,1,1,1].
6. **Assign docking.** A ride "docked" when its end point is within
   ``DOCK_RADIUS_M`` of any site centroid. Each docked end is attributed to its
   **nearest** site so per-site counts partition the total exactly (see note
   below).
7. **Aggregate** per day / ISO week / month and render SVG.

Note on nearest-site attribution
--------------------------------
Three site pairs sit closer together than 2 x ``DOCK_RADIUS_M`` (97.1 m, 95.6 m
and 83.7 m apart), so their capture buffers overlap. Counting an end once per
site it falls near would double-count 61 ends and make the per-site column sum to
more than the true docked total. Attributing each end to the nearest site keeps
the per-site table a true partition of the headline figure.

Energy model
------------
``kWh_need = rides * WH_PER_RIDE / 1000``. ``WH_PER_RIDE`` is all-in: traction
drain during the ride plus between-ride standby draw amortised per ride.

Charging CSV schema (``--charging``)
------------------------------------
The Metabase charging export is session-level, one row per plug-in::

    rack_id, receptacle_id, duration_in_s, energy_used_in_wh, city, operator,
    vehicle_id, started_at, finished_at, vehicle_qr

Quirks the loader handles explicitly, because each one silently corrupts the
numbers if missed: a UTF-8 BOM on the header line (``encoding='utf-8-sig'``),
thousands separators inside ``duration_in_s`` (``thousands=','``), and
``started_at`` / ``finished_at`` formatted as ``"June 15, 2026, 07:24"``
(``'%B %d, %Y, %H:%M'``).

**Timezone.** The charging export is in UTC; the ride exports are Finnish local
time (UTC+3, EEST). ``CHARGING_TZ_SHIFT_H`` hours are added to every charging
timestamp before any joining or bucketing. See that constant for how the offset
was established and how it is re-asserted at generation time.

Vehicles are linked between the two sources through the digits in
``vehicle_qr`` (``https://m.ryde.vip/scooter.html?n=135811`` -> ``135811``),
which match the ride exports' ``vehicle_id``.

Position log vs analysis set
----------------------------
Two different ride frames are used on purpose:

* the **analysis set** -- after the sub-``MIN_RIDE_M`` exclusion -- carries every
  ride-volume, energy and docking-rate metric;
* the **position log** -- the same rides *before* that exclusion -- answers only
  "where and when was this vehicle last seen". Applying the sub-100 m rule here
  would delete exactly the short repositioning hops that precede a plug-in and
  would make the preceding-ride gap look older and further away than it was.
  Section 6 states the effect of this choice on its own headline number.

CLI
---
``python station_report.py --rides A.csv [--rides B.csv] [--charging C.csv]
[--racks R.csv] [--charging-tz-shift H] [--out path.html] [--self-test]``

Bare ``python station_report.py`` uses the default Vaasa ride, charging and rack
files.
"""

from __future__ import annotations

import argparse
import html as _html
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from operator_dataset_toggle import _parse_time_series  # noqa: E402

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

WH_PER_RIDE = 46          # Wh consumed per ride, all-in (ride drain + standby)
DOCK_RADIUS_M = 50.0      # ride-end within this distance of a site == docked
MERGE_RADIUS_M = 50.0     # racks within this distance merge into one site
MIN_RIDE_M = 100.0        # rides shorter than this (straight-line) are excluded
EARTH_RADIUS_M = 6371000.0

NEAR_ZERO_END_THRESHOLD = 10   # sites at/below this many ends get flagged

# --- Charging <-> rides alignment ------------------------------------------- #
# The charging export is UTC; the ride exports are Finnish local time (UTC+3,
# EEST). ESTABLISHED EMPIRICALLY, not assumed: sweeping candidate offsets from
# -5 h to +5 h and counting charging sessions whose vehicle's own preceding ride
# ended within RIDER_DOCK_RADIUS_M of that rack and less than
# RIDER_DOCK_GAP_MIN before the session started gives a single sharp maximum at
# +3 h (162 matches, median rack distance 84 m) against 11-36 matches and
# 590-1100 m at every other offset. A wrong offset would not merely blur this
# join, it would invert Section 6's conclusion, so the sweep is re-run at
# generation time and the report shouts if the winner is no longer +3 h.
CHARGING_TZ_SHIFT_H = 3
TZ_SCAN_OFFSETS = list(range(-5, 6))

CHARGING_OPERATOR = "Ryde"     # ride exports are Ryde's fleet; see Section 0

RIDER_DOCK_RADIUS_M = 50.0     # rider-docked: last ride ended this close to the rack
RIDER_DOCK_GAP_MIN = 15.0      # ...and less than this many minutes before plug-in
REVERSE_PLUG_WINDOW_MIN = 30.0  # reverse view: ride-end -> plug-in within this
GAP_BUCKET_EDGES_MIN = [15.0, 60.0, 240.0, 1440.0]
GAP_BUCKET_LABELS = ["<15 min, other rack", "15-60 min", "1-4 h", "4-24 h",
                     "no ride in 24 h"]

# Racks whose usage profile is a depot/yard rather than a public station. Kept in
# every energy total, excluded from rider-behaviour metrics; both disclosed.
DEPOT_RACK_IDS = {"016-0008-0064"}

# A rack whose first session is this far after the first session in the whole
# file was commissioned mid-window; its low volume is newness, not failure.
COMMISSION_LAG_DAYS = 7

# 22 physical rack coordinates in Vaasa, (latitude, longitude).
RACK_COORDS: list[tuple[float, float]] = [
    (63.096921850, 21.616303750),
    (63.095746267, 21.607832867),
    (63.100608583, 21.628855033),
    (63.102767683, 21.624930933),
    (63.096963450, 21.619538450),
    (63.096337233, 21.616485167),
    (63.100304367, 21.624574683),
    (63.096967383, 21.618556517),
    (63.097145733, 21.621802733),
    (63.098636217, 21.609760750),
    (63.100481767, 21.624683150),
    (63.096361300, 21.614268133),
    (63.105118617, 21.595999400),
    (63.095771583, 21.615077083),
    (63.097092967, 21.621845317),
    (63.097534733, 21.617957983),
    (63.096566283, 21.616655150),
    (63.104909417, 21.596305917),
    (63.097084417, 21.616137067),
    (63.096480383, 21.614398900),
    (63.096419817, 21.615096433),
    (63.095512000, 21.615707983),
]

DEFAULT_RIDE_FILES = [
    "/Users/santtoshmohan/Downloads/Vaasa Trips 14.06-28.06 (1).csv",
    "/Users/santtoshmohan/Downloads/Trips Vaasa 29.06-06.08.csv",
]
DEFAULT_CHARGING_FILE = (
    "/Users/santtoshmohan/Downloads/"
    "charging_report_2026-08-07T11_10_28.250929411+02_00.csv"
)
DEFAULT_RACK_FILE = _REPO_ROOT / "data" / "vaasa_racks.csv"
DEFAULT_OUT = _REPO_ROOT / "reports" / "station_performance_report.html"

# Column alias map -> canonical name (matched case-insensitively).
COLUMN_ALIASES: dict[str, str] = {
    "qrcode": "vehicle_id", "qr_code": "vehicle_id", "vehicle_id": "vehicle_id",
    "vehicleid": "vehicle_id", "vehicle": "vehicle_id", "scooter_id": "vehicle_id",
    "starttime": "start_time", "start_time": "start_time", "start_date": "start_time",
    "endtime": "end_time", "end_time": "end_time", "end_date": "end_time",
    "start_lat": "start_lat", "start_latitude": "start_lat",
    "vehicle_start_latitude": "start_lat",
    "start_lon": "start_lon", "start_lng": "start_lon", "start_longitude": "start_lon",
    "vehicle_start_longitude": "start_lon",
    "end_lat": "end_lat", "end_latitude": "end_lat", "vehicle_end_latitude": "end_lat",
    "end_lon": "end_lon", "end_lng": "end_lon", "end_longitude": "end_lon",
    "vehicle_end_longitude": "end_lon",
}
CANONICAL_COLUMNS = [
    "vehicle_id", "start_time", "end_time",
    "start_lat", "start_lon", "end_lat", "end_lon",
]

# Validated categorical palette (light surface). Slots 1-3 clear the all-pairs
# CVD and normal-vision floors; aqua sits below 3:1 contrast so every chart that
# uses it carries direct labels or an accompanying table (the "relief" rule).
C_BLUE = "#2a78d6"
C_ORANGE = "#eb6834"
C_AQUA = "#1baf7a"
C_SURFACE = "#ffffff"
C_GRID = "#e8e8e4"
C_AXIS = "#c9c9c2"
C_INK = "#0b0b0b"
C_INK2 = "#52514e"
C_MUTED = "#8a8a83"
C_PARTIAL = "#f3f2ee"
C_BAND = "#cddff5"        # uncertainty-band fill (Section 5)

# The gap buckets in Section 6 are ORDINAL (<15 min ... no ride in 24 h), so they
# get a single-hue sequential ramp rather than categorical slots -- an ordered
# quantity encoded in unordered colours is unreadable. "Rider-docked" is not part
# of that order (it is the finding), so it takes the orange accent and separates
# cleanly from every blue in the ramp.
C_SEQ = ["#d3e3f6", "#a5c5e9", "#77a6db", "#4a8bd0", "#1f5fa8"]

GRAN_LABELS = {"day": "Day", "week": "Week", "month": "Month"}


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #

def haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in metres. Fully vectorised; broadcasts."""
    rad = np.pi / 180.0
    lat1 = np.asarray(lat1, dtype=float) * rad
    lon1 = np.asarray(lon1, dtype=float) * rad
    lat2 = np.asarray(lat2, dtype=float) * rad
    lon2 = np.asarray(lon2, dtype=float) * rad
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


@dataclass
class Rack:
    """One physical rack from the registry CSV."""
    rack_id: str
    lat: float
    lon: float
    receptacles: int


@dataclass
class Site:
    index: int
    lat: float
    lon: float
    rack_count: int
    rack_indices: list[int] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"S{self.index + 1:02d}"


def cluster_racks(racks: list[tuple[float, float]],
                  radius_m: float = MERGE_RADIUS_M) -> list[Site]:
    """Single-link (transitive) clustering of racks; site position = centroid.

    Two racks within ``radius_m`` belong to the same site, and that relation is
    transitive, so a chain of nearby racks forms one site.
    """
    n = len(racks)
    if n == 0:
        return []
    lat = np.array([r[0] for r in racks], dtype=float)
    lon = np.array([r[1] for r in racks], dtype=float)
    dist = haversine_m(lat[:, None], lon[:, None], lat[None, :], lon[None, :])

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if dist[i, j] <= radius_m:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    ordered = sorted(groups.values(), key=lambda g: (-len(g), g[0]))
    return [
        Site(index=k, lat=float(lat[g].mean()), lon=float(lon[g].mean()),
             rack_count=len(g), rack_indices=sorted(g))
        for k, g in enumerate(ordered)
    ]


def clustering_stability(racks: list[tuple[float, float]],
                         radius_m: float = MERGE_RADIUS_M
                         ) -> tuple[float, float, int, int]:
    """Over what range of merge radii does the site count stay put?

    The site count is a step function of the merge radius that can only change at
    an actual rack-pair distance, and it is monotone non-increasing. This walks
    those breakpoints and returns ``(lo, hi, count, count_just_below_lo)``: the
    count holds for any radius in ``[lo, hi)``. Reported so that a reader can see
    whether the chosen radius sits on a plateau or on a knife edge.
    """
    n = len(racks)
    lat = np.array([r[0] for r in racks], dtype=float)
    lon = np.array([r[1] for r in racks], dtype=float)
    d = haversine_m(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    breaks = sorted({float(d[i, j]) for i in range(n) for j in range(i + 1, n)})
    here = len(cluster_racks(racks, radius_m))
    lo = min((b for b in breaks if b <= radius_m
              and len(cluster_racks(racks, b)) == here), default=radius_m)
    hi = min((b for b in breaks if b > radius_m
              and len(cluster_racks(racks, b)) != here), default=float("inf"))
    below = len(cluster_racks(racks, math.nextafter(lo, 0.0)))
    return lo, hi, here, below


# --------------------------------------------------------------------------- #
# Loading & cleaning
# --------------------------------------------------------------------------- #

def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename by *name* (case-insensitive alias map), never by position."""
    mapping = {}
    for col in df.columns:
        key = str(col).strip().lower().replace(" ", "_")
        if key in COLUMN_ALIASES:
            mapping[col] = COLUMN_ALIASES[key]
    out = df.rename(columns=mapping)
    missing = [c for c in CANONICAL_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"missing required columns after normalisation: {missing}")
    return out[CANONICAL_COLUMNS].copy()


def load_rides(paths: list[str | Path]) -> tuple[pd.DataFrame, list[dict]]:
    """Read every ride CSV as strings, normalise columns, stack. Returns raw frame."""
    frames, per_file = [], []
    for p in paths:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(f"ride file not found: {path}")
        raw = pd.read_csv(path, dtype=str)
        norm = normalise_columns(raw)
        norm["source_file"] = path.name
        frames.append(norm)
        per_file.append({"file": path.name, "rows": int(len(norm)),
                         "columns": list(raw.columns)})
    combined = pd.concat(frames, ignore_index=True)
    return combined, per_file


@dataclass
class Waterfall:
    raw_rows: int = 0
    dropped_duplicate: int = 0
    dropped_unparseable: int = 0
    dropped_short: int = 0
    kept: int = 0
    missing_end_coords: int = 0
    rides_under_1min: int = 0

    def reconciles(self) -> bool:
        return self.raw_rows == (self.kept + self.dropped_duplicate
                                 + self.dropped_unparseable + self.dropped_short)


def _clean_rides(raw: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Dedupe + parse timestamps + coerce coordinates. No length filtering.

    Shared by ``prepare_rides`` (which then applies the sub-``MIN_RIDE_M`` rule)
    and ``position_log`` (which must not).
    """
    df = raw.copy()

    dup_mask = df.duplicated(subset=CANONICAL_COLUMNS, keep="first")
    n_dup = int(dup_mask.sum())
    df = df.loc[~dup_mask].copy()

    df["start_time"] = _parse_time_series(df["start_time"]).dt.tz_localize(None)
    df["end_time"] = _parse_time_series(df["end_time"]).dt.tz_localize(None)
    bad_ts = df["start_time"].isna() | df["end_time"].isna()
    n_bad = int(bad_ts.sum())
    df = df.loc[~bad_ts].copy()

    for col in ("start_lat", "start_lon", "end_lat", "end_lon"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["trip_m"] = haversine_m(df["start_lat"].to_numpy(), df["start_lon"].to_numpy(),
                               df["end_lat"].to_numpy(), df["end_lon"].to_numpy())
    return df, n_dup, n_bad


def position_log(raw: pd.DataFrame) -> pd.DataFrame:
    """Vehicle position history: every parsed ride-end with coordinates.

    Deliberately built BEFORE the sub-``MIN_RIDE_M`` exclusion. This frame is
    never used for ride counts, only to answer "where and when was this vehicle
    last seen" for the Section 6 join. Filtering it would drop the short
    repositioning hops that immediately precede a plug-in, which is precisely the
    behaviour that section measures.
    """
    df, _, _ = _clean_rides(raw)
    out = df.loc[df["end_lat"].notna() & df["end_lon"].notna(),
                 ["vehicle_id", "end_time", "end_lat", "end_lon"]].copy()
    out["vehicle_id"] = out["vehicle_id"].astype(str)
    return out.sort_values("end_time").reset_index(drop=True)


def prepare_rides(raw: pd.DataFrame,
                  min_ride_m: float = MIN_RIDE_M) -> tuple[pd.DataFrame, Waterfall]:
    """Dedupe -> parse timestamps -> drop sub-``min_ride_m`` rides.

    Rides with missing end coordinates survive (their length is unmeasurable) and
    are counted separately; they are later excluded from the docking denominator.
    """
    wf = Waterfall(raw_rows=int(len(raw)))
    df, wf.dropped_duplicate, wf.dropped_unparseable = _clean_rides(raw)

    # NaN distance (missing end coords) compares False -> such rides are kept.
    short_mask = df["trip_m"] < min_ride_m
    wf.dropped_short = int(short_mask.sum())
    df = df.loc[~short_mask].copy()

    wf.kept = int(len(df))
    wf.missing_end_coords = int(df["end_lat"].isna().sum())
    dur_s = (df["end_time"] - df["start_time"]).dt.total_seconds()
    wf.rides_under_1min = int((dur_s < 60).sum())

    df["duration_s"] = dur_s
    df["date"] = df["start_time"].dt.normalize()
    df["week"] = df["date"] - pd.to_timedelta(df["start_time"].dt.weekday, unit="D")
    df["month"] = df["start_time"].dt.to_period("M").dt.start_time
    return df.reset_index(drop=True), wf


def assign_docking(df: pd.DataFrame, sites: list[Site],
                   radius_m: float = DOCK_RADIUS_M) -> pd.DataFrame:
    """Attach ``docked`` / ``dock_site`` / ``start_docked`` columns.

    Each qualifying end is attributed to its NEAREST site so per-site counts sum
    to the headline docked total (site buffers overlap in three places).
    """
    out = df.copy()
    slat = np.array([s.lat for s in sites], dtype=float)
    slon = np.array([s.lon for s in sites], dtype=float)

    out["has_end"] = out["end_lat"].notna() & out["end_lon"].notna()
    out["docked"] = False
    out["dock_site"] = -1
    if out["has_end"].any():
        idx = out.index[out["has_end"]]
        d = haversine_m(out.loc[idx, "end_lat"].to_numpy()[:, None],
                        out.loc[idx, "end_lon"].to_numpy()[:, None],
                        slat[None, :], slon[None, :])
        nearest = d.argmin(axis=1)
        mind = d.min(axis=1)
        ok = mind <= radius_m
        out.loc[idx, "docked"] = ok
        out.loc[idx[ok], "dock_site"] = nearest[ok]

    out["has_start"] = out["start_lat"].notna() & out["start_lon"].notna()
    out["start_docked"] = False
    if out["has_start"].any():
        idx = out.index[out["has_start"]]
        d = haversine_m(out.loc[idx, "start_lat"].to_numpy()[:, None],
                        out.loc[idx, "start_lon"].to_numpy()[:, None],
                        slat[None, :], slon[None, :])
        out.loc[idx, "start_docked"] = d.min(axis=1) <= radius_m
    return out


# --------------------------------------------------------------------------- #
# Metrics (pure functions over a prepared frame)
# --------------------------------------------------------------------------- #

def period_bounds(period_start: pd.Timestamp, gran: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    if gran == "day":
        return period_start, period_start + pd.Timedelta(days=1)
    if gran == "week":
        return period_start, period_start + pd.Timedelta(days=7)
    if gran == "month":
        return period_start, period_start + pd.offsets.MonthBegin(1)
    raise ValueError(gran)


def partial_flags(periods: list[pd.Timestamp], gran: str,
                  data_start: pd.Timestamp, data_end: pd.Timestamp) -> list[bool]:
    """A period is partial when the observation window does not cover all of it."""
    flags = []
    for p in periods:
        lo, hi = period_bounds(p, gran)
        flags.append(bool(lo < data_start or hi > data_end))
    return flags


def fleet_overview(df: pd.DataFrame, gran: str) -> pd.DataFrame:
    """rides / distinct active vehicles / rides per active vehicle per day."""
    grouped = df.groupby(gran if gran != "day" else "date")
    out = pd.DataFrame({
        "rides": grouped.size(),
        "vehicles": grouped["vehicle_id"].nunique(),
        "observed_days": grouped["date"].nunique(),
    })
    out.index.name = "period"
    out["rides_per_vehicle_per_day"] = (
        out["rides"] / out["vehicles"].replace(0, np.nan) / out["observed_days"]
    )
    return out.sort_index()


def energy_need_kwh(rides: pd.Series | np.ndarray | int | float,
                    wh_per_ride: int = WH_PER_RIDE):
    """kWh required to serve ``rides`` rides."""
    return rides * wh_per_ride / 1000.0


def weekly_docking(df: pd.DataFrame) -> pd.DataFrame:
    """Weekly docking success, with the bounds implied by unassessable rides.

    ``dock_rate_pct`` is the point estimate on the assessable denominator, i.e.
    it assumes rides with no end coordinate docked at the same rate as the rest.
    That assumption is not safe here (the missing rides are biased towards
    stations), so two bounds accompany it:

    * ``lo_pct`` -- every unassessable ride failed to dock;
    * ``hi_pct`` -- every unassessable ride docked.

    The truth is inside ``[lo_pct, hi_pct]`` by construction, whatever caused the
    coordinates to go missing. When those brackets overlap across weeks, no
    week-to-week ordering can be claimed.
    """
    g = df.groupby("week")
    out = pd.DataFrame({
        "rides": g.size(),
        "assessable": g["has_end"].sum(),
        "docked": g["docked"].sum(),
    })
    out["unassessable"] = out["rides"] - out["assessable"]
    out["dock_rate_pct"] = 100.0 * out["docked"] / out["assessable"].replace(0, np.nan)
    rides = out["rides"].replace(0, np.nan)
    out["lo_pct"] = 100.0 * out["docked"] / rides
    out["hi_pct"] = 100.0 * (out["docked"] + out["unassessable"]) / rides
    out["unassessable_pct"] = 100.0 * out["unassessable"] / rides
    out.index.name = "period"
    return out.sort_index()


def bounds_overlap(frame: pd.DataFrame, a: pd.Timestamp, b: pd.Timestamp) -> bool:
    """Do the [lo, hi] brackets of two weeks overlap? If so, neither is higher."""
    ra, rb = frame.loc[a], frame.loc[b]
    return bool(ra["lo_pct"] <= rb["hi_pct"] and rb["lo_pct"] <= ra["hi_pct"])


def full_periods(frame: pd.DataFrame, partial: list[bool]) -> pd.DataFrame:
    """Rows of a period-indexed frame whose period is fully inside the window.

    Every headline KPI derived from a weekly series uses this. A partial period
    is a shorter observation, not a smaller value, so pooling it into a mean
    quietly drags the KPI towards whatever the stub happened to contain.
    """
    return frame.loc[~np.asarray(partial, dtype=bool)]


def buffer_end_counts(df: pd.DataFrame, sites: list[Site],
                      radius_m: float = DOCK_RADIUS_M) -> np.ndarray:
    """Ends within ``radius_m`` of each site, counted independently per site.

    Unlike the nearest-site attribution this does NOT partition -- an end inside
    two overlapping buffers is counted twice, so the column sums to more than the
    docked total. That is exactly why both columns exist: nearest-attribution
    answers "what share of docking did this site win", the plain buffer answers
    "is this site being used at all". A site that loses every contested end to a
    busier neighbour 84 m away reads as dead in the first column and alive in the
    second, and only the second is evidence about the hardware.
    """
    counts = np.zeros(len(sites), dtype=int)
    e = df.loc[df["has_end"]]
    if e.empty or not sites:
        return counts
    slat = np.array([s.lat for s in sites], dtype=float)
    slon = np.array([s.lon for s in sites], dtype=float)
    d = haversine_m(e["end_lat"].to_numpy()[:, None], e["end_lon"].to_numpy()[:, None],
                    slat[None, :], slon[None, :])
    return (d <= radius_m).sum(axis=0).astype(int)


def per_site_table(df: pd.DataFrame, sites: list[Site]) -> pd.DataFrame:
    """Per-site capture. Carries BOTH the attributed and the buffer count.

    ``flagged`` (the "installed but unused" flag) is driven by the buffer count,
    never by the attributed count -- see ``buffer_end_counts``.
    """
    counts = df.loc[df["docked"], "dock_site"].value_counts()
    total = int(counts.sum())
    buf = buffer_end_counts(df, sites)
    rows = []
    for s in sites:
        n = int(counts.get(s.index, 0))
        rows.append({
            "site": s.name,
            "lat": s.lat, "lon": s.lon,
            "racks": s.rack_count,
            "ends": n,
            "buffer_ends": int(buf[s.index]),
            "share_pct": (100.0 * n / total) if total else 0.0,
            "flagged": int(buf[s.index]) <= NEAR_ZERO_END_THRESHOLD,
        })
    return pd.DataFrame(rows).sort_values("ends", ascending=False).reset_index(drop=True)


def weekly_vehicle_docking(df: pd.DataFrame) -> pd.DataFrame:
    """Weekly split of riding vehicles by whether they ever docked.

    Vocabulary matters here and is fixed across the whole report: **active**
    means "rode at all" (Sections 1 and 4); **docked** means "ended at least one
    ride within DOCK_RADIUS_M of a site" (this function). Calling the second one
    "active" too made 96% in one section and 42% in another look like a fleet
    collapse when they measure different things entirely.

    Denominator = vehicles that rode at all that week.
    """
    per = df.groupby(["week", "vehicle_id"])["docked"].any().reset_index()
    g = per.groupby("week")["docked"]
    out = pd.DataFrame({"docked": g.sum(), "riding": g.size()})
    out["never_docked"] = out["riding"] - out["docked"]
    out["docked_pct"] = 100.0 * out["docked"] / out["riding"].replace(0, np.nan)
    out.index.name = "period"
    return out.sort_index()


def fleet_utilisation(df: pd.DataFrame, gran: str) -> pd.DataFrame:
    """Both utilisation denominators, plus fleet-activity share as a percentage."""
    ov = fleet_overview(df, gran)
    total_fleet = int(df["vehicle_id"].nunique())
    ov["total_fleet"] = total_fleet
    ov["rides_per_fleet_vehicle_per_day"] = (
        ov["rides"] / total_fleet / ov["observed_days"] if total_fleet else np.nan
    )
    ov["fleet_active_pct"] = 100.0 * ov["vehicles"] / total_fleet if total_fleet else np.nan
    return ov


# --------------------------------------------------------------------------- #
# Charging loader (tolerant)
# --------------------------------------------------------------------------- #

CHG_REQUIRED_COLUMNS = ["rack_id", "receptacle_id", "energy_used_in_wh", "city",
                        "operator", "vehicle_id", "started_at", "finished_at",
                        "vehicle_qr"]
CHG_TIME_FORMAT = "%B %d, %Y, %H:%M"


class ChargingDataError(Exception):
    """Charging export failed a schema or sanity check.

    Raised rather than allowed to surface as a traceback: ``main`` catches it and
    renders the message inside the report, so a broken export produces a report
    that says what is broken instead of no report at all.
    """


@dataclass
class Charging:
    """A loaded, timezone-corrected, Ryde-only charging export."""
    sessions: pd.DataFrame        # one row per session, local time in `started`
    tz_shift_h: int
    dropped_other_operator: int
    other_operators: list[str]
    dropped_no_qr: int
    unknown_racks: list[str]

    @property
    def total_kwh(self) -> float:
        return float(self.sessions["kwh"].sum())

    @property
    def n_sessions(self) -> int:
        return int(len(self.sessions))


def load_charging(path: str | Path, racks: list[Rack],
                  tz_shift_h: int = CHARGING_TZ_SHIFT_H) -> Charging:
    """Load the Metabase charging export, corrected onto ride-file local time.

    Every file quirk is handled explicitly because each one fails silently rather
    than loudly: the UTF-8 BOM would rename ``rack_id`` to ``\\ufeffrack_id``, the
    thousands separators would make ``duration_in_s`` a string column, and the
    ``"June 15, 2026, 07:24"`` stamps parse to NaT under any default format.
    """
    p = Path(path)
    if not p.exists():
        raise ChargingDataError(f"charging file not found: {p}")
    try:
        df = pd.read_csv(p, encoding="utf-8-sig", thousands=",")
    except Exception as exc:                                  # noqa: BLE001
        raise ChargingDataError(f"charging CSV could not be read: {exc}") from exc

    missing = [c for c in CHG_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ChargingDataError(
            f"charging CSV is missing required column(s) {missing}. Columns "
            f"present: {list(df.columns)}")

    # Defensive operator filter: the ride exports are Ryde's fleet, so mixing a
    # second operator's supply into this demand would inflate every coverage
    # figure in the report.
    op = df["operator"].astype(str).str.strip()
    keep = op == CHARGING_OPERATOR
    others = sorted(set(op[~keep]))
    dropped_op = int((~keep).sum())
    df = df.loc[keep].copy()
    if df.empty:
        raise ChargingDataError(
            f"charging CSV contains no rows for operator '{CHARGING_OPERATOR}' "
            f"(operators present: {others or 'none'})")

    started = pd.to_datetime(df["started_at"], format=CHG_TIME_FORMAT, errors="coerce")
    finished = pd.to_datetime(df["finished_at"], format=CHG_TIME_FORMAT, errors="coerce")
    if started.isna().any():
        bad = df.loc[started.isna(), "started_at"].head(3).tolist()
        raise ChargingDataError(
            f"{int(started.isna().sum())} of {len(df)} started_at values do not "
            f"match the expected format '{CHG_TIME_FORMAT}'; examples: {bad}")

    energy_wh = pd.to_numeric(df["energy_used_in_wh"], errors="coerce")
    if energy_wh.isna().any():
        raise ChargingDataError(
            f"{int(energy_wh.isna().sum())} non-numeric energy_used_in_wh values")
    if (energy_wh < 0).any():
        raise ChargingDataError("negative energy_used_in_wh values present")

    shift = pd.Timedelta(hours=tz_shift_h)
    out = pd.DataFrame({
        "rack_id": df["rack_id"].astype(str).str.strip(),
        "receptacle_id": df["receptacle_id"],
        "kwh": energy_wh / 1000.0,
        "started": started + shift,
        "finished": finished + shift,
        "duration_s": pd.to_numeric(df["duration_in_s"], errors="coerce"),
        # The ride exports key vehicles by the bare digits that the QR URL ends
        # with (".../scooter.html?n=135811" -> "135811").
        "veh": df["vehicle_qr"].astype(str).str.extract(r"(\d{5,})")[0],
    })

    by_id = {r.rack_id: r for r in racks}
    unknown = sorted(set(out["rack_id"]) - set(by_id))
    if unknown:
        raise ChargingDataError(
            f"charging CSV references {len(unknown)} rack_id(s) absent from the "
            f"rack registry: {unknown[:5]}")
    out["rack_lat"] = out["rack_id"].map({k: v.lat for k, v in by_id.items()})
    out["rack_lon"] = out["rack_id"].map({k: v.lon for k, v in by_id.items()})

    dropped_no_qr = int(out["veh"].isna().sum())
    out["date"] = out["started"].dt.normalize()
    out["week"] = out["date"] - pd.to_timedelta(out["started"].dt.weekday, unit="D")
    out["month"] = out["started"].dt.to_period("M").dt.start_time
    out["is_depot"] = out["rack_id"].isin(DEPOT_RACK_IDS)

    return Charging(sessions=out.sort_values("started").reset_index(drop=True),
                    tz_shift_h=tz_shift_h,
                    dropped_other_operator=dropped_op, other_operators=others,
                    dropped_no_qr=dropped_no_qr, unknown_racks=unknown)


def attach_sites(chg: Charging, racks: list[Rack], sites: list[Site]) -> None:
    """Map each session's rack onto its merged site, in place."""
    rack_to_site = {}
    for s in sites:
        for i in s.rack_indices:
            rack_to_site[racks[i].rack_id] = s.index
    chg.sessions["site"] = chg.sessions["rack_id"].map(rack_to_site)


def charging_by_gran(chg: Charging, gran: str) -> pd.Series:
    key = {"day": "date", "week": "week", "month": "month"}[gran]
    return chg.sessions.groupby(key)["kwh"].sum().sort_index()


# --------------------------------------------------------------------------- #
# Timezone alignment: measured, not assumed
# --------------------------------------------------------------------------- #

def _preceding_ride(sessions: pd.DataFrame, poslog: pd.DataFrame,
                    shift_h: float = 0.0) -> pd.DataFrame:
    """For each session, the vehicle's own most recent ride-end before it.

    ``merge_asof(by=vehicle, direction='backward')`` -- the standard tool for
    "last event of the same key at or before this time".
    """
    left = sessions.dropna(subset=["veh"]).copy()
    left["t"] = left["started"] + pd.Timedelta(hours=shift_h)
    left = left.sort_values("t")
    right = poslog.rename(columns={"end_time": "ride_end"})
    m = pd.merge_asof(left, right, left_on="t", right_on="ride_end",
                      left_by="veh", right_by="vehicle_id", direction="backward")
    matched = m["ride_end"].notna().to_numpy()
    dist = np.full(len(m), np.nan)
    if matched.any():
        dist[matched] = haversine_m(
            m.loc[matched, "end_lat"].to_numpy(), m.loc[matched, "end_lon"].to_numpy(),
            m.loc[matched, "rack_lat"].to_numpy(), m.loc[matched, "rack_lon"].to_numpy())
    m["rack_dist_m"] = dist
    m["gap_min"] = (m["t"] - m["ride_end"]).dt.total_seconds() / 60.0
    return m


def scan_tz_offsets(chg: Charging, poslog: pd.DataFrame,
                    offsets: list[int] = TZ_SCAN_OFFSETS) -> pd.DataFrame:
    """Re-derive the ride/charging clock offset from the data itself.

    A session that a *rider* plugged in must be preceded, within minutes, by that
    same vehicle's ride ending at that same rack. Under the correct offset those
    pairs line up and the count spikes; under a wrong one the "preceding ride" is
    some unrelated earlier trip, kilometres away. ``offset_h`` is the total shift
    applied to the raw UTC stamps, so a healthy scan puts its maximum exactly on
    the configured ``CHARGING_TZ_SHIFT_H``.
    """
    rows = []
    for off in offsets:
        m = _preceding_ride(chg.sessions, poslog, shift_h=off - chg.tz_shift_h)
        near = (m["gap_min"] < RIDER_DOCK_GAP_MIN)
        hit = near & (m["rack_dist_m"] <= RIDER_DOCK_RADIUS_M)
        med = float(np.nanmedian(m.loc[near, "rack_dist_m"])) if near.any() else np.nan
        rows.append({"offset_h": off, "matches": int(hit.sum()),
                     "median_dist_m": med})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Who is docking: charging sessions attributed to riders vs operations
# --------------------------------------------------------------------------- #

def _gap_bucket(gap_min: np.ndarray, rider: np.ndarray) -> np.ndarray:
    """Ordinal bucket per session. ``rider`` sessions are removed from the order."""
    out = np.full(len(gap_min), GAP_BUCKET_LABELS[-1], dtype=object)
    e = GAP_BUCKET_EDGES_MIN
    with np.errstate(invalid="ignore"):
        out[(gap_min < e[0])] = GAP_BUCKET_LABELS[0]
        out[(gap_min >= e[0]) & (gap_min < e[1])] = GAP_BUCKET_LABELS[1]
        out[(gap_min >= e[1]) & (gap_min < e[2])] = GAP_BUCKET_LABELS[2]
        out[(gap_min >= e[2]) & (gap_min < e[3])] = GAP_BUCKET_LABELS[3]
    out[np.isnan(gap_min)] = GAP_BUCKET_LABELS[-1]
    out[rider] = "rider-docked"
    return out


def classify_sessions(chg: Charging, poslog: pd.DataFrame) -> pd.DataFrame:
    """Label every charging session by how its vehicle got to the rack.

    ``rider-docked`` = the vehicle's own preceding ride ended within
    ``RIDER_DOCK_RADIUS_M`` of *that same rack* and less than
    ``RIDER_DOCK_GAP_MIN`` before the session started. Everything else is bucketed
    by how stale that preceding ride is.

    This is a *necessary* condition, not a sufficient one, and it is stated as
    such in the report: a van can collect a scooter parked beside a rack and plug
    it in two minutes later, and it will be counted as rider-docked. The rule
    therefore over-counts riders, which makes the headline a ceiling.
    """
    m = _preceding_ride(chg.sessions, poslog, shift_h=0.0)
    rider = ((m["rack_dist_m"] <= RIDER_DOCK_RADIUS_M)
             & (m["gap_min"] < RIDER_DOCK_GAP_MIN)).fillna(False).to_numpy()
    m["rider_docked"] = rider
    m["bucket"] = _gap_bucket(m["gap_min"].to_numpy(dtype=float), rider)
    return m


def docking_attribution(classified: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Weekly counts per bucket, plus whole-period shares."""
    order = ["rider-docked"] + GAP_BUCKET_LABELS
    pv = (classified.pivot_table(index="week", columns="bucket", values="kwh",
                                 aggfunc="size")
          .reindex(columns=order).fillna(0.0).sort_index())
    shares = 100.0 * classified["bucket"].value_counts().reindex(order).fillna(0) \
        / max(1, len(classified))
    return pv, shares


def reverse_plugin_rate(poslog: pd.DataFrame, chg: Charging, racks: list[Rack],
                        since: pd.Timestamp | None = None) -> pd.DataFrame:
    """Of ride-ends parked at a rack, how many are followed by a plug-in?

    The mirror image of ``classify_sessions``: it asks the question from the
    rider's side. To be a true mirror the plug-in must happen at the *same rack*
    the rider parked beside — the forward rule requires the ride to have ended
    within ``RIDER_DOCK_RADIUS_M`` of the rack that logged the session, so
    accepting a plug-in at any rack would count a different event (the vehicle
    was moved, then charged elsewhere).

    Returns both, because the gap between them is itself informative:
      ``pct``           - same-rack plug-ins (the strict mirror)
      ``pct_any_rack``  - plug-in at any rack (upper bound; includes vehicles
                          collected from this rack and charged at another)
    """
    cols = ["ends", "plugged", "pct", "plugged_any", "pct_any_rack"]
    if poslog.empty:
        return pd.DataFrame(columns=cols)
    rlat = np.array([r.lat for r in racks], dtype=float)
    rlon = np.array([r.lon for r in racks], dtype=float)
    rid = np.array([r.rack_id for r in racks], dtype=object)
    d = haversine_m(poslog["end_lat"].to_numpy()[:, None],
                    poslog["end_lon"].to_numpy()[:, None], rlat[None, :], rlon[None, :])
    near = d.min(axis=1) <= RIDER_DOCK_RADIUS_M
    at_rack = poslog.loc[near].copy()
    # which rack the rider actually parked beside (nearest wins)
    at_rack["parked_rack_id"] = rid[d[near].argmin(axis=1)]
    if since is not None:
        at_rack = at_rack.loc[at_rack["end_time"] >= since]
    if at_rack.empty:
        return pd.DataFrame(columns=cols)

    at_rack = at_rack.sort_values("end_time")
    starts = (chg.sessions.dropna(subset=["veh"])[["veh", "started", "rack_id"]]
              .rename(columns={"rack_id": "charged_rack_id"})
              .sort_values("started"))
    r = pd.merge_asof(at_rack, starts, left_on="end_time", right_on="started",
                      left_by="vehicle_id", right_by="veh", direction="forward")
    within = ((r["started"] - r["end_time"]).dt.total_seconds() / 60.0
              <= REVERSE_PLUG_WINDOW_MIN).fillna(False)
    r["plugged_any"] = within
    r["plugged"] = within & (r["charged_rack_id"] == r["parked_rack_id"])
    r["week"] = (r["end_time"].dt.normalize()
                 - pd.to_timedelta(r["end_time"].dt.weekday, unit="D"))
    g = r.groupby("week")
    out = pd.DataFrame({"ends": g.size(),
                        "plugged": g["plugged"].sum(),
                        "plugged_any": g["plugged_any"].sum()})
    out["pct"] = 100.0 * out["plugged"] / out["ends"].replace(0, np.nan)
    out["pct_any_rack"] = 100.0 * out["plugged_any"] / out["ends"].replace(0, np.nan)
    out.index.name = "period"
    return out.sort_index()


def rack_commissioning(chg: Charging, racks: list[Rack],
                       lag_days: int = COMMISSION_LAG_DAYS) -> pd.DataFrame:
    """First-session date per rack, so newness is not read as failure."""
    g = chg.sessions.groupby("rack_id").agg(
        first_session=("started", "min"), last_session=("started", "max"),
        sessions=("kwh", "size"), kwh=("kwh", "sum"))
    fleet_start = chg.sessions["started"].min()
    g["days_late"] = (g["first_session"] - fleet_start).dt.total_seconds() / 86400.0
    g["late"] = g["days_late"] >= lag_days
    by_id = {r.rack_id: r for r in racks}
    g["receptacles"] = [by_id[i].receptacles for i in g.index]
    g["kwh_per_receptacle"] = g["kwh"] / g["receptacles"]
    g["is_depot"] = [i in DEPOT_RACK_IDS for i in g.index]
    return g.sort_values("kwh", ascending=False)


def depot_profile(chg: Charging, poslog: pd.DataFrame, racks: list[Rack],
                  rack_id: str) -> dict:
    """Evidence for/against a rack being a depot rather than a public station."""
    s = chg.sessions
    here = set(s.loc[s["rack_id"] == rack_id, "veh"].dropna())
    elsewhere = set(s.loc[s["rack_id"] != rack_id, "veh"].dropna())
    rack = next((r for r in racks if r.rack_id == rack_id), None)
    ends_near = 0
    if rack is not None and not poslog.empty:
        d = haversine_m(poslog["end_lat"].to_numpy(), poslog["end_lon"].to_numpy(),
                        rack.lat, rack.lon)
        ends_near = int((d <= RIDER_DOCK_RADIUS_M).sum())
    n_here = max(1, len(here))
    return {
        "rack_id": rack_id,
        "sessions": int((s["rack_id"] == rack_id).sum()),
        "kwh": float(s.loc[s["rack_id"] == rack_id, "kwh"].sum()),
        "vehicles": len(here),
        "also_elsewhere": len(here & elsewhere),
        "also_elsewhere_pct": 100.0 * len(here & elsewhere) / n_here,
        "ride_ends_within_radius": ends_near,
    }


# --------------------------------------------------------------------------- #
# SVG charting
# --------------------------------------------------------------------------- #

def esc(s) -> str:
    return _html.escape(str(s), quote=True)


def fmt_int(v) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "-"
    return f"{int(round(v)):,}"


def fmt_1(v) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "-"
    return f"{v:,.1f}"


def fmt_2(v) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "-"
    return f"{v:,.2f}"


def fmt_pct(v) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "-"
    return f"{v:,.1f}%"


def _nice_step(rough: float) -> float:
    if rough <= 0:
        return 1.0
    exp = math.floor(math.log10(rough))
    frac = rough / (10 ** exp)
    for f in (1.0, 2.0, 2.5, 5.0, 10.0):
        if frac <= f:
            return f * (10 ** exp)
    return 10 ** (exp + 1)


def axis_ticks(vmin: float, vmax: float, target: int = 5) -> list[float]:
    if not math.isfinite(vmin) or not math.isfinite(vmax):
        return [0.0, 1.0]
    if vmax <= vmin:
        vmax = vmin + 1.0
    step = _nice_step((vmax - vmin) / max(1, target))
    start = math.floor(vmin / step) * step
    end = math.ceil(vmax / step) * step
    ticks, v, guard = [], start, 0
    while v <= end + step * 1e-9 and guard < 200:
        ticks.append(round(v, 10))
        v += step
        guard += 1
    return ticks


@dataclass
class Panel:
    """One plot row inside an SVG figure (small multiples share the x-axis)."""
    title: str
    series: list[dict]                 # {name, values, color, kind}
    fmt: object = fmt_int
    y_suffix: str = ""
    stacked: bool = False
    height: int = 190
    y_max_hint: float | None = None


def _pt(x: float, y: float) -> str:
    return f"{x:.2f},{y:.2f}"


def _label_indices(values: list, partial: list[bool] | None = None) -> list[int]:
    """Selective direct labels: the maximum and the final point.

    **Partial periods are never labelled.** A one-day stub sitting in a weekly
    series is routinely the highest per-vehicle-per-day figure on the chart --
    same numerator arithmetic, a seventh of the denominator -- so annotating it
    prints a "peak" that no full week ever reached. Excluding partial periods
    from both the max search and the final-point label means no number is ever
    printed on a shaded band.
    """
    ok = []
    for i, v in enumerate(values):
        if v is None or not math.isfinite(v):
            continue
        if partial is not None and i < len(partial) and partial[i]:
            continue
        ok.append(i)
    if not ok:
        return []
    return sorted({max(ok, key=lambda i: values[i]), ok[-1]})


def svg_figure(x_labels: list[str], panels: list[Panel], *,
               width: int = 1020,
               partial: list[bool] | None = None,
               x_tick_every: int | None = None,
               chart_id: str = "c") -> str:
    """Render one or more stacked panels sharing a single x-axis.

    Small multiples rather than a second y-axis: series whose magnitudes differ
    (rides vs vehicles vs rides-per-vehicle) each get their own panel, so no
    chart ever carries two y-scales.
    """
    n = len(x_labels)
    if n == 0:
        return '<p class="empty">No data.</p>'

    pad_l, pad_r, pad_t, pad_b = 74, 26, 8, 46
    legend_h = 26
    gap = 34
    plot_w = width - pad_l - pad_r
    total_h = pad_t + sum(p.height + legend_h + gap for p in panels) - gap + pad_b

    if x_tick_every is None:
        x_tick_every = max(1, math.ceil(n / 14))

    # One band-centred coordinate system for everything. Bars, line vertices,
    # tick labels and partial-period shading must agree exactly -- otherwise
    # stacked panels sharing an x-axis drift out of alignment and the "partial
    # period" shading lands next to the point it is meant to mark. Centring every
    # mark in its band also keeps edge bars inside the plot area.
    slot_w = plot_w / n

    def xpos(i: int) -> float:
        return pad_l + slot_w * (i + 0.5)

    def band(i: int) -> tuple[float, float]:
        return pad_l + slot_w * i, slot_w

    parts: list[str] = [
        f'<svg viewBox="0 0 {width} {total_h:.0f}" width="100%" '
        f'preserveAspectRatio="xMinYMin meet" role="img" '
        f'xmlns="http://www.w3.org/2000/svg" class="chart">'
    ]
    parts.append(
        '<style>'
        '.chart{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,'
        'Helvetica,Arial,sans-serif}'
        '.chart .ax{font-size:11px;fill:%s}'
        '.chart .pt{font-size:11px;font-weight:600;fill:%s;paint-order:stroke;'
        'stroke:%s;stroke-width:3px;stroke-linejoin:round}'
        '.chart .lg{font-size:12px;fill:%s}'
        '.chart .ti{font-size:12.5px;font-weight:650;fill:%s}'
        '</style>' % (C_MUTED, C_INK, C_SURFACE, C_INK2, C_INK)
    )

    y_cursor = pad_t
    for pi, panel in enumerate(panels):
        top = y_cursor + legend_h
        bottom = top + panel.height
        is_last = pi == len(panels) - 1

        # ---- panel title + legend row
        parts.append(f'<text class="ti" x="{pad_l}" y="{y_cursor + 12:.0f}">'
                     f'{esc(panel.title)}</text>')
        # A legend is always present for >= 2 series; a single-series panel is
        # named by its own title, so no legend box is drawn.
        if len(panel.series) >= 2:
            ty = y_cursor + 12
            approx = sum(len(s["name"]) * 6.6 + 26 for s in panel.series)
            lx = max(pad_l + 8, width - pad_r - approx)
            for s in panel.series:
                parts.append(
                    f'<rect x="{lx:.1f}" y="{ty - 8:.1f}" width="10" height="10" '
                    f'rx="2.5" fill="{s["color"]}"/>')
                parts.append(f'<text class="lg" x="{lx + 15:.1f}" y="{ty:.1f}">'
                             f'{esc(s["name"])}</text>')
                lx += 15 + len(s["name"]) * 6.6 + 18

        # ---- y scale
        vals: list[float] = []
        if panel.stacked:
            for i in range(n):
                tot = sum((s["values"][i] or 0) for s in panel.series)
                vals.append(tot)
        else:
            for s in panel.series:
                vals += [v for v in s["values"] if v is not None and math.isfinite(v)]
                # A band's lower edge can sit below every line on the panel; if it
                # is left out of the scale the band is clipped at the axis and
                # silently understates the uncertainty it exists to show.
                vals += [v for v in s.get("lower", []) or []
                         if v is not None and math.isfinite(v)]
        vmax = max(vals) if vals else 1.0
        if panel.y_max_hint is not None and vmax <= panel.y_max_hint:
            # Hard cap (percentage axes): divide the cap evenly rather than
            # adding headroom, so a 0-100% axis never grows a nonsensical 150%
            # tick above it.
            cap = panel.y_max_hint
            ticks = axis_ticks(0.0, cap, 4)
            if ticks[-1] > cap:
                ticks = [cap * k / 4.0 for k in range(5)]
        else:
            ticks = axis_ticks(0.0, vmax * 1.08, 4)
        tmax = ticks[-1] if ticks else 1.0

        def ypos(v: float, _t=tmax, _top=top, _bot=bottom) -> float:
            if not math.isfinite(v):
                return _bot
            return _bot - (_bot - _top) * (v / _t if _t else 0)

        # ---- partial-period shading (behind everything)
        if partial:
            for i, is_p in enumerate(partial):
                if not is_p or i >= n:
                    continue
                bx, bw = band(i)
                parts.append(
                    f'<rect x="{bx:.2f}" y="{top:.2f}" width="{bw:.2f}" '
                    f'height="{bottom - top:.2f}" fill="{C_PARTIAL}"/>')

        # ---- gridlines (solid hairlines, one shade off surface)
        for t in ticks:
            gy = ypos(t)
            parts.append(
                f'<line x1="{pad_l}" y1="{gy:.2f}" x2="{width - pad_r}" '
                f'y2="{gy:.2f}" stroke="{C_GRID}" stroke-width="1"/>')
            parts.append(f'<text class="ax" x="{pad_l - 9}" y="{gy + 3.5:.2f}" '
                         f'text-anchor="end">{esc(panel.fmt(t))}{esc(panel.y_suffix)}'
                         f'</text>')

        # ---- marks
        if panel.stacked:
            base = [0.0] * n
            for s in panel.series:
                pts_top, pts_bot = [], []
                for i in range(n):
                    v = s["values"][i] or 0
                    pts_bot.append((xpos(i), ypos(base[i])))
                    base[i] += v
                    pts_top.append((xpos(i), ypos(base[i])))
                poly = " ".join(_pt(x, y) for x, y in pts_top)
                poly += " " + " ".join(_pt(x, y) for x, y in reversed(pts_bot))
                parts.append(f'<polygon points="{poly}" fill="{s["color"]}" '
                             f'fill-opacity="0.9"/>')
                # 2px surface gap between stacked fills
                parts.append(
                    f'<polyline points="{" ".join(_pt(x, y) for x, y in pts_top)}" '
                    f'fill="none" stroke="{C_SURFACE}" stroke-width="2"/>')
        else:
            # Bands paint first so lines and points read on top of them.
            for s in panel.series:
                if s.get("kind") != "band":
                    continue
                up = [(xpos(i), ypos(v)) for i, v in enumerate(s["values"])
                      if v is not None and math.isfinite(v)]
                lo = [(xpos(i), ypos(v)) for i, v in enumerate(s["lower"])
                      if v is not None and math.isfinite(v)]
                if len(up) >= 2 and len(lo) >= 2:
                    poly = (" ".join(_pt(x, y) for x, y in up) + " "
                            + " ".join(_pt(x, y) for x, y in reversed(lo)))
                    parts.append(f'<polygon points="{poly}" fill="{s["color"]}" '
                                 f'fill-opacity="0.55"/>')
            for s in panel.series:
                kind = s.get("kind", "line")
                if kind == "band":
                    continue
                if kind == "bar":
                    for i in range(n):
                        v = s["values"][i]
                        if v is None or not math.isfinite(v):
                            continue
                        bx, bw = band(i)
                        inset = min(6.0, bw * 0.22) + 1.0
                        w = max(1.5, bw - 2 * inset)
                        yv = ypos(v)
                        r = min(4.0, w / 2.0)
                        parts.append(
                            f'<rect x="{bx + inset:.2f}" y="{yv:.2f}" '
                            f'width="{w:.2f}" height="{max(0.0, bottom - yv):.2f}" '
                            f'rx="{r:.1f}" fill="{s["color"]}"/>')
                else:
                    pts = [(xpos(i), ypos(v)) for i, v in enumerate(s["values"])
                           if v is not None and math.isfinite(v)]
                    if len(pts) >= 2:
                        parts.append(
                            f'<polyline points="{" ".join(_pt(x, y) for x, y in pts)}" '
                            f'fill="none" stroke="{s["color"]}" stroke-width="2" '
                            f'stroke-linejoin="round" stroke-linecap="round"/>')
                    if len(pts) <= 20:
                        for x, y in pts:
                            parts.append(
                                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" '
                                f'fill="{s["color"]}" stroke="{C_SURFACE}" '
                                f'stroke-width="2"/>')

        # ---- selective direct labels (the maximum and the final point only)
        # On a stacked panel the only number that belongs at the top of the stack
        # is the stack TOTAL -- labelling an individual series there would print
        # one series' value at the height of the whole stack.
        if panel.stacked:
            totals = [sum((q["values"][i] or 0) for q in panel.series)
                      for i in range(n)]
            label_sets = [(totals, totals)]
        else:
            label_sets = [(s["values"], s["values"])
                          for s in panel.series[:2] if s.get("kind") != "band"]

        for values, positions in label_sets:
            for i in _label_indices(values, partial):
                v, pv = values[i], positions[i]
                if v is None or not math.isfinite(v):
                    continue
                anchor = "end" if i == n - 1 and n > 1 else "middle"
                dx = -4 if anchor == "end" else 0
                parts.append(
                    f'<text class="pt" x="{xpos(i) + dx:.2f}" '
                    f'y="{ypos(pv) - 9:.2f}" text-anchor="{anchor}">'
                    f'{esc(panel.fmt(v))}{esc(panel.y_suffix)}</text>')

        # ---- axes
        parts.append(f'<line x1="{pad_l}" y1="{bottom:.2f}" x2="{width - pad_r}" '
                     f'y2="{bottom:.2f}" stroke="{C_AXIS}" stroke-width="1"/>')

        if is_last:
            for i in range(0, n, x_tick_every):
                parts.append(
                    f'<text class="ax" x="{xpos(i):.2f}" y="{bottom + 16:.2f}" '
                    f'text-anchor="middle">{esc(x_labels[i])}</text>')
            if partial and any(partial):
                parts.append(
                    f'<rect x="{pad_l}" y="{bottom + 26:.2f}" width="11" '
                    f'height="11" fill="{C_PARTIAL}" stroke="{C_AXIS}" '
                    f'stroke-width="1"/>')
                parts.append(
                    f'<text class="ax" x="{pad_l + 16}" y="{bottom + 35:.2f}">'
                    f'shaded = partial period (window does not cover the whole '
                    f'period; not a real decline, and never labelled or counted '
                    f'as a peak)</text>')

        y_cursor = bottom + gap
    parts.append("</svg>")
    return "".join(parts)


def toggle_block(chart_id: str, svgs: dict[str, str]) -> str:
    """Day/week/month switch over pre-rendered SVGs (all rendered server-side)."""
    btns, panels = [], []
    for i, gran in enumerate(("day", "week", "month")):
        if gran not in svgs:
            continue
        on = " is-on" if i == 0 else ""
        btns.append(f'<button type="button" class="tg-btn{on}" data-gran="{gran}">'
                    f'{GRAN_LABELS[gran]}</button>')
        panels.append(f'<div class="panel{on}" data-gran="{gran}">{svgs[gran]}</div>')
    return (f'<div class="toggle" data-group="{chart_id}">{"".join(btns)}</div>'
            f'<div class="panels" data-group="{chart_id}">{"".join(panels)}</div>')


def period_labels(periods: list[pd.Timestamp], gran: str) -> list[str]:
    if gran == "day":
        return [p.strftime("%d %b") for p in periods]
    if gran == "week":
        return [f"{p.strftime('%d %b')}" for p in periods]
    return [p.strftime("%b %Y") for p in periods]


# --------------------------------------------------------------------------- #
# HTML assembly
# --------------------------------------------------------------------------- #

CSS = """
:root{--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8a83;--line:#e8e8e4;
--surface:#ffffff;--surface2:#faf9f6;--blue:#2a78d6;--orange:#eb6834;--aqua:#1baf7a}
*{box-sizing:border-box}
body{margin:0;background:#f4f3ef;color:var(--ink);
font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
font-size:15px;line-height:1.55;-webkit-font-smoothing:antialiased;
/* WebKit inflates the font of text-heavy blocks on narrow viewports. Left on,
   the prose in Sections 5 and 6 balloons to several times the size of the
   charts it is explaining. */
-webkit-text-size-adjust:100%;text-size-adjust:100%}
.wrap{max-width:1120px;margin:0 auto;padding:32px 20px 80px}
header.masthead{background:var(--surface);border:1px solid var(--line);
border-radius:14px;padding:26px 28px;margin-bottom:22px}
h1{font-size:26px;margin:0 0 6px;letter-spacing:-0.01em}
.sub{color:var(--ink2);margin:0}
h2{font-size:19px;margin:0 0 4px;letter-spacing:-0.01em}
h3{font-size:14px;margin:22px 0 8px;color:var(--ink2);
text-transform:uppercase;letter-spacing:.06em}
section{background:var(--surface);border:1px solid var(--line);border-radius:14px;
padding:24px 28px;margin-bottom:20px}
.secnum{display:inline-block;font-size:12px;font-weight:700;color:var(--muted);
letter-spacing:.08em;margin-bottom:2px}
.note{color:var(--ink2);font-size:13.5px;margin:8px 0 0}
.defn{background:var(--surface2);border-left:3px solid var(--blue);
padding:10px 14px;border-radius:0 8px 8px 0;font-size:13.5px;color:var(--ink2);
margin:14px 0}
.stats{display:flex;flex-wrap:wrap;gap:10px;margin:18px 0 6px}
.stat{flex:1 1 150px;background:var(--surface2);border:1px solid var(--line);
border-radius:10px;padding:12px 14px}
.stat .k{font-size:11.5px;color:var(--muted);text-transform:uppercase;
letter-spacing:.05em}
.stat .v{font-size:22px;font-weight:680;letter-spacing:-0.02em;margin-top:2px}
.stat .u{font-size:12px;color:var(--ink2)}
.hero{font-size:44px;font-weight:700;letter-spacing:-0.03em;line-height:1.1}
table{border-collapse:collapse;width:100%;font-size:13.5px;margin-top:10px}
th,td{text-align:right;padding:7px 10px;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
th{font-size:11.5px;text-transform:uppercase;letter-spacing:.05em;
color:var(--muted);font-weight:650}
tbody tr:hover{background:var(--surface2)}
td.flag{color:var(--orange);font-weight:600}
.tablewrap{overflow-x:auto}
.toggle{display:inline-flex;gap:2px;background:var(--surface2);
border:1px solid var(--line);border-radius:9px;padding:3px;margin:16px 0 6px}
.tg-btn{font:inherit;font-size:13px;border:0;background:transparent;color:var(--ink2);
padding:5px 15px;border-radius:6px;cursor:pointer}
.tg-btn.is-on{background:var(--surface);color:var(--ink);font-weight:620;
box-shadow:0 1px 2px rgba(0,0,0,.09)}
.panel{display:none}
.panel.is-on{display:block}
code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
font-size:12.5px;background:#f0efec;padding:1.5px 5px;border-radius:4px}
pre{background:#f7f6f3;border:1px solid var(--line);border-radius:8px;
padding:12px 14px;overflow-x:auto;font-size:12.5px;
font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.wf{display:flex;flex-wrap:wrap;gap:8px;align-items:stretch;margin-top:12px}
.wf .step{flex:1 1 130px;background:var(--surface2);border:1px solid var(--line);
border-radius:10px;padding:10px 12px}
.wf .step .n{font-size:19px;font-weight:680;letter-spacing:-0.02em}
.wf .step .l{font-size:11.5px;color:var(--muted)}
.wf .step.drop .n{color:var(--orange)}
.assume{margin:8px 0 0 18px;padding:0}
.assume li{margin:5px 0;font-size:13.5px;color:var(--ink2)}
.empty{color:var(--muted);font-style:italic}
footer{color:var(--muted);font-size:12.5px;text-align:center;padding:10px 0 0}
.legendnote{font-size:12.5px;color:var(--muted);margin-top:6px}
.err{background:#fdf1ef;border:1px solid #e9c3bc;border-left:4px solid #c0392b;
border-radius:0 10px 10px 0;padding:14px 18px;margin:14px 0}
.err .tag{display:inline-block;background:#c0392b;color:#fff;font-size:11px;
font-weight:700;letter-spacing:.06em;padding:3px 9px;border-radius:20px;
text-transform:uppercase}
.err h4{margin:10px 0 6px;font-size:15px}
.err p{margin:6px 0;font-size:13.5px;color:var(--ink2)}
.caveat{background:var(--surface2);border-left:3px solid var(--orange);
padding:10px 14px;border-radius:0 8px 8px 0;font-size:13.5px;color:var(--ink2);
margin:14px 0}
td.dim{color:var(--muted)}
.keyfind{background:var(--surface2);border:1px solid var(--line);border-radius:12px;
padding:16px 20px;margin:14px 0}
.keyfind .lede{font-size:16px;line-height:1.5}
"""

JS = """
(function () {
  var btns = document.querySelectorAll('.tg-btn');
  for (var i = 0; i < btns.length; i++) {
    btns[i].addEventListener('click', function () {
      var group = this.parentNode.getAttribute('data-group');
      var gran = this.getAttribute('data-gran');
      var gb = document.querySelectorAll('.toggle[data-group="' + group + '"] .tg-btn');
      for (var j = 0; j < gb.length; j++) { gb[j].classList.remove('is-on'); }
      this.classList.add('is-on');
      var ps = document.querySelectorAll('.panels[data-group="' + group + '"] .panel');
      for (var k = 0; k < ps.length; k++) {
        if (ps[k].getAttribute('data-gran') === gran) { ps[k].classList.add('is-on'); }
        else { ps[k].classList.remove('is-on'); }
      }
    });
  }
})();
"""


def stat(k: str, v: str, u: str = "") -> str:
    return (f'<div class="stat"><div class="k">{esc(k)}</div>'
            f'<div class="v">{esc(v)}</div>'
            f'<div class="u">{esc(u)}</div></div>')


def error_card(title: str, message: str, extra: str = "") -> str:
    return (f'<div class="err"><span class="tag">Data problem</span>'
            f'<h4>{esc(title)}</h4><p>{esc(message)}</p>{extra}</div>')


def build_report(df: pd.DataFrame, poslog: pd.DataFrame, sites: list[Site],
                 racks: list[Rack], wf: Waterfall, per_file: list[dict],
                 charging: Charging | None, charging_error: str | None,
                 tz_scan: pd.DataFrame | None,
                 ride_paths: list[str], charging_path: str | None
                 ) -> tuple[str, dict]:
    """Assemble the whole HTML document. Returns (html, headline_numbers)."""
    # Rides are attributed to the period of their START time, so the observation
    # window is bounded by start times. (The final ride starts 06 Aug and ends
    # 07 Aug 00:06; using end times would stretch the window to a phantom 55th
    # day containing six minutes of data.)
    data_start = df["start_time"].min().normalize()
    data_end = df["start_time"].max().normalize() + pd.Timedelta(days=1)
    n_days = int((data_end - data_start).days)

    total_rides = int(len(df))
    total_vehicles = int(df["vehicle_id"].nunique())
    assessable = int(df["has_end"].sum())
    unassessable = total_rides - assessable
    docked_total = int(df["docked"].sum())
    dock_rate = 100.0 * docked_total / assessable if assessable else float("nan")
    total_kwh = float(energy_need_kwh(total_rides))

    grans = ("day", "week", "month")
    periods: dict[str, list[pd.Timestamp]] = {}
    labels: dict[str, list[str]] = {}
    parts_flags: dict[str, list[bool]] = {}
    util: dict[str, pd.DataFrame] = {}
    for g in grans:
        u = fleet_utilisation(df, g)
        util[g] = u
        periods[g] = list(u.index)
        labels[g] = period_labels(periods[g], g)
        parts_flags[g] = partial_flags(periods[g], g, data_start, data_end)

    # ---- charging-derived quantities, computed once ----
    have_chg = charging is not None
    supply: dict[str, list[float | None]] = {}
    pct_catered: dict[str, list[float | None]] = {}
    total_supply_kwh = charging.total_kwh if have_chg else float("nan")
    overall_catered = (100.0 * total_supply_kwh / total_kwh) if have_chg else float("nan")
    chg_start = charging.sessions["started"].min() if have_chg else None
    chg_end = charging.sessions["started"].max() if have_chg else None
    if have_chg:
        for g in grans:
            by = charging_by_gran(charging, g).reindex(periods[g])
            vals: list[float | None] = []
            for p, raw_v in zip(periods[g], by.to_numpy()):
                _, hi = period_bounds(p, g)
                if hi <= chg_start:
                    # No charging infrastructure existed yet. Plotting 0 here
                    # would draw a supply collapse that never happened.
                    vals.append(None)
                else:
                    vals.append(0.0 if pd.isna(raw_v) else float(raw_v))
            supply[g] = vals
            need_g = list(energy_need_kwh(util[g]["rides"]))
            pct_catered[g] = [None if (s is None or not nd) else 100.0 * s / nd
                              for s, nd in zip(vals, need_g)]
        # Sessions after the last ride day sit outside every plotted period.
        chg_outside = int(((charging.sessions["started"] < data_start)
                           | (charging.sessions["started"] >= data_end)).sum())
        chg_outside_kwh = float(charging.sessions.loc[
            (charging.sessions["started"] < data_start)
            | (charging.sessions["started"] >= data_end), "kwh"].sum())
    else:
        chg_outside, chg_outside_kwh = 0, 0.0

    out: list[str] = []
    out.append('<div class="wrap">')

    # ---------------- 0. Header + method ----------------
    period_txt = (f"{data_start.strftime('%d %b %Y')} - "
                  f"{(data_end - pd.Timedelta(days=1)).strftime('%d %b %Y')}")
    out.append('<header class="masthead">')
    out.append("<h1>Station Performance Report &mdash; Vaasa</h1>")
    out.append(f'<p class="sub">{esc(period_txt)} &middot; {n_days} days &middot; '
               f'{len(sites)} merged station sites from {len(racks)} racks '
               f'&middot; {DOCK_RADIUS_M:.0f} m docking radius</p>')
    out.append('<div class="stats">')
    out.append(stat("Rides analysed", fmt_int(total_rides), "after all filters"))
    out.append(stat("Distinct vehicles", fmt_int(total_vehicles), "in period"))
    out.append(stat("Energy need", f"{fmt_int(total_kwh)} kWh",
                    f"@ {WH_PER_RIDE} Wh/ride"))
    if have_chg:
        out.append(stat("Energy supplied", f"{fmt_1(total_supply_kwh)} kWh",
                        f"{fmt_pct(overall_catered)} of need"))
    else:
        out.append(stat("Energy supplied", "-", "no charging export"))
    out.append("</div></header>")

    out.append("<section>")
    out.append('<span class="secnum">SECTION 0</span>')
    out.append("<h2>Method, filters and data quality</h2>")
    out.append('<p class="note">Every number in this report is computed on the '
               'filtered ride set described below. The waterfall reconciles '
               'exactly &mdash; no rows are silently lost.</p>')

    out.append("<h3>Filter waterfall</h3>")
    out.append('<div class="wf">')
    out.append(f'<div class="step"><div class="n">{fmt_int(wf.raw_rows)}</div>'
               f'<div class="l">Raw rows read</div></div>')
    def drop_step(count: int, label: str) -> str:
        # "-0" reads as an error; a zero drop is shown as a plain 0.
        n_txt = f"-{fmt_int(count)}" if count else "0"
        return (f'<div class="step drop"><div class="n">{n_txt}</div>'
                f'<div class="l">{label}</div></div>')

    out.append(drop_step(wf.dropped_duplicate,
                         "Exact duplicates across the file join"))
    out.append(drop_step(wf.dropped_unparseable, "Unparseable timestamps"))
    out.append(drop_step(wf.dropped_short,
                         f"Rides &lt; {MIN_RIDE_M:.0f} m start&rarr;end"))
    out.append(f'<div class="step"><div class="n">{fmt_int(wf.kept)}</div>'
               f'<div class="l">Rides kept &mdash; the analysis set</div></div>')
    out.append("</div>")
    ok = wf.reconciles()
    out.append(f'<p class="note">Reconciliation: {fmt_int(wf.raw_rows)} raw = '
               f'{fmt_int(wf.kept)} kept + {fmt_int(wf.dropped_duplicate)} duplicate '
               f'+ {fmt_int(wf.dropped_unparseable)} unparseable + '
               f'{fmt_int(wf.dropped_short)} short &rarr; '
               f'<strong>{"balanced" if ok else "MISMATCH"}</strong>.</p>')

    out.append("<h3>Data quality</h3>")
    out.append('<div class="stats">')
    pct_missing = 100.0 * wf.missing_end_coords / wf.kept if wf.kept else 0.0
    out.append(stat("Missing end coords", fmt_int(wf.missing_end_coords),
                    f"{fmt_pct(pct_missing)} of kept rides"))
    out.append(stat("Unparseable timestamps", fmt_int(wf.dropped_unparseable),
                    "dropped before analysis"))
    out.append(stat("Rides under 1 minute", fmt_int(wf.rides_under_1min),
                    "kept, but likely false starts"))
    out.append(stat("Sub-100 m rides", fmt_int(wf.dropped_short),
                    "excluded globally"))
    out.append("</div>")
    out.append('<p class="note">Rides with missing end coordinates are retained in '
               'ride/energy counts (they really happened) but are <strong>excluded '
               'from the docking denominator</strong> because their end position is '
               'unknown. They are reported as &ldquo;unassessable&rdquo; in '
               'Section 5.</p>')

    out.append("<h3>Source files</h3>")
    out.append('<div class="tablewrap"><table><thead><tr><th>File</th>'
               "<th>Rows</th><th>Column order as shipped</th></tr></thead><tbody>")
    for f in per_file:
        out.append(f'<tr><td>{esc(f["file"])}</td><td>{fmt_int(f["rows"])}</td>'
                   f'<td style="text-align:left;font-size:12px">'
                   f'{esc(", ".join(f["columns"]))}</td></tr>')
    out.append("</tbody></table></div>")

    # ---- charging source, scope and the timezone assertion ----
    out.append("<h3>Charging export, scope and clock alignment</h3>")
    if charging_error:
        out.append(error_card(
            "The charging export could not be used", charging_error,
            extra=('<p>Sections 3, 4 and 6 are blank as a result. Every '
                   'ride-side number in this report is unaffected &mdash; they '
                   'are computed from the ride exports alone.</p>')))
    elif not have_chg:
        out.append('<p class="note">No charging export was supplied '
                   '(<code>--charging</code>), so Sections 3, 4 and 6 are '
                   'blank.</p>')
    else:
        out.append('<div class="stats">')
        out.append(stat("Charging sessions", fmt_int(charging.n_sessions),
                        f"operator = {CHARGING_OPERATOR} only"))
        out.append(stat("Racks reporting", fmt_int(charging.sessions["rack_id"].nunique()),
                        f"of {len(racks)} in the registry"))
        out.append(stat("Energy supplied", f"{fmt_1(charging.total_kwh)} kWh",
                        "sum over all sessions"))
        out.append(stat("Charging window",
                        f"{chg_start.strftime('%d %b')}-{chg_end.strftime('%d %b')}",
                        "local time, after correction"))
        out.append("</div>")

        # Compare against the shift that was ACTUALLY applied, not the module
        # default: a run with --charging-tz-shift 0 must fail this check, and
        # comparing to the constant would have let it pass while silently
        # destroying Section 6.
        applied = charging.tz_shift_h
        best = None
        if tz_scan is not None and not tz_scan.empty:
            best = int(tz_scan.loc[tz_scan["matches"].idxmax(), "offset_h"])
        if best is not None and best != applied:
            out.append(error_card(
                "Timezone assertion FAILED - treat Sections 3, 4 and 6 as suspect",
                (f"This report applied a {applied:+d} h correction to charging "
                 f"timestamps, but re-deriving the offset from this export puts "
                 f"the best alignment at {best:+d} h. Either the export is in a "
                 f"different timezone than the one this report was built "
                 f"against, or the wrong --charging-tz-shift was passed. Every "
                 f"rider-vs-operations figure in Section 6 depends on this and "
                 f"is currently wrong."),
                extra=('<p>Re-run with <code>--charging-tz-shift '
                       f'{best}</code> after confirming, or fix the export.</p>')))
        else:
            out.append(
                f'<p class="note"><strong>Clock alignment verified.</strong> The '
                f'charging export is in UTC and the ride exports are Finnish '
                f'local time (UTC+3, EEST), so '
                f'<strong>{applied:+d} h is added to every charging '
                f'timestamp</strong> before any join or bucketing. That offset is '
                f'not assumed &mdash; it is measured, and re-measured on every '
                f'run. Sweeping candidate offsets from '
                f'{TZ_SCAN_OFFSETS[0]:+d} h to {TZ_SCAN_OFFSETS[-1]:+d} h and '
                f'counting sessions whose vehicle&rsquo;s own preceding ride '
                f'ended within {RIDER_DOCK_RADIUS_M:.0f} m of that rack and less '
                f'than {RIDER_DOCK_GAP_MIN:.0f} min earlier gives a single sharp '
                f'maximum:</p>')
            if tz_scan is not None:
                out.append('<div class="tablewrap"><table><thead><tr>'
                           "<th>Offset applied</th>")
                for _, r in tz_scan.iterrows():
                    out.append(f'<th>{int(r["offset_h"]):+d} h</th>')
                out.append("</tr></thead><tbody><tr><td>Rider-docked matches</td>")
                for _, r in tz_scan.iterrows():
                    cls = ' class="flag"' if int(r["offset_h"]) == applied else ""
                    out.append(f'<td{cls}>{fmt_int(r["matches"])}</td>')
                out.append('</tr><tr><td>Median rack distance, gap &lt; 15 min</td>')
                for _, r in tz_scan.iterrows():
                    out.append(f'<td>{fmt_int(r["median_dist_m"])} m</td>')
                out.append("</tr></tbody></table></div>")
            out.append('<p class="note">An offset that is merely close would not '
                       'do this: the runner-up offsets produce a handful of '
                       'matches at 600&ndash;1,100 m, i.e. unrelated earlier '
                       'trips. Getting this wrong would not blur Section 6, it '
                       'would invert it.</p>')

        if charging.dropped_other_operator:
            out.append(f'<p class="note">Filtered out '
                       f'{fmt_int(charging.dropped_other_operator)} session(s) '
                       f'from other operator(s): '
                       f'{esc(", ".join(charging.other_operators))}.</p>')
        if charging.dropped_no_qr:
            out.append(f'<p class="note">{fmt_int(charging.dropped_no_qr)} '
                       f'session(s) carry no parseable vehicle QR and are kept in '
                       f'energy totals but cannot be joined to a ride; they are '
                       f'excluded from Section 6 only.</p>')
        if chg_outside:
            out.append(f'<p class="note">{fmt_int(chg_outside)} session(s) '
                       f'({fmt_1(chg_outside_kwh)} kWh) fall outside the ride '
                       f'window and so sit off the right-hand edge of the charts. '
                       f'They remain in the whole-period supply total.</p>')

    out.append("<h3>Assumptions and definitions</h3>")
    out.append('<ul class="assume">')
    out.append(f"<li><strong>Period</strong>: both exports combined into one "
               f"contiguous window, {esc(period_txt)} ({n_days} days).</li>")
    out.append(f"<li><strong>Docking radius</strong>: {DOCK_RADIUS_M:.0f} m exactly. "
               f"A ride is docked when its end point lies within this distance of a "
               f"merged site centroid.</li>")
    out.append(f"<li><strong>Station merging</strong>: single-link clustering at "
               f"{MERGE_RADIUS_M:.0f} m over {len(racks)} racks &rarr; "
               f"{len(sites)} sites (rack counts "
               f"{', '.join(str(s.rack_count) for s in sites)}). Site position is "
               f"the centroid of its racks.</li>")
    hinge_lo, hinge_hi, hinge_n, hinge_below = clustering_stability(
        [(r.lat, r.lon) for r in racks], MERGE_RADIUS_M)
    hi_txt = "no further merge occurs" if not math.isfinite(hinge_hi) else \
        f"{hinge_hi:.2f} m"
    out.append(f"<li><strong>Clustering sensitivity</strong>: the "
               f"{hinge_n}-site count hinges on one rack pair "
               f"{hinge_lo:.2f} m apart. At any merge radius below that the same "
               f"racks resolve into {hinge_below} sites; the count holds at "
               f"{hinge_n} from {hinge_lo:.2f} m up to {esc(hi_txt)}, so the "
               f"{MERGE_RADIUS_M:.0f} m choice sits inside a plateau rather than "
               f"on a knife edge. The threshold is not being changed &mdash; this "
               f"is disclosure, not a retraction.</li>")
    out.append(f"<li><strong>Sub-{MIN_RIDE_M:.0f} m exclusion</strong>: rides whose "
               f"straight-line start&rarr;end haversine distance is under "
               f"{MIN_RIDE_M:.0f} m are removed from <em>every</em> metric. Rides "
               f"with no end coordinate cannot be measured and are therefore not "
               f"removed by this rule.</li>")
    out.append(f"<li><strong>Energy</strong>: kWh need = rides &times; "
               f"{WH_PER_RIDE} Wh / 1000. The {WH_PER_RIDE} Wh figure is all-in "
               f"(traction drain during the ride plus between-ride standby draw "
               f"amortised per ride).</li>")
    out.append("<li><strong>Nearest-site attribution, and why the table carries "
               "two counts</strong>: three site pairs sit closer than twice the "
               "docking radius (97.1 m, 95.6 m and 83.7 m apart), so their "
               "capture buffers overlap. Each docked end is attributed to its "
               "<em>nearest</em> site, which makes the share column in Section 5 "
               "a true partition rather than a double count &mdash; but it also "
               "means a site next to a busier neighbour loses every contested "
               "end. Section 5 therefore also shows the plain "
               f"{DOCK_RADIUS_M:.0f} m buffer count, and the "
               "&ldquo;unused&rdquo; flag is driven by that second column "
               "only.</li>")
    out.append("<li><strong>Weeks</strong> are ISO weeks (Monday start) computed "
               "from local naive timestamps. Partial weeks at either end of the "
               "window are shaded on every chart &mdash; they are not real "
               "declines &mdash; and are never labelled or treated as a peak or "
               "trough.</li>")
    out.append("<li><strong>Full-weeks-only KPI rule</strong>: every headline "
               "number derived from a weekly series is computed over "
               "<em>complete</em> weeks only, and its label says so. A partial "
               "week is a shorter observation, not a smaller value; pooling the "
               "1-day and 4-day stubs at the ends of this window into a mean "
               "drags it several points away from anything the fleet actually "
               "did.</li>")
    out.append(f"<li><strong>Scope: Ryde only.</strong> The ride exports are "
               f"Ryde&rsquo;s fleet, and the charging export is filtered to "
               f"<code>operator = {CHARGING_OPERATOR}</code> to match. Both sides "
               f"of every energy comparison therefore describe the same "
               f"operator. Comparing all-operator supply against Ryde-only demand "
               f"would inflate apparent coverage purely by mixing populations "
               f"(the racks also serve another operator whose rides are not in "
               f"this dataset), which is why it is not done here.</li>")
    if have_chg and DEPOT_RACK_IDS:
        out.append(f"<li><strong>Depot exclusion</strong>: rack(s) "
                   f"{esc(', '.join(sorted(DEPOT_RACK_IDS)))} charge heavily but "
                   f"sit essentially unvisited by riders, and the profile matches "
                   f"a yard rather than a public station (Section 4). They stay "
                   f"in every energy total and are excluded from the "
                   f"rider-behaviour metrics in Section 6; both figures are shown "
                   f"either way.</li>")
    out.append("<li><strong>Rides are attributed to the period of their start "
               "time.</strong></li>")
    out.append("</ul>")
    out.append("</section>")

    # ---------------- 1. Fleet & ride overview ----------------
    out.append("<section>")
    out.append('<span class="secnum">SECTION 1</span>')
    out.append("<h2>Fleet and ride overview</h2>")
    out.append('<div class="defn">Three measures at very different magnitudes '
               '(rides in thousands, vehicles in hundreds, rides-per-vehicle in '
               'single digits) are shown as stacked panels sharing one x-axis '
               'rather than on a shared or dual y-axis, which would invent a '
               'correlation that is not in the data.</div>')
    svgs = {}
    for g in grans:
        u = util[g]
        svgs[g] = svg_figure(
            labels[g],
            [
                Panel("Rides", [{"name": "Rides", "values": list(u["rides"]),
                                 "color": C_BLUE,
                                 "kind": "bar" if g != "day" else "line"}],
                      fmt=fmt_int),
                Panel("Distinct active vehicles",
                      [{"name": "Vehicles", "values": list(u["vehicles"]),
                        "color": C_ORANGE, "kind": "line"}], fmt=fmt_int),
                Panel("Rides per active vehicle per day",
                      [{"name": "Rides/vehicle/day",
                        "values": list(u["rides_per_vehicle_per_day"]),
                        "color": C_AQUA, "kind": "line"}], fmt=fmt_2),
            ],
            partial=parts_flags[g], chart_id=f"fleet-{g}")
    out.append(toggle_block("fleet", svgs))
    avg_rpv = total_rides / total_vehicles / n_days if total_vehicles else float("nan")
    out.append('<div class="stats">')
    out.append(stat("Total rides", fmt_int(total_rides), "analysis set"))
    out.append(stat("Distinct vehicles", fmt_int(total_vehicles), "whole period"))
    out.append(stat("Rides / day", fmt_int(total_rides / n_days), "mean"))
    out.append(stat("Rides / vehicle / day", fmt_2(avg_rpv), "fleet-wide mean"))
    out.append("</div>")
    out.append("</section>")

    # ---------------- 2. Energy demand ----------------
    out.append("<section>")
    out.append('<span class="secnum">SECTION 2</span>')
    out.append("<h2>Energy demand</h2>")
    out.append(f'<div class="defn">kWh need = rides &times; {WH_PER_RIDE} Wh / 1000. '
               f'This is demand implied by ridership, not metered consumption.</div>')
    svgs = {}
    for g in grans:
        u = util[g]
        kwh = list(energy_need_kwh(u["rides"]))
        svgs[g] = svg_figure(
            labels[g],
            [Panel("kWh needed", [{"name": "kWh need", "values": kwh,
                                   "color": C_BLUE,
                                   "kind": "bar" if g != "day" else "line"}],
                   fmt=fmt_int, height=230)],
            partial=parts_flags[g], chart_id=f"energy-{g}")
    out.append(toggle_block("energy", svgs))
    out.append('<div class="stats">')
    out.append(stat("Total energy need", f"{fmt_int(total_kwh)} kWh", "whole period"))
    out.append(stat("Mean per day", f"{fmt_1(total_kwh / n_days)} kWh", ""))
    out.append(stat("Mean per week", f"{fmt_1(total_kwh / (n_days / 7))} kWh", ""))
    out.append(stat("Wh per ride", str(WH_PER_RIDE), "assumed constant"))
    out.append("</div>")
    out.append("</section>")

    # ---------------- 3. Supplied vs needed ----------------
    out.append("<section>")
    out.append('<span class="secnum">SECTION 3</span>')
    out.append("<h2>Energy supplied vs needed</h2>")
    if not have_chg:
        out.append(error_card(
            "Section unavailable",
            charging_error or "No charging export was supplied (--charging)."))
    else:
        out.append(f'<div class="defn">Demand is modelled '
                   f'(rides &times; {WH_PER_RIDE} Wh); supply is metered '
                   f'(the sum of <code>energy_used_in_wh</code> over Ryde '
                   f'charging sessions). The two are separate panels sharing one '
                   f'x-axis, not two y-axes on one plot: kWh and % are different '
                   f'quantities and overlaying them would let the eye read a '
                   f'crossing point that does not exist.</div>')
        out.append('<div class="stats">')
        out.append(f'<div class="stat" style="flex:1 1 240px"><div class="k">'
                   f'Share of energy demand catered by stations</div>'
                   f'<div class="hero">{esc(fmt_pct(overall_catered))}</div>'
                   f'<div class="u">{fmt_1(total_supply_kwh)} kWh supplied vs '
                   f'{fmt_1(total_kwh)} kWh needed</div></div>')
        # stat() escapes its arguments, so this needs the real character rather
        # than an HTML entity, which would render as literal "&times;".
        out.append(stat("Energy need", f"{fmt_int(total_kwh)} kWh",
                        f"{fmt_int(total_rides)} rides × {WH_PER_RIDE} Wh"))
        out.append(stat("Energy supplied", f"{fmt_1(total_supply_kwh)} kWh",
                        f"{fmt_int(charging.n_sessions)} sessions"))
        out.append(stat("Shortfall", f"{fmt_int(total_kwh - total_supply_kwh)} kWh",
                        "covered by battery swaps"))
        out.append("</div>")
        svgs = {}
        for g in grans:
            u = util[g]
            need = list(energy_need_kwh(u["rides"]))
            svgs[g] = svg_figure(
                labels[g],
                [Panel("kWh needed (modelled from rides)",
                       [{"name": "Needed", "values": need, "color": C_BLUE,
                         "kind": "bar" if g != "day" else "line"}],
                       fmt=fmt_int, height=175),
                 Panel("kWh supplied (metered at the stations)",
                       [{"name": "Supplied", "values": supply[g], "color": C_ORANGE,
                         "kind": "bar" if g != "day" else "line"}],
                       fmt=fmt_int, height=175),
                 Panel("% of demand catered",
                       [{"name": "% catered", "values": pct_catered[g],
                         "color": C_AQUA, "kind": "line"}], fmt=fmt_1,
                       y_suffix="%", height=160)],
                partial=parts_flags[g], chart_id=f"supply-{g}")
        out.append(toggle_block("supply", svgs))
        out.append(f'<p class="legendnote">Ryde charging begins '
                   f'{chg_start.strftime("%d %b %Y")}. Periods that end before '
                   f'that are drawn as <strong>gaps, not zeros</strong> &mdash; '
                   f'{data_start.strftime("%d %b")} had no stations to supply '
                   f'anything, which is a different statement from stations '
                   f'supplying nothing.</p>')
        out.append(f'<p class="note">The {fmt_int(total_kwh - total_supply_kwh)} '
                   f'kWh gap is not necessarily unmet demand: it is energy that '
                   f'reached the fleet by some route other than these stations, '
                   f'overwhelmingly battery swaps. What this section measures is '
                   f'the split between the two, not a service failure.</p>')
    out.append("</section>")

    # ---------------- 4. % catered vs utilisation ----------------
    out.append("<section>")
    out.append('<span class="secnum">SECTION 4</span>')
    out.append("<h2>Percent of kWh catered vs utilisation rate</h2>")
    out.append('<div class="defn">Utilisation is charted on <strong>both '
               'denominators</strong> because the choice is not yet locked: '
               '<em>rides per active vehicle per day</em> (denominator = vehicles '
               'that rode that period) and <em>rides per fleet vehicle per day</em> '
               '(denominator = all distinct vehicles seen in the whole period). '
               'Fleet-active share is also shown as a percentage so it can sit on '
               'the same 0&ndash;100% axis as percent-catered without a second '
               'y-scale.</div>')
    svgs = {}
    for g in grans:
        u = util[g]
        pct_panel = [{"name": "Fleet active share",
                      "values": list(u["fleet_active_pct"]),
                      "color": C_AQUA, "kind": "line"}]
        if have_chg:
            pct_panel.insert(0, {"name": "% of kWh demand catered",
                                 "values": pct_catered[g], "color": C_ORANGE,
                                 "kind": "line"})
        svgs[g] = svg_figure(
            labels[g],
            [Panel("Utilisation - rides per vehicle per day (both denominators)",
                   [{"name": "per active vehicle",
                     "values": list(u["rides_per_vehicle_per_day"]),
                     "color": C_BLUE, "kind": "line"},
                    {"name": "per fleet vehicle",
                     "values": list(u["rides_per_fleet_vehicle_per_day"]),
                     "color": C_ORANGE, "kind": "line"}], fmt=fmt_2, height=200),
             Panel("Percentages on one 0-100% axis: kWh catered and fleet activity",
                   pct_panel, fmt=fmt_1, y_suffix="%",
                   height=170, y_max_hint=100.0)],
            partial=parts_flags[g], chart_id=f"util-{g}")
    out.append(toggle_block("util", svgs))
    if not have_chg:
        out.append(error_card(
            "Percent-catered unavailable",
            charging_error or "No charging export was supplied (--charging).",
            extra=("<p>The utilisation panel above is computed from the ride "
                   "exports alone and is unaffected.</p>")))
    else:
        out.append('<p class="note">Both percentage series sit on one axis '
                   'because they are both percentages &mdash; but they are not '
                   'two views of one thing. Fleet-active share moves with '
                   'ridership; percent-catered moves with how much charging '
                   'hardware exists and how much of it gets used. Reading a '
                   'causal link off their co-movement would be unsupported: this '
                   'window contains one summer and no intervention.</p>')

        # ---- per-station supply ----
        out.append("<h3>Supply by station</h3>")
        s = charging.sessions
        by_site = s.groupby("site").agg(kwh=("kwh", "sum"), sessions=("kwh", "size"))
        site_rec = {st.index: sum(racks[i].receptacles for i in st.rack_indices)
                    for st in sites}
        buf = buffer_end_counts(df, sites)
        rows = []
        for st in sites:
            k = float(by_site["kwh"].get(st.index, 0.0))
            ns = int(by_site["sessions"].get(st.index, 0))
            rec = site_rec[st.index]
            rack_ids = [racks[i].rack_id for i in st.rack_indices]
            rows.append({
                "site": st.name, "racks": st.rack_count, "recept": rec,
                "kwh": k, "sessions": ns,
                "kwh_per_rec": k / rec if rec else float("nan"),
                "share": 100.0 * k / total_supply_kwh if total_supply_kwh else 0.0,
                "ends": int(buf[st.index]),
                "depot": any(r in DEPOT_RACK_IDS for r in rack_ids),
                "rack_ids": rack_ids,
            })
        rows.sort(key=lambda r: -r["kwh"])
        out.append('<div class="tablewrap"><table><thead><tr><th>Site</th>'
                   "<th>Racks</th><th>Receptacles</th><th>Sessions</th>"
                   "<th>kWh supplied</th><th>kWh / receptacle</th>"
                   f"<th>Share of supply</th><th>Ride-ends within "
                   f"{DOCK_RADIUS_M:.0f} m</th></tr></thead><tbody>")
        for r in rows:
            nm = r["site"] + (" &middot; depot" if r["depot"] else "")
            out.append(
                f'<tr><td>{nm}</td><td>{r["racks"]}</td><td>{r["recept"]}</td>'
                f'<td>{fmt_int(r["sessions"])}</td><td>{fmt_1(r["kwh"])}</td>'
                f'<td>{fmt_1(r["kwh_per_rec"])}</td>'
                f'<td>{fmt_pct(r["share"])}</td>'
                f'<td>{fmt_int(r["ends"])}</td></tr>')
        out.append(f'<tr><td><strong>Total</strong></td>'
                   f'<td>{sum(r["racks"] for r in rows)}</td>'
                   f'<td>{sum(r["recept"] for r in rows)}</td>'
                   f'<td>{fmt_int(sum(r["sessions"] for r in rows))}</td>'
                   f'<td><strong>{fmt_1(sum(r["kwh"] for r in rows))}</strong></td>'
                   f'<td></td><td>100.0%</td><td></td></tr>')
        out.append("</tbody></table></div>")

        # ---- commissioning dates ----
        comm = rack_commissioning(charging, racks)
        late = comm.loc[comm["late"]]
        out.append("<h3>Rack commissioning</h3>")
        out.append(f'<p class="note">A rack that only came online in the last '
                   f'weeks of the window has a low session count because it is '
                   f'new, not because it is failing. First-session dates make '
                   f'that distinction; {len(late)} of {len(comm)} racks started '
                   f'more than {COMMISSION_LAG_DAYS} days after the first session '
                   f'in the export.</p>')
        out.append('<div class="tablewrap"><table><thead><tr><th>Rack</th>'
                   "<th>Site</th><th>First session</th><th>Sessions</th>"
                   "<th>kWh</th><th>kWh / receptacle</th><th>Note</th>"
                   "</tr></thead><tbody>")
        rack_site = {racks[i].rack_id: st.name for st in sites for i in st.rack_indices}
        for rid, r in comm.sort_values("first_session").iterrows():
            notes = []
            if bool(r["late"]):
                notes.append(f'commissioned mid-window '
                             f'({int(round(r["days_late"]))} days in)')
            if bool(r["is_depot"]):
                notes.append("probable depot/yard")
            cls = ' class="flag"' if notes else ' class="dim"'
            out.append(
                f'<tr><td>{esc(rid)}</td><td>{esc(rack_site.get(rid, "-"))}</td>'
                f'<td>{r["first_session"].strftime("%d %b %Y")}</td>'
                f'<td>{fmt_int(r["sessions"])}</td><td>{fmt_1(r["kwh"])}</td>'
                f'<td>{fmt_1(r["kwh_per_receptacle"])}</td>'
                f'<td{cls} style="text-align:left">{esc("; ".join(notes))}</td>'
                f'</tr>')
        out.append("</tbody></table></div>")

        # ---- depot evidence ----
        for rid in sorted(DEPOT_RACK_IDS):
            prof = depot_profile(charging, poslog, racks, rid)
            if not prof["sessions"]:
                continue
            out.append(
                f'<div class="caveat"><strong>Rack {esc(rid)} looks like a '
                f'depot, not a station.</strong> It ran '
                f'{fmt_int(prof["sessions"])} sessions for '
                f'{fmt_1(prof["kwh"])} kWh &mdash; ordinary volume &mdash; but '
                f'only {fmt_int(prof["ride_ends_within_radius"])} ride-ends in '
                f'the entire window finished within {RIDER_DOCK_RADIUS_M:.0f} m '
                f'of it, and {fmt_pct(prof["also_elsewhere_pct"])} of the '
                f'{fmt_int(prof["vehicles"])} vehicles charged there also charge '
                f'at public racks. Vehicles are arriving by van. It is kept in '
                f'every energy total on this page and excluded from the '
                f'rider-behaviour metrics in Section 6, where both figures are '
                f'shown.</div>')
    out.append("</section>")

    # ---------------- 5. Docking success ----------------
    wd = weekly_docking(df)
    wk_labels = period_labels(list(wd.index), "week")
    wk_partial = partial_flags(list(wd.index), "week", data_start, data_end)
    out.append("<section>")
    out.append('<span class="secnum">SECTION 5</span>')
    out.append("<h2>Docking success rate (weekly)</h2>")
    out.append(f'<div class="defn"><strong>Numerator</strong>: ride-ends within '
               f'{DOCK_RADIUS_M:.0f} m of any merged site. '
               f'<strong>Denominator</strong>: rides with valid end coordinates '
               f'({fmt_int(assessable)} of {fmt_int(total_rides)}). The '
               f'{fmt_int(unassessable)} rides with missing end coordinates are '
               f'<em>excluded from the denominator</em> and reported separately as '
               f'unassessable.</div>')
    wd_full = full_periods(wd, wk_partial)
    peak_wk = wd_full["dock_rate_pct"].idxmax()
    trough_wk = wd_full["dock_rate_pct"].idxmin()
    overlap = bounds_overlap(wd, peak_wk, trough_wk)
    # Quote the missing-data share for the SAME two weeks the docking point
    # estimate is quoted for, so the sentence and the table below it agree.
    unass_lo = float(wd_full.loc[peak_wk, "unassessable_pct"])
    unass_hi = float(wd_full.loc[trough_wk, "unassessable_pct"])

    # Full weeks only, matching the rule declared in Section 0 (a partial week
    # would otherwise drag the headline without the reader seeing why).
    dock_docked_full = int(wd_full["docked"].sum())
    dock_assess_full = int(wd_full["assessable"].sum())
    dock_rate_full = (100.0 * dock_docked_full / dock_assess_full
                      if dock_assess_full else float("nan"))
    out.append('<div class="stats">')
    out.append(f'<div class="stat" style="flex:1 1 220px"><div class="k">'
               f'Docking success, full weeks only (pooled)</div>'
               f'<div class="hero">{esc(fmt_pct(dock_rate_full))}'
               f'</div><div class="u">{fmt_int(dock_docked_full)} of '
               f'{fmt_int(dock_assess_full)} assessable ends &middot; '
               f'all {len(wd)} weeks incl. partial: {fmt_pct(dock_rate)}</div></div>')
    out.append(stat("Unassessable rides", fmt_int(unassessable), "no end coordinate"))
    _bufs = per_site_table(df, sites)["buffer_ends"]
    out.append(stat("Sites with meaningful rider traffic",
                    str(int((_bufs > NEAR_ZERO_END_THRESHOLD).sum())),
                    f"of {len(sites)} (more than {NEAR_ZERO_END_THRESHOLD} ends in the window)"))
    out.append("</div>")

    out.append(svg_figure(
        wk_labels,
        [Panel("Weekly docking rate, with the range the missing data allows",
               [{"name": "Range if every unassessable ride failed / docked",
                 "values": list(wd["hi_pct"]), "lower": list(wd["lo_pct"]),
                 "color": C_BAND, "kind": "band"},
                {"name": "Point estimate (assessable ends only)",
                 "values": list(wd["dock_rate_pct"]),
                 "color": C_BLUE, "kind": "line"}], fmt=fmt_1, y_suffix="%",
               height=215),
         Panel("Share of rides with no end coordinate (the confound)",
               [{"name": "Unassessable", "values": list(wd["unassessable_pct"]),
                 "color": C_ORANGE, "kind": "bar"}], fmt=fmt_1, y_suffix="%",
               height=155)],
        partial=wk_partial, chart_id="dock"))

    out.append('<div class="keyfind">')
    out.append(f'<p class="lede"><strong>The apparent second-half decline in '
               f'docking is not established by this data.</strong> The point '
               f'estimate falls from {fmt_pct(wd_full["dock_rate_pct"].max())} '
               f'({peak_wk.strftime("%d %b")}) to '
               f'{fmt_pct(wd_full["dock_rate_pct"].min())} '
               f'({trough_wk.strftime("%d %b")}), but over the same weeks the '
               f'share of rides with no end coordinate rises roughly '
               f'{unass_hi / max(unass_lo, 1e-9):.1f}-fold, from '
               f'{fmt_pct(unass_lo)} to '
               f'{fmt_pct(unass_hi)}. Those rides are not missing at random: a '
               f'ride with no end coordinate <em>starts</em> within '
               f'{DOCK_RADIUS_M:.0f} m of a site '
               f'{fmt_pct(100.0 * df.loc[~df["has_end"], "start_docked"].mean())} '
               f'of the time, against '
               f'{fmt_pct(100.0 * df.loc[df["has_end"], "start_docked"].mean())} '
               f'for rides that do report one &mdash; they are biased '
               f'<em>towards</em> stations, exactly where docking is decided.</p>')
    out.append(f'<p class="note">Bounding the true rate between "every '
               f'unassessable ride failed" and "every unassessable ride docked" '
               f'gives [{fmt_1(wd.loc[peak_wk, "lo_pct"])}%, '
               f'{fmt_1(wd.loc[peak_wk, "hi_pct"])}%] for the peak week and '
               f'[{fmt_1(wd.loc[trough_wk, "lo_pct"])}%, '
               f'{fmt_1(wd.loc[trough_wk, "hi_pct"])}%] for the trough. '
               f'{"Those brackets overlap, so the two weeks cannot be ordered at all" if overlap else "Those brackets are disjoint"} '
               f'&mdash; the movement is indistinguishable from a GPS-logging '
               f'regression that began in mid-July. Investigate the logging '
               f'pipeline before treating this as rider behaviour.</p>')
    out.append("</div>")

    out.append('<div class="tablewrap"><table><thead><tr><th>Week starting</th>'
               "<th>Rides</th><th>Assessable</th><th>Docked</th>"
               "<th>Point estimate</th><th>Lower bound</th><th>Upper bound</th>"
               "<th>Unassessable</th><th></th></tr></thead><tbody>")
    for (p, r), is_p in zip(wd.iterrows(), wk_partial):
        out.append(f'<tr><td>{p.strftime("%d %b %Y")}</td>'
                   f'<td>{fmt_int(r["rides"])}</td>'
                   f'<td>{fmt_int(r["assessable"])}</td>'
                   f'<td>{fmt_int(r["docked"])}</td>'
                   f'<td>{fmt_pct(r["dock_rate_pct"])}</td>'
                   f'<td>{fmt_pct(r["lo_pct"])}</td>'
                   f'<td>{fmt_pct(r["hi_pct"])}</td>'
                   f'<td>{fmt_pct(r["unassessable_pct"])}</td>'
                   f'<td class="dim" style="text-align:left">'
                   f'{"partial week" if is_p else ""}</td></tr>')
    out.append("</tbody></table></div>")

    st = per_site_table(df, sites)
    out.append("<h3>Per-site capture</h3>")
    out.append(f'<div class="defn">Two counts, because one alone misleads. '
               f'<strong>Attributed</strong> gives each docked end to its single '
               f'nearest site, so that column partitions the '
               f'{fmt_int(docked_total)} docked ends exactly and the shares sum '
               f'to 100%. <strong>Within {DOCK_RADIUS_M:.0f} m</strong> counts '
               f'every end inside a site&rsquo;s own buffer regardless of what '
               f'else is nearby, so it does not partition &mdash; and it is the '
               f'only one of the two that answers &ldquo;is this hardware being '
               f'used&rdquo;.</div>')
    out.append('<div class="tablewrap"><table><thead><tr><th>Site</th>'
               "<th>Centroid (lat, lon)</th><th>Racks</th>"
               "<th>Ends attributed (nearest site)</th>"
               f"<th>Ends within {DOCK_RADIUS_M:.0f} m</th>"
               "<th>Share of docked ends</th><th>Status</th></tr></thead><tbody>")
    for _, r in st.iterrows():
        flagged = bool(r["flagged"])
        if flagged:
            status = "no meaningful rider traffic - investigate"
        elif r["buffer_ends"] >= 3 * max(1, r["ends"]):
            status = ("in use - most of its traffic is attributed to a closer "
                      "neighbouring site")
        else:
            status = "in use"
        cls = ' class="flag"' if flagged else ""
        out.append(
            f'<tr><td>{esc(r["site"])}</td>'
            f'<td style="text-align:left">{r["lat"]:.6f}, {r["lon"]:.6f}</td>'
            f'<td>{int(r["racks"])}</td><td>{fmt_int(r["ends"])}</td>'
            f'<td>{fmt_int(r["buffer_ends"])}</td>'
            f'<td>{fmt_pct(r["share_pct"])}</td>'
            f'<td{cls} style="text-align:left">{esc(status)}</td></tr>')
    out.append("</tbody></table></div>")
    gap_sites = st.loc[st["buffer_ends"] >= 3 * st["ends"].clip(lower=1)]
    if not gap_sites.empty:
        g0 = gap_sites.iloc[0]
        # The neighbour and its distance are derived, not written in: hard-coding
        # them would quietly lie the moment the rack registry changes.
        s0 = sites[int(g0["site"][1:]) - 1]
        nb = min((o for o in sites if o.index != s0.index),
                 key=lambda o: float(haversine_m(s0.lat, s0.lon, o.lat, o.lon)))
        nb_m = float(haversine_m(s0.lat, s0.lon, nb.lat, nb.lon))
        out.append(f'<p class="note">Why both columns are needed: '
                   f'{esc(g0["site"])} is attributed {fmt_int(g0["ends"])} ends '
                   f'but has {fmt_int(g0["buffer_ends"])} inside its own '
                   f'{DOCK_RADIUS_M:.0f} m buffer, because {esc(nb.name)} sits '
                   f'{nb_m:.1f} m away and wins the tie on nearly every one. '
                   f'Judged on the attributed column alone it would be reported '
                   f'as dead hardware; it is not.</p>')
    out.append(f'<p class="note">The unused flag fires below '
               f'{NEAR_ZERO_END_THRESHOLD} ends <em>within the buffer</em>. '
               f'Rack commissioning dates in Section 4 separate "never '
               f'commissioned" from "commissioned and ignored", and the charging '
               f'data separates "ignored by riders" from "not a public station at '
               f'all".</p>')
    out.append("</section>")

    # ---------------- 6. Who is actually docking? ----------------
    rider_pct = rider_pct_all = float("nan")
    reverse_pct = float("nan")
    out.append("<section>")
    out.append('<span class="secnum">SECTION 6</span>')
    out.append("<h2>Who is actually docking?</h2>")
    if not have_chg:
        out.append(error_card(
            "Section unavailable",
            charging_error or "No charging export was supplied (--charging).",
            extra=("<p>This section needs charging sessions joined to rides; "
                   "ride data alone shows where vehicles stop, not whether "
                   "anyone plugged them in.</p>")))
    else:
        cls_all = classify_sessions(charging, poslog)
        cls = cls_all.loc[~cls_all["rack_id"].isin(DEPOT_RACK_IDS)]
        rider_pct = 100.0 * float(cls["rider_docked"].mean())
        rider_pct_all = 100.0 * float(cls_all["rider_docked"].mean())
        n_rider = int(cls["rider_docked"].sum())
        pv, shares = docking_attribution(cls)
        # Of sessions whose vehicle HAS a known ride in the previous 24 h.
        known = cls.loc[cls["bucket"] != GAP_BUCKET_LABELS[-1]]
        stale_4_24 = (100.0 * (known["bucket"] == "4-24 h").sum()
                      / max(1, len(known)))

        # Depot racks are excluded here for the same reason as the forward view:
        # vehicles arrive there by van, so they are not rider behaviour.
        rev_racks = [r for r in racks if r.rack_id not in DEPOT_RACK_IDS]
        rev = reverse_plugin_rate(poslog, charging, rev_racks,
                                  since=chg_start.normalize())
        reverse_pct = (100.0 * rev["plugged"].sum() / rev["ends"].sum()
                       if not rev.empty else float("nan"))
        reverse_pct_any = (100.0 * rev["plugged_any"].sum() / rev["ends"].sum()
                           if not rev.empty else float("nan"))

        # Same join against the *filtered* analysis set, purely to quantify what
        # the position-log choice is worth (quoted in the limitations list).
        poslog_f = df.loc[df["has_end"],
                          ["vehicle_id", "end_time", "end_lat", "end_lon"]].copy()
        poslog_f["vehicle_id"] = poslog_f["vehicle_id"].astype(str)
        cls_f = classify_sessions(charging, poslog_f.sort_values("end_time"))
        rider_pct_filtered = 100.0 * float(
            cls_f.loc[~cls_f["rack_id"].isin(DEPOT_RACK_IDS), "rider_docked"].mean())

        out.append(f'<div class="defn">A session is <strong>rider-docked</strong> '
                   f'when the vehicle&rsquo;s own preceding ride ended within '
                   f'{RIDER_DOCK_RADIUS_M:.0f} m of <em>that same rack</em> and '
                   f'less than {RIDER_DOCK_GAP_MIN:.0f} minutes before the '
                   f'session started &mdash; someone rode it there and plugged it '
                   f'in. Every other session is bucketed by how stale that '
                   f'preceding ride is. The join is '
                   f'<code>merge_asof(by=vehicle, direction=&quot;backward&quot;)</code> '
                   f'on the vehicle&rsquo;s last ride-end before each session.</div>')

        out.append('<div class="keyfind">')
        out.append(f'<p class="lede">Riders park at the stations. They very '
                   f'rarely plug in. Of {fmt_int(len(cls))} charging sessions, '
                   f'<strong>{fmt_pct(rider_pct)}</strong> ({fmt_int(n_rider)}) '
                   f'follow the vehicle&rsquo;s own ride ending at that rack '
                   f'minutes earlier. Looked at from the other direction, of the '
                   f'{fmt_int(int(rev["ends"].sum()))} ride-ends that finished '
                   f'within {RIDER_DOCK_RADIUS_M:.0f} m of a rack while the '
                   f'stations were live, <strong>{fmt_pct(reverse_pct)}</strong> '
                   f'were followed by a charging session within '
                   f'{REVERSE_PLUG_WINDOW_MIN:.0f} minutes.</p>')
        out.append(f'<p class="note">The remaining sessions are consistent with '
                   f'operations staff collecting and docking vehicles: '
                   f'{fmt_pct(stale_4_24)} of sessions with a known preceding '
                   f'ride start 4&ndash;24 hours after it, and '
                   f'{fmt_pct(float(shares[GAP_BUCKET_LABELS[-1]]))} follow no '
                   f'ride at all in the previous 24 hours. This is a description '
                   f'of how the system is being used, not a verdict on it: the '
                   f'stations are moving real energy '
                   f'({fmt_1(total_supply_kwh)} kWh), just overwhelmingly through '
                   f'a staffed workflow rather than a rider-facing one.</p>')
        out.append("</div>")

        pv_labels = period_labels(list(pv.index), "week")
        pv_partial = partial_flags(list(pv.index), "week", data_start, data_end)
        pv_pct = 100.0 * pv.div(pv.sum(axis=1).replace(0, np.nan), axis=0)
        colours = [C_ORANGE] + C_SEQ
        series = [{"name": c, "values": list(pv[c]), "color": col}
                  for c, col in zip(pv.columns, colours)]
        rider_line = [{"name": "Rider-docked share",
                       "values": list(pv_pct["rider-docked"]),
                       "color": C_ORANGE, "kind": "line"}]
        out.append(svg_figure(
            pv_labels,
            [Panel("Charging sessions per week, by how the vehicle got there",
                   series, fmt=fmt_int, stacked=True, height=225),
             Panel("Rider-docked share of sessions", rider_line, fmt=fmt_1,
                   y_suffix="%", height=150)],
            partial=pv_partial, chart_id="whodocks"))
        out.append('<p class="legendnote">The gap buckets are an ordered '
                   'quantity, so they use one hue running light to dark. '
                   'Rider-docked is not part of that order and takes the '
                   'contrasting colour.</p>')

        out.append("<h3>Session attribution over the whole window</h3>")
        out.append('<div class="tablewrap"><table><thead><tr>'
                   "<th>How the vehicle reached the rack</th><th>Sessions</th>"
                   "<th>Share</th></tr></thead><tbody>")
        counts = cls["bucket"].value_counts()
        for name in ["rider-docked"] + GAP_BUCKET_LABELS:
            n = int(counts.get(name, 0))
            cls_a = ' class="flag"' if name == "rider-docked" else ""
            out.append(f'<tr><td{cls_a} style="text-align:left">{esc(name)}</td>'
                       f'<td>{fmt_int(n)}</td>'
                       f'<td>{fmt_pct(100.0 * n / max(1, len(cls)))}</td></tr>')
        out.append(f'<tr><td><strong>Total</strong></td>'
                   f'<td><strong>{fmt_int(len(cls))}</strong></td>'
                   f'<td>100.0%</td></tr>')
        out.append("</tbody></table></div>")

        if not rev.empty:
            out.append("<h3>The reverse view, week by week</h3>")
            out.append(f'<p class="note">Of ride-ends that finished within '
                       f'{RIDER_DOCK_RADIUS_M:.0f} m of a rack, the share '
                       f'followed by a charging session for that vehicle within '
                       f'{REVERSE_PLUG_WINDOW_MIN:.0f} minutes. The strict mirror '
                       f'of the forward rule requires the plug-in to happen at '
                       f'<em>the same rack</em> the rider parked beside '
                       f'({fmt_pct(reverse_pct)}); allowing a plug-in at any rack '
                       f'raises it to {fmt_pct(reverse_pct_any)}, the difference '
                       f'being vehicles collected from one rack and charged at '
                       f'another. Both are shown; the same-rack figure is the '
                       f'like-for-like comparison. Rack-level buffers, depot '
                       f'rack(s) excluded.</p>')
            out.append(svg_figure(
                period_labels(list(rev.index), "week"),
                [Panel("Ride-ends at a rack that are followed by a plug-in",
                       [{"name": "Same rack (strict mirror)",
                         "values": list(rev["pct"]),
                         "color": C_AQUA, "kind": "bar"},
                        {"name": "Any rack (upper bound)",
                         "values": list(rev["pct_any_rack"]),
                         "color": C_MUTED, "kind": "line"}], fmt=fmt_1,
                       y_suffix="%", height=170)],
                partial=partial_flags(list(rev.index), "week", data_start, data_end),
                chart_id="reverse"))

        out.append("<h3>What this rule can and cannot tell you</h3>")
        out.append('<ul class="assume">')
        out.append(f"<li><strong>It over-counts riders, so "
                   f"{fmt_pct(rider_pct)} is a ceiling.</strong> A van collecting "
                   f"a scooter that was parked beside a rack and plugging it in "
                   f"within {RIDER_DOCK_GAP_MIN:.0f} minutes satisfies the rule "
                   f"exactly as a rider does. Nothing in either export "
                   f"distinguishes them.</li>")
        out.append(f"<li><strong>Consumer GPS error is typically a few tens of "
                   f"metres</strong> in a built-up area, against a "
                   f"{RIDER_DOCK_RADIUS_M:.0f} m radius. Ends recorded just "
                   f"outside the buffer that were physically at the rack are "
                   f"missed; ends recorded just inside it that were across the "
                   f"street are counted. This blurs the figure by a point or so "
                   f"in each direction, not by an order of magnitude.</li>")
        out.append(f"<li><strong>Only the single most recent ride is "
                   f"considered.</strong> If a vehicle was ridden to the rack, "
                   f"then moved 20 m by someone else, the intervening movement "
                   f"becomes the preceding ride.</li>")
        out.append(f"<li><strong>The position log deliberately keeps rides under "
                   f"{MIN_RIDE_M:.0f} m</strong>, which the rest of the report "
                   f"excludes: a short hop to a rack is exactly the event being "
                   f"measured. Applying the sub-{MIN_RIDE_M:.0f} m exclusion here "
                   f"too would report {fmt_pct(rider_pct_filtered)} instead "
                   f"&mdash; lower, and wrong, because it dates each vehicle from "
                   f"an older trip than the one that actually put it at the "
                   f"rack.</li>")
        out.append(f"<li><strong>{fmt_int(charging.dropped_no_qr)} session(s) "
                   f"cannot be joined</strong> for lack of a parseable vehicle QR "
                   f"and are excluded from this section only.</li>")
        out.append(f"<li><strong>Depot handling</strong>: the charts and shares "
                   f"above exclude rack(s) "
                   f"{esc(', '.join(sorted(DEPOT_RACK_IDS)))}. Including them, the "
                   f"rider-docked share is {fmt_pct(rider_pct_all)} of "
                   f"{fmt_int(len(cls_all))} sessions &mdash; the direction of "
                   f"the finding does not depend on the exclusion.</li>")
        out.append("</ul>")
    out.append("</section>")

    # ---------------- 7. Docked vs never-docked vehicles ----------------
    va = weekly_vehicle_docking(df)
    va_labels = period_labels(list(va.index), "week")
    va_partial = partial_flags(list(va.index), "week", data_start, data_end)
    va_full = full_periods(va, va_partial)
    docked_share_full = float(100.0 * va_full["docked"].sum() / va_full["riding"].sum())
    docked_share_all = float(100.0 * va["docked"].sum() / va["riding"].sum())
    out.append("<section>")
    out.append('<span class="secnum">SECTION 7</span>')
    out.append("<h2>Docked vs never-docked vehicles (weekly)</h2>")
    out.append(f'<div class="defn">A vehicle <strong>docked</strong> in a week if '
               f'at least one of its rides that week ended within '
               f'{DOCK_RADIUS_M:.0f} m of a site; it <strong>never docked</strong> '
               f'if it rode that week and no ride ended near a site. '
               f'<strong>Denominator</strong>: vehicles that rode at all that '
               f'week. <strong>Note the vocabulary</strong>: a vehicle that never '
               f'docks is not idle. Sections 1 and 4 use <em>active</em> to mean '
               f'&ldquo;rode at all&rdquo;, which is a different and much larger '
               f'population &mdash; this section is about where rides end, not '
               f'whether they happen.</div>')
    out.append(svg_figure(
        va_labels,
        [Panel("Vehicles that rode, split by whether they ever docked",
               [{"name": "Docked at least once",
                 "values": list(va["docked"]), "color": C_BLUE},
                {"name": "Rode but never docked",
                 "values": list(va["never_docked"]), "color": C_ORANGE}],
               fmt=fmt_int, stacked=True, height=200),
         Panel("Docked-vehicle share",
               [{"name": "Docked share", "values": list(va["docked_pct"]),
                 "color": C_AQUA, "kind": "line"}], fmt=fmt_1, y_suffix="%",
               height=165, y_max_hint=100.0)],
        partial=va_partial, chart_id="docked"))
    out.append('<div class="stats">')
    out.append(stat("Docked-vehicle share, full weeks only (pooled)",
                    fmt_pct(docked_share_full),
                    f"{fmt_int(va_full['docked'].sum())} of "
                    f"{fmt_int(va_full['riding'].sum())} vehicle-weeks"))
    out.append(stat("Same figure, mean of weekly rates",
                    fmt_pct(float(va_full["docked_pct"].mean())),
                    "full weeks only"))
    out.append(stat("Peak weekly riding fleet",
                    fmt_int(va_full["riding"].max()), "vehicles, full weeks"))
    out.append("</div>")
    out.append(f'<p class="note">Both headline figures above cover complete weeks '
               f'only. Pooling the partial weeks at the ends of the window in as '
               f'well would give {fmt_pct(docked_share_all)}, which is not a '
               f'statement about any week the fleet actually operated &mdash; '
               f'it is the arithmetic of a 1-day stub and a 4-day stub diluting '
               f'the real ones.</p>')
    out.append('<div class="tablewrap"><table><thead><tr><th>Week starting</th>'
               "<th>Rode</th><th>Docked</th><th>Never docked</th>"
               "<th>Docked share</th><th></th></tr></thead><tbody>")
    for (p, r), is_p in zip(va.iterrows(), va_partial):
        out.append(f'<tr><td>{p.strftime("%d %b %Y")}</td>'
                   f'<td>{fmt_int(r["riding"])}</td><td>{fmt_int(r["docked"])}</td>'
                   f'<td>{fmt_int(r["never_docked"])}</td>'
                   f'<td>{fmt_pct(r["docked_pct"])}</td>'
                   f'<td class="dim" style="text-align:left">'
                   f'{"partial week - excluded from the KPIs above" if is_p else ""}'
                   f'</td></tr>')
    out.append("</tbody></table></div>")
    out.append("</section>")

    generated = time.strftime("%Y-%m-%d %H:%M")
    out.append(f'<footer>Generated {esc(generated)} by <code>station_report.py</code> '
               f'&middot; self-contained, no external resources &middot; sources: '
               f'{esc(", ".join(Path(p).name for p in ride_paths))}</footer>')
    out.append("</div>")

    doc = (
        "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<title>Station Performance Report &mdash; Vaasa</title>"
        f"<style>{CSS}</style></head><body>"
        + "".join(out)
        + f"<script>{JS}</script></body></html>"
    )

    headline = {
        "period": period_txt,
        "days": n_days,
        "total_rides": total_rides,
        "total_vehicles": total_vehicles,
        "assessable": assessable,
        "unassessable": unassessable,
        "docked_total": docked_total,
        "dock_rate_pct": dock_rate,
        "docked_share_full_weeks_pct": docked_share_full,
        "docked_share_all_weeks_pct": docked_share_all,
        "total_kwh_need": total_kwh,
        "total_kwh_supplied": total_supply_kwh,
        "pct_catered": overall_catered,
        "charging_sessions": charging.n_sessions if have_chg else 0,
        "rider_docked_pct": rider_pct,
        "rider_docked_pct_incl_depot": rider_pct_all,
        "reverse_plugin_pct": reverse_pct,
        "dock_peak_week": peak_wk.strftime("%d %b"),
        "dock_peak_bounds": (float(wd.loc[peak_wk, "lo_pct"]),
                             float(wd.loc[peak_wk, "hi_pct"])),
        "dock_trough_week": trough_wk.strftime("%d %b"),
        "dock_trough_bounds": (float(wd.loc[trough_wk, "lo_pct"]),
                               float(wd.loc[trough_wk, "hi_pct"])),
        "dock_bounds_overlap": overlap,
        "waterfall_ok": ok,
        "charging_error": charging_error,
    }
    return doc, headline


# --------------------------------------------------------------------------- #
# Self-tests
# --------------------------------------------------------------------------- #

def run_self_tests(ride_paths: list[str], racks: list[Rack], sites: list[Site],
                   charging_path: str | None = None,
                   tz_shift_h: int = CHARGING_TZ_SHIFT_H) -> int:
    failures = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        if not ok:
            failures += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")

    print("\nTEST C - filter waterfall reconciles exactly")
    raw, _ = load_rides(ride_paths)
    df, wf = prepare_rides(raw)
    check("raw == kept + duplicate + unparseable + short",
          wf.reconciles(),
          f"{wf.raw_rows} == {wf.kept} + {wf.dropped_duplicate} + "
          f"{wf.dropped_unparseable} + {wf.dropped_short}")
    df = assign_docking(df, sites)
    poslog = position_log(raw)

    print("\nTEST D - docking rate on the 86k file alone vs known 8.9%")
    solo = [p for p in ride_paths if "29.06" in Path(p).name] or [ride_paths[-1]]
    raw2, _ = load_rides(solo)
    df_nf, wf_nf = prepare_rides(raw2, min_ride_m=0.0)     # filter disabled
    df_nf = assign_docking(df_nf, sites)
    a_nf = int(df_nf["has_end"].sum())
    d_nf = int(df_nf["docked"].sum())
    r_nf = 100.0 * d_nf / a_nf
    df_f, _ = prepare_rides(raw2, min_ride_m=MIN_RIDE_M)
    df_f = assign_docking(df_f, sites)
    a_f = int(df_f["has_end"].sum())
    d_f = int(df_f["docked"].sum())
    r_f = 100.0 * d_f / a_f
    print(f"    without <{MIN_RIDE_M:.0f}m filter: {d_nf}/{a_nf} = {r_nf:.2f}%")
    print(f"    with    <{MIN_RIDE_M:.0f}m filter: {d_f}/{a_f} = {r_f:.2f}%")
    print(f"    delta: {r_f - r_nf:+.2f} pp")
    check("without-filter rate reproduces 8.9% +/- 0.2pp", abs(r_nf - 8.9) <= 0.2,
          f"got {r_nf:.2f}%")

    print("\nTEST E - clustering reproduces 12 sites with expected rack counts")
    counts = [s.rack_count for s in sites]
    expected = [4, 3, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1]
    check("12 sites", len(sites) == 12, f"got {len(sites)}")
    check("rack counts match", sorted(counts, reverse=True) == expected,
          f"got {sorted(counts, reverse=True)}")
    lo, hi, here, below = clustering_stability([(r.lat, r.lon) for r in racks])
    print(f"    site count is {here} for merge radius in [{lo:.2f}, {hi:.2f}) m; "
          f"{below} just below {lo:.2f} m")
    check("50 m sits strictly inside the plateau, not on its edge",
          lo < MERGE_RADIUS_M < hi, f"[{lo:.2f}, {hi:.2f})")

    print("\nTEST F - Section 5 bounds: the docking 'decline' is not established")
    data_start = df["start_time"].min().normalize()
    data_end = df["start_time"].max().normalize() + pd.Timedelta(days=1)
    wd = weekly_docking(df)
    wk_partial = partial_flags(list(wd.index), "week", data_start, data_end)
    wdf = full_periods(wd, wk_partial)
    pk, tr = wdf["dock_rate_pct"].idxmax(), wdf["dock_rate_pct"].idxmin()
    print(f"    peak   {pk:%d %b}: point {wd.loc[pk, 'dock_rate_pct']:.2f}%  "
          f"bounds [{wd.loc[pk, 'lo_pct']:.2f}, {wd.loc[pk, 'hi_pct']:.2f}]")
    print(f"    trough {tr:%d %b}: point {wd.loc[tr, 'dock_rate_pct']:.2f}%  "
          f"bounds [{wd.loc[tr, 'lo_pct']:.2f}, {wd.loc[tr, 'hi_pct']:.2f}]")
    print("    weekly unassessable %: "
          + ", ".join(f"{v:.2f}" for v in wd["unassessable_pct"]))
    check("peak-week bounds overlap trough-week bounds (no trend claimable)",
          bounds_overlap(wd, pk, tr))
    check("lo <= point <= hi for every week",
          bool(((wd["lo_pct"] <= wd["dock_rate_pct"] + 1e-9)
                & (wd["dock_rate_pct"] <= wd["hi_pct"] + 1e-9)).all()))
    m_rate = 100.0 * df.loc[~df["has_end"], "start_docked"].mean()
    h_rate = 100.0 * df.loc[df["has_end"], "start_docked"].mean()
    check("missing-end rides are biased towards stations (not missing at random)",
          m_rate > h_rate, f"{m_rate:.2f}% vs {h_rate:.2f}% start within "
                           f"{DOCK_RADIUS_M:.0f} m")

    print("\nTEST G - full-weeks-only KPI rule")
    va = weekly_vehicle_docking(df)
    va_partial = partial_flags(list(va.index), "week", data_start, data_end)
    vaf = full_periods(va, va_partial)
    pooled_full = 100.0 * vaf["docked"].sum() / vaf["riding"].sum()
    pooled_all = 100.0 * va["docked"].sum() / va["riding"].sum()
    print(f"    pooled, full weeks only : {pooled_full:.2f}%")
    print(f"    mean of weekly rates    : {vaf['docked_pct'].mean():.2f}%")
    print(f"    pooled, all weeks (NOT reported): {pooled_all:.2f}%")
    check("full-week pooled differs from the all-week figure",
          abs(pooled_full - pooled_all) > 1.0,
          f"{pooled_full:.2f}% vs {pooled_all:.2f}%")
    check("partial weeks excluded", int((~np.array(va_partial)).sum()) == len(vaf))

    print("\nTEST H - peak annotations skip partial periods")
    u = fleet_utilisation(df, "week")
    up = partial_flags(list(u.index), "week", data_start, data_end)
    vals = list(u["rides_per_vehicle_per_day"])
    naive = max(range(len(vals)), key=lambda i: vals[i])
    chosen = _label_indices(vals, up)
    print(f"    unguarded max would be {vals[naive]:.2f} on "
          f"{u.index[naive]:%d %b} (partial={up[naive]})")
    print(f"    labelled: " + ", ".join(f"{u.index[i]:%d %b}={vals[i]:.2f}"
                                        for i in chosen))
    check("no partial period is labelled", not any(up[i] for i in chosen))
    check("the 1-day stub is not reported as the peak", naive not in chosen)

    if not charging_path:
        print("\n(no charging export - skipping charging tests)")
        return failures

    print("\nTEST I - charging export loads, and the +3 h clock offset holds")
    try:
        chg = load_charging(charging_path, racks, tz_shift_h)
    except ChargingDataError as exc:
        check("charging export loads", False, str(exc))
        return failures
    attach_sites(chg, racks, sites)
    check("charging export loads", True,
          f"{chg.n_sessions:,} sessions, {chg.sessions['rack_id'].nunique()} racks, "
          f"{chg.total_kwh:,.1f} kWh")
    check(f"operator filter leaves {CHARGING_OPERATOR} only",
          set(chg.sessions["rack_id"]).issubset({r.rack_id for r in racks}))
    scan = scan_tz_offsets(chg, poslog)
    for _, r in scan.iterrows():
        mark = "  <-- configured" if int(r["offset_h"]) == tz_shift_h else ""
        print(f"    offset {int(r['offset_h']):+d} h: {int(r['matches']):4d} "
              f"matches, median {r['median_dist_m']:7.1f} m{mark}")
    best = int(scan.loc[scan["matches"].idxmax(), "offset_h"])
    check(f"best offset is the configured +{tz_shift_h} h", best == tz_shift_h,
          f"got {best:+d} h")
    runner = scan.loc[scan["offset_h"] != best, "matches"].max()
    check("the winner is unambiguous (>=2x the runner-up)",
          scan["matches"].max() >= 2 * runner,
          f"{int(scan['matches'].max())} vs {int(runner)}")

    print("\nTEST J - energy balance")
    need = float(energy_need_kwh(len(df)))
    catered = 100.0 * chg.total_kwh / need
    print(f"    demand {need:,.1f} kWh = {len(df):,} rides x {WH_PER_RIDE} Wh")
    print(f"    supply {chg.total_kwh:,.1f} kWh over {chg.n_sessions:,} sessions")
    check("share of demand catered reproduces 30.8% +/- 0.1pp",
          abs(catered - 30.8) <= 0.1, f"got {catered:.2f}%")

    print("\nTEST K - per-site kWh partitions the total exactly")
    by_site = chg.sessions.groupby("site")["kwh"].sum()
    check("every session maps to a site", int(chg.sessions["site"].isna().sum()) == 0)
    check("sum of per-site kWh == total kWh",
          abs(float(by_site.sum()) - chg.total_kwh) < 1e-6,
          f"{by_site.sum():,.4f} vs {chg.total_kwh:,.4f}")
    by_rack = chg.sessions.groupby("rack_id")["kwh"].sum()
    check("sum of per-rack kWh == total kWh",
          abs(float(by_rack.sum()) - chg.total_kwh) < 1e-6)

    print("\nTEST L - rider-docked share")
    cls_all = classify_sessions(chg, poslog)
    cls = cls_all.loc[~cls_all["rack_id"].isin(DEPOT_RACK_IDS)]
    r_all = 100.0 * float(cls_all["rider_docked"].mean())
    r_ex = 100.0 * float(cls["rider_docked"].mean())
    print(f"    all racks      : {int(cls_all['rider_docked'].sum())}/"
          f"{len(cls_all)} = {r_all:.2f}%")
    print(f"    excluding depot: {int(cls['rider_docked'].sum())}/"
          f"{len(cls)} = {r_ex:.2f}%")
    check("rider-docked share (all racks) reproduces ~3.0%",
          abs(r_all - 3.03) <= 0.1, f"got {r_all:.2f}%")
    check("excluding the depot does not change the conclusion", r_ex < 5.0,
          f"got {r_ex:.2f}%")
    buckets = cls_all["bucket"].value_counts()
    check("every session lands in exactly one bucket",
          int(buckets.sum()) == len(cls_all))
    # The section excludes depot racks from rider-behaviour metrics, so the
    # self-test exercises the same rack list the report renders.
    rev_racks = [r for r in racks if r.rack_id not in DEPOT_RACK_IDS]
    rev = reverse_plugin_rate(poslog, chg, rev_racks,
                              since=chg.sessions["started"].min().normalize())
    rpct = 100.0 * rev["plugged"].sum() / rev["ends"].sum()
    rpct_any = 100.0 * rev["plugged_any"].sum() / rev["ends"].sum()
    print(f"    reverse, same rack (strict mirror): {int(rev['plugged'].sum())}/"
          f"{int(rev['ends'].sum())} = {rpct:.2f}%")
    print(f"    reverse, any rack (upper bound)   : "
          f"{int(rev['plugged_any'].sum())}/{int(rev['ends'].sum())} = {rpct_any:.2f}%")
    check("reverse plug-in rate, strict same-rack mirror, is ~1.6%",
          abs(rpct - 1.64) <= 0.2, f"got {rpct:.2f}%")
    check("any-rack upper bound exceeds the strict mirror", rpct_any > rpct,
          f"{rpct_any:.2f}% vs {rpct:.2f}%")
    check("both reverse figures stay far below the forward rider-docked share",
          rpct_any < 5.0, f"got {rpct_any:.2f}%")

    print("\nTEST M - depot and commissioning flags")
    comm = rack_commissioning(chg, racks)
    late = list(comm.loc[comm["late"]].index)
    print(f"    commissioned >= {COMMISSION_LAG_DAYS} days late: "
          + ", ".join(f"{i} ({comm.loc[i, 'first_session']:%d %b})" for i in late))
    check("mid-window commissioning is detected", len(late) >= 1)
    for rid in sorted(DEPOT_RACK_IDS):
        prof = depot_profile(chg, poslog, racks, rid)
        print(f"    {rid}: {prof['sessions']} sessions, "
              f"{prof['kwh']:.1f} kWh, {prof['ride_ends_within_radius']} ride-ends "
              f"within {RIDER_DOCK_RADIUS_M:.0f} m, "
              f"{prof['also_elsewhere_pct']:.1f}% of its vehicles also use other racks")
        check(f"{rid} has real charging volume but no rider traffic",
              prof["sessions"] > 100 and prof["ride_ends_within_radius"] < 10)
    return failures


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def load_racks(path: str | Path) -> list[Rack]:
    """Read the rack registry: id, coordinates and receptacle count per rack."""
    df = pd.read_csv(Path(path))
    cols = {str(c).strip().lower(): c for c in df.columns}
    lat_c = next((cols[k] for k in ("lat", "latitude", "rack_lat") if k in cols), None)
    lon_c = next((cols[k] for k in ("lon", "lng", "longitude", "rack_lon")
                  if k in cols), None)
    id_c = next((cols[k] for k in ("rack_id", "rack", "id") if k in cols), None)
    rec_c = next((cols[k] for k in ("receptacles", "receptacle_count", "slots")
                  if k in cols), None)
    if lat_c is None or lon_c is None:
        raise ValueError("rack CSV needs latitude and longitude columns")
    out: list[Rack] = []
    for i, row in df.iterrows():
        lat, lon = pd.to_numeric(row[lat_c], errors="coerce"), \
            pd.to_numeric(row[lon_c], errors="coerce")
        if pd.isna(lat) or pd.isna(lon):
            continue
        rid = str(row[id_c]).strip() if id_c else f"rack-{i:02d}"
        rec = int(pd.to_numeric(row[rec_c], errors="coerce")) if rec_c else 0
        out.append(Rack(rack_id=rid, lat=float(lat), lon=float(lon), receptacles=rec))
    return out


def fallback_racks() -> list[Rack]:
    """The hard-coded coordinates, used only when no registry file is present."""
    return [Rack(rack_id=f"rack-{i:02d}", lat=a, lon=b, receptacles=0)
            for i, (a, b) in enumerate(RACK_COORDS)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Generate the standalone Standab station performance report.")
    ap.add_argument("--rides", action="append", default=None,
                    help="ride export CSV (repeatable)")
    ap.add_argument("--charging", default=None,
                    help="charging session CSV (Metabase export)")
    ap.add_argument("--racks", default=None,
                    help="rack registry CSV (rack_id, lat, lng, receptacles)")
    ap.add_argument("--charging-tz-shift", type=int, default=CHARGING_TZ_SHIFT_H,
                    help=("hours added to charging timestamps to reach ride-file "
                          f"local time (default {CHARGING_TZ_SHIFT_H}; the value "
                          "is re-derived from the data and disagreement is "
                          "reported loudly)"))
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output HTML path")
    ap.add_argument("--self-test", action="store_true",
                    help="run reconciliation and cross-check tests, then exit")
    args = ap.parse_args(argv)

    ride_paths = args.rides or list(DEFAULT_RIDE_FILES)
    rack_path = args.racks or (str(DEFAULT_RACK_FILE)
                               if DEFAULT_RACK_FILE.exists() else None)
    racks = load_racks(rack_path) if rack_path else fallback_racks()
    sites = cluster_racks([(r.lat, r.lon) for r in racks], MERGE_RADIUS_M)
    chg_path = args.charging or (DEFAULT_CHARGING_FILE
                                 if Path(DEFAULT_CHARGING_FILE).exists() else None)

    if args.self_test:
        print(f"Racks: {len(racks)} -> sites: {len(sites)}")
        rc = run_self_tests(ride_paths, racks, sites, chg_path,
                            args.charging_tz_shift)
        print(f"\n{'ALL SELF-TESTS PASSED' if rc == 0 else f'{rc} FAILURE(S)'}")
        return 1 if rc else 0

    t0 = time.time()
    raw, per_file = load_rides(ride_paths)
    df, wf = prepare_rides(raw)
    if not wf.reconciles():
        raise AssertionError(
            f"filter waterfall does not reconcile: raw={wf.raw_rows} "
            f"kept={wf.kept} dup={wf.dropped_duplicate} "
            f"bad_ts={wf.dropped_unparseable} short={wf.dropped_short}")
    df = assign_docking(df, sites)
    poslog = position_log(raw)

    # A broken charging export must not cost us the whole report: it is caught
    # here and rendered as a visible failure inside the HTML instead.
    charging: Charging | None = None
    charging_error: str | None = None
    tz_scan: pd.DataFrame | None = None
    if chg_path:
        try:
            charging = load_charging(chg_path, racks, args.charging_tz_shift)
            attach_sites(charging, racks, sites)
            tz_scan = scan_tz_offsets(charging, poslog)
        except ChargingDataError as exc:
            charging, tz_scan = None, None
            charging_error = str(exc)
            print(f"  WARNING: charging export unusable - {exc}")

    doc, headline = build_report(df, poslog, sites, racks, wf, per_file,
                                 charging, charging_error, tz_scan,
                                 ride_paths, chg_path)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc, encoding="utf-8")
    elapsed = time.time() - t0

    print(f"Wrote {out_path}")
    print(f"  wall clock        : {elapsed:.2f} s")
    print(f"  output size       : {out_path.stat().st_size:,} bytes")
    print(f"  period            : {headline['period']} ({headline['days']} days)")
    print(f"  rides after filters : {headline['total_rides']:,}")
    print(f"  distinct vehicles : {headline['total_vehicles']:,}")
    print(f"  docking success   : {headline['dock_rate_pct']:.2f}% "
          f"({headline['docked_total']:,}/{headline['assessable']:,})")
    print(f"  unassessable ends : {headline['unassessable']:,}")
    print(f"  docking bounds    : {headline['dock_peak_week']} "
          f"[{headline['dock_peak_bounds'][0]:.2f}, "
          f"{headline['dock_peak_bounds'][1]:.2f}] vs "
          f"{headline['dock_trough_week']} "
          f"[{headline['dock_trough_bounds'][0]:.2f}, "
          f"{headline['dock_trough_bounds'][1]:.2f}] -> "
          f"{'OVERLAP (no trend)' if headline['dock_bounds_overlap'] else 'disjoint'}")
    print(f"  docked share (full weeks): "
          f"{headline['docked_share_full_weeks_pct']:.2f}%")
    print(f"  total kWh need    : {headline['total_kwh_need']:,.1f}")
    if headline["charging_error"]:
        print(f"  charging          : UNUSABLE - {headline['charging_error']}")
    else:
        print(f"  kWh supplied      : {headline['total_kwh_supplied']:,.1f} "
              f"({headline['charging_sessions']:,} sessions)")
        print(f"  % of demand catered : {headline['pct_catered']:.2f}%")
        print(f"  rider-docked share: {headline['rider_docked_pct']:.2f}% "
              f"(incl. depot {headline['rider_docked_pct_incl_depot']:.2f}%)")
        print(f"  reverse plug-in rate: {headline['reverse_plugin_pct']:.2f}%")
    print(f"  waterfall balanced: {headline['waterfall_ok']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
