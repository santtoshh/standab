"""Helpers for the upload + column-mapping wizard in [app.py](app.py).

This module is the single source of truth for:
  1. peeking at an uploaded CSV/XLSX (`peek_dataset`) so the frontend
     can render the column-mapping confirmation page,
  2. writing a canonical-named CSV from the user's confirmed mapping
     (`materialize_dataset`) so the analysis notebook can read it as-is.

The auto-detection reuses the alias lists in
[operator_dataset_toggle.py](operator_dataset_toggle.py) so behaviour stays
identical to what the notebook already does silently.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from operator_dataset_toggle import (
    CANONICAL_FIELD_ALIASES,
    REQUIRED_BASE_FIELDS,
    REQUIRED_GEOJSON_FIELDS,
    REQUIRED_LATLNG_FIELDS,
    _extract_coords,
)

PEEK_SAMPLE_ROWS = 5

EXCEL_EXTS = {".xlsx", ".xls"}
CSV_EXTS = {".csv", ".tsv", ".txt"}

EXTRA_PASSTHROUGH_FIELDS: tuple[str, ...] = (
    "ride_id",
    "vehicle_type",
    "city",
    "country",
    "start_battery",
    "end_battery",
    "distance_meters",
    "ride_duration_mins",
    "avg_speed_kmph",
)

CANONICAL_FIELDS: tuple[str, ...] = (
    "vehicle_id",
    "start_time",
    "end_time",
    "start_lat",
    "start_lng",
    "end_lat",
    "end_lng",
    "start_loc",
    "end_loc",
)

# --- Parking hubs (optional map overlay) ----------------------------------
# The overlay is purely visual: one red marker per location with a popup.
# Minimum to render a marker is lat + lng; name makes the popup useful.
PARKING_CANONICAL_FIELDS: tuple[str, ...] = (
    "name",
    "lat",
    "lng",
    "address",
    "zone",
    "capacity",
    "id",
)

PARKING_REQUIRED_FIELDS: tuple[str, ...] = ("lat", "lng")

PARKING_FIELD_ALIASES: dict[str, list[str]] = {
    "name": [
        "name",
        "location",
        "location_name",
        "title",
        "label",
        "standort",
        "place",
        "hub",
        "station",
        "strasse",
        "straße",
        "street",
    ],
    "lat": ["lat", "latitude", "breitengrad", "y"],
    "lng": [
        "lng",
        "lon",
        "long",
        "longitude",
        "laengengrad",
        "längengrad",
        "x",
    ],
    "address": [
        "address",
        "adresse",
        "street2",
        "strasse_1",
        "straße_1",
        "strasse.1",
        "straße.1",
        "cross_street",
        "description",
        "desc",
    ],
    "zone": ["zone", "area", "gebiet", "district", "stadtteil", "region"],
    "capacity": [
        "capacity",
        "spots",
        "spaces",
        "slots",
        "abstellpunkte",
        "count",
    ],
    "id": [
        "id",
        "identifier",
        "nummer",
        "no",
        "number",
        "lfd_nummer",
        "lfd._nummer",
        "lfd_nr",
    ],
}


def _normalize_header(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def _is_excel(path: Path) -> bool:
    return path.suffix.lower() in EXCEL_EXTS


def _read_head(path: Path, sheet: str | None = None, nrows: int = PEEK_SAMPLE_ROWS) -> pd.DataFrame:
    if _is_excel(path):
        kwargs: dict[str, Any] = {"nrows": nrows}
        if sheet is not None:
            kwargs["sheet_name"] = sheet
        return pd.read_excel(path, **kwargs)
    return pd.read_csv(path, nrows=nrows, engine="python")


def _read_full(path: Path, sheet: str | None = None) -> pd.DataFrame:
    if _is_excel(path):
        kwargs: dict[str, Any] = {}
        if sheet is not None:
            kwargs["sheet_name"] = sheet
        return pd.read_excel(path, **kwargs)
    return pd.read_csv(path, engine="python")


def _list_sheets(path: Path) -> list[str] | None:
    if not _is_excel(path):
        return None
    xl = pd.ExcelFile(path)
    return list(xl.sheet_names)


def _suggest_mapping(columns: list[str]) -> dict[str, Any]:
    """Map each canonical field to a source column using the alias lists.

    Returns the original-cased source column name when a match is found, or
    ``None``. ``coord_mode`` is set to ``"latlng"`` when at least one of the
    four lat/lng fields auto-maps; otherwise to ``"geojson"`` if any *_loc
    field auto-maps; otherwise defaults to ``"latlng"``.
    """
    norm_to_orig: dict[str, str] = {}
    for orig in columns:
        norm = _normalize_header(orig)
        norm_to_orig.setdefault(norm, orig)

    suggested: dict[str, Any] = {field: None for field in CANONICAL_FIELDS}

    for field, aliases in CANONICAL_FIELD_ALIASES.items():
        if field not in CANONICAL_FIELDS:
            continue
        for alias in aliases:
            if alias in norm_to_orig:
                suggested[field] = norm_to_orig[alias]
                break

    has_latlng = any(
        suggested.get(f) for f in REQUIRED_LATLNG_FIELDS
    )
    has_geojson = all(suggested.get(f) for f in REQUIRED_GEOJSON_FIELDS)
    if has_geojson and not has_latlng:
        suggested["coord_mode"] = "geojson"
    else:
        suggested["coord_mode"] = "latlng"
    return suggested


def _coerce_sample_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (np.floating,)):
        f = float(value)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (dict, list)):
        return value
    return str(value)


def _df_to_sample(df: pd.DataFrame) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for _, record in df.iterrows():
        rows.append([_coerce_sample_value(record[col]) for col in df.columns])
    return rows


def _build_warnings(suggested: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    coord_mode = suggested.get("coord_mode", "latlng")
    for field in REQUIRED_BASE_FIELDS:
        if not suggested.get(field):
            warnings.append(
                f"Could not auto-detect a column for required field '{field}'."
            )
    if coord_mode == "latlng":
        missing = [f for f in REQUIRED_LATLNG_FIELDS if not suggested.get(f)]
        if missing:
            warnings.append(
                "Lat/Lng coordinate columns missing auto-detection: "
                + ", ".join(missing)
                + ". Switch to GeoJSON mode if your file uses start_loc/end_loc."
            )
    else:
        missing = [f for f in REQUIRED_GEOJSON_FIELDS if not suggested.get(f)]
        if missing:
            warnings.append(
                "GeoJSON coordinate columns missing auto-detection: "
                + ", ".join(missing)
                + "."
            )
    return warnings


def peek_dataset(path: str | Path, sheet: str | None = None) -> dict[str, Any]:
    """Sniff a CSV/XLSX file and return columns + sample rows + auto-mapping.

    Used by ``POST /preview`` to render the column-mapping confirmation
    page. Reads only the header + first ``PEEK_SAMPLE_ROWS`` rows; safe to
    call on multi-GB inputs.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")

    sheets = _list_sheets(p)
    chosen_sheet: str | None
    if sheet is not None:
        chosen_sheet = sheet
    elif sheets:
        chosen_sheet = sheets[0]
    else:
        chosen_sheet = None

    df_head = _read_head(p, sheet=chosen_sheet, nrows=PEEK_SAMPLE_ROWS)
    columns = [str(c) for c in df_head.columns]

    suggested = _suggest_mapping(columns)
    sample_rows = _df_to_sample(df_head)
    warnings = _build_warnings(suggested)

    return {
        "columns": columns,
        "sample_rows": sample_rows,
        "sheets": sheets,
        "active_sheet": chosen_sheet,
        "suggested": suggested,
        "warnings": warnings,
    }


def validate_mapping(mapping: dict[str, Any], coord_mode: str) -> list[str]:
    """Return a list of validation errors for a confirmed mapping.

    Empty list = OK to materialize.
    """
    errors: list[str] = []
    for field in REQUIRED_BASE_FIELDS:
        if not mapping.get(field):
            errors.append(f"Required field '{field}' is not mapped.")

    if coord_mode == "geojson":
        required = REQUIRED_GEOJSON_FIELDS
    else:
        coord_mode = "latlng"
        required = REQUIRED_LATLNG_FIELDS
    for field in required:
        if not mapping.get(field):
            errors.append(f"Required field '{field}' is not mapped.")

    seen: dict[str, str] = {}
    for canonical, source in mapping.items():
        if not source:
            continue
        if source in seen:
            errors.append(
                f"Source column '{source}' is mapped to multiple canonical fields "
                f"('{seen[source]}' and '{canonical}')."
            )
        else:
            seen[source] = canonical
    return errors


def materialize_dataset(
    src_path: str | Path,
    dest_path: str | Path,
    mapping: dict[str, Any],
    coord_mode: str = "latlng",
    sheet: str | None = None,
) -> Path:
    """Apply ``mapping`` to ``src_path`` and write a canonical CSV at ``dest_path``.

    ``mapping`` is ``{canonical_field: source_column_name | None}``. In GeoJSON
    mode, ``start_loc`` / ``end_loc`` are decoded into the four lat/lng columns
    using ``_extract_coords`` so that the downstream notebook works regardless
    of the original input shape.
    """
    src = Path(src_path)
    dest = Path(dest_path)

    coord_mode = (coord_mode or "latlng").lower()
    if coord_mode not in {"latlng", "geojson"}:
        coord_mode = "latlng"

    errors = validate_mapping(mapping, coord_mode)
    if errors:
        raise ValueError("; ".join(errors))

    df = _read_full(src, sheet=sheet)

    inverse: dict[str, str] = {}
    for canonical, source in mapping.items():
        if not source:
            continue
        if source not in df.columns:
            raise ValueError(
                f"Column '{source}' (mapped to '{canonical}') is not present in the source file."
            )
        inverse[source] = canonical

    used_sources = set(inverse.keys())
    keep_cols: list[str] = list(used_sources)

    norm_lookup = {_normalize_header(c): c for c in df.columns}
    extras_kept: list[str] = []
    for extra in EXTRA_PASSTHROUGH_FIELDS:
        if extra in df.columns and extra not in used_sources:
            extras_kept.append(extra)
            continue
        norm = _normalize_header(extra)
        if norm in norm_lookup and norm_lookup[norm] not in used_sources:
            extras_kept.append(norm_lookup[norm])

    df_out = df[keep_cols + extras_kept].copy()
    df_out = df_out.rename(columns=inverse)

    for canonical_extra in EXTRA_PASSTHROUGH_FIELDS:
        if canonical_extra in df_out.columns:
            continue
        norm = _normalize_header(canonical_extra)
        for col in list(df_out.columns):
            if _normalize_header(col) == norm:
                df_out = df_out.rename(columns={col: canonical_extra})
                break

    if coord_mode == "geojson":
        if "start_loc" in df_out.columns:
            start_pairs = df_out["start_loc"].apply(_extract_coords)
            df_out["start_lat"] = start_pairs.apply(lambda p: p[0])
            df_out["start_lng"] = start_pairs.apply(lambda p: p[1])
        if "end_loc" in df_out.columns:
            end_pairs = df_out["end_loc"].apply(_extract_coords)
            df_out["end_lat"] = end_pairs.apply(lambda p: p[0])
            df_out["end_lng"] = end_pairs.apply(lambda p: p[1])

    final_cols: list[str] = []
    for canonical in CANONICAL_FIELDS:
        if canonical in df_out.columns:
            final_cols.append(canonical)
    for extra in EXTRA_PASSTHROUGH_FIELDS:
        if extra in df_out.columns and extra not in final_cols:
            final_cols.append(extra)
    if final_cols:
        df_out = df_out[final_cols]

    dest.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(dest, index=False)
    return dest


# --- Parking hubs import --------------------------------------------------

def _suggest_parking_mapping(columns: list[str]) -> dict[str, Any]:
    """Map each parking canonical field to a source column via the alias lists."""
    norm_to_orig: dict[str, str] = {}
    for orig in columns:
        norm = _normalize_header(orig)
        norm_to_orig.setdefault(norm, orig)

    suggested: dict[str, Any] = {field: None for field in PARKING_CANONICAL_FIELDS}
    for field in PARKING_CANONICAL_FIELDS:
        for alias in PARKING_FIELD_ALIASES.get(field, []):
            if alias in norm_to_orig:
                suggested[field] = norm_to_orig[alias]
                break
    return suggested


def _build_parking_warnings(suggested: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    for field in PARKING_REQUIRED_FIELDS:
        if not suggested.get(field):
            warnings.append(
                f"Could not auto-detect a column for required field '{field}'."
            )
    if not suggested.get("name"):
        warnings.append(
            "No 'name' column auto-detected; map popups will have no title."
        )
    return warnings


def peek_parking_dataset(path: str | Path, sheet: str | None = None) -> dict[str, Any]:
    """Sniff a parking CSV/XLSX and return columns + sample rows + auto-mapping."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Parking file not found: {p}")

    sheets = _list_sheets(p)
    if sheet is not None:
        chosen_sheet = sheet
    elif sheets:
        chosen_sheet = sheets[0]
    else:
        chosen_sheet = None

    df_head = _read_head(p, sheet=chosen_sheet, nrows=PEEK_SAMPLE_ROWS)
    columns = [str(c) for c in df_head.columns]

    suggested = _suggest_parking_mapping(columns)
    sample_rows = _df_to_sample(df_head)
    warnings = _build_parking_warnings(suggested)

    return {
        "columns": columns,
        "sample_rows": sample_rows,
        "sheets": sheets,
        "active_sheet": chosen_sheet,
        "suggested": suggested,
        "warnings": warnings,
    }


def validate_parking_mapping(mapping: dict[str, Any]) -> list[str]:
    """Return validation errors for a parking mapping. Empty list = OK."""
    errors: list[str] = []
    for field in PARKING_REQUIRED_FIELDS:
        if not mapping.get(field):
            errors.append(f"Required field '{field}' is not mapped.")

    seen: dict[str, str] = {}
    for canonical, source in mapping.items():
        if not source:
            continue
        if source in seen:
            errors.append(
                f"Source column '{source}' is mapped to multiple fields "
                f"('{seen[source]}' and '{canonical}')."
            )
        else:
            seen[source] = canonical
    return errors


def materialize_parking_dataset(
    src_path: str | Path,
    dest_path: str | Path,
    mapping: dict[str, Any],
    sheet: str | None = None,
) -> Path:
    """Write a canonical parking CSV (name, lat, lng, address, zone, capacity, id).

    Coordinates are assumed to already be WGS-84 decimal degrees. Rows without
    a finite lat/lng are dropped (they cannot be placed on the map).
    """
    src = Path(src_path)
    dest = Path(dest_path)

    errors = validate_parking_mapping(mapping)
    if errors:
        raise ValueError("; ".join(errors))

    df = _read_full(src, sheet=sheet)

    inverse: dict[str, str] = {}
    for canonical, source in mapping.items():
        if not source:
            continue
        if source not in df.columns:
            raise ValueError(
                f"Column '{source}' (mapped to '{canonical}') is not present in the source file."
            )
        inverse[source] = canonical

    df_out = df[list(inverse.keys())].copy().rename(columns=inverse)

    df_out["lat"] = pd.to_numeric(df_out.get("lat"), errors="coerce")
    df_out["lng"] = pd.to_numeric(df_out.get("lng"), errors="coerce")
    df_out = df_out[df_out["lat"].notna() & df_out["lng"].notna()].copy()

    if "capacity" in df_out.columns:
        df_out["capacity"] = pd.to_numeric(df_out["capacity"], errors="coerce").astype(
            "Int64"
        )

    if df_out.empty:
        raise ValueError(
            "No parking rows have valid lat/lng after applying the mapping."
        )

    final_cols = [c for c in PARKING_CANONICAL_FIELDS if c in df_out.columns]
    df_out = df_out[final_cols]

    dest.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(dest, index=False)
    return dest
