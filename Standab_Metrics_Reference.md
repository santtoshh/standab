# Standab Pre-Pilot Analysis — Metrics & Calculation Reference

> **Purpose:** Cross-reference document for all metrics, formulas, and logic used in the HTML output.
> Each section corresponds to a UI panel. Columns: **Metric**, **Logic/Formula**, **Remarks**.

---

## 1. General Statistics (Info Bar — Expanded)

Baseline values computed in Python, then updated live by JavaScript after **Apply** using the **Activity multiplier** (`m`).

| # | Metric | Logic / Formula | Remarks |
|---|--------|-----------------|---------|
| 1.1 | **Total rides** | Python: count of ride-end rows in `[period_start, period_end)` within search boundary. JS after Apply: `total_rides × m` | `m` = Activity multiplier, clamped [0.5, 5]. |
| 1.2 | **Unique vehicles (period)** | `nunique(vehicle_id)` across full period | **Not** scaled by multiplier. Represents distinct vehicle IDs that had at least one ride. |
| 1.3 | **Avg rides/vehicle (period)** | `total_rides / unique_vehicles`. JS: `(total_rides × m) / unique_vehicles` | Numerator scales; denominator does not. |
| 1.4 | **Avg rides / active vehicle / day** | For each calendar day: `rides_that_day / vehicles_that_rode_that_day`. Then: `mean()` over all days. JS: `value × m` | "Active" = vehicle with ≥1 ride on that specific day. This is **not** the same as dividing total rides by total unique vehicles. |

---

## 2. Monthly Table (Info Bar — Expanded)

One row per calendar month. Python pre-computes; JavaScript recalculates with multiplier after Apply.

| # | Metric | Logic / Formula | Remarks |
|---|--------|-----------------|---------|
| 2.1 | **Rides** | `groupby(month).size()`. JS: `rides × m` | Total ride-ends in that month. |
| 2.2 | **Unique vehicles** | `groupby(month).vehicle_id.nunique()` | **Not** scaled. Distinct vehicles with ≥1 ride in the month. |
| 2.3 | **Rides/day** | `rides / calendar_days_in_month`. JS: `(rides × m) / days` | `days` is capped to the overlap between that month and the full `TIME_PERIOD` (handles partial first/last months). |
| 2.4 | **UR (rides/day/veh)** | `rides_per_day / unique_vehicles_that_month` | Simple utilisation rate against the monthly roster. A value >1 means more rides/day than monthly unique vehicles. |
| 2.5 | **Avg active veh/day** | For each day in that month: count distinct vehicles with ≥1 ride. Then: `mean()` over days. | **Not** scaled. Typically much lower than monthly unique vehicles because not all vehicles ride every day. |
| 2.6 | **Avg rides/active veh/day** | For each day in that month: `rides / active_vehicles`. Then: `mean()` over days. JS: `value × m` | When a vehicle rides, how many trips does it make on average? |

---

## 3. Selected Coverage per Bucket (Info Bar — Chart & Tables)

Updated after **Apply**. Uses the greedy-selected hexes and the bucket partition.

| # | Metric | Logic / Formula | Remarks |
|---|--------|-----------------|---------|
| 3.1 | **Coverage % (per bucket)** | `(distinct vehicles visiting any selected hex in bucket) / (total distinct vehicles in bucket) × 100` | Plotted as the line chart. Each bucket spans `bucketDays` active days. |
| 3.2 | **Avg coverage (reference line)** | `mean(coverage_pct across all buckets)` | Horizontal dashed line on chart. |
| 3.3 | **Top 10 / Bottom 10 buckets** | Sorted by coverage %. Columns: Bucket #, Dates, Coverage %, Vehicles (= total fleet in that bucket) | Helps identify which periods have weakest/strongest station coverage. |

---

## 4. Selection Controls (Sidebar — Controls Tab)

### 4a. Summary Strip

| # | Metric | Logic / Formula | Remarks |
|---|--------|-----------------|---------|
| 4.1 | **Bucket** | `bucketDays + "d"` | Partition size in active days. |
| 4.2 | **Top-N** | User slider (max 500) | Max hexes the greedy algorithm can select. |
| 4.3 | **Selected** | Count of hexes with rank ≤ Top-N after greedy | May be < Top-N if fewer hexes exist. |
| 4.4 | **Avg selected coverage** | `mean over buckets of (vehicles_on_selected_hexes / fleet_in_bucket × 100)` | Per-bucket fleet coverage, averaged. |
| 4.5 | **Period unique coverage** | `(union of all vehicle IDs across selected hexes) / (union of all vehicle IDs across ALL hexes) × 100` | Uses **full feature payload**, not just the time window. Can differ from bucket-based metric. |
| 4.6 | **Catchment coverage** | Same as 4.4 but `selected_hexes` expanded to include H3 neighbors within the **catchment radius** | Only shown when "Include unselected neighbors" is checked. |

### 4b. Greedy Algorithm (Station Selection)

| # | Metric | Logic / Formula | Remarks |
|---|--------|-----------------|---------|
| 4.7 | **Per-bucket hex score** | `wDemand × chargecount + wFleet × fleetcoverage` | `chargecount` = scaled end-ride count for hex in bucket. `fleetcoverage` = `(new unique vehicles / fleet_in_bucket) × 100`. |
| 4.8 | **Greedy ranking** | Per bucket: pick hex with highest score, mark its vehicles as "visited", repeat. Position in pick order = rank within bucket. | Mirrors Python optimiser logic. "New unique vehicles" decreases as earlier hexes claim vehicles. |
| 4.9 | **Accumulated score** | `sum over all buckets of (sigma − position)` where `sigma` = total hexes eligible | Hexes ranked globally by this accumulated score. |
| 4.10 | **Hex fill color** | `score / max_score` → mapped to 9-step YlOrRd color ramp | Darker red = higher relative score. |

### 4c. Input Parameters

| # | Parameter | Role | Remarks |
|---|-----------|------|---------|
| 4.11 | **Simulated demand (×)** | Activity multiplier `m`. Scales ride **counts** and derived rates. | Does **not** scale vehicle IDs or coverage %. Clamped [0.5, 5]. |
| 4.12 | **Time window (Start/End)** | Filters data to active days within this range. | Bucket construction uses only these days. |
| 4.13 | **Exclude time range** | Days within this sub-range are dropped before bucketing. | Useful for removing holiday/event anomalies. |
| 4.14 | **Dwell filter (Min/Max)** | Only hexes whose **Python-computed** `median_dwell_mins` falls within range are eligible for greedy. | Filter is on eligibility only; does not recompute dwell client-side. |
| 4.15 | **End ride demand weight** | `wDemand` in greedy formula (4.7) | Slider [0, 1]. |
| 4.16 | **Fleet coverage weight** | `wFleet` in greedy formula (4.7) | Slider [0, 1]. |
| 4.17 | **Days per bucket** | Partition active days into consecutive buckets of this length. | Last bucket may be shorter. |
| 4.18 | **Exclude days with 0 rides** | Drops days with zero ride-ends before building buckets. | Changes bucket boundaries and coverage denominators. |
| 4.19 | **OD day type** | Filters origin-destination travel pattern data by All / Weekday / Weekend. | Only affects Detail tab OD section. |
| 4.20 | **Catchment radius (m)** | Expands selected hexes to include H3 neighbors within this distance. | Used for catchment coverage (4.6) and summary catchment metrics. |
| 4.21 | **Include unselected neighbors** | Whether unselected neighbor hexes are included in catchment calculations. | |
| 4.22 | **kWh per charge** | Energy per ride-end for station sizing. Default: **0.24**. | Used in Summary tab kWh calculations. |
| 4.23 | **Target coverage finder** | Greedy-adds hexes in score order until avg per-bucket coverage ≥ target %. | Returns minimum station count or "not achievable". |

---

## 5. Detail Tab (Sidebar — Per-Hex Panel)

Shown when a hex is selected. All metrics use the **current time window** unless noted "full period".

### 5a. Key Performance Indicators

| # | Metric | Logic / Formula | Remarks |
|---|--------|-----------------|---------|
| 5.1 | **Ends/day** | `sum(end_counts × m) / active_window_days` | Recalculated from raw daily data after Apply. |
| 5.2 | **Avg k-day coverage** | `mean over buckets of (unique_vehicles_at_hex / fleet_in_bucket × 100)` | Label dynamically shows bucket size (e.g. "Avg 5-day coverage"). |
| 5.3 | **Dwell ≥ 60 (%, full period)** | From Python: `dwells_ge_60 / dwells_defined × 100` | **Full period**, not filtered by time window slider. |
| 5.4 | **Avg daily net flow** | `(total_ends − total_starts) / window_days × m` | Positive = net inflow (more rides ending than starting). |

### 5b. Demand & Flow Detail

| # | Metric | Logic / Formula | Remarks |
|---|--------|-----------------|---------|
| 5.5 | **Ends / Starts / Net bar** | SVG bar: `ends`, `starts`, `net = ends − starts` per window | Visual relative comparison. |
| 5.6 | **Monthly trend sparkline** | Weekday vs weekend **medians** per month of daily end counts | Uses median, not mean, to reduce outlier sensitivity. |
| 5.7 | **Monthly stats table** | Per month: min / median / avg / max / P80 / days ≥ P80 of daily end counts | "Days ≥ P80" = count of days at or above the 80th-percentile end count. |

### 5c. Top Travel Patterns (OD)

| # | Metric | Logic / Formula | Remarks |
|---|--------|-----------------|---------|
| 5.8 | **Top 5 outbound / inbound** | Aggregate `OD_BY_DAY` for hex; `trips_scaled = trips × m`; `% = trips / total_starts (or total_ends)` | Unique vehicles shown but **not** scaled. Filtered by OD day type. |

### 5d. Charging Windows (Dwell Analysis)

| # | Metric | Logic / Formula | Remarks |
|---|--------|-----------------|---------|
| 5.9 | **Charging windows** | From raw ride ends: `dwell_same_loc_mins` (only if next start within 50 m). Day = 07:00–21:59 UTC, Night = 22:00–06:59. | Recomputed from raw data in **selected window** — can differ from Python full-period dwell. |
| 5.10 | **Monthly dwell table** | Day/Night: avg, median, P90 in **hours** (`value/60`). Day ≥60% = share of dwells ≥ 60 min. | "≥60%" is share of events with dwell ≥ 60 minutes, not a time threshold. |

### 5e. Peak Presence

| # | Metric | Logic / Formula | Remarks |
|---|--------|-----------------|---------|
| 5.11 | **Monthly avg daily peak 1-min presence** | For each day: max simultaneous vehicles present in hex (1-min granularity). Vehicle present from `end_time` to `next_start_time`. Per month: mean of daily peaks. | Does **not** use the 50 m same-location rule. Edge months may impute missing days with month mean. |

---

## 6. Summary Tab (Sidebar — Table & Exports)

Per-station row. Updated after Apply.

| # | Metric | Logic / Formula | Remarks |
|---|--------|-----------------|---------|
| 6.1 | **Hex rank** | Greedy rank from accumulated score | |
| 6.2 | **Est. avg hex ends/month** | `(total_hex_ends_in_window / window_days) × 30` | Estimated monthly average; uses 30-day month. Label says "Est. avg" to reflect this. |
| 6.3 | **Hex dwell ≥ 60 (%, full period)** | Same as 5.3: `dwells_ge_60 / dwells_defined × 100` | Full period, not window-filtered. |
| 6.4 | **Neighbors in range** | H3 hexes within catchment radius. CSV marks selected neighbors with `*`. | |
| 6.5 | **Est. avg catchment ends/month** | `(deduplicated_ends / window_days) × 30`. **Nearest-station assignment**: each hex assigned to closest selected station (min H3 distance; tie-break: lower rank). | Deduplication ensures each hex's ends count toward only **one** station. |
| 6.6 | **Catchment dwell ≥ 60** | Ends-weighted mean of `hex_dwell_ge_60_pct` over all hexes assigned to station | Weighted by end count, not simple average. |
| 6.7 | **Est. kWh/month** | `catchment_ends_per_month × kWh_per_charge` | |
| 6.8 | **System total row** | Sums of hex-ends, deduplicated catchment ends, and system kWh | System catchment ends are globally deduplicated (each hex counted once). |

### CSV / XLSX Export

| # | Detail | Remarks |
|---|--------|---------|
| 6.9 | **CSV** | Mirrors the summary table columns. System total row uses `kWh_input || 3.5` as fallback. | **Note:** fallback `3.5` differs from UI default `0.24` — if the kWh input is accidentally cleared, the CSV total will use 3.5. |
| 6.10 | **Business Case XLSX** | Per-station sheet: `endsMonth`, `kwhMonth`, revenue via `billingLadderCost(kwhMonth)` across fixed billing tiers. Other cells are static placeholders. | Revenue tiers are hardcoded in JS. |

---

## 7. Map Tooltips

| # | Metric | Logic / Formula | Remarks |
|---|--------|-----------------|---------|
| 7.1 | **Rank** | Greedy rank (or "NA" if unranked) | |
| 7.2 | **Score** | `hex_score / max_score` (normalized 0–1) | |
| 7.3 | **Ends/day** | Same as 5.1 | |
| 7.4 | **Avg k-day coverage %** | Same as 5.2 | |
| 7.5 | **Net flow/day** | Same as 5.4 | |
| 7.6 | **Dwell ≥ 60 %** | Same as 5.3 (full period) | |
| 7.7 | **Weekday / Weekend ends/day** | `weekday_end_day_avg × m`, `weekend_end_day_avg × m` | Shown on separate weekpart overlay layer. |

---

## 8. Python Pre-Processing (Upstream Calculations)

These feed into the GeoJSON payload consumed by the JavaScript.

| # | Metric | Logic / Formula | Remarks |
|---|--------|-----------------|---------|
| 8.1 | **`dwell_same_loc_mins`** | `(next_start_time − end_time)` in minutes, only if haversine(end, next_start) ≤ 50 m | Negative values → NaN. Requires `next_start_time` to exist. |
| 8.2 | **`median_dwell_mins`** | `groupby(end_h3).dwell_same_loc_mins.median()` | Used by dwell filter (4.14). |
| 8.3 | **`share_dwells_ge_60_pct`** | `dwells_ge_60 / dwells_defined × 100` per hex | `dwells_defined` = count of non-null dwell values. |
| 8.4 | **`end_rides_day_avg`** | Mean of daily end counts (zero-filled calendar) per hex | |
| 8.5 | **`weekday_end_day_avg` / `weekend_end_day_avg`** | Same as 8.4 split by day type | |
| 8.6 | **`net_trips_day_avg`** | `mean(daily_ends − daily_starts)` per hex | |
| 8.7 | **`station_score` / `rank`** | Python optimiser output. Score = sum over intervals of `(sigma − pick_position)`. Rank by score desc, end_rides desc. | Overridden by JS after Apply if parameters change. |
| 8.8 | **`avg_5day_vehicle_coverage_pct`** | Per cycle interval: `nunique(vehicles_at_hex) / nunique(vehicles_in_interval) × 100`. Then `mean()` over intervals. | `CYCLE_DURATION` defaults to 5 days. Initial value; JS overrides after Apply. |
| 8.9 | **`incremental_coverage_pct`** | As hexes taken in rank order: `(new unique vehicles) / total_period_fleet × 100` | Cumulative version also computed. |

---

## Key Assumptions & Edge Cases

| # | Item | Detail |
|---|------|--------|
| A | **Activity multiplier scope** | Scales **counts** (rides, ends, starts) and derived rates. Does **not** scale vehicle IDs, unique vehicle counts, or coverage percentages. |
| B | **Period unique coverage vs. bucket coverage** | Period unique coverage (4.5) uses the **full feature payload** (all hexes, all time), not just the selected window. Bucket-based coverage (4.4) uses only the windowed data. These can diverge. |
| C | **Dwell: full period vs. window** | KPI "Dwell ≥ 60" (5.3, 6.3) uses **Python full-period** data. Detail tab "Charging windows" (5.9) recomputes from **raw rides in the selected window**. Values can differ. |
| D | **Peak presence** | Assumes vehicle stays at end-hex until next start. Does **not** apply the 50 m same-location rule (stated in UI). |
| E | **CSV kWh fallback** | System total row in CSV uses `kWh_input || 3.5`. If the kWh input is cleared, fallback is 3.5 instead of the UI default 0.24. |
| F | **Partial months** | `days` for rides/day is capped to the overlap between the calendar month and `TIME_PERIOD`, handling partial first/last months. |
| G | **Nearest-station deduplication** | In Summary catchment metrics, each hex is assigned to exactly one station (closest by H3 distance; tie-break: lower rank). This prevents double-counting. |
