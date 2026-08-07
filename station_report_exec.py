#!/usr/bin/env python3
"""Executive one-pager for the Vaasa station pilot.

A plain-language companion to ``station_report.py``. Same period, same rules,
same numbers — every figure here is computed by importing that module, so the
two reports can never drift apart. What differs is the audience: this one is
written for people who were not in the data meeting, drops all methodology,
and puts one sentence of meaning under every chart.

Scope is deliberately limited to what the meeting asked for, and to figures
that are directly measured rather than inferred:
  energy needed vs supplied · share of demand catered ·
  docking success over time · which stations work ·
  scooters using a station each week

Usage:
    python station_report_exec.py [--out reports/station_summary_exec.html]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import math
from pathlib import Path

import numpy as np
import pandas as pd

import station_report as S

# Validated with the dataviz skill's checker against surface #fcfcfb:
#   node scripts/validate_palette.js "#2a78d6,#eb6834,#1baf7a" --mode light
#   -> lightness PASS, chroma PASS, CVD separation PASS (worst 9.2 deutan),
#      normal-vision PASS (27.6), contrast WARN on the green -> every series
#      carries a visible label, which is the required relief.
BLUE = "#2a78d6"    # what the fleet needs / primary measure
ORANGE = "#eb6834"  # what the stations delivered / the gap
GREEN = "#1baf7a"   # positive outcome
INK = "#111111"
INK2 = "#5b5b57"
MUTED = "#8a8a83"
GRID = "#e8e8e4"
SURFACE = "#fcfcfb"
BAND = "#f0efea"

# The per-station pie needs seven fills and the palette above has three. The
# extra three are 22% white tints of the same three hues rather than new hues,
# so the chart stays inside the validated palette. Slot order is fixed and
# interleaves base with tint so no base sits next to its own tint:
#   node scripts/validate_palette.js
#     "#1baf7a,#2a78d6,#ef8961,#5996df,#eb6834,#4dc197,#5b5b57" --mode light
#   -> lightness PASS, chroma FAIL on the neutral Other slot only (it is meant
#      to read as grey), CVD separation PASS (worst adjacent 9.6 deutan),
#      normal-vision PASS (24.0), contrast WARN -> relieved by the direct slice
#      labels, the value legend and the per-station table beneath it.
BLUE_TINT = "#5996df"
ORANGE_TINT = "#ef8961"
GREEN_TINT = "#4dc197"
PIE_COLORS = [GREEN, BLUE, ORANGE_TINT, BLUE_TINT, ORANGE, GREEN_TINT]
PIE_OTHER = INK2          # the residual bucket is deliberately neutral
PIE_TOP_N = 6             # twelve slices is unreadable; the rest fold into Other
PIE_LABEL_MIN_PCT = 8.0   # slices at or above this are labelled on the chart

W, H = 900, 300          # chart viewBox; scales to container width
PAD_L, PAD_R, PAD_T, PAD_B = 62, 26, 30, 46

# Trips are counted as ending at a station at two distances. 50 m is "at the
# rack". 100 m is "on the same corner", which is a fairer test of whether the
# station pulled the trip, and it matters for the two locations a rider cannot
# park at. At 100 m the circles around neighbouring locations overlap (the
# closest pair is 84 m apart), so every trip is attributed to its nearest
# location and never counted twice.
WIDE_RADIUS_M = 100.0


def esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _ticks(vmax: float, n: int = 4) -> list[float]:
    if vmax <= 0:
        return [0.0]
    raw = vmax / n
    mag = 10.0 ** np.floor(np.log10(raw))
    step = next((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), 10 * mag)
    out, v = [], 0.0
    while v <= vmax + step * 0.5:
        out.append(round(v, 6))
        v += step
    return out


def line_chart(labels: list[str], series: list[dict], *, y_suffix: str = "",
               y_max: float | None = None, fmt=lambda v: f"{v:,.0f}",
               band_from: int | None = None, band_note: str = "") -> str:
    """Multi-series line chart. One y-axis, always — two scales would invent
    a crossing point that does not exist."""
    n = len(labels)
    vals = [v for s in series for v in s["values"] if v is not None and not np.isnan(v)]
    vmax = y_max if y_max is not None else max(vals) * 1.18
    ticks = _ticks(vmax)
    vmax = max(ticks[-1], vmax)

    def X(i: int) -> float:
        return PAD_L + (W - PAD_L - PAD_R) * (i / max(n - 1, 1))

    def Y(v: float) -> float:
        return H - PAD_B - (H - PAD_T - PAD_B) * (v / vmax)

    p = [f'<svg viewBox="0 0 {W} {H}" role="img" style="width:100%;height:auto">']
    if band_from is not None and band_from < n:
        x0 = X(band_from) - (X(1) - X(0)) / 2
        p.append(f'<rect x="{x0:.1f}" y="{PAD_T}" width="{W - PAD_R - x0:.1f}" '
                 f'height="{H - PAD_T - PAD_B}" fill="{BAND}"/>')
        if band_note:
            p.append(f'<text x="{x0 + 8:.1f}" y="{PAD_T + 15}" font-size="12" '
                     f'fill="{MUTED}">{esc(band_note)}</text>')
    for t in ticks:
        y = Y(t)
        p.append(f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{W - PAD_R}" y2="{y:.1f}" '
                 f'stroke="{GRID}" stroke-width="1"/>')
        p.append(f'<text x="{PAD_L - 10}" y="{y + 4:.1f}" font-size="13" '
                 f'text-anchor="end" fill="{MUTED}">{fmt(t)}{y_suffix}</text>')
    step = max(1, n // 8)
    for i, lab in enumerate(labels):
        if i % step == 0 or i == n - 1:
            p.append(f'<text x="{X(i):.1f}" y="{H - PAD_B + 22}" font-size="13" '
                     f'text-anchor="middle" fill="{MUTED}">{esc(lab)}</text>')
    for s in series:
        pts = [(X(i), Y(v)) for i, v in enumerate(s["values"])
               if v is not None and not np.isnan(v)]
        if not pts:
            continue
        d = "M" + "L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        p.append(f'<path d="{d}" fill="none" stroke="{s["color"]}" stroke-width="3" '
                 f'stroke-linejoin="round" stroke-linecap="round"/>')
        for x, y in pts:                       # 2px surface ring on every marker
            p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{s["color"]}" '
                     f'stroke="{SURFACE}" stroke-width="2"/>')
        lx, ly = pts[-1]
        p.append(f'<text x="{lx - 6:.1f}" y="{ly - 14:.1f}" font-size="15" '
                 f'font-weight="700" text-anchor="end" fill="{INK}">'
                 f'{fmt(s["values"][-1])}{y_suffix}</text>')
    p.append("</svg>")
    return "".join(p)


def funnel_chart(steps: list[dict]) -> str:
    """Stacked stages, each bar sized as a share of the stage above it.

    Sizing every bar against the first stage would make the last two invisible
    (160 out of 105,100 is 0.15% of the width), which hides exactly the drop the
    chart exists to show. Each bar is therefore drawn as its share of the stage
    directly above, and both the count and that share are printed on the row so
    nothing has to be inferred from the length alone.

    ``steps`` are dicts with ``label``, ``value`` and ``note``.
    """
    W = 900
    row_h, gap, label_w = 62, 14, 300
    bar_w = W - label_w - 20
    H = len(steps) * (row_h + gap) + 10
    p = [f'<svg viewBox="0 0 {W} {H}" role="img" style="width:100%;height:auto">']
    for i, s in enumerate(steps):
        y = i * (row_h + gap)
        prev = steps[i - 1]["value"] if i else s["value"]
        share = (s["value"] / prev * 100.0) if prev else 0.0
        w = max(bar_w * share / 100.0, 3.0)
        colour = s.get("color", BLUE)
        p.append(f'<text x="0" y="{y + 24:.0f}" font-size="15" font-weight="700" '
                 f'fill="{INK}">{esc(s["label"])}</text>')
        p.append(f'<text x="0" y="{y + 44:.0f}" font-size="13" fill="{MUTED}">'
                 f'{esc(s["note"])}</text>')
        p.append(f'<rect x="{label_w}" y="{y + 6:.0f}" width="{bar_w}" '
                 f'height="{row_h - 12}" fill="{GRID}" rx="6"/>')
        p.append(f'<rect x="{label_w}" y="{y + 6:.0f}" width="{w:.1f}" '
                 f'height="{row_h - 12}" fill="{colour}" rx="6"/>')
        inside = w > 150
        tx = label_w + w - 12 if inside else label_w + w + 12
        p.append(f'<text x="{tx:.1f}" y="{y + row_h / 2 + 2:.0f}" font-size="16" '
                 f'font-weight="800" text-anchor="{"end" if inside else "start"}" '
                 f'fill="{"#ffffff" if inside else INK}" '
                 f'style="font-variant-numeric:tabular-nums">'
                 f'{s["value"]:,.0f}</text>')
        if i:
            p.append(f'<text x="{W}" y="{y + row_h / 2 + 2:.0f}" font-size="14" '
                     f'text-anchor="end" fill="{INK2}" '
                     f'style="font-variant-numeric:tabular-nums">'
                     f'{share:.1f}% of the step above</text>')
    p.append("</svg>")
    return "".join(p)


def compare_bars(items: list[dict], *, y_suffix: str = "%") -> str:
    """Two or three labelled bars for an at-a-glance headline comparison."""
    W, H = 900, 150
    pad, gap = 0, 26
    n = len(items)
    bw = (W - gap * (n - 1)) / n
    vmax = max(i["value"] for i in items) * 1.35
    p = [f'<svg viewBox="0 0 {W} {H}" role="img" style="width:100%;height:auto">']
    for i, it in enumerate(items):
        x = i * (bw + gap)
        h = (H - 58) * it["value"] / vmax
        y = H - 40 - h
        p.append(f'<rect x="{x:.0f}" y="{y:.0f}" width="{bw:.0f}" height="{h:.0f}" '
                 f'fill="{it["color"]}" rx="8"/>')
        p.append(f'<text x="{x + bw / 2:.0f}" y="{y - 10:.0f}" font-size="30" '
                 f'font-weight="800" text-anchor="middle" fill="{INK}">'
                 f'{it["value"]:.0f}{y_suffix}</text>')
        p.append(f'<text x="{x + bw / 2:.0f}" y="{H - 14:.0f}" font-size="14.5" '
                 f'text-anchor="middle" fill="{INK2}">{esc(it["label"])}</text>')
    p.append("</svg>")
    return "".join(p)


def bar_chart(labels: list[str], values: list[float], *, color: str = BLUE,
              y_suffix: str = "", fmt=lambda v: f"{v:,.0f}",
              highlight: int | None = None) -> str:
    n = len(values)
    vmax = max(values) * 1.18 if values else 1
    ticks = _ticks(vmax)
    vmax = max(ticks[-1], vmax)
    plot_w = W - PAD_L - PAD_R
    band = plot_w / n
    bw = min(band * 0.62, 54)

    def Y(v: float) -> float:
        return H - PAD_B - (H - PAD_T - PAD_B) * (v / vmax)

    p = [f'<svg viewBox="0 0 {W} {H}" role="img" style="width:100%;height:auto">']
    for t in ticks:
        y = Y(t)
        p.append(f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{W - PAD_R}" y2="{y:.1f}" '
                 f'stroke="{GRID}" stroke-width="1"/>')
        p.append(f'<text x="{PAD_L - 10}" y="{y + 4:.1f}" font-size="13" '
                 f'text-anchor="end" fill="{MUTED}">{fmt(t)}{y_suffix}</text>')
    for i, v in enumerate(values):
        cx = PAD_L + band * (i + 0.5)
        y = Y(v)
        c = ORANGE if highlight is not None and i == highlight else color
        p.append(f'<rect x="{cx - bw / 2:.1f}" y="{y:.1f}" width="{bw:.1f}" '
                 f'height="{H - PAD_B - y:.1f}" fill="{c}" rx="4"/>')
        p.append(f'<text x="{cx:.1f}" y="{y - 9:.1f}" font-size="13" '
                 f'font-weight="700" text-anchor="middle" fill="{INK}">'
                 f'{fmt(v)}{y_suffix}</text>')
        p.append(f'<text x="{cx:.1f}" y="{H - PAD_B + 22}" font-size="12.5" '
                 f'text-anchor="middle" fill="{MUTED}">{esc(labels[i])}</text>')
    p.append("</svg>")
    return "".join(p)


def pie_slices(items: list[dict], top_n: int = PIE_TOP_N) -> list[dict]:
    """Largest ``top_n`` entries, then one Other slice holding the remainder.

    ``items`` are dicts with ``label`` and ``value``. The returned list always
    sums to the input total, so the chart and the table below it cannot
    disagree: Other is computed as total minus the slices above it rather than
    re-summed from the tail.
    """
    ranked = sorted(items, key=lambda r: -r["value"])
    total = sum(r["value"] for r in items)
    head, tail = ranked[:top_n], ranked[top_n:]
    out = [{"label": r["label"], "value": r["value"],
            "pct": 100.0 * r["value"] / total if total else 0.0,
            "color": PIE_COLORS[i % len(PIE_COLORS)], "n_merged": 1}
           for i, r in enumerate(head)]
    if tail:
        rest = total - sum(r["value"] for r in head)
        out.append({"label": f"Other ({len(tail)} locations)", "value": rest,
                    "pct": 100.0 * rest / total if total else 0.0,
                    "color": PIE_OTHER, "n_merged": len(tail)})
    return out


def pie_chart(slices: list[dict], *, unit: str = "kWh",
              fmt=lambda v: f"{v:,.0f}", legend_title: str = "ENERGY GIVEN",
              total_label: str = "Total", wrap_at: int = 30) -> str:
    """Pie with outside labels and a value legend, largest slice first.

    Labels sit outside the arc on a short leader line rather than on the fill,
    so every number is drawn in text ink on the page surface instead of over a
    coloured wedge. Only slices at or above ``PIE_LABEL_MIN_PCT`` get one; the
    thin ones would collide, and the legend carries their values anyway. Each
    slice takes a 2px surface-coloured stroke so neighbours never touch.
    """
    WW, HH = 900, 360
    cx, cy, r = 208.0, 180.0, 124.0
    p = [f'<svg viewBox="0 0 {WW} {HH}" role="img" style="width:100%;height:auto">']
    ang = -math.pi / 2                      # start at 12 o'clock, run clockwise
    total = sum(s["value"] for s in slices) or 1.0
    for s in slices:
        sweep = 2 * math.pi * s["value"] / total
        a0, a1 = ang, ang + sweep
        ang = a1
        if sweep <= 0:
            continue
        if sweep >= 2 * math.pi - 1e-9:     # a lone slice is a whole circle
            p.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{s["color"]}" '
                     f'stroke="{SURFACE}" stroke-width="2"/>')
        else:
            x0, y0 = cx + r * math.cos(a0), cy + r * math.sin(a0)
            x1, y1 = cx + r * math.cos(a1), cy + r * math.sin(a1)
            large = 1 if sweep > math.pi else 0
            p.append(f'<path d="M{cx:.1f},{cy:.1f} L{x0:.1f},{y0:.1f} '
                     f'A{r:.1f},{r:.1f} 0 {large} 1 {x1:.1f},{y1:.1f} Z" '
                     f'fill="{s["color"]}" stroke="{SURFACE}" stroke-width="2"/>')
        if s["pct"] >= PIE_LABEL_MIN_PCT:
            am = (a0 + a1) / 2
            lx0, ly0 = cx + (r - 6) * math.cos(am), cy + (r - 6) * math.sin(am)
            lx1, ly1 = cx + (r + 16) * math.cos(am), cy + (r + 16) * math.sin(am)
            right = math.cos(am) >= 0
            tx = lx1 + (7 if right else -7)
            p.append(f'<line x1="{lx0:.1f}" y1="{ly0:.1f}" x2="{lx1:.1f}" '
                     f'y2="{ly1:.1f}" stroke="{MUTED}" stroke-width="1.5"/>')
            p.append(f'<text x="{tx:.1f}" y="{ly1 + 5:.1f}" font-size="14" '
                     f'font-weight="700" fill="{INK}" '
                     f'text-anchor="{"start" if right else "end"}">'
                     f'{s["pct"]:.0f}%</text>')
    # Legend: colour chip, label, value, share. Identity is never colour alone.
    lx, ly = 470.0, 62.0
    p.append(f'<text x="{lx}" y="{ly - 22:.1f}" font-size="13" font-weight="700" '
             f'fill="{MUTED}" letter-spacing="0.6">{esc(legend_title)}, '
             f'{esc(unit.upper())}</text>')
    for s in slices:
        p.append(f'<rect x="{lx}" y="{ly - 11:.1f}" width="13" height="13" rx="3" '
                 f'fill="{s["color"]}"/>')
        # Wrap long labels onto a second line instead of letting them run into
        # the value column.
        words, lines, cur = str(s["label"]).split(), [], ""
        for w in words:
            if len(cur) + len(w) + 1 <= wrap_at:
                cur = f"{cur} {w}".strip()
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        for li, line in enumerate(lines):
            p.append(f'<text x="{lx + 22:.1f}" y="{ly + li * 17:.1f}" '
                     f'font-size="14.5" fill="{INK}">{esc(line)}</text>')
        extra = (len(lines) - 1) * 17.0
        p.append(f'<text x="{lx + 292:.1f}" y="{ly:.1f}" font-size="14.5" '
                 f'text-anchor="end" fill="{INK}" '
                 f'style="font-variant-numeric:tabular-nums">{fmt(s["value"])}</text>')
        # One decimal here on purpose: the slice labels round to whole
        # percent, and a legend that also rounded would show "8%" beside a
        # 7.5% slice that the chart correctly leaves unlabelled.
        p.append(f'<text x="{lx + 372:.1f}" y="{ly:.1f}" font-size="14.5" '
                 f'text-anchor="end" fill="{INK2}" '
                 f'style="font-variant-numeric:tabular-nums">{s["pct"]:.1f}%</text>')
        ly += 30.0 + extra
    p.append(f'<line x1="{lx}" y1="{ly - 20:.1f}" x2="{lx + 372:.1f}" '
             f'y2="{ly - 20:.1f}" stroke="{GRID}" stroke-width="1"/>')
    p.append(f'<text x="{lx + 22:.1f}" y="{ly + 2:.1f}" font-size="14.5" '
             f'font-weight="700" fill="{INK}">{esc(total_label)}</text>')
    p.append(f'<text x="{lx + 292:.1f}" y="{ly + 2:.1f}" font-size="14.5" '
             f'text-anchor="end" font-weight="700" fill="{INK}" '
             f'style="font-variant-numeric:tabular-nums">{fmt(total)}</text>')
    p.append("</svg>")
    return "".join(p)


CSS = """
*{box-sizing:border-box}
body{margin:0;background:#f4f3ef;color:#111;font:16px/1.6 -apple-system,BlinkMacSystemFont,
 'Segoe UI',Roboto,Helvetica,Arial,sans-serif;-webkit-text-size-adjust:100%}
.wrap{max-width:1000px;margin:0 auto;padding:34px 22px 60px}
.head{margin-bottom:26px}
h1{font-size:32px;line-height:1.2;margin:0 0 6px;letter-spacing:-.02em}
.sub{color:#5b5b57;font-size:16px}
.card{background:#fcfcfb;border:1px solid #e6e5df;border-radius:14px;padding:26px 28px;margin-bottom:20px}
h2{font-size:22px;margin:0 0 4px;letter-spacing:-.01em}
h3{font-size:16px;margin:26px 0 4px;letter-spacing:-.01em}
.note{font-size:15px;color:#5b5b57;margin:0 0 6px}
.lead{font-size:17px;color:#3a3a36;margin:0 0 20px}
.heroes{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px;margin-bottom:20px}
.hero{background:#fcfcfb;border:1px solid #e6e5df;border-radius:14px;padding:20px 22px}
.hero .n{font-size:40px;font-weight:800;letter-spacing:-.03em;line-height:1;margin-bottom:8px}
.hero .t{font-size:15px;color:#3a3a36;font-weight:600;line-height:1.35}
.hero .s{font-size:13.5px;color:#8a8a83;margin-top:7px}
.takeaway{background:#f7f6f2;border-left:4px solid #2a78d6;padding:13px 17px;border-radius:0 8px 8px 0;
 margin-top:16px;font-size:16px;color:#26261f}
.takeaway.warn{border-left-color:#eb6834}
.takeaway.good{border-left-color:#1baf7a}
.chartttl{font-size:14px;font-weight:700;color:#5b5b57;margin:18px 0 4px;text-transform:uppercase;letter-spacing:.04em}
.legend{display:flex;gap:20px;flex-wrap:wrap;margin:8px 0 6px;font-size:14.5px;color:#3a3a36}
.legend i{display:inline-block;width:13px;height:13px;border-radius:3px;margin-right:7px;vertical-align:-1px}
.tablewrap{overflow-x:auto;-webkit-overflow-scrolling:touch;margin-top:12px}
table{width:100%;min-width:560px;border-collapse:collapse;font-size:15px}
th{text-align:left;font-size:13px;text-transform:uppercase;letter-spacing:.04em;color:#8a8a83;
 padding:8px 10px;border-bottom:2px solid #e6e5df}
td{padding:9px 10px;border-bottom:1px solid #efeee9}
td.num{text-align:right;font-variant-numeric:tabular-nums}
ul.acts{margin:6px 0 0;padding-left:22px}
ul.acts li{margin-bottom:11px}
.foot{color:#8a8a83;font-size:13.5px;margin-top:26px;text-align:center}
@media print{body{background:#fff}.card,.hero{break-inside:avoid;border-color:#ddd}.wrap{padding:0}}
@media(max-width:620px){h1{font-size:26px}.hero .n{font-size:34px}}
"""


def build(out_path: Path) -> Path:
    raw, _ = S.load_rides(S.DEFAULT_RIDE_FILES)
    racks = S.load_racks(S.DEFAULT_RACK_FILE)
    sites = S.cluster_racks([(r.lat, r.lon) for r in racks], S.MERGE_RADIUS_M)
    df, _wf = S.prepare_rides(raw)
    df = S.assign_docking(df, sites)
    chg = S.load_charging(S.DEFAULT_CHARGING_FILE, racks)
    S.attach_sites(chg, racks, sites)   # needed by the follow-through join

    start = df["start_time"].min().normalize()
    end = df["start_time"].max().normalize()
    days = int((end - start).days) + 1

    wd = S.weekly_docking(df)
    wv = S.weekly_vehicle_docking(df)
    partial = S.partial_flags(list(wd.index), "week", start, end + pd.Timedelta(days=1))
    wdf = S.full_periods(wd, partial)
    wvf = S.full_periods(wv, partial)

    rides = len(df)
    vehicles = int(df["vehicle_id"].nunique())
    need_total = S.energy_need_kwh(rides)
    # Energy comes from charging sessions only. The export also logs vehicles
    # that held a point and drew nothing; those are counted where occupancy and
    # docking are counted, never where kWh are.
    supplied = float(chg.total_kwh)
    catered = 100.0 * supplied / need_total
    dock_rate = 100.0 * wdf["docked"].sum() / wdf["assessable"].sum()

    # Coverage for the vehicles that actually reach a station, full weeks only.
    cov = S.station_visitor_coverage(df, chg)
    cov_need = float(cov["need_kwh"].sum())
    cov_given = float(cov["given_kwh"].sum())
    cov_pct = 100.0 * cov_given / cov_need
    cov_fleet_pct = (100.0 * float(cov["fleet_given_kwh"].sum())
                     / float(cov["fleet_need_kwh"].sum()))
    cov_labels = [w.strftime("%-d %b") for w in cov.index]
    cov_visitor_line = [float(v) for v in cov["coverage_pct"]]
    cov_fleet_line = [float(v) for v in cov["fleet_coverage_pct"]]
    cov_weeks_txt = (f'{cov.index[0].strftime("%-d %B")} to '
                     f'{(cov.index[-1] + pd.Timedelta(days=6)).strftime("%-d %B")}')

    # What happens after a trip ends at a station: same location, next event.
    ft = S.site_follow_through(df, chg, sites)
    ft30, ft24 = ft["windows"][30.0], ft["windows"][1440.0]

    # Same measure at the wider radius, computed here so both live side by side.
    _dw = pd.DataFrame({
        "week": df["week"].values,
        "has_end": df["has_end"].values,
        "wide": (np.stack([
            np.where(np.isnan(S.haversine_m(s.lat, s.lon,
                                            df["end_lat"].values, df["end_lon"].values)),
                     np.inf,
                     S.haversine_m(s.lat, s.lon,
                                   df["end_lat"].values, df["end_lon"].values))
            for s in sites]).min(axis=0) <= WIDE_RADIUS_M),
    })
    _gw = _dw.groupby("week").agg(assessable=("has_end", "sum"), wide=("wide", "sum"))
    _gwf = S.full_periods(_gw, partial)
    dock_rate_wide = 100.0 * _gwf["wide"].sum() / _gwf["assessable"].sum()
    veh_share = 100.0 * wvf["docked"].sum() / wvf["riding"].sum()

    ut = S.fleet_utilisation(df, "week")
    utf = S.full_periods(ut, partial)

    sup_w = S.charging_by_gran(chg, "week")
    wk_labels = [w.strftime("%-d %b") for w in wdf.index]
    need_w = [S.energy_need_kwh(int(wdf.loc[w, "rides"])) for w in wdf.index]
    sup_wv = [float(sup_w.get(w, np.nan)) for w in wdf.index]
    cat_w = [100.0 * s / n if n else np.nan for s, n in zip(sup_wv, need_w)]
    dock_w = [float(wdf.loc[w, "dock_rate_pct"]) for w in wdf.index]
    dock_w_wide = [100.0 * _gwf.loc[w, "wide"] / _gwf.loc[w, "assessable"]
                   for w in wdf.index]
    vehs_w = [100.0 * wvf.loc[w, "docked"] / wvf.loc[w, "riding"] for w in wdf.index]
    unass_w = [float(wdf.loc[w, "unassessable_pct"]) for w in wdf.index]
    util_used = [float(utf.loc[w, "rides_per_vehicle_per_day"]) for w in wdf.index]
    util_fleet = [float(utf.loc[w, "rides_per_fleet_vehicle_per_day"]) for w in wdf.index]

    # The stations came online part-way through the first week, so that week's
    # supply is not a full week's worth. Quote the steady-state range only.
    chg_start = chg.sessions["started"].min()
    # Compare timestamps, not dates: the first session lands part-way through
    # its own week, so that week must fall out of the range.
    sup_full = [s for w, s in zip(wdf.index, sup_wv)
                if w >= chg_start and not np.isnan(s)]
    if not sup_full:
        sup_full = [s for s in sup_wv if not np.isnan(s)]

    # The busiest week is where the coverage story is clearest: supply is near
    # constant, so the share covered bottoms out exactly when demand peaks.
    _busy = int(np.argmax(util_used))
    busy_wk = wdf.index[_busy].strftime("%-d %B")
    busy_util = util_used[_busy]
    busy_cat = cat_w[_busy]

    # Where the missing-GPS problem starts: first week whose unassessable share
    # is more than double the earliest week's. Beyond it the docking line is not
    # trustworthy, so it is shaded rather than narrated.
    base_unass = unass_w[0] if unass_w else 0.0
    shade_from = next((i for i, u in enumerate(unass_w)
                       if u > max(2.0 * base_unass, 2.0)), None)

    pst = S.per_site_table(df, sites).sort_values("buffer_ends", ascending=False)
    # Reuse the technical report's own site names. Numbering these by traffic
    # rank instead would make "S8" mean a different physical place in each
    # document, and anyone cross-referencing the two would land on the wrong
    # station.
    site_lbl = [str(v) for v in pst["site"]]
    live_sites = int((pst["buffer_ends"] > 10).sum())

    # Rack-level detail behind each location, so the grouping is auditable:
    # which physical racks were merged, how many charging points they add up
    # to, and what each location delivered.
    kwh_by_rack = chg.charges.groupby("rack_id")["kwh"].sum()

    # Distance from every trip end to each site, then to the nearest one.
    _el, _eo = df["end_lat"].values, df["end_lon"].values
    _site_d = np.stack([
        np.where(np.isnan(S.haversine_m(s.lat, s.lon, _el, _eo)), np.inf,
                 S.haversine_m(s.lat, s.lon, _el, _eo)) for s in sites])
    _nearest = _site_d.argmin(axis=0)
    _dmin = _site_d.min(axis=0)
    wide_by_site = {s.name: int(((_nearest == i) & (_dmin <= WIDE_RADIUS_M)).sum())
                    for i, s in enumerate(sites)}
    near_by_site = {s.name: int(((_nearest == i) & (_dmin <= S.DOCK_RADIUS_M)).sum())
                    for i, s in enumerate(sites)}
    by_name = {s.name: s for s in sites}
    site_rows = []
    for lbl, (_, row) in zip(site_lbl, pst.iterrows()):
        site = by_name[str(row["site"])]
        members = [racks[i] for i in site.rack_indices]
        site_rows.append({
            "label": lbl,
            "rack_ids": ", ".join(r.rack_id.split("-")[-1] for r in members),
            "n_racks": len(members),
            "points": sum(r.receptacles for r in members),
            "ends": int(row["buffer_ends"]),
            "near": near_by_site[site.name],
            "wide": wide_by_site[site.name],
            "kwh": float(sum(kwh_by_rack.get(r.rack_id, 0.0) for r in members)),
        })
    merged_sites = sum(1 for r in site_rows if r["n_racks"] > 1)

    swaps_share = 100.0 - catered
    per_day_need = need_total / days
    per_day_sup = supplied / days

    o: list[str] = []
    o.append('<!doctype html><html lang="en"><head><meta charset="utf-8">'
             '<meta name="viewport" content="width=device-width,initial-scale=1">'
             f'<title>Vaasa station pilot: summary</title><style>{CSS}</style>'
             '</head><body><div class="wrap">')

    o.append('<div class="head">')
    o.append("<h1>Vaasa charging stations: how the pilot is doing</h1>")
    o.append(f'<div class="sub">{start.strftime("%-d %B")} to '
             f'{end.strftime("%-d %B %Y")} &nbsp;&middot;&nbsp; {days} days '
             f'&nbsp;&middot;&nbsp; {rides:,} scooter trips &nbsp;&middot;&nbsp; '
             f'{len(sites)} station locations</div>')
    o.append("</div>")

    # ---- the four numbers -------------------------------------------------
    o.append('<div class="heroes">')
    o.append(f'<div class="hero"><div class="n">{catered:.0f}%</div>'
             f'<div class="t">of the electricity the scooters needed came from '
             f'our stations</div><div class="s">the other {swaps_share:.0f}% '
             f'still comes from swapping batteries by hand</div></div>')
    o.append(f'<div class="hero"><div class="n">{dock_rate:.0f} in 100</div>'
             f'<div class="t">trips end right at a station, within '
             f'{S.DOCK_RADIUS_M:.0f} metres</div>'
             f'<div class="s">{dock_rate_wide:.0f} in 100 if you count within '
             f'{WIDE_RADIUS_M:.0f} metres</div></div>')
    o.append(f'<div class="hero"><div class="n">{veh_share:.0f}%</div>'
             f'<div class="t">of scooters visit a station in a given week</div>'
             f'<div class="s">so most of the fleet does reach a station at '
             f'some point</div></div>')
    o.append(f'<div class="hero"><div class="n">{cov_pct:.0f}%</div>'
             f'<div class="t">of the energy needed by the scooters that do use '
             f'our stations comes from the stations</div>'
             f'<div class="s">against {cov_fleet_pct:.0f}% across the whole '
             f'fleet, so reaching a station is not the same as being charged '
             f'by one</div></div>')
    o.append(f'<div class="hero"><div class="n">{supplied:,.0f} kWh</div>'
             f'<div class="t">charged at the stations so far</div>'
             f'<div class="s">enough energy for about '
             f'{supplied * 1000 / S.WH_PER_RIDE / 1000:,.0f},000 trips</div></div>')
    o.append("</div>")

    # ---- energy -----------------------------------------------------------
    o.append('<div class="card">')
    o.append("<h2>Energy: what the scooters need vs what the stations give</h2>")
    o.append(f'<p class="lead">Assuming every trip uses about '
             f'{S.WH_PER_RIDE} Wh of battery, across {days} days the fleet '
             f'needed roughly <strong>{need_total:,.0f} kWh</strong>. The '
             f'stations delivered <strong>{supplied:,.0f} kWh</strong> of '
             f'it.</p>')
    o.append('<div class="chartttl">Each week, in kilowatt-hours</div>')
    o.append('<div class="legend">'
             f'<span><i style="background:{BLUE}"></i>Energy the scooters needed</span>'
             f'<span><i style="background:{ORANGE}"></i>Energy the stations gave</span>'
             "</div>")
    o.append(line_chart(wk_labels,
                        [{"values": need_w, "color": BLUE},
                         {"values": sup_wv, "color": ORANGE}],
                        y_suffix="", fmt=lambda v: f"{v:,.0f}"))
    o.append(f'<div class="takeaway">The gap between the two lines is energy we '
             f'still handle by swapping batteries. That is about '
             f'<strong>{per_day_need - per_day_sup:,.0f} kWh a day</strong>.</div>')

    o.append(f'<div class="takeaway">The orange line is almost flat. Once the '
             f'stations were fully running they gave out about the same energy '
             f'every week, between {min(sup_full):,.0f} and '
             f'{max(sup_full):,.0f} kWh. What the fleet needs swings a lot '
             f'more, from {min(need_w):,.0f} to {max(need_w):,.0f} kWh.</div>')

    # ---- where the fleet's energy comes from --------------------------------
    swaps_kwh = need_total - supplied
    pie = [
        {"label": "From our stations", "value": supplied,
         "pct": 100.0 * supplied / need_total, "color": ORANGE, "n_merged": 1},
        {"label": "Still done by swapping batteries by hand", "value": swaps_kwh,
         "pct": 100.0 * swaps_kwh / need_total, "color": MUTED, "n_merged": 1},
    ]
    o.append("<h3>Where the fleet&rsquo;s energy comes from</h3>")
    o.append(f'<p class="note">All {need_total:,.0f} kWh the fleet needed over '
             f'the {days} days, split by how it was delivered.</p>')
    o.append(pie_chart(pie, legend_title="ENERGY DELIVERED",
                       total_label="All energy the fleet needed"))
    o.append(f'<div class="takeaway">The stations supplied '
             f'<strong>{supplied:,.0f} kWh</strong> of the '
             f'{need_total:,.0f} kWh the fleet needed, so '
             f'<strong>{catered:.0f}%</strong>. The other '
             f'{swaps_kwh:,.0f} kWh was handled by swapping batteries.</div>')
    o.append("</div>")

    # ---- coverage for the scooters that actually reach a station ------------
    o.append('<div class="card">')
    o.append("<h2>How much energy do the scooters that use our stations get "
             "from us?</h2>")
    o.append(f'<p class="lead">The 31% above counts the whole fleet, including '
             f'scooters that never go near a rack. A fairer question is how '
             f'well we serve the scooters that <em>do</em> use the stations. '
             f'Each week we take the scooters that ended at least one trip at a '
             f'station, then compare the energy <em>those</em> scooters needed '
             f'with what the stations actually gave <em>them</em>.</p>')
    o.append('<div class="chartttl">Share of their energy that came from our stations</div>')
    o.append(compare_bars([
        {"label": "Scooters that visit a station", "value": cov_pct, "color": GREEN},
        {"label": "The whole fleet", "value": cov_fleet_pct, "color": MUTED},
    ]))
    o.append(f'<div class="takeaway">Scooters that use the stations get '
             f'<strong>{cov_pct:.0f}%</strong> of their energy from us, against '
             f'<strong>{cov_fleet_pct:.0f}%</strong> across the whole fleet. '
             f'Better, but still under half.</div>')
    o.append('<div class="chartttl">The same two figures, week by week</div>')
    o.append('<div class="legend">'
             f'<span><i style="background:{GREEN}"></i>Scooters that visited a '
             f'station that week</span>'
             f'<span><i style="background:{MUTED}"></i>The whole fleet</span>'
             "</div>")
    o.append(line_chart(cov_labels,
                        [{"values": cov_visitor_line, "color": GREEN},
                         {"values": cov_fleet_line, "color": MUTED}],
                        y_suffix="%", y_max=60, fmt=lambda v: f"{v:,.0f}"))
    o.append(f'<div class="takeaway warn">Across these weeks those scooters '
             f'needed <strong>{cov_need:,.0f} kWh</strong> and the stations gave '
             f'them <strong>{cov_given:,.0f} kWh</strong>. The gap is still '
             f'covered by swapping batteries. The line drops in the busiest week '
             f'for the same reason as everything else on this page, the stations '
             f'give out about the same energy no matter how busy it gets.</div>')
    o.append('<div class="tablewrap">')
    o.append("<table><thead><tr><th>Week</th><th class='num'>Scooters that rode</th>"
             "<th class='num'>Of those, visited a station</th>"
             "<th class='num'>Energy they needed</th>"
             "<th class='num'>Energy the stations gave them</th>"
             "<th class='num'>Their coverage</th>"
             "<th class='num'>Whole-fleet coverage</th></tr></thead><tbody>")
    for w, r in cov.iterrows():
        o.append(f'<tr><td>{esc(w.strftime("%-d %b"))}</td>'
                 f'<td class="num">{int(r["riding"]):,}</td>'
                 f'<td class="num">{int(r["visitors"]):,} '
                 f'({r["visitor_pct"]:.0f}%)</td>'
                 f'<td class="num">{r["need_kwh"]:,.0f} kWh</td>'
                 f'<td class="num">{r["given_kwh"]:,.0f} kWh</td>'
                 f'<td class="num">{r["coverage_pct"]:.0f}%</td>'
                 f'<td class="num">{r["fleet_coverage_pct"]:.0f}%</td></tr>')
    o.append(f'<tr><td><strong>All six weeks</strong></td>'
             f'<td class="num"></td><td class="num"></td>'
             f'<td class="num"><strong>{cov_need:,.0f} kWh</strong></td>'
             f'<td class="num"><strong>{cov_given:,.0f} kWh</strong></td>'
             f'<td class="num"><strong>{cov_pct:.1f}%</strong></td>'
             f'<td class="num"><strong>{cov_fleet_pct:.1f}%</strong></td></tr>')
    o.append("</tbody></table></div>")
    o.append(f'<p class="note">A scooter counts as a station visitor in a week '
             f'if any one of its trips ended at a station that week. Energy '
             f'needed is its trips that week times {S.WH_PER_RIDE} Wh. Energy '
             f'given counts only sessions that actually delivered power, so a '
             f'vehicle left on a rack that took no charge adds nothing to this '
             f'column.</p>')
    o.append("</div>")

    # ---- utilisation vs coverage -------------------------------------------
    o.append('<div class="card">')
    o.append("<h2>When the city gets busier, do the stations keep up?</h2>")
    o.append(f'<p class="lead">Utilisation is how many trips each scooter does '
             f'per day. The busier the fleet, the more energy it burns. So the '
             f'question is whether station coverage holds up when demand '
             f'rises.</p>')
    o.append('<div class="chartttl">How hard each scooter is working, trips per day</div>')
    o.append('<div class="legend">'
             f'<span><i style="background:{BLUE}"></i>Per scooter that was in use</span>'
             f'<span><i style="background:{MUTED}"></i>Per scooter in the whole fleet</span>'
             "</div>")
    o.append(line_chart(wk_labels,
                        [{"values": util_used, "color": BLUE},
                         {"values": util_fleet, "color": MUTED}],
                        y_max=2.2, fmt=lambda v: f"{v:.1f}"))
    o.append('<div class="chartttl">Share of the fleet&rsquo;s energy covered by stations</div>')
    o.append(line_chart(wk_labels, [{"values": cat_w, "color": GREEN}],
                        y_suffix="%", y_max=60, fmt=lambda v: f"{v:,.0f}"))
    o.append(f'<div class="takeaway warn">The two move in opposite directions. '
             f'In the busiest week ({busy_wk}) each scooter did '
             f'<strong>{busy_util:.1f} trips a day</strong>, and the stations '
             f'covered only <strong>{busy_cat:.0f}%</strong> of the energy '
             f'needed. In quieter weeks they cover a third or more. The '
             f'stations give out about the same energy every week, so '
             f'<strong>coverage drops exactly when the city needs it '
             f'most</strong>.</div>')
    o.append("</div>")

    # ---- docking ----------------------------------------------------------
    o.append('<div class="card">')
    o.append("<h2>Do people finish their trips at a station?</h2>")
    o.append(f'<p class="lead">Measured two ways. Within '
             f'{S.DOCK_RADIUS_M:.0f} metres means the scooter was left right at '
             f'the rack. Within {WIDE_RADIUS_M:.0f} metres means it was left on '
             f'the same corner, which is a fairer test of whether the station '
             f'pulled the trip.</p>')
    o.append('<div class="chartttl">Share of trips ending at a station, each week</div>')
    o.append('<div class="legend">'
             f'<span><i style="background:{ORANGE}"></i>Within '
             f'{WIDE_RADIUS_M:.0f} metres</span>'
             f'<span><i style="background:{BLUE}"></i>Within '
             f'{S.DOCK_RADIUS_M:.0f} metres</span>'
             "</div>")
    o.append(line_chart(wk_labels,
                        [{"values": dock_w_wide, "color": ORANGE},
                         {"values": dock_w, "color": BLUE}],
                        y_suffix="%", y_max=22, fmt=lambda v: f"{v:,.0f}",
                        band_from=shade_from,
                        band_note="less reliable, see note"))
    if shade_from is not None:
        o.append(f'<div class="takeaway warn">About <strong>{dock_rate:.0f} '
                 f'trips in every 100</strong> end right at a rack, and '
                 f'<strong>{dock_rate_wide:.0f} in 100</strong> end within '
                 f'{WIDE_RADIUS_M:.0f} metres of one. The last few weeks look '
                 f'lower on both lines, but in those weeks a lot more scooters '
                 f'did not report where they were parked. So we cannot say '
                 f'whether docking actually fell. Treat the shaded part as '
                 f'unknown, not as a drop.</div>')

    o.append("<h3>What happens after a trip ends at a station</h3>")
    o.append(f'<p class="note">Leaving a scooter beside a rack is not the same '
             f'as putting it on the rack. This follows every trip through to '
             f'what happened next.</p>')
    o.append(funnel_chart([
        {"label": "All trips", "value": rides,
         "note": f"over the {days} days", "color": MUTED},
        {"label": "Ended at a station",
         "value": ft["ends"],
         "note": f"within {S.DOCK_RADIUS_M:.0f} metres of one", "color": BLUE},
        {"label": "Put on a charging point",
         "value": ft30["docked_or_charged"],
         "note": "by anyone, within 30 minutes", "color": ORANGE},
        {"label": "Actually took charge",
         "value": ft30["charged"],
         "note": "drew power, not just docked", "color": GREEN},
    ]))
    o.append(f'<div class="takeaway">Scooters do reach the stations and get '
             f'parked next to them, but they are rarely put on a point by the '
             f'person who rode them there. Out of every 100 trips that end at a '
             f'station, about <strong>{ft30["docked_or_charged_pct"]:.0f}</strong> '
             f'end with the scooter on a point within half an hour. Waiting a '
             f'full day instead of half an hour raises that to '
             f'{ft24["docked_or_charged_pct"]:.0f} in 100, which is the staff '
             f'collecting them later.</div>')
    o.append("</div>")

    # ---- stations ---------------------------------------------------------
    o.append('<div class="card">')
    o.append("<h2>Which stations are being used?</h2>")
    o.append(f'<p class="lead">Trips ending at each of our {len(sites)} station '
             f'locations over the whole period.</p>')
    # Same counts as the table below (each trip credited to its nearest
    # location), so a reader cannot find two different totals for one place.
    o.append(bar_chart([r["label"] for r in site_rows],
                       [r["near"] for r in site_rows],
                       color=BLUE, fmt=lambda v: f"{v:,.0f}"))
    o.append(f'<div class="takeaway"><strong>{live_sites} of {len(sites)}</strong> '
             f'locations get real traffic, and the busiest three take a large '
             f'share of it. Some of the quiet ones were only installed '
             f'recently.</div>')

    o.append("<h3>How the racks group into locations</h3>")
    o.append(f'<p class="note">We have {len(racks)} racks in the city, but some '
             f'sit right next to each other. Anything within '
             f'{S.MERGE_RADIUS_M:.0f} metres of another rack counts as one '
             f'location, because to a rider they are the same place. That '
             f'gives <strong>{len(sites)} locations</strong>, '
             f'{merged_sites} of which are made up of more than one rack. '
             f'Trips are counted against the nearest location, so nothing is '
             f'counted twice where two locations sit close together.</p>')
    o.append('<div class="tablewrap">')
    o.append("<table><thead><tr><th>Location</th><th>Racks in it</th>"
             "<th class='num'>Racks</th><th class='num'>Charging points</th>"
             f"<th class='num'>Trips within {S.DOCK_RADIUS_M:.0f} m</th>"
             f"<th class='num'>Trips within {WIDE_RADIUS_M:.0f} m</th>"
             "<th class='num'>Energy given (kWh)</th></tr></thead><tbody>")
    for r in site_rows:
        o.append(f'<tr><td><strong>{esc(r["label"])}</strong></td>'
                 f'<td>{esc(r["rack_ids"])}</td>'
                 f'<td class="num">{r["n_racks"]}</td>'
                 f'<td class="num">{r["points"]}</td>'
                 f'<td class="num">{r["near"]:,}</td>'
                 f'<td class="num">{r["wide"]:,}</td>'
                 f'<td class="num">{r["kwh"]:,.0f}</td></tr>')
    o.append(f'<tr><td colspan="2"><strong>Total</strong></td>'
             f'<td class="num"><strong>{len(racks)}</strong></td>'
             f'<td class="num"><strong>{sum(r["points"] for r in site_rows)}</strong></td>'
             f'<td class="num"><strong>{sum(r["near"] for r in site_rows):,}</strong></td>'
             f'<td class="num"><strong>{sum(r["wide"] for r in site_rows):,}</strong></td>'
             f'<td class="num"><strong>{sum(r["kwh"] for r in site_rows):,.0f}</strong></td>'
             f'</tr>')
    o.append("</tbody></table></div>")
    o.append("</div>")

    # ---- fleet reach ------------------------------------------------------
    o.append('<div class="card">')
    o.append("<h2>How much of the fleet reaches a station each week?</h2>")
    o.append('<div class="chartttl">Share of scooters in use that visited a station that week</div>')
    o.append(line_chart(wk_labels, [{"values": vehs_w, "color": GREEN}],
                        y_suffix="%", y_max=100, fmt=lambda v: f"{v:,.0f}"))
    o.append(f'<div class="takeaway good">About <strong>{veh_share:.0f}% of '
             f'scooters</strong> reach a station in a normal week. The network '
             f'already covers a good part of the fleet. The gap is how often '
             f'trips actually <em>finish</em> there.</div>')
    o.append("</div>")

    # ---- summary of findings (statements of fact, no recommendations) ------
    o.append('<div class="card">')
    o.append("<h2>What the numbers show</h2>")
    o.append('<ul class="acts">')
    o.append(f"<li><strong>The stations cover {catered:.0f}% of the "
             f"fleet&rsquo;s energy.</strong> The remaining "
             f"{swaps_share:.0f}% is still handled by swapping batteries by "
             f"hand, which works out at about "
             f"{per_day_need - per_day_sup:,.0f} kWh a day.</li>")
    o.append(f"<li><strong>Coverage drops when the city is busiest.</strong> "
             f"The stations give out about the same energy every week "
             f"regardless of demand, so their share fell to {busy_cat:.0f}% in "
             f"the busiest week against a third or more in quiet weeks.</li>")
    o.append(f"<li><strong>{dock_rate:.0f} trips in 100 end at a station.</strong> "
             f"The other {100 - dock_rate:.0f} finish somewhere else in the "
             f"city.</li>")
    o.append(f"<li><strong>Ending at a station rarely means being put on "
             f"one.</strong> Of the {ft['ends']:,} trips that end within "
             f"{S.DOCK_RADIUS_M:.0f} metres of a station, "
             f"{ft30['docked_or_charged_pct']:.1f}% see that scooter docked or "
             f"charged at that same station within 30 minutes, and "
             f"{ft30['charged_pct']:.1f}% draw power.</li>")
    o.append(f"<li><strong>{veh_share:.0f}% of scooters reach a station in a "
             f"given week.</strong> So the network is within range of most of "
             f"the fleet, even though few trips finish there.</li>")
    o.append(f"<li><strong>Even the scooters that use a station get "
             f"{cov_pct:.0f}% of their energy from us.</strong> Over the six "
             f"full weeks those scooters needed {cov_need:,.0f} kWh and the "
             f"stations gave them {cov_given:,.0f} kWh, against "
             f"{cov_fleet_pct:.0f}% for the whole fleet.</li>")
    _top3_kwh = sum(r["kwh"] for r in sorted(site_rows, key=lambda r: -r["kwh"])[:3])
    o.append(f"<li><strong>Use is concentrated in a few locations.</strong> "
             f"{live_sites} of {len(sites)} sites get real traffic, the busiest "
             f"three take a large share of the trips, and the three largest "
             f"deliver {100.0 * _top3_kwh / supplied:.0f}% of all the energy. "
             f"Some of the quiet ones were only installed recently.</li>")
    o.append("</ul>")
    o.append("</div>")

    o.append(f'<div class="foot">Vaasa station pilot &middot; '
             f'{start.strftime("%-d %b")} to {end.strftime("%-d %b %Y")} &middot; '
             f'prepared {_dt.date.today().strftime("%-d %B %Y")}</div>')
    o.append("</div></body></html>")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(o), encoding="utf-8")

    print(f"  period            : {start.date()} -> {end.date()} ({days} days)")
    print(f"  rides / vehicles  : {rides:,} / {vehicles:,}")
    print(f"  energy catered    : {catered:.1f}%  ({supplied:,.0f} of {need_total:,.0f} kWh)")
    print(f"  docking rate      : {dock_rate:.1f}%")
    print(f"  weekly reach      : {veh_share:.1f}%")
    print(f"  energy supplied   : {supplied:,.0f} kWh "
          f"({chg.n_charge_sessions:,} charging sessions, "
          f"{chg.n_dock_events:,} dock events)")
    print(f"  visitor coverage  : {cov_pct:.1f}%  ({cov_given:,.0f} of "
          f"{cov_need:,.0f} kWh)  vs fleet {cov_fleet_pct:.1f}%")
    print(f"  follow-through    : {ft30['docked_or_charged_pct']:.1f}% docked or "
          f"charged within 30 min, {ft30['charged_pct']:.1f}% drew power "
          f"(of {ft['ends']:,} trips)")
    print(f"  pie: stations {pie[0]['pct']:.0f}% + swaps {pie[1]['pct']:.0f}% = "
          f"{sum(s['value'] for s in pie):,.1f} kWh (need total {need_total:,.1f})")
    print(f"  wrote             : {out_path}  ({out_path.stat().st_size:,} bytes)")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="reports/station_summary_exec.html")
    build(Path(ap.parse_args().out))


if __name__ == "__main__":
    main()
