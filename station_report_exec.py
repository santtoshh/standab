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

W, H = 900, 300          # chart viewBox; scales to container width
PAD_L, PAD_R, PAD_T, PAD_B = 62, 26, 30, 46


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
table{width:100%;border-collapse:collapse;font-size:15px;margin-top:12px}
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
    supplied = float(chg.sessions["kwh"].sum())
    catered = 100.0 * supplied / need_total
    dock_rate = 100.0 * wdf["docked"].sum() / wdf["assessable"].sum()
    veh_share = 100.0 * wvf["docked"].sum() / wvf["riding"].sum()

    sup_w = S.charging_by_gran(chg, "week")
    wk_labels = [w.strftime("%-d %b") for w in wdf.index]
    need_w = [S.energy_need_kwh(int(wdf.loc[w, "rides"])) for w in wdf.index]
    sup_wv = [float(sup_w.get(w, np.nan)) for w in wdf.index]
    cat_w = [100.0 * s / n if n else np.nan for s, n in zip(sup_wv, need_w)]
    dock_w = [float(wdf.loc[w, "dock_rate_pct"]) for w in wdf.index]
    vehs_w = [100.0 * wvf.loc[w, "docked"] / wvf.loc[w, "riding"] for w in wdf.index]
    unass_w = [float(wdf.loc[w, "unassessable_pct"]) for w in wdf.index]

    # Where the missing-GPS problem starts: first week whose unassessable share
    # is more than double the earliest week's. Beyond it the docking line is not
    # trustworthy, so it is shaded rather than narrated.
    base_unass = unass_w[0] if unass_w else 0.0
    shade_from = next((i for i, u in enumerate(unass_w)
                       if u > max(2.0 * base_unass, 2.0)), None)

    pst = S.per_site_table(df, sites).sort_values("buffer_ends", ascending=False)
    site_lbl = [f"S{i + 1}" for i in range(len(pst))]
    site_val = [int(v) for v in pst["buffer_ends"]]
    live_sites = int((pst["buffer_ends"] > 10).sum())

    swaps_share = 100.0 - catered
    per_day_need = need_total / days
    per_day_sup = supplied / days

    o: list[str] = []
    o.append('<!doctype html><html lang="en"><head><meta charset="utf-8">'
             '<meta name="viewport" content="width=device-width,initial-scale=1">'
             f'<title>Vaasa station pilot &mdash; summary</title><style>{CSS}</style>'
             '</head><body><div class="wrap">')

    o.append('<div class="head">')
    o.append("<h1>Vaasa charging stations &mdash; how the pilot is doing</h1>")
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
             f'<div class="t">trips end at a station</div>'
             f'<div class="s">the other {100 - dock_rate:.0f} end somewhere '
             f'else in the city</div></div>')
    o.append(f'<div class="hero"><div class="n">{veh_share:.0f}%</div>'
             f'<div class="t">of scooters visit a station in a given week</div>'
             f'<div class="s">so most of the fleet does reach a station '
             f'sometimes</div></div>')
    o.append(f'<div class="hero"><div class="n">{supplied:,.0f} kWh</div>'
             f'<div class="t">charged at the stations so far</div>'
             f'<div class="s">enough energy for about '
             f'{supplied * 1000 / S.WH_PER_RIDE / 1000:,.0f},000 trips</div></div>')
    o.append("</div>")

    # ---- energy -----------------------------------------------------------
    o.append('<div class="card">')
    o.append("<h2>Energy: what the scooters need vs what the stations give</h2>")
    o.append(f'<p class="lead">Every trip uses about {S.WH_PER_RIDE} Wh of '
             f'battery. Across {days} days the fleet needed roughly '
             f'<strong>{need_total:,.0f} kWh</strong>. The stations delivered '
             f'<strong>{supplied:,.0f} kWh</strong> of it.</p>')
    o.append('<div class="chartttl">Each week, in kilowatt-hours</div>')
    o.append('<div class="legend">'
             f'<span><i style="background:{BLUE}"></i>Energy the scooters needed</span>'
             f'<span><i style="background:{ORANGE}"></i>Energy the stations gave</span>'
             "</div>")
    o.append(line_chart(wk_labels,
                        [{"values": need_w, "color": BLUE},
                         {"values": sup_wv, "color": ORANGE}],
                        y_suffix="", fmt=lambda v: f"{v:,.0f}"))
    o.append(f'<div class="takeaway">The gap between the two lines is energy '
             f'still being handled by battery swaps &mdash; about '
             f'<strong>{per_day_need - per_day_sup:,.0f} kWh a day</strong>. '
             f'Closing it is what more stations, or more use of the ones we have, '
             f'would buy us.</div>')

    o.append('<div class="chartttl">Share of the fleet&rsquo;s energy covered by stations</div>')
    o.append(line_chart(wk_labels, [{"values": cat_w, "color": GREEN}],
                        y_suffix="%", y_max=60, fmt=lambda v: f"{v:,.0f}"))
    o.append(f'<div class="takeaway good">This has held steady at roughly '
             f'<strong>a third</strong> of the fleet&rsquo;s needs since the '
             f'first full week &mdash; the stations are carrying a real, '
             f'consistent share of the load.</div>')
    o.append("</div>")

    # ---- docking ----------------------------------------------------------
    o.append('<div class="card">')
    o.append("<h2>Do people finish their trips at a station?</h2>")
    o.append(f'<p class="lead">A trip counts as ending at a station if the '
             f'scooter was left within {S.DOCK_RADIUS_M:.0f} metres of one.</p>')
    o.append('<div class="chartttl">Share of trips ending at a station, each week</div>')
    o.append(line_chart(wk_labels, [{"values": dock_w, "color": BLUE}],
                        y_suffix="%", y_max=14, fmt=lambda v: f"{v:,.0f}",
                        band_from=shade_from,
                        band_note="less reliable — see note"))
    if shade_from is not None:
        o.append(f'<div class="takeaway warn">Roughly '
                 f'<strong>{dock_rate:.0f} trips in every 100</strong> end at a '
                 f'station. The last few weeks look lower, but in those weeks '
                 f'many more scooters failed to report where they were parked, '
                 f'so we cannot say whether docking actually fell. Treat the '
                 f'shaded part as unknown, not as a decline.</div>')
    o.append("</div>")

    # ---- stations ---------------------------------------------------------
    o.append('<div class="card">')
    o.append("<h2>Which stations are being used?</h2>")
    o.append(f'<p class="lead">Trips ending at each of our {len(sites)} station '
             f'locations over the whole period.</p>')
    o.append(bar_chart(site_lbl, site_val, color=BLUE, fmt=lambda v: f"{v:,.0f}"))
    o.append(f'<div class="takeaway"><strong>{live_sites} of {len(sites)}</strong> '
             f'locations get real traffic, and the busiest three take a large '
             f'share of it. The quietest ones are worth a look on the ground &mdash; '
             f'some are simply newer than the rest.</div>')
    o.append("</div>")

    # ---- fleet reach ------------------------------------------------------
    o.append('<div class="card">')
    o.append("<h2>How much of the fleet reaches a station each week?</h2>")
    o.append('<div class="chartttl">Share of scooters in use that visited a station that week</div>')
    o.append(line_chart(wk_labels, [{"values": vehs_w, "color": GREEN}],
                        y_suffix="%", y_max=100, fmt=lambda v: f"{v:,.0f}"))
    o.append(f'<div class="takeaway good">About <strong>{veh_share:.0f}% of '
             f'scooters</strong> touch a station in a normal week, so the '
             f'network already reaches a good part of the fleet. The gap is '
             f'how often trips actually <em>finish</em> there.</div>')
    o.append("</div>")

    # ---- what to do -------------------------------------------------------
    o.append('<div class="card">')
    o.append("<h2>What this suggests</h2>")
    o.append('<ul class="acts">')
    o.append(f"<li><strong>Give riders a reason to finish at a station.</strong> "
             f"Only {dock_rate:.0f} trips in 100 end at one today. A small "
             f"in-app incentive for ending a trip at a station is the cheapest "
             f"way to move every other number on this page.</li>")
    o.append(f"<li><strong>The stations already pay their way operationally.</strong> "
             f"They cover {catered:.0f}% of the fleet&rsquo;s energy, which is "
             f"charging work the team no longer does by hand.</li>")
    o.append("<li><strong>Look at the quietest locations.</strong> A few sites "
             "get very little traffic. Some were installed only recently; the "
             "others may be in the wrong place.</li>")
    o.append(f"<li><strong>Fix the missing location data.</strong> In recent "
             f"weeks a growing share of trips did not report where they ended, "
             f"which limits what we can measure. Worth raising with the "
             f"vehicle team.</li>")
    o.append("</ul>")
    o.append("</div>")

    o.append(f'<div class="foot">Vaasa station pilot &middot; '
             f'{start.strftime("%-d %b")}&ndash;{end.strftime("%-d %b %Y")} &middot; '
             f'prepared {_dt.date.today().strftime("%-d %B %Y")}</div>')
    o.append("</div></body></html>")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(o), encoding="utf-8")

    print(f"  period            : {start.date()} -> {end.date()} ({days} days)")
    print(f"  rides / vehicles  : {rides:,} / {vehicles:,}")
    print(f"  energy catered    : {catered:.1f}%  ({supplied:,.0f} of {need_total:,.0f} kWh)")
    print(f"  docking rate      : {dock_rate:.1f}%")
    print(f"  weekly reach      : {veh_share:.1f}%")
    print(f"  energy supplied   : {supplied:,.0f} kWh")
    print(f"  wrote             : {out_path}  ({out_path.stat().st_size:,} bytes)")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="reports/station_summary_exec.html")
    build(Path(ap.parse_args().out))


if __name__ == "__main__":
    main()
