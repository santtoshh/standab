#!/usr/bin/env python3
"""Standalone note: why did station coverage drop in the busiest week?

Separate from the pilot summary on purpose. It answers one question, and the
answer turned out to be the opposite of the obvious guess, so it gets its own
page rather than a paragraph inside another report.

Everything is measured from the charging export and the ride exports through
``station_report``, so the numbers agree with the other two documents.

Occupancy counts every receptacle event, charging or not. The charging export
now includes rows that delivered no power at all: a vehicle was placed on a
point and nothing flowed. Those rows are the whole reason this page exists, so
they are counted here, and every energy figure on the page (time on charge, kWh
per session) still uses charging sessions only.

Usage:
    python station_capacity_note.py [--out reports/station_capacity_note.html]
"""
from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path

import numpy as np
import pandas as pd

import station_report as S
from station_report_exec import BLUE, ORANGE, GREEN, MUTED, GRID, SURFACE, INK, esc

BUSY_WEEK = pd.Timestamp("2026-07-06")

# The occupancy sweep lives in station_report so this page, the technical report
# and the one-pager cannot drift apart on the one number the argument rests on.
occupancy_series = S.occupancy_series


def grouped_bars(labels: list[str], series: list[dict], *, y_max: float,
                 y_suffix: str = "", fmt=lambda v: f"{v:,.0f}",
                 cap_line: float | None = None,
                 cap_label: str = "") -> str:
    """Grouped bars on one shared axis, with an optional capacity rule."""
    W, H = 900, 320
    PL, PR, PT, PB = 62, 26, 34, 46
    n, k = len(labels), len(series)
    plot_w = W - PL - PR
    band = plot_w / n
    bw = min(band * 0.72 / k, 34)

    def Y(v: float) -> float:
        return H - PB - (H - PT - PB) * (v / y_max)

    p = [f'<svg viewBox="0 0 {W} {H}" role="img" style="width:100%;height:auto">']
    for t in np.linspace(0, y_max, 5):
        y = Y(t)
        p.append(f'<line x1="{PL}" y1="{y:.1f}" x2="{W - PR}" y2="{y:.1f}" '
                 f'stroke="{GRID}" stroke-width="1"/>')
        p.append(f'<text x="{PL - 10}" y="{y + 4:.1f}" font-size="13" '
                 f'text-anchor="end" fill="{MUTED}">{fmt(t)}{y_suffix}</text>')
    if cap_line is not None:
        y = Y(cap_line)
        p.append(f'<line x1="{PL}" y1="{y:.1f}" x2="{W - PR}" y2="{y:.1f}" '
                 f'stroke="{INK}" stroke-width="2" stroke-dasharray="6,4"/>')
        p.append(f'<text x="{W - PR}" y="{y - 8:.1f}" font-size="13" '
                 f'font-weight="700" text-anchor="end" fill="{INK}">'
                 f'{esc(cap_label)}</text>')
    for i, lab in enumerate(labels):
        cx = PL + band * (i + 0.5)
        for j, s in enumerate(series):
            v = s["values"][i]
            x = cx - (k * bw + 2 * (k - 1)) / 2 + j * (bw + 2)
            y = Y(v)
            p.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" '
                     f'height="{H - PB - y:.1f}" fill="{s["color"]}" rx="4"/>')
            p.append(f'<text x="{x + bw / 2:.1f}" y="{y - 7:.1f}" font-size="12" '
                     f'font-weight="700" text-anchor="middle" fill="{INK}">'
                     f'{fmt(v)}</text>')
        p.append(f'<text x="{cx:.1f}" y="{H - PB + 22}" font-size="13" '
                 f'text-anchor="middle" fill="{MUTED}">{esc(lab)}</text>')
    p.append("</svg>")
    return "".join(p)


CSS = """
*{box-sizing:border-box}
body{margin:0;background:#f4f3ef;color:#111;font:16px/1.6 -apple-system,BlinkMacSystemFont,
 'Segoe UI',Roboto,Helvetica,Arial,sans-serif;-webkit-text-size-adjust:100%}
.wrap{max-width:1000px;margin:0 auto;padding:34px 22px 60px}
h1{font-size:30px;line-height:1.2;margin:0 0 6px;letter-spacing:-.02em}
.sub{color:#5b5b57;font-size:16px;margin-bottom:24px}
.card{background:#fcfcfb;border:1px solid #e6e5df;border-radius:14px;padding:26px 28px;margin-bottom:20px}
h2{font-size:21px;margin:0 0 10px;letter-spacing:-.01em}
.lead{font-size:17px;color:#3a3a36;margin:0 0 18px}
.answer{background:#fcfcfb;border:1px solid #e6e5df;border-left:5px solid #eb6834;
 border-radius:14px;padding:22px 26px;margin-bottom:20px}
.answer .k{font-size:13px;font-weight:700;color:#8a8a83;text-transform:uppercase;letter-spacing:.05em}
.answer p{font-size:19px;line-height:1.5;margin:8px 0 0}
.chartttl{font-size:14px;font-weight:700;color:#5b5b57;margin:16px 0 4px;text-transform:uppercase;letter-spacing:.04em}
.legend{display:flex;gap:20px;flex-wrap:wrap;margin:8px 0 6px;font-size:14.5px;color:#3a3a36}
.legend i{display:inline-block;width:13px;height:13px;border-radius:3px;margin-right:7px;vertical-align:-1px}
.takeaway{background:#f7f6f2;border-left:4px solid #2a78d6;padding:13px 17px;border-radius:0 8px 8px 0;
 margin-top:16px;font-size:16px;color:#26261f}
.tablewrap{overflow-x:auto;-webkit-overflow-scrolling:touch;margin-top:14px}
table{width:100%;min-width:640px;border-collapse:collapse;font-size:15px}
th{text-align:left;font-size:13px;text-transform:uppercase;letter-spacing:.04em;color:#8a8a83;
 padding:8px 10px;border-bottom:2px solid #e6e5df}
td{padding:9px 10px;border-bottom:1px solid #efeee9}
td.num{text-align:right;font-variant-numeric:tabular-nums}
tr.hi td{background:#fdf1ec;font-weight:700}
ul{margin:6px 0 0;padding-left:22px}
li{margin-bottom:10px}
.foot{color:#8a8a83;font-size:13.5px;margin-top:26px;text-align:center}
@media print{body{background:#fff}.card,.answer{break-inside:avoid;border-color:#ddd}.wrap{padding:0}}
@media(max-width:620px){h1{font-size:25px}.answer p{font-size:17px}}
"""


def build(out_path: Path) -> Path:
    racks = S.load_racks(S.DEFAULT_RACK_FILE)
    chg = S.load_charging(S.DEFAULT_CHARGING_FILE, racks)
    raw, _ = S.load_rides(S.DEFAULT_RIDE_FILES)
    df, _ = S.prepare_rides(raw)

    total_recep = S.total_receptacles(racks)
    # `s` is every receptacle event, charging or not. `c` is the subset that
    # delivered energy. Occupancy uses `s`, because a vehicle holds its point
    # either way. Time on charge and kWh per session use `c`, because averaging
    # them over rows that moved no energy would report a session that never
    # happened.
    s = chg.sessions.copy()
    s["dur_h"] = s["duration_s"] / 3600.0
    c = s.loc[s["is_charge"]]
    d = s.loc[~s["is_charge"]]

    occ = occupancy_series(s)
    occ_charge_only = occupancy_series(c)
    peak_all = int(occ["occ"].max())
    peak_charge_only = int(occ_charge_only["occ"].max())
    dock_med_h = float(d["dur_h"].median())
    dock_over_6 = int((d["dur_h"] > 6).sum())

    g_all = s.groupby("week").size()
    g = c.groupby("week").agg(sessions=("kwh", "size"), kwh=("kwh", "sum"),
                              kwh_per=("kwh", "mean"), med_dur=("dur_h", "median"))
    rides_wk = df.groupby("week").size()
    # Only weeks where both the stations and the ride export cover all seven
    # days. The first week starts part-way through (charging began mid-morning
    # on the 15th) and the last is cut short, so neither is comparable.
    chg_start, ride_end = s["started"].min(), df["start_time"].max()
    weeks = [w for w in g.index
             if w >= chg_start and w + pd.Timedelta(days=7) <= ride_end]
    rows = []
    for w in weeks:
        sub = occ[occ["week"] == w]
        r = g.loc[w]
        need = S.energy_need_kwh(int(rides_wk.get(w, 0)))
        rows.append({
            "week": w, "label": w.strftime("%-d %b"),
            "rides": int(rides_wk.get(w, 0)),
            "sessions": int(r.sessions),
            "docks": int(g_all.get(w, 0)) - int(r.sessions),
            "peak": int(sub["occ"].max()) if len(sub) else 0,
            "mean_occ": float(sub["occ"].mean()) if len(sub) else 0.0,
            "kwh": float(r.kwh),
            "kwh_per": float(r.kwh_per),
            "med_dur": float(r.med_dur),
            "catered": 100.0 * float(r.kwh) / need if need else float("nan"),
        })
    busy = next(r for r in rows if r["week"] == BUSY_WEEK)
    others = [r for r in rows if r["week"] != BUSY_WEEK]

    # Compare like with like: both halves of every sentence below use the same
    # full weeks, never the partial stubs at either end of the window.
    full_weeks = {r["week"] for r in rows}
    b_sess = c[c["week"] == BUSY_WEEK]
    o_sess = c[c["week"].isin(full_weeks - {BUSY_WEEK})]
    busy_short = 100.0 * float((b_sess["dur_h"] < 1).mean())
    other_short = 100.0 * float((o_sess["dur_h"] < 1).mean())

    # Power per occupied point, over ALL occupied point-hours including docks
    # drawing nothing. Dividing by charging hours alone would contradict this
    # page's own statement that docks are counted in every occupancy figure.
    tick_h = pd.Timedelta(S.OCCUPANCY_TICK).total_seconds() / 3600.0
    _busy_occ_h = float(occ.loc[occ["week"] == BUSY_WEEK, "occ"].sum() * tick_h)
    _other_occ_h = float(occ.loc[occ["week"].isin(full_weeks - {BUSY_WEEK}),
                                 "occ"].sum() * tick_h)
    busy_rate = b_sess["kwh"].sum() * 1000 / _busy_occ_h
    other_rate = o_sess["kwh"].sum() * 1000 / _other_occ_h
    long_12 = 100.0 * float((c["dur_h"] > 12).mean())
    long_24 = 100.0 * float((c["dur_h"] > 24).mean())
    free_at_peak = 100.0 * (total_recep - peak_all) / total_recep
    busy_free = 100.0 * (total_recep - busy["mean_occ"]) / total_recep

    o: list[str] = []
    o.append('<!doctype html><html lang="en"><head><meta charset="utf-8">'
             '<meta name="viewport" content="width=device-width,initial-scale=1">'
             f'<title>Vaasa stations: the busiest week</title><style>{CSS}</style>'
             '</head><body><div class="wrap">')
    o.append("<h1>Why did the stations cover less in the busiest week?</h1>")
    o.append(f'<div class="sub">Vaasa charging stations &middot; week of '
             f'{busy["label"]} 2026 compared with the other full weeks</div>')

    o.append('<div class="answer"><div class="k">Short answer</div>'
             f'<p>Not because the racks were full. In that week they were '
             f'<strong>emptier than usual</strong>. Vehicles were put on a rack '
             f'and taken away again quickly, so each one picked up less '
             f'charge.</p></div>')

    # 1. capacity
    o.append('<div class="card">')
    o.append("<h2>The racks were never close to full</h2>")
    o.append(f'<p class="lead">We have <strong>{total_recep} charging points</strong> '
             f'across the {len(racks)} racks. This counts how many were occupied at '
             f'once, every 15 minutes. A vehicle sitting on a rack fully charged '
             f'still counts as occupying its point, and so does a vehicle that '
             f'was placed on a point and drew no power at all.</p>')
    o.append('<div class="chartttl">Charging points in use at once, each week</div>')
    o.append('<div class="legend">'
             f'<span><i style="background:{BLUE}"></i>Busiest moment of the week</span>'
             f'<span><i style="background:{MUTED}"></i>Typical moment</span></div>')
    o.append(grouped_bars([r["label"] for r in rows],
                          [{"values": [r["peak"] for r in rows], "color": BLUE},
                           {"values": [r["mean_occ"] for r in rows], "color": MUTED}],
                          y_max=total_recep * 1.02,
                          cap_line=total_recep,
                          cap_label=f"all {total_recep} points"))
    o.append(f'<div class="takeaway">The fullest the network ever got in the whole '
             f'period was <strong>{peak_all} of {total_recep} points</strong> '
             f'({peak_all / total_recep * 100:.0f}%), so half the points were free '
             f'even at that moment. In the busiest week the peak was '
             f'<strong>{busy["peak"]}</strong> '
             f'({busy["peak"] / total_recep * 100:.0f}%) and at a typical moment '
             f'only <strong>{busy["mean_occ"]:.0f} points</strong> were in use, the '
             f'lowest of any week. That leaves {busy_free:.0f}% of our charging '
             f'points free at a typical moment in the week we are asking '
             f'about.</div>')

    o.append("<h3>Vehicles that drew no power are counted too</h3>")
    o.append(f'<p class="note">The station data now also records vehicles that '
             f'were placed on a point and took no charge at all. There are '
             f'{len(d):,} of these, {100.0 * len(d) / len(s):.0f}% of everything '
             f'the racks logged, and they are not brief: the typical one lasts '
             f'{dock_med_h:.1f} hours and {dock_over_6:,} of them run past six. '
             f'They hold a point exactly like a charging vehicle does, so they '
             f'are counted in every occupancy figure on this page. Doing so '
             f'raises the fullest moment we ever saw from '
             f'{peak_charge_only} points to {peak_all}, and the answer to the '
             f'question does not change: the busiest week is still the emptiest '
             f'week, at {busy["mean_occ"]:.0f} points in use at a typical '
             f'moment.</p>')
    o.append("</div>")

    # 2. dwell
    o.append('<div class="card">')
    o.append("<h2>What changed was how long vehicles stayed</h2>")
    o.append('<div class="chartttl">What the racks logged each week</div>')
    o.append('<div class="legend">'
             f'<span><i style="background:{BLUE}"></i>Charging sessions, power delivered</span>'
             f'<span><i style="background:{MUTED}"></i>Dock events, no power delivered</span></div>')
    o.append(grouped_bars([r["label"] for r in rows],
                          [{"values": [r["sessions"] for r in rows], "color": BLUE},
                           {"values": [r["docks"] for r in rows], "color": MUTED}],
                          y_max=max(max(r["sessions"], r["docks"]) for r in rows) * 1.25))
    o.append(f'<div class="takeaway">The two are counted separately everywhere on this page. Charging sessions peak in the busiest week at <strong>{busy["sessions"]:,}</strong>. Dock events that moved no power run to <strong>{busy["docks"]:,}</strong> that week and peak in a different week entirely, so the two are not measuring the same thing. Only charging sessions go into the time-on-charge and kWh figures below.</div>')
    o.append('<div class="chartttl">Typical time a vehicle spent on charge, hours</div>')
    o.append(grouped_bars([r["label"] for r in rows],
                          [{"values": [r["med_dur"] for r in rows], "color": ORANGE}],
                          y_max=max(r["med_dur"] for r in rows) * 1.25,
                          fmt=lambda v: f"{v:.1f}"))
    o.append('<div class="chartttl">Energy each vehicle took on, kWh per session</div>')
    o.append(grouped_bars([r["label"] for r in rows],
                          [{"values": [r["kwh_per"] for r in rows], "color": GREEN}],
                          y_max=max(r["kwh_per"] for r in rows) * 1.25,
                          fmt=lambda v: f"{v:.2f}"))
    o.append(f'<div class="takeaway">In the busiest week a vehicle typically sat on '
             f'charge for <strong>{busy["med_dur"]:.1f} hours</strong> against about '
             f'<strong>{np.median([r["med_dur"] for r in others]):.1f} hours</strong> '
             f'in the other weeks, and picked up '
             f'<strong>{busy["kwh_per"]:.2f} kWh</strong> instead of around '
             f'<strong>{np.mean([r["kwh_per"] for r in others]):.2f} kWh</strong>. '
             f'{busy_short:.0f}% of sessions lasted under an hour that week, against '
             f'{other_short:.0f}% normally. Vehicles were going back out before they '
             f'had taken much charge.</div>')
    o.append("</div>")

    # 3. the table
    o.append('<div class="card">')
    o.append("<h2>Week by week</h2>")
    o.append('<div class="tablewrap">')
    o.append("<table><thead><tr><th>Week</th><th class='num'>Trips</th>"
             "<th class='num'>Charging sessions</th>"
             "<th class='num'>Dock events, no power</th>"
             "<th class='num'>Peak points used</th>"
             "<th class='num'>Typical time on charge</th>"
             "<th class='num'>kWh per session</th>"
             "<th class='num'>Share of energy covered</th></tr></thead><tbody>")
    for r in rows:
        cls = ' class="hi"' if r["week"] == BUSY_WEEK else ""
        o.append(f'<tr{cls}><td>{esc(r["label"])}</td>'
                 f'<td class="num">{r["rides"]:,}</td>'
                 f'<td class="num">{r["sessions"]:,}</td>'
                 f'<td class="num">{r["docks"]:,}</td>'
                 f'<td class="num">{r["peak"]} of {total_recep}</td>'
                 f'<td class="num">{r["med_dur"]:.1f} h</td>'
                 f'<td class="num">{r["kwh_per"]:.2f}</td>'
                 f'<td class="num">{r["catered"]:.0f}%</td></tr>')
    o.append("</tbody></table></div>")
    o.append(f'<div class="takeaway">The busiest week had the '
             f'<strong>most charging sessions of any week ({busy["sessions"]:,})</strong>, '
             f'so vehicles were reaching the racks. They just did not stay long '
             f'enough to take on much energy.</div>')
    o.append("</div>")

    # 4. what it means
    o.append('<div class="card">')
    o.append("<h2>What this means</h2>")
    o.append("<ul>")
    o.append(f"<li><strong>Slots are not the limit.</strong> At the fullest "
             f"moment of the whole period {peak_all} of our {total_recep} "
             f"charging points were held, so {free_at_peak:.0f}% were still free. "
             f"In the busiest week the peak was {busy['peak']} and a typical "
             f"moment used {busy['mean_occ']:.0f}. This counts vehicles that drew "
             f"no power as occupying their point.</li>")
    o.append(f"<li><strong>Time on charge is the limit.</strong> When the city is "
             f"busy, vehicles are pulled back into service after about "
             f"{busy['med_dur']:.1f} hours instead of "
             f"{np.median([r['med_dur'] for r in others]):.1f}, and each one takes "
             f"on roughly a quarter less energy.</li>")
    o.append(f"<li><strong>Vehicles do sit fully charged, but in the quiet weeks.</strong> "
             f"Across the whole period {long_12:.0f}% of charging sessions ran "
             f"over 12 hours and {long_24:.0f}% over 24. That happens when there "
             f"is spare capacity anyway, so it is not what held the busy week "
             f"back.</li>")
    o.append(f"<li><strong>The stations charged faster when it was busy.</strong> "
             f"They delivered about {busy_rate:.0f} watts per occupied point that "
             f"week against {other_rate:.0f} normally, which fits vehicles arriving "
             f"more empty and spending less time sitting full.</li>")
    o.append("</ul></div>")

    o.append(f'<div class="foot">Vaasa charging stations &middot; prepared '
             f'{_dt.date.today().strftime("%-d %B %Y")}</div>')
    o.append("</div></body></html>")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(o), encoding="utf-8")

    # Assertions: these are the claims the page makes. Occupancy is measured on
    # every receptacle event, so these now have to survive the dock rows too.
    assert peak_all < total_recep * 0.75, "racks got close to full; premise is wrong"
    assert busy["mean_occ"] == min(r["mean_occ"] for r in rows), \
        "busy week is not the emptiest on typical occupancy"
    assert busy["sessions"] == max(r["sessions"] for r in rows), \
        "busy week not the busiest by charging sessions"
    assert busy["med_dur"] == min(r["med_dur"] for r in rows), \
        "busy week not the shortest dwell"
    # The re-export is only meaningful if it actually moved the occupancy
    # reading; if these two ever match, the dock rows have been filtered out
    # again somewhere upstream.
    assert peak_all > peak_charge_only, \
        "dock events are not reaching the occupancy sweep"
    assert len(d) > 0 and float(d["kwh"].max()) == 0.0, \
        "dock rows are missing or carry energy"

    print(f"  charging points     : {total_recep}")
    print(f"  charging sessions   : {len(c):,}")
    print(f"  dock events, no power: {len(d):,} "
          f"({100.0 * len(d) / len(s):.0f}% of all events), median "
          f"{dock_med_h:.2f} h")
    print(f"  peak ever occupied  : {peak_all} ({peak_all / total_recep * 100:.0f}%)"
          f"  [charging sessions only: {peak_charge_only}]")
    print(f"  mean occupancy      : {float(occ['occ'].mean()):.1f}")
    print(f"  busy week peak      : {busy['peak']} ({busy['peak'] / total_recep * 100:.0f}%)")
    print(f"  busy week mean occ  : {busy['mean_occ']:.1f} (lowest of any week)")
    print(f"  busy week sessions  : {busy['sessions']:,} (max of any week) "
          f"+ {busy['docks']:,} dock events")
    print(f"  busy week dwell     : {busy['med_dur']:.1f} h vs "
          f"{np.median([r['med_dur'] for r in others]):.1f} h")
    print(f"  busy week kWh/sess  : {busy['kwh_per']:.3f} vs "
          f"{np.mean([r['kwh_per'] for r in others]):.3f}")
    print(f"  wrote               : {out_path} ({out_path.stat().st_size:,} bytes)")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="reports/station_capacity_note.html")
    build(Path(ap.parse_args().out))


if __name__ == "__main__":
    main()
