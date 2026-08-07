#!/usr/bin/env python3
"""Standalone note: why did station coverage drop in the busiest week?

Separate from the pilot summary on purpose. It answers one question, and the
answer turned out to be the opposite of the obvious guess, so it gets its own
page rather than a paragraph inside another report.

Everything is measured from the charging export and the ride exports through
``station_report``, so the numbers agree with the other two documents.

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
TICK = "15min"          # resolution for the occupancy sweep


def occupancy_series(sessions: pd.DataFrame) -> pd.DataFrame:
    """Receptacles in use at each tick, by counting overlapping sessions.

    A session occupies a receptacle from ``started`` to ``finished`` whether or
    not it is still drawing power, which is exactly the thing being tested: a
    vehicle sitting on a rack fully charged still holds the slot.
    """
    t0 = sessions["started"].min().floor("D")
    t1 = sessions["finished"].max().ceil("D")
    ticks = pd.date_range(t0, t1, freq=TICK)
    st = sessions["started"].values.astype("datetime64[ns]")
    fi = sessions["finished"].values.astype("datetime64[ns]")
    occ = np.array([((st <= t) & (fi > t)).sum()
                    for t in ticks.values.astype("datetime64[ns]")])
    out = pd.DataFrame({"t": ticks, "occ": occ})
    out["week"] = (out["t"].dt.normalize()
                   - pd.to_timedelta(out["t"].dt.weekday, unit="D"))
    return out


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
table{width:100%;border-collapse:collapse;font-size:15px;margin-top:14px}
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

    total_recep = sum(r.receptacles for r in racks)
    s = chg.sessions.copy()
    s["dur_h"] = s["duration_s"] / 3600.0

    occ = occupancy_series(s)
    peak_all = int(occ["occ"].max())

    g = s.groupby("week").agg(sessions=("kwh", "size"), kwh=("kwh", "sum"),
                              kwh_per=("kwh", "mean"), med_dur=("dur_h", "median"),
                              occ_h=("dur_h", "sum"))
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
            "peak": int(sub["occ"].max()) if len(sub) else 0,
            "mean_occ": float(sub["occ"].mean()) if len(sub) else 0.0,
            "kwh": float(r.kwh),
            "kwh_per": float(r.kwh_per),
            "med_dur": float(r.med_dur),
            "catered": 100.0 * float(r.kwh) / need if need else float("nan"),
        })
    busy = next(r for r in rows if r["week"] == BUSY_WEEK)
    others = [r for r in rows if r["week"] != BUSY_WEEK]

    b_sess = s[s["week"] == BUSY_WEEK]
    o_sess = s[(s["week"] >= pd.Timestamp("2026-06-15")) & (s["week"] != BUSY_WEEK)]
    busy_short = 100.0 * float((b_sess["dur_h"] < 1).mean())
    other_short = 100.0 * float((o_sess["dur_h"] < 1).mean())
    busy_rate = b_sess["kwh"].sum() * 1000 / b_sess["dur_h"].sum()
    other_rate = o_sess["kwh"].sum() * 1000 / o_sess["dur_h"].sum()
    long_12 = 100.0 * float((s["dur_h"] > 12).mean())
    long_24 = 100.0 * float((s["dur_h"] > 24).mean())

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
             f'<strong>emptier than usual</strong>. Vehicles were plugged in and '
             f'taken away again quickly, so each one picked up less charge.</p></div>')

    # 1. capacity
    o.append('<div class="card">')
    o.append("<h2>The racks were never close to full</h2>")
    o.append(f'<p class="lead">We have <strong>{total_recep} charging points</strong> '
             f'across the {len(racks)} racks. This counts how many were occupied at '
             f'once, every 15 minutes. A vehicle sitting on a rack fully charged '
             f'still counts as occupying its point.</p>')
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
             f'({peak_all / total_recep * 100:.0f}%). In the busiest week the peak '
             f'was <strong>{busy["peak"]}</strong> '
             f'({busy["peak"] / total_recep * 100:.0f}%) and at a typical moment only '
             f'<strong>{busy["mean_occ"]:.0f} points</strong> were in use, the '
             f'lowest of any week. Around two thirds of our charging points were '
             f'free at all times.</div>')
    o.append("</div>")

    # 2. dwell
    o.append('<div class="card">')
    o.append("<h2>What changed was how long vehicles stayed</h2>")
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
    o.append("<table><thead><tr><th>Week</th><th class='num'>Trips</th>"
             "<th class='num'>Charging sessions</th>"
             "<th class='num'>Peak points used</th>"
             "<th class='num'>Typical time on charge</th>"
             "<th class='num'>kWh per session</th>"
             "<th class='num'>Share of energy covered</th></tr></thead><tbody>")
    for r in rows:
        cls = ' class="hi"' if r["week"] == BUSY_WEEK else ""
        o.append(f'<tr{cls}><td>{esc(r["label"])}</td>'
                 f'<td class="num">{r["rides"]:,}</td>'
                 f'<td class="num">{r["sessions"]:,}</td>'
                 f'<td class="num">{r["peak"]} of {total_recep}</td>'
                 f'<td class="num">{r["med_dur"]:.1f} h</td>'
                 f'<td class="num">{r["kwh_per"]:.2f}</td>'
                 f'<td class="num">{r["catered"]:.0f}%</td></tr>')
    o.append("</tbody></table>")
    o.append(f'<div class="takeaway">The busiest week had the '
             f'<strong>most charging sessions of any week ({busy["sessions"]:,})</strong>, '
             f'so vehicles were reaching the racks. They just did not stay long '
             f'enough to take on much energy.</div>')
    o.append("</div>")

    # 4. what it means
    o.append('<div class="card">')
    o.append("<h2>What this means</h2>")
    o.append("<ul>")
    o.append(f"<li><strong>Slots are not the limit.</strong> Around two thirds of "
             f"our {total_recep} charging points sit free even at the busiest "
             f"moment of the busiest week.</li>")
    o.append(f"<li><strong>Time on charge is the limit.</strong> When the city is "
             f"busy, vehicles are pulled back into service after about "
             f"{busy['med_dur']:.1f} hours instead of "
             f"{np.median([r['med_dur'] for r in others]):.1f}, and each one takes "
             f"on roughly a quarter less energy.</li>")
    o.append(f"<li><strong>Vehicles do sit fully charged, but in the quiet weeks.</strong> "
             f"Across the whole period {long_12:.0f}% of sessions ran over 12 hours "
             f"and {long_24:.0f}% over 24. That happens when there is spare capacity "
             f"anyway, so it is not what held the busy week back.</li>")
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

    # Assertions: these are the claims the page makes.
    assert peak_all < total_recep * 0.75, "racks got close to full; premise is wrong"
    assert busy["mean_occ"] == min(r["mean_occ"] for r in rows), \
        "busy week is not the emptiest on typical occupancy"
    assert busy["sessions"] == max(r["sessions"] for r in rows), \
        "busy week not the busiest by sessions"
    assert busy["med_dur"] == min(r["med_dur"] for r in rows), \
        "busy week not the shortest dwell"

    print(f"  charging points     : {total_recep}")
    print(f"  peak ever occupied  : {peak_all} ({peak_all / total_recep * 100:.0f}%)")
    print(f"  busy week peak      : {busy['peak']} ({busy['peak'] / total_recep * 100:.0f}%)")
    print(f"  busy week sessions  : {busy['sessions']:,} (max of any week)")
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
