"""Figure + summary table for the P3 x P4 sweeps (jobs 2664328, 2665264).

Parses every RESULT line from results/p3_admit*_*.log, averages repeats per
(MPS %, admission limit, quantum), writes results/p3_sweep_summary.csv and
results/p3_sweep.png (house style of the integrator report's figures).

Usage: python plot_p3_sweep.py
"""
import csv, re, statistics
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
LINE = re.compile(r"RESULT rep=(\d+) admit=(\S+) .*?node (\d+) ns/day \| cpu ([\d.]+) cores .*?pct=(\S+) .*?quantum=(\d+)")

rows = defaultdict(list)
for log in sorted(RES.glob("p3_admit*_*.log")):
    job = log.stem.split("_")[-1]
    for m in LINE.finditer(log.read_text()):
        rep, admit, node, cpu, pct, quantum = m.groups()
        rows[(pct, admit, int(quantum))].append((int(node), float(cpu), job))

ADMIT_ORDER = ["all", "32", "16", "8", "6", "4", "2"]
PCT_ORDER = ["inherit", "50", "25", "15", "10"]
summary = []
for (pct, admit, quantum), vals in rows.items():
    nodes = [v[0] for v in vals]
    summary.append({"mps_pct": pct, "admit": admit, "quantum": quantum, "n": len(nodes),
                    "node_ns_day_mean": round(statistics.mean(nodes)), "node_min": min(nodes), "node_max": max(nodes),
                    "cpu_cores_mean": round(statistics.mean(v[1] for v in vals), 1),
                    "jobs": "+".join(sorted({v[2] for v in vals}))})
summary.sort(key=lambda r: (PCT_ORDER.index(r["mps_pct"]), r["quantum"], ADMIT_ORDER.index(r["admit"])))
with open(RES / "p3_sweep_summary.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(summary[0]))
    w.writeheader()
    w.writerows(summary)

baseline = next(r for r in summary if r["mps_pct"] == "inherit" and r["admit"] == "all")["node_ns_day_mean"]
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#555555", "#e2e2e2"
COLORS = {"inherit": "#2a78d6", "50": "#eb6834", "25": "#1baf7a", "15": "#eda100", "10": "#e87ba4"}
LABEL = {"inherit": "MPS default", "50": "MPS 50 %", "25": "MPS 25 %", "15": "MPS 15 %", "10": "MPS 10 %"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
fig, ax = plt.subplots(figsize=(11.5, 6.6), dpi=120)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
x_of = {a: i for i, a in enumerate(ADMIT_ORDER)}
ax.axhline(baseline, color=MUTED, lw=1.5, ls=(0, (5, 4)), zorder=1)
ax.text(0.15, baseline - 90, f"today: all 59 admitted, {baseline:,}", ha="left", va="top",
        color=MUTED, fontsize=11)
label_y = {}
for pct in PCT_ORDER:
    pts = sorted((x_of[r["admit"]], r["node_ns_day_mean"]) for r in summary
                 if r["mps_pct"] == pct and r["quantum"] == 250)
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=COLORS[pct], lw=2, marker="o", ms=8, mec=SURFACE, mew=2, zorder=3)
    label_y[pct] = (xs[-1], ys[-1])
q = next(r for r in summary if r["mps_pct"] == "25" and r["admit"] == "8" and r["quantum"] == 50)
ax.plot([x_of["8"]], [q["node_ns_day_mean"]], marker="D", ms=9, color=COLORS["25"], mec=SURFACE, mew=2, zorder=4)
ax.annotate(f"50-step turns: {q['node_ns_day_mean']:,} (+{q['node_ns_day_mean'] / baseline - 1:.0%})",
            xy=(x_of["8"], q["node_ns_day_mean"]), xytext=(x_of["8"] - 1.55, q["node_ns_day_mean"] + 230),
            fontsize=11, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
# five series: identity via legend + markers; exact values in the report table (contrast relief)
ax.set_xticks(range(len(ADMIT_ORDER)))
ax.set_xticklabels(["all (59)", "32", "16", "8", "6", "4", "2"], color=MUTED)
ax.set_xlim(-0.3, len(ADMIT_ORDER) - 0.7)
ax.set_ylim(0, 5200)
ax.set_xlabel("Active replicas per GPU (all 59 stay resident)", color=MUTED)
ax.yaxis.grid(True, color=GRID, lw=1)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="y", colors=MUTED, length=0)
ax.tick_params(axis="x", length=0)
ax.set_title("Node throughput (ns/day), 236 contexts under MPS", loc="left", fontweight="bold", color=INK, pad=14)
handles = [plt.Line2D([], [], color=COLORS[p], lw=2, marker="o", ms=7) for p in PCT_ORDER]
ax.legend(handles, [LABEL[p] for p in PCT_ORDER], loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=5,
          frameon=False, fontsize=11)
fig.tight_layout()
fig.savefig(RES / "p3_sweep.png", facecolor=SURFACE)
print(f"baseline {baseline}; wrote {RES/'p3_sweep_summary.csv'} and {RES/'p3_sweep.png'}")
for r in summary:
    print(r)
