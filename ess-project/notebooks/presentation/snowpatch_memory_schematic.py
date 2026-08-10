import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch
from matplotlib.lines import Line2D

FIGDIR = Path("/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/figures/draft")

INK   = "#14344a"          # valley wall colour from the slide
GREEN = "#2f6b34"
GRAY  = "#9aa3a8"
WARM  = "#c0392b"          # south-facing label
COOL  = "#1f6fa8"          # north-facing label

# SWE multipliers applied to the north-facing snow patch, shallow -> deep
MULT   = np.array([0.5, 1.0, 1.5, 2.0])
SNOWC  = ["#cfe3f2", "#9ec9e6", "#5fa3d0", "#2c6fa8"]   # one colour per multiplier

A = 1.30                                  # valley: y = A x^2
BAND = [0.42, 0.98]                       # elevation-band dividers
XMAX = 1.20


def surface(x):
    return A * x ** 2


def normal(x):
    """Unit outward normal to the valley surface at x (points away from the fill)."""
    dy = 2 * A * x
    n = np.array([-dy, 1.0])
    return n / np.linalg.norm(n)


def tangent(x):
    dy = 2 * A * x
    t = np.array([1.0, dy])
    return t / np.linalg.norm(t)


def tree(ax, x, y, h=0.22, w=0.085):
    """Three-tier conifer standing upright with its trunk base at (x, y)."""
    ax.plot([x, x], [y, y + 0.30 * h], color="#5b3a1e", lw=1.6, zorder=4,
            solid_capstyle="butt")
    for k, (top, bot, ww) in enumerate([(1.00, 0.62, 0.62), (0.74, 0.44, 0.82), (0.52, 0.24, 1.0)]):
        ax.add_patch(Polygon([[x, y + top * h],
                              [x - ww * w, y + bot * h],
                              [x + ww * w, y + bot * h]],
                             fc=GREEN, ec="white", lw=0.5, zorder=5))


fig = plt.figure(figsize=(9.5, 9.0))
gs = fig.add_gridspec(2, 1, height_ratios=[1.05, 0.85], hspace=0.28)

# ======================= top: valley cross-section =========================
ax = fig.add_subplot(gs[0])

xs = np.linspace(-XMAX, XMAX, 400)
ax.plot(xs, surface(xs), color=INK, lw=3.2, zorder=6, solid_capstyle="round")

for b, lab in zip(BAND, ["Elevation band $i$", "Elevation band $i+1$"]):
    ax.plot([-1.45, 1.45], [b, b], color=GRAY, ls=(0, (7, 6)), lw=1.6, zorder=1)
    ax.text(-1.00, b + 0.045, lab, ha="right", va="bottom", fontsize=10.5, color="#5b666d")

# trees: denser and taller on the shaded north-facing (right) wall
for xt in (-0.78, -0.68):
    tree(ax, xt, surface(xt), h=0.20)
for xt in (0.50, 0.60, 0.70, 0.82, 0.92):
    tree(ax, xt, surface(xt), h=0.24)

ax.text(-1.62, 1.55, "South\nfacing", color=WARM, fontsize=15, weight="bold",
        ha="center", va="center")
ax.text(1.72, 1.05, "North\nfacing", color=COOL, fontsize=15, weight="bold",
        ha="center", va="center")

# ---- the snow patch: fixed footprint, stacked thickness = the experiment ----
X0, HALF_LEN = 0.99, 0.19        # patch centre on the slope, half length along-slope
t0 = tangent(X0); n0 = normal(X0)
c0 = np.array([X0, surface(X0)])
base_a, base_b = c0 - HALF_LEN * t0, c0 + HALF_LEN * t0
THICK = 0.17                      # slab thickness for the 1.0x case

for m, col in zip(MULT[::-1], SNOWC[::-1]):        # deepest first, shallowest on top
    d = THICK * m
    ax.add_patch(Polygon([base_a, base_b, base_b + d * n0, base_a + d * n0],
                         fc=col, ec="white", lw=1.1, zorder=7))
ax.add_patch(Polygon([base_a, base_b, base_b + THICK * MULT[-1] * n0,
                      base_a + THICK * MULT[-1] * n0],
                     fc="none", ec=INK, lw=1.6, zorder=9))

# depth arrow, perpendicular to the slope, with the multiplier ticks
arr_base = base_b + 0.085 * t0
tip = arr_base + THICK * (MULT[-1] + 0.30) * n0
ax.add_patch(FancyArrowPatch(arr_base, tip, arrowstyle="-|>", mutation_scale=15,
                             lw=1.7, color=INK, zorder=10, shrinkA=0, shrinkB=0))
for m in MULT:
    p = arr_base + THICK * m * n0
    tick = 0.030 * t0
    ax.plot(*np.c_[p - tick, p + tick], color=INK, lw=1.5, zorder=10)
    # fan the labels up-slope so they clear each other on this steep wall
    lab = p + (0.07 + 0.115 * list(MULT).index(m)) * t0
    ax.plot(*np.c_[p + 0.02 * t0, lab - 0.02 * t0], color=GRAY, lw=0.7, zorder=9)
    ax.text(*lab, f"{m:g}×", fontsize=9.5, color=INK, ha="center", va="bottom", zorder=10)
ax.text(0.28, 1.92, "snow-patch depth\n(× baseline SWE)", fontsize=10.5, color=INK,
        ha="center", va="bottom", linespacing=1.25)
ax.annotate("", xy=tuple(tip + 0.03 * n0), xytext=(0.42, 1.92),
            arrowprops=dict(arrowstyle="-", color=INK, lw=1.0,
                            connectionstyle="arc3,rad=-0.15"), zorder=10)

ax.set_xlim(-1.95, 2.05); ax.set_ylim(-0.10, 2.45)
ax.set_aspect("equal"); ax.set_axis_off()
ax.set_title("Experiment: hold the patch footprint fixed, vary its depth",
             fontsize=12.5, color=INK, pad=6)

# ===================== bottom: expected memory signal ======================
axq = fig.add_subplot(gs[1])

t = np.arange(0, 214)                          # days from 1 Apr
peak, width = 58, 22
base = 6 + 42 * np.exp(-0.5 * ((t - peak) / width) ** 2) + 5 * np.exp(-t / 120)

for m, col in zip(MULT, SNOWC):
    extra = (m - 1.0)
    # deeper patch -> a later, longer melt contribution and a fatter recession
    contrib = extra * (14 * np.exp(-0.5 * ((t - (peak + 26)) / 30) ** 2)
                       + 9 * np.exp(-(t - 40).clip(0) / 46))
    axq.plot(t, base + contrib, color=col, lw=2.3,
             label=f"{m:g}× SWE", zorder=4 if m != 1.0 else 5)

# where the deepest run is still separated from baseline
deep = base + 1.0 * (14 * np.exp(-0.5 * ((t - (peak + 26)) / 30) ** 2)
                     + 9 * np.exp(-(t - 40).clip(0) / 46))
delta = deep - base
thresh = 0.05 * delta.max()
horizon = t[delta > thresh][-1]
axq.axvspan(0, horizon, color="#eef4f8", zorder=0)
axq.axvline(horizon, color=GRAY, ls=(0, (5, 4)), lw=1.4, zorder=2)
axq.annotate("memory horizon\n(traces reconverge)", xy=(horizon, 9),
             xytext=(horizon - 74, 24), fontsize=9.5, color="#5b666d", ha="center",
             arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.2))

axq.set_xlim(0, t[-1]); axq.set_ylim(0, 82)
ticks = [0, 30, 61, 91, 122, 153, 183, 213]
axq.set_xticks(ticks)
axq.set_xticklabels(["Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov"])
axq.set_ylabel("streamflow (m³ s⁻¹)", fontsize=10.5, color=INK)
axq.set_title("Expected signal: divergence through melt, then a decaying memory in the recession",
              fontsize=12.5, color=INK, pad=8)
axq.legend(frameon=False, fontsize=9.5, title="north-facing patch", title_fontsize=9.5,
           loc="upper left")
for s in ("top", "right"):
    axq.spines[s].set_visible(False)
axq.tick_params(labelsize=9.5, colors="#5b666d")

# inset: difference from the 1x baseline — the memory signal itself
axd = axq.inset_axes([0.56, 0.42, 0.40, 0.36])
for m, col in zip(MULT, SNOWC):
    extra = (m - 1.0)
    d = extra * (14 * np.exp(-0.5 * ((t - (peak + 26)) / 30) ** 2)
                 + 9 * np.exp(-(t - 40).clip(0) / 46))
    axd.plot(t, d, color=col, lw=1.6)
axd.axhline(0, color=GRAY, lw=0.9)
axd.axvline(horizon, color=GRAY, ls=(0, (4, 3)), lw=1.0)
axd.set_xlim(0, t[-1]); axd.set_xticks([]); axd.set_yticks([0])
axd.set_yticklabels(["0"], fontsize=8, color="#5b666d")
axd.set_title("Δ from 1× baseline", fontsize=8.5, color="#5b666d", pad=3)
for s in ("top", "right"):
    axd.spines[s].set_visible(False)

FIGDIR.mkdir(parents=True, exist_ok=True)
fig.savefig(FIGDIR / "snowpatch_memory_schematic.png", dpi=200, bbox_inches="tight",
            facecolor="white")
print("saved")
