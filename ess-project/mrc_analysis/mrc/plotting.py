"""Diagnostic figures and an optional interactive QC viewer.

All figure functions return a Matplotlib Figure and never call ``plt.show``;
the CLI saves them. The QC viewer (off by default) is a separate, explicitly
interactive entry point for manually time-shifting recessions in one window.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

import matplotlib
# No forced backend here: batch/CLI use sets "Agg" (see cli.py) while notebooks
# keep their inline backend. Matplotlib auto-selects Agg when no display exists.
import matplotlib.pyplot as plt

from .config import Config
from .mrc import MRCResult, TailFit
from .recession_id import Recession
from .temporal import TrendResult


def plot_ranked_recessions(recessions: List[Recession], tails: List[TailFit],
                           *, stagger: float = 8.0, title: str = "") -> plt.Figure:
    """Figure 1: ranked recession traces on semi-log, staggered by length.

    Recessions are sorted longest-first and offset horizontally by a fixed
    stagger so individual traces are legible. Conforming recessions are solid,
    non-conforming dashed/grey.
    """
    order = np.argsort([-r.length for r in recessions])
    fig, ax = plt.subplots(figsize=(9, 6))
    for rank, idx in enumerate(order):
        r = recessions[idx]
        tf = tails[idx]
        x = r.t + rank * stagger
        color = plt.cm.viridis(rank / max(len(order) - 1, 1))
        if tf.conforming:
            ax.semilogy(x, r.Q, "-", color=color, lw=1.2, alpha=0.9)
        else:
            ax.semilogy(x, r.Q, "--", color="0.6", lw=0.8, alpha=0.6)
    ax.set_xlabel("days since recession start (staggered)")
    ax.set_ylabel("Q (log scale)")
    ax.set_title(title or f"Ranked recession traces (n={len(recessions)})")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    return fig


def _start_label(r: Recession, fmt: str) -> str:
    if fmt == "date":
        return str(r.start.date())
    if fmt == "doy":
        return str(r.doy_start)
    if fmt == "doy_wy":
        return f"{r.doy_start} (WY{r.water_year})"
    if fmt == "seg":
        return str(r.seg_id)
    # "doy_year" (default): day-of-year with its CALENDAR year, since doy is
    # calendar-based (a doy-322 start is Nov of year Y, but water year Y+1).
    return f"{r.doy_start} ({r.start.year})"


def plot_matched_mrc(recessions: List[Recession], tails: List[TailFit],
                     result: MRCResult, *, title: str = "",
                     label_starts: bool = False, label_fmt: str = "doy_year",
                     label_fontsize: int = 7,
                     show_text: bool = False) -> plt.Figure:
    """Figure 2: matched master recession curve for one window.

    Each used recession's tail is drawn at its optimized time offset; the fitted
    master curve is overlaid, annotated with k, tau, RMSE, and counts.

    Set ``label_starts=True`` to tag the first point of each recession trace, so
    outlying traces can be traced back to when they started. ``label_fmt`` picks
    the text:

    - ``"doy_year"`` (default) -- ``"201 (1995)"``, day of year + calendar year
    - ``"doy"``      -- ``"201"``
    - ``"doy_wy"``   -- ``"201 (WY1995)"``
    - ``"date"``     -- ``"1995-07-20"``
    - ``"seg"``      -- ``"12"`` (segment id; cross-reference the metrics table)

    Labels are dense by construction -- these traces overlap -- so shrink
    ``label_fontsize`` or use ``"seg"`` when it gets unreadable.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    seg_by_id = {r.seg_id: (r, tf) for r, tf in zip(recessions, tails)}
    for offset, seg_id in zip(result.offsets, result.used_seg_ids):
        r, _ = seg_by_id[seg_id]
        # Redraw the tail portion consistent with the matching (last tail pts).
        ax.semilogy(r.t + offset, r.Q, "-", ms=3, alpha=0.3, color="0.4")
        if label_starts and r.t.size:
            ax.annotate(_start_label(r, label_fmt),
                        (r.t[0] + offset, r.Q[0]),
                        textcoords="offset points", xytext=(3, 3),
                        fontsize=label_fontsize, color="tab:blue", alpha=0.9,
                        clip_on=True)
    if result.master_t.size:
        ax.semilogy(result.master_t, result.master_Q, "-", color="crimson", lw=2.2,
                    label="master curve")
    txt = (f"k = {result.k:.4f} / day\n"
           f"tau = {result.tau:.1f} days\n"
           f"RMSE(days) = {result.rmse:.3f}\n"
           f"n used = {result.n_used} / conforming {result.n_conforming}")
    if show_text:
        ax.text(0.98, 0.95, txt, transform=ax.transAxes, ha="right", va="top",
                fontsize=9, bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax.set_xlabel("shifted time (days)")
    ax.set_ylabel("Q (log scale)")
    ax.set_title(title or "Matched master recession curve")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


def plot_mrc_overlay(entries, *, normalize: bool = True, title: str = "",
                     trace_alpha: float = 0.10, linestyles=("-", "--"),
                     colors=("black", "crimson"), figsize=(9, 6)) -> plt.Figure:
    """Overlay master recession curves from several runs (e.g. two basins).

    ``entries`` is a sequence of ``(label, recessions, tails, result)`` tuples.
    Each entry contributes very light grey background traces (its used recessions,
    at their matched offsets) plus one master curve, alternating through
    ``linestyles`` and ``colors``. The legend carries each master's k, tau, n, RMSE.

    With ``normalize`` (default) each curve is anchored at its own high-flow start:
    time is measured from that point and flow is scaled by Q0 there, so the curves
    are directly comparable on a semilog axis (the slope is -k) even when the two
    basins differ by orders of magnitude in discharge and each matching strip chose
    an arbitrary time origin. Set ``normalize=False`` to plot raw Q vs shifted time.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#f2f2f2")                       # grey base

    for i, (label, recessions, tails, result) in enumerate(entries):
        ls = linestyles[i % len(linestyles)]
        col = colors[i % len(colors)]
        if not result.master_t.size:
            continue
        t0 = float(result.master_t[0])
        q0 = float(result.master_Q[0]) if normalize else 1.0

        seg_by_id = {r.seg_id: r for r in recessions}
        for offset, seg_id in zip(result.offsets, result.used_seg_ids):
            r = seg_by_id[seg_id]
            x = r.t + offset - (t0 if normalize else 0.0)
            ax.semilogy(x, r.Q / q0, "-", color=col, lw=0.8, alpha=trace_alpha, zorder=1)

        mx = result.master_t - (t0 if normalize else 0.0)
        ax.semilogy(mx, result.master_Q / q0, ls, color=col, lw=2.4, zorder=3,
                    label=(f"{label}:  $\\tau$={int(result.tau)} d   "))

                    # label=(f"{label}:  $\\tau$={result.tau:.1f} d   "
                    #        f"n={result.n_used}   RMSE={result.rmse:.2f} d"))

    ax.set_xlabel("days since master-curve start" if normalize else "shifted time (days)")
    ax.set_ylabel("Q / Q$_0$" if normalize else "Q")
    ax.set_title(title or "Master recession curves")
    ax.grid(True, which="both", color="white", alpha=0.7, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    return fig


def plot_k_timeseries(window_df: pd.DataFrame, trend: Optional[TrendResult] = None,
                      *, title: str = "") -> plt.Figure:
    """Figure 3: master k vs time with Sen-slope trend line and a simple CI band.

    The CI band is a +/- 1 RMSE-of-residual envelope around the Sen line — a
    visual guide, not a rigorous interval (see the trend caveat).
    """
    df = window_df.dropna(subset=["k"]).sort_values("year_mid")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["year_mid"], df["k"], "o-", color="steelblue", ms=5, label="master k per window")

    if trend is not None and np.isfinite(trend.sen_slope) and len(df):
        x = df["year_mid"].to_numpy()
        yline = trend.sen_slope * x + trend.sen_intercept
        ax.plot(x, yline, "--", color="crimson", lw=1.8,
                label=f"Sen slope = {trend.sen_slope:.4g}/yr (p={trend.p_value:.3f})")
        resid = df["k"].to_numpy() - yline
        band = np.nanstd(resid)
        ax.fill_between(x, yline - band, yline + band, color="crimson", alpha=0.12,
                        label="+/-1 sigma residual")

    ax.set_xlabel("window mid-year")
    ax.set_ylabel("master recession constant k (1/day)")
    ax.set_title(title or "Stationarity of the master recession constant")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    if trend is not None and trend.caveat:
        fig.text(0.01, 0.005, "Caveat: " + trend.caveat, fontsize=6.5,
                 ha="left", va="bottom", wrap=True, color="0.35")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def save_standard_figures(outdir: Path, recessions, tails, full_result,
                          window_df, trend, cfg: Config) -> List[Path]:
    """Render and save the three standard figures; return their paths."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fmt = cfg.output.figure_format
    dpi = cfg.output.figure_dpi
    paths = []

    f1 = plot_ranked_recessions(recessions, tails)
    p1 = outdir / f"fig1_ranked_recessions.{fmt}"
    f1.savefig(p1, dpi=dpi); plt.close(f1); paths.append(p1)

    f2 = plot_matched_mrc(recessions, tails, full_result)
    p2 = outdir / f"fig2_matched_mrc.{fmt}"
    f2.savefig(p2, dpi=dpi); plt.close(f2); paths.append(p2)

    f3 = plot_k_timeseries(window_df, trend)
    p3 = outdir / f"fig3_k_vs_time.{fmt}"
    f3.savefig(p3, dpi=dpi); plt.close(f3); paths.append(p3)
    return paths


# --------------------------------------------------------------------------- #
# Brutsaert-Nieber (-dQ/dt vs Q)
# --------------------------------------------------------------------------- #
def plot_bn(result, *, title: str = "") -> plt.Figure:
    """Panel A: the -dQ/dt vs Q cloud with the fitted lower envelope and a b=1
    (linear-reservoir) reference line -- an independent linearity check."""
    fig, ax = plt.subplots(figsize=(8, 6))
    c = result.cloud
    ax.scatter(c["Q"], c["minus_dQdt"], s=6, alpha=0.15, color="0.5", label="recession days")
    if len(result.envelope):
        ax.scatter(result.envelope["Q"], result.envelope["minus_dQdt"], s=28,
                   color="tab:blue", edgecolor="k", zorder=5, label="lower envelope")
    f = result.overall
    qq = np.array([c["Q"].min(), c["Q"].max()])
    if np.isfinite(f.b):
        ax.plot(qq, f.a * qq ** f.b, "-", color="crimson", lw=2,
                label=f"fit: b={f.b:.2f}, τ={f.tau_ref:.0f} d @ Q={f.q_ref:.2g}")
        # b=1 reference anchored at the fit's value at q_ref (a1 = a * q_ref^(b-1)).
        a1 = f.a * f.q_ref ** (f.b - 1.0)
        ax.plot(qq, a1 * qq, "--", color="tab:blue", lw=1.6, label="b = 1 (linear reservoir)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Q"); ax.set_ylabel("-dQ/dt")
    ax.set_title(title or "Brutsaert-Nieber recession analysis")
    ax.grid(True, which="both", alpha=0.2); ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def plot_bn_split(result, *, title: str = "") -> plt.Figure:
    """Panel B: the cloud split by wetness state, with a per-group envelope fit.
    Separated envelopes (same b, different intercept) = effective τ shifts with
    storage state; overlapping envelopes = τ stationary w.r.t. wetness."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"wet": "tab:blue", "dry": "tab:orange", "mid": "0.5"}
    cycle = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    if "group" not in result.cloud.columns:
        ax.text(0.5, 0.5, "wetness split disabled (bn.wetness.enabled=false)",
                ha="center", va="center", transform=ax.transAxes)
        return fig
    for i, (label, sub) in enumerate(result.cloud.groupby("group")):
        if label == "unknown":
            continue
        col = colors.get(label, cycle[i % len(cycle)])
        ax.scatter(sub["Q"], sub["minus_dQdt"], s=6, alpha=0.15, color=col)
        fit = result.groups.get(label)
        if fit and np.isfinite(fit.b):
            qq = np.array([sub["Q"].min(), sub["Q"].max()])
            ax.plot(qq, fit.a * qq ** fit.b, "-", color=col, lw=2.2,
                    label=f"{label}: b={fit.b:.2f}, τ={fit.tau_ref:.0f} d (n={fit.n_cloud})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Q"); ax.set_ylabel("-dQ/dt")
    ax.set_title(title or "Brutsaert-Nieber by wetness state (drought test)")
    ax.grid(True, which="both", alpha=0.2); ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Optional interactive QC viewer (off by default)
# --------------------------------------------------------------------------- #
def launch_qc_viewer(recessions: List[Recession], tails: List[TailFit],
                     result: MRCResult, cfg: Config) -> None:
    """Interactive viewer to manually time-shift recessions for QC.

    Matplotlib backend: sliders adjust each used recession's offset live and
    redraw. QC only — adjustments are not persisted. Requires an interactive
    matplotlib backend (call outside headless environments).
    """
    backend = cfg.output.qc_viewer.backend
    if backend == "plotly":
        _qc_viewer_plotly(recessions, tails, result)
        return

    import matplotlib
    matplotlib.use("TkAgg", force=True)  # needs a display
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    seg_by_id = {r.seg_id: r for r in recessions}
    used = [(sid, off) for sid, off in zip(result.used_seg_ids, result.offsets)]
    used = used[:8]  # cap sliders for usability

    fig, ax = plt.subplots(figsize=(10, 7))
    plt.subplots_adjust(bottom=0.05 + 0.03 * len(used))
    lines = {}
    for sid, off in used:
        r = seg_by_id[sid]
        (ln,) = ax.semilogy(r.t + off, r.Q, ".-", ms=3, label=f"seg {sid}")
        lines[sid] = ln
    if result.master_t.size:
        ax.semilogy(result.master_t, result.master_Q, "-", color="crimson", lw=2)
    ax.set_xlabel("shifted time (days)"); ax.set_ylabel("Q (log)")
    ax.set_title("QC viewer — drag sliders to time-shift recessions (not saved)")

    sliders = []
    for i, (sid, off) in enumerate(used):
        axs = plt.axes([0.15, 0.02 + 0.03 * i, 0.7, 0.02])
        s = Slider(axs, f"seg {sid}", -200, 200, valinit=float(off))
        def _update(val, sid=sid):
            r = seg_by_id[sid]
            lines[sid].set_xdata(r.t + sliders_by_id[sid].val)
            fig.canvas.draw_idle()
        s.on_changed(_update)
        sliders.append(s)
    sliders_by_id = {sid: s for (sid, _), s in zip(used, sliders)}
    plt.show()


def _qc_viewer_plotly(recessions, tails, result) -> None:  # pragma: no cover
    """Plotly fallback: writes an interactive HTML with per-recession sliders."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError("plotly backend requested but plotly is not installed") from exc
    seg_by_id = {r.seg_id: r for r in recessions}
    fig = go.Figure()
    for sid, off in zip(result.used_seg_ids, result.offsets):
        r = seg_by_id[sid]
        fig.add_trace(go.Scatter(x=r.t + off, y=r.Q, mode="markers+lines", name=f"seg {sid}"))
    if result.master_t.size:
        fig.add_trace(go.Scatter(x=result.master_t, y=result.master_Q,
                                 mode="lines", name="master", line=dict(color="crimson")))
    fig.update_yaxes(type="log")
    fig.update_layout(title="MRC QC viewer (plotly)", xaxis_title="shifted time (days)",
                      yaxis_title="Q")
    out = Path("mrc_qc_viewer.html")
    fig.write_html(str(out))
    print(f"Wrote interactive QC viewer to {out.resolve()}")


def _shade(hex_color: str, amount: float) -> tuple:
    """Blend a hex colour toward white (amount>0) or black (amount<0). |amount| in [0,1]."""
    import matplotlib.colors as mcolors
    r, g, b = mcolors.to_rgb(hex_color)
    if amount >= 0:
        return tuple(c + (1.0 - c) * amount for c in (r, g, b))
    return tuple(c * (1.0 + amount) for c in (r, g, b))


def plot_mrc_facet(groups, *, normalize: bool = True, trace_alpha: float = 0.07,
                   markers=("o", "s", "^"), shades=(-0.35, 0.05, 0.38),
                   figsize=(9.0, 8.4), golden: float = 1.618,
                   clip_to_reference: bool = True, ylim=(0.01, 3.0),
                   tau_panel: bool = True, tau_shared: bool = False,
                   basin_labels: bool = True, marker_every: int = 14,
                   ms: float = 6.0, marker_scale=None, tau_width: float = 0.20
                   ) -> plt.Figure:
    """Master recession curves, one ROW per basin, with a narrow tau panel beside each.

    Layout: groups are stacked vertically (first group on top), each row being a
    recession panel plus a slim tau strip on its right. The strip's y axis is tau in
    days and all of a row's markers sit at the SAME x, so near-identical timescales
    stack instead of being nudged apart -- the overlap is the message.

    The master curve is drawn as POINTS, not a line: it is a fitted curve through
    assembled recession tails, so discrete markers state what it is without implying a
    continuously observed trajectory. Faint background lines are the individual
    recessions that were assembled into it.

    ``groups`` is a sequence of ``(group_label, base_colour, entries)`` where ``entries``
    is the ``(label, recessions, tails, result)`` sequence ``plot_mrc_overlay`` takes; the
    FIRST entry is treated as the reference (observed). ``shades`` blends the base colour
    toward black (negative) or white (positive), one step per entry, so each row keeps
    its basin hue while the variant reads off lightness plus marker shape.

    ``clip_to_reference`` truncates every curve in a group at the reference curve's
    lowest normalised flow, so model curves are not drawn past the range the observations
    cover. ``tau_shared`` puts every row's tau on one scale (comparable across basins);
    the default gives each row its own tight range, which separates near-equal values.

    Legends, titles and value annotations are intentionally omitted -- add them at figure
    level so they can be positioned freely.
    """
    n = len(groups)
    # A filled triangle of a given `ms` reads smaller than a circle or square: markersize
    # is the bounding-box edge and the triangle covers about half of it.
    scale = {"^": 1.35, "v": 1.35, "o": 1.0, "s": 0.95}
    if marker_scale:
        scale.update(marker_scale)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(n, 2 if tau_panel else 1,
                          width_ratios=[1.0, tau_width] if tau_panel else [1.0],
                          hspace=0.14, wspace=0.05)
    axes, strips, curve_end = [], [], []

    for gi, (glabel, base_col, entries) in enumerate(groups):
        ax = fig.add_subplot(gs[gi, 0])
        ax.set_facecolor("#f2f2f2")
        ax.set_box_aspect(1.0 / golden)

        floor = -np.inf
        if clip_to_reference and entries:
            _l, _rs, _tl, r0 = entries[0]
            if r0.master_Q.size:
                q0r = float(r0.master_Q[0]) if normalize else 1.0
                floor = float(np.min(r0.master_Q / q0r))

        ends = []
        for i, (label, recessions, tails, result) in enumerate(entries):
            if not result.master_t.size:
                ends.append(np.nan)
                continue
            col = _shade(base_col, shades[i % len(shades)])
            t0 = float(result.master_t[0])
            q0 = float(result.master_Q[0]) if normalize else 1.0

            seg_by_id = {r.seg_id: r for r in recessions}
            for offset, seg_id in zip(result.offsets, result.used_seg_ids):
                r = seg_by_id[seg_id]
                x = r.t + offset - (t0 if normalize else 0.0)
                y = r.Q / q0
                keep = y >= floor
                ax.semilogy(x[keep], y[keep], "-", color=col, lw=0.7,
                            alpha=trace_alpha, zorder=1)

            mx = result.master_t - (t0 if normalize else 0.0)
            my = result.master_Q / q0
            keep = my >= floor
            mx, my = mx[keep], my[keep]
            step = max(1, int(round(mx.size / max(marker_every, 1))))
            mk = markers[i % len(markers)]
            ax.semilogy(mx[::step], my[::step], mk, ls="none", color=col,
                        ms=ms * scale.get(mk, 1.0), mfc=col, mec="white", mew=0.9,
                        zorder=4 - i * 0.1)
            ends.append(float(result.tau))
        curve_end.append(ends)

        ax.grid(True, which="both", color="white", alpha=0.75, lw=0.8)
        ax.set_axisbelow(True)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_ylabel("log(Q) (m$^3$/s) " if normalize else "log(Q) (m$^3$/s) ")
        if gi < n - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Days since recession start")
        if basin_labels:
            ax.set_title(glabel, loc="left", fontweight="bold")
        axes.append(ax)

    for ax in axes[1:]:                       # one Q/Q0 scale and one day axis for all rows
        ax.sharey(axes[0]); ax.sharex(axes[0])

    if not tau_panel:
        fig.tight_layout()
        return fig

    all_tau = [t for row in curve_end for t in row if np.isfinite(t)]
    lo_all, hi_all = min(all_tau), max(all_tau)
    for gi, (glabel, base_col, entries) in enumerate(groups):
        tax = fig.add_subplot(gs[gi, 1])
        tax.set_facecolor("#f2f2f2")
        taus = curve_end[gi]
        for i, tv in enumerate(taus):
            if not np.isfinite(tv):
                continue
            col = _shade(base_col, shades[i % len(shades)])
            mk = markers[i % len(markers)]
            tax.plot(0.0, tv, mk, ms=(ms + 4.0) * scale.get(mk, 1.0), color=col,
                     mfc=col, mec="white", mew=1.2, zorder=4)
        good = [t for t in taus if np.isfinite(t)]
        if tau_shared:
            lo, hi = lo_all, hi_all
        else:
            lo, hi = min(good), max(good)
        pad = max((hi - lo) * 0.10, 2.0)
        tax.set_xlim(-1, 1)
        tax.set_ylim(lo - pad, hi + pad)
        tax.set_xticks([])
        tax.set_ylabel(r"Recession Timescale, $1/k$"+"\n(days)")
        tax.yaxis.set_label_position("right")
        tax.yaxis.tick_right()
        tax.grid(True, axis="y", color="white", alpha=0.75, lw=0.8)
        tax.set_axisbelow(True)
        for s in ("top", "left", "bottom"):
            tax.spines[s].set_visible(False)
        strips.append(tax)

    fig.tight_layout()
    return fig
