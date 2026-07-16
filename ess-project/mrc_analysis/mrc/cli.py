"""End-to-end orchestrator for the MRC pipeline.

Usage
-----
    python -m mrc.cli run --config config/default.yaml --output-dir outputs
    python -m mrc.cli alpha --config config/default.yaml   # alpha-sensitivity only

The ``run`` command executes the full pipeline reproducibly: it seeds the RNG,
snapshots the resolved config to the output directory, loads (or synthesizes)
data, screens recessions, builds the full-record and windowed MRCs, runs the
trend test, and writes tidy tables + figures.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # batch/headless: render figures without a display

import numpy as np
import pandas as pd

from .config import Config, load_config
from . import io as mio
from . import filter as mfilter
from . import recession_id as rid
from . import mrc as mmrc
from . import temporal as mtemporal
from . import bn as mbn
from . import plotting as mplot

log = logging.getLogger("mrc")


# --------------------------------------------------------------------------- #
# Setup helpers
# --------------------------------------------------------------------------- #
def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _output_dir(cfg: Config) -> Path:
    out = Path(cfg.run.output_dir) / cfg.run.label
    out.mkdir(parents=True, exist_ok=True)
    return out


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def run_pipeline(cfg: Config, outdir: Optional[Path] = None) -> dict:
    """Execute the full pipeline; return a dict of result objects/tables."""
    np.random.seed(cfg.run.seed)
    outdir = Path(outdir) if outdir is not None else _output_dir(cfg)

    # Snapshot the resolved config for reproducibility.
    cfg.dump_yaml(outdir / "config_used.yaml")
    log.info("Resolved config snapshot -> %s", outdir / "config_used.yaml")

    # --- Load data -------------------------------------------------------- #
    flow = mio.load_streamflow(cfg)
    forcing = mio.load_forcing(cfg)
    df = mio.merge_forcing(flow, forcing)
    log.info("Loaded record: %s to %s (%d days), forcing=%s",
             df.index.min().date(), df.index.max().date(), len(df),
             forcing is not None)

    # --- Gaps ------------------------------------------------------------- #
    gaps = mio.detect_gaps(df, "Q")
    gap_df = mio.summarize_gaps(gaps)
    if cfg.preprocess.gap_report:
        log.info("Detected %d gaps (%d missing days total)",
                 len(gaps), int(gap_df["length_days"].sum()) if len(gap_df) else 0)

    # --- Baseflow separation --------------------------------------------- #
    df = mfilter.add_baseflow_columns(df, cfg)

    # --- Flag report ------------------------------------------------------ #
    _report_flags(df, cfg)

    # --- Dynamic (SWE-meltout) recession season -------------------------- #
    season_df = None
    if getattr(cfg.recession.season, "mode", "fixed") == "swe_meltout":
        season_df = rid.season_starts_frame(cfg)
        n_fb = int(season_df["used_fallback"].sum())
        log.info("SWE-meltout season: %d years (%d fallback), mean start doy=%.0f "
                 "(meltout + %d d)", len(season_df), n_fb,
                 season_df["doy_start"].mean(), cfg.recession.season.meltout.buffer_days)

    # --- Recession identification ---------------------------------------- #
    recessions = rid.identify_recessions(df, cfg)
    log.info("Identified %d recession segments", len(recessions))

    # --- Tail fits + full-record MRC ------------------------------------- #
    full_result, tails = mmrc.build_mrc(recessions, cfg)
    rec_metrics = mmrc.recession_metrics_frame(recessions, tails)
    log.info("Full-record master: k=%.4f tau=%.1f rmse=%.3f (used %d/%d conforming)",
             full_result.k, full_result.tau, full_result.rmse,
             full_result.n_used, full_result.n_conforming)

    # --- Alpha sensitivity ----------------------------------------------- #
    alpha_df = rid.alpha_sensitivity(df, cfg)

    # --- Temporal / stationarity ----------------------------------------- #
    window_df = mtemporal.window_mrc_series(recessions, cfg)
    per_year_df = mtemporal.per_year_mrc_series(recessions, cfg)
    trend = mtemporal.trend_on_windows(window_df, cfg)
    log.info("Trend on windowed k: %s (Sen slope=%.4g/yr, p=%.3f, n=%d)",
             trend.trend, trend.sen_slope, trend.p_value, trend.n)
    if trend.caveat:
        log.warning("Trend caveat: %s", trend.caveat)

    # --- Brutsaert-Nieber (-dQ/dt vs Q) linearity diagnostic ------------- #
    bn_result = None
    if getattr(cfg, "bn", None) is not None and cfg.bn.enabled:
        bn_result = mbn.bn_analysis(recessions, cfg, df=df)
        f = bn_result.overall
        log.info("Brutsaert-Nieber: b=%.3f (1=linear reservoir), tau=%.1f d @ Q=%.3g, "
                 "r2=%.3f, %d cloud pts", f.b, f.tau_ref, f.q_ref, f.r2, f.n_cloud)
        if bn_result.groups:
            g = ", ".join(f"{k}: b={v.b:.2f} tau={v.tau_ref:.0f}d"
                          for k, v in bn_result.groups.items())
            log.info("Brutsaert-Nieber by wetness -> %s", g)

    results = {
        "df": df, "gaps": gap_df, "recessions": recessions, "tails": tails,
        "recession_metrics": rec_metrics, "full_result": full_result,
        "alpha_sensitivity": alpha_df, "window_mrc": window_df,
        "per_year_mrc": per_year_df, "trend": trend, "season_starts": season_df,
        "bn": bn_result,
    }

    # --- Write outputs ---------------------------------------------------- #
    if cfg.output.write_tables:
        _write_tables(outdir, results)
    if cfg.output.write_figures:
        paths = mplot.save_standard_figures(
            outdir / "figures", recessions, tails, full_result, window_df, trend, cfg)
        if bn_result is not None:
            figdir = outdir / "figures"
            fmt, dpi = cfg.output.figure_format, cfg.output.figure_dpi
            fb = mplot.plot_bn(bn_result); pb = figdir / f"fig4_brutsaert_nieber.{fmt}"
            fb.savefig(pb, dpi=dpi); mplot.plt.close(fb); paths.append(pb)
            if bn_result.groups:
                fs = mplot.plot_bn_split(bn_result); ps = figdir / f"fig5_bn_wetness_split.{fmt}"
                fs.savefig(ps, dpi=dpi); mplot.plt.close(fs); paths.append(ps)
        log.info("Wrote %d figures -> %s", len(paths), outdir / "figures")

    # --- Optional QC viewer ---------------------------------------------- #
    if cfg.output.qc_viewer.enabled:
        log.info("Launching QC viewer (interactive)")
        mplot.launch_qc_viewer(recessions, tails, full_result, cfg)

    return results


def _report_flags(df: pd.DataFrame, cfg: Config) -> None:
    if not cfg.flags.report_affected or "flag" not in df.columns:
        return
    flags = cfg.flags
    col = df["flag"].astype("string").fillna("")
    ice = np.zeros(len(df), dtype=bool)
    for code in flags.ice_affected:
        ice |= col.str.contains(code, case=False, na=False).to_numpy()
    est = np.zeros(len(df), dtype=bool)
    for code in flags.estimated:
        est |= col.str.contains(code, case=False, na=False).to_numpy()
    log.info("Qualification flags: %d ice-affected, %d estimated days "
             "(drop_ice=%s, drop_est=%s)",
             int(ice.sum()), int(est.sum()),
             flags.drop_ice_affected, flags.drop_estimated)


def _write_tables(outdir: Path, results: dict) -> None:
    tdir = outdir / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    results["recession_metrics"].to_csv(tdir / "per_recession_metrics.csv", index=False)
    results["window_mrc"].to_csv(tdir / "per_window_mrc.csv", index=False)
    results["per_year_mrc"].to_csv(tdir / "per_year_mrc.csv", index=False)
    results["alpha_sensitivity"].to_csv(tdir / "alpha_sensitivity.csv", index=False)
    results["gaps"].to_csv(tdir / "gaps.csv", index=False)
    pd.DataFrame([results["trend"].as_dict()]).to_csv(tdir / "trend_test.csv", index=False)
    if results.get("season_starts") is not None:
        results["season_starts"].to_csv(tdir / "season_starts.csv", index=False)
    if results.get("bn") is not None:
        mbn.fits_frame(results["bn"]).to_csv(tdir / "bn_fits.csv", index=False)
        results["bn"].cloud.to_csv(tdir / "bn_cloud.csv", index=False)
    log.info("Wrote tidy tables -> %s", tdir)


# --------------------------------------------------------------------------- #
# CLI entry
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mrc", description="Master recession curve analysis")
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, default=None, help="YAML config path")
    common.add_argument("--output-dir", type=Path, default=None, help="override run.output_dir")
    common.add_argument("--label", type=str, default=None, help="override run.label")
    common.add_argument("--log-level", type=str, default=None, help="override run.log_level")

    sub.add_parser("run", parents=[common], help="run the full pipeline")
    sub.add_parser("alpha", parents=[common], help="alpha-sensitivity of recession selection only")
    return p


def _apply_overrides(cfg: Config, args) -> Config:
    ov = {"run": {}}
    if args.output_dir is not None:
        ov["run"]["output_dir"] = str(args.output_dir)
    if args.label is not None:
        ov["run"]["label"] = args.label
    if args.log_level is not None:
        ov["run"]["log_level"] = args.log_level
    if ov["run"]:
        return load_config(args.config, overrides=ov)
    return cfg


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    cfg = _apply_overrides(cfg, args)
    _setup_logging(cfg.run.log_level)

    if args.command == "run":
        run_pipeline(cfg)
    elif args.command == "alpha":
        flow = mio.load_streamflow(cfg)
        forcing = mio.load_forcing(cfg)
        df = mio.merge_forcing(flow, forcing)
        df = mfilter.add_baseflow_columns(df, cfg)
        table = rid.alpha_sensitivity(df, cfg)
        print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
