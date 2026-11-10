"""Daily cryospheric streamflow contribution of Lyell (LBM) from the whole season (May 1 - Oct 31), reference = Dana Fork (DFkBC; T120 shown as a non-independent check).
s0 = ratio of cumulative specific discharge Lyell/ref over May 1 -> date of the reference's 7-d peak (both snowmelt-driven). Daily CSC(t) = q_L - s0*q_ref (mm/d), fraction f(t) = CSC/q_L (7-d smoothed).
Reference transition = first day after its peak that its 7-d mean < 10% of the peak (snowmelt wave over). Post-transition Lyell volume = sum q_L after that date.
-> figures/draft/tuo_dmc_daily_csc.png, tuo_dmc_daily_csc_table.csv, tuo_dmc_daily_csc_years.png"""
import warnings; warnings.filterwarnings("ignore")
exec(open(__file__.replace("tuo_dmc_daily_csc", "tuo_dmc_csc_multi")).read().split("def run")[0])
FR = 0.10; rows, ser = [], {}
for ref in ["DFkBC", "T120"]:
    for y in range(2005, 2022):
        w = pd.DataFrame({"G": Q["LBM"], "N": Q[ref]}).loc[f"{y}-05-01":f"{y}-10-31"]
        ok = w.notna().all(axis=1)
        if len(w) < 150 or (~ok).sum() > 12 or not ok.iloc[:30].all() or ok.iloc[-45:-14].sum() < 30: continue
        w = w.interpolate(limit=10).dropna(); n7 = w.N.rolling(7, center=True, min_periods=4).mean(); pk = n7.idxmax(); pv = n7.max()
        after = n7.loc[pk:]; tr = after.index[(after < FR*pv).values][0] if (after < FR*pv).any() else None
        s0 = w.G.loc[:pk].sum()/w.N.loc[:pk].sum(); csc = w.G-s0*w.N; f = (csc.rolling(7, center=True, min_periods=4).mean()/w.G.rolling(7, center=True, min_periods=4).mean()).clip(upper=1)
        r = dict(ref=ref, year=y, mega=y in EXC, peak_doy=pk.dayofyear, peak_ref_mm=pv, s0=s0)
        if tr is not None:
            post = slice(tr, None); r.update(trans_doy=tr.dayofyear, lyell_post_mm=w.G.loc[post].sum(), lyell_post_mean=w.G.loc[post].mean(), lyell_total_mm=w.G.sum(), post_frac=w.G.loc[post].sum()/w.G.sum(), csc_season_mm=csc.sum(), csc_season_frac=csc.sum()/w.G.sum(), ref_post_mm=w.N.loc[post].sum()*s0, csc_post_frac=csc.loc[post].sum()/w.G.loc[post].sum())
        rows.append(r); ser[(ref, y)] = (w, csc, f, tr, pk)
T = pd.DataFrame(rows); T.to_csv(D+"tuo_dmc_daily_csc_table.csv", index=False); pd.set_option("display.width", 200)
print(T[T.ref == "DFkBC"][["year", "peak_doy", "trans_doy", "s0", "lyell_post_mm", "lyell_post_mean", "post_frac", "csc_season_frac", "csc_post_frac"]].round(2).to_string(index=False))
print(T[T.ref == "T120"][["year", "peak_doy", "trans_doy", "s0", "lyell_post_mm", "post_frac", "csc_season_frac", "csc_post_frac"]].round(2).to_string(index=False))
for ref in ["DFkBC", "T120"]:
    t = T[T.ref == ref]; t2 = t[~t.mega]; print(ref, "all: n", len(t), "trans_doy med", t.trans_doy.median(), "csc_season_frac med", round(t.csc_season_frac.median(), 2), "| excl 2011/2017: n", len(t2), "trans_doy med", t2.trans_doy.median(), "csc_season_frac med", round(t2.csc_season_frac.median(), 2), "range", round(t2.csc_season_frac.min(), 2), round(t2.csc_season_frac.max(), 2))
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
ys = sorted(set(T[T.ref == "DFkBC"].year)); fig, ax = plt.subplots(2, 5, figsize=(17, 6.6)); ax = ax.ravel()
for a, y in zip(ax, ys):
    w, csc, f, tr, pk = ser[("DFkBC", y)]; d = w.index.dayofyear; a.semilogy(d, w.G.rolling(3, center=True, min_periods=1).mean(), color="#2a78d6", label="Lyell"); a.semilogy(d, w.N.rolling(3, center=True, min_periods=1).mean(), color="#d55e00", label="Dana"); a.semilogy(d, csc.clip(lower=1e-3).rolling(3, center=True, min_periods=1).mean(), color="#009e73", lw=1.2, label="Lyell CSC")
    if tr is not None: a.axvline(tr.dayofyear, color="#555", ls="--", lw=.8)
    a.axvline(pk.dayofyear, color="#999", ls=":", lw=.8); a.set_ylim(0.02, 30); a.set_title(f"{y}{' (mega)' if y in EXC else ''}", fontsize=8, loc="left"); a.grid(color="#e3e2de", lw=.5, which="both")
ax[0].legend(fontsize=6); fig.suptitle("Daily specific discharge (mm/d): Lyell, Dana, Lyell CSC = Lyell − s0·Dana (dotted: Dana peak, dashed: Dana snowmelt-wave end)"); fig.tight_layout(); fig.savefig(D+"tuo_dmc_daily_csc.png", dpi=125, bbox_inches="tight")
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8)); t = T[T.ref == "DFkBC"]
for yy, (m, c) in {"all": (t, "#2a78d6")}.items():
    ax[0].plot(t.year, t.trans_doy, "o-", color="#2a78d6", label="Dana"); tt = T[T.ref == "T120"]; ax[0].plot(tt.year, tt.trans_doy, "s--", color="#999", label="T120"); ax[1].plot(t.year, t.lyell_post_mm, "o-", color="#2a78d6"); ax[2].plot(t.year, t.csc_season_frac, "o-", color="#009e73")
for a, tl in zip(ax, ["reference snowmelt-wave end (doy)", "Lyell volume after transition (mm)", "season CSC fraction of Lyell flow"]):
    a.set_title(tl, loc="left"); a.grid(color="#e3e2de", lw=.5)
    for y in EXC: a.axvline(y, color="#d55e00", alpha=.25, lw=6)
ax[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(D+"tuo_dmc_daily_csc_years.png", dpi=125, bbox_inches="tight")
