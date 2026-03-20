import numpy as np
from scipy.signal import lfilter
from scipy.optimize import minimize
import warnings

"""
Linear reservoir models for hydrologic response. These are simple, conceptual models that can capture the 
recession behavior of runoff. The linear reservoir assumes that the outflow is proportional to the storage, 
which leads to an exponential decay of flow over time. The two-reservoir model allows for a fast and slow component, 
representing different flow paths in the watershed like snowmelt/rainfall and groundwater.

Example usage:
x = linear_reservoir_daily(discharge.sel(time=slice("2010-10-01","2012-10-01")).resample(time='1D').mean().values, k=3.8e-2)
routed_df = pd.DataFrame({
    'time': discharge.resample(time='1D').sum()['time'].values,
    'routed_discharge': x.ravel()
})

results = calibrate_reservoirs(q_in=discharge.sel(time=slice("2005-10-01","2015-10-01")).resample(time='1D').mean().values, 
                               obs=obs_df['discharge_cms'].loc['2005-10-01':'2015-10-01'].resample('1D').mean().values, metric='nse')

print(f"k_fast : {results['k_fast']:.4f} day-1  "
      f"({results['residence_fast_days']:.1f} day residence time)")
print(f"k_slow : {results['k_slow']:.4f} day-1  "
      f"({results['residence_slow_days']:.1f} day residence time)")
print(f"f_fast : {results['f_fast']:.2f}")
print(f"NSE    : {results['nse']:.3f}")
print(f"KGE    : {results['kge']:.3f}")
print(f"logNSE : {results['log_nse']:.3f}")
"""

def linear_reservoir_daily(q_in, k):
    """
    Vectorized linear reservoir for daily timestep.
    
    q_in : daily flow array (any consistent units)
    k    : recession constant (day-1). Mean residence time = 1/k days.
    
    Update order: inflow first, then release fraction k.
    At steady state: q_out = q_in.
    """
    q = np.asarray(q_in, dtype=np.float64).ravel()
    q = np.nan_to_num(q, nan=0.0)

    # S[t] = (1-k)*S[t-1] + q[t]
    # q_out[t] = k * S[t]
    # --> S[t] = sum_{i=0}^{t} (1-k)^(t-i) * q[i]
    # --> this is a causal IIR filter: b=[1], a=[1, -(1-k)]
    # then q_out = k * S

    a = 1.0 - k
    S = lfilter([1.0], [1.0, -a], q)
    q_out = k * S

    return np.maximum(q_out, 0.0)

def two_reservoir_daily(q_in, k_fast, k_slow, f=0.7):
    """
    Two linear reservoirs in parallel.
    
    f        : fraction of q_in routed to the fast reservoir
    k_fast   : fast recession constant (day-1) — surface/near-surface
    k_slow   : slow recession constant (day-1) — groundwater-like
    """
    q = np.asarray(q_in, dtype=np.float64).ravel()
    q = np.nan_to_num(q, nan=0.0)

    q_fast = linear_reservoir_daily(f * q, k_fast)
    q_slow = linear_reservoir_daily((1 - f) * q, k_slow)

    return q_fast + q_slow

# ── reservoir functions ──────────────────────────────────────────────

def linear_reservoir_daily(q_in, k):
    q = np.asarray(q_in, dtype=np.float64).ravel()
    q = np.nan_to_num(q, nan=0.0)
    a = 1.0 - k
    S = lfilter([1.0], [1.0, -a], q)
    return np.maximum(k * S, 0.0)

def two_reservoir_daily(q_in, k_fast, k_slow, f):
    q = np.asarray(q_in, dtype=np.float64).ravel()
    q = np.nan_to_num(q, nan=0.0)
    return (linear_reservoir_daily(f * q, k_fast) +
            linear_reservoir_daily((1 - f) * q, k_slow))

# ── metrics ──────────────────────────────────────────────────────────

def nse(obs, sim):
    mask = np.isfinite(obs) & np.isfinite(sim)
    o, s = obs[mask], sim[mask]
    denom = np.sum((o - np.mean(o))**2)
    if denom == 0:
        return np.nan
    return 1.0 - np.sum((o - s)**2) / denom

def kge(obs, sim):
    mask = np.isfinite(obs) & np.isfinite(sim)
    o, s = obs[mask], sim[mask]
    r = np.corrcoef(o, s)[0, 1]
    alpha = np.std(s) / np.std(o)
    beta  = np.mean(s) / np.mean(o)
    return 1.0 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

def log_nse(obs, sim, eps=0.01):
    """NSE on log-transformed flows — emphasises low-flow / recession fit."""
    mask = np.isfinite(obs) & np.isfinite(sim) & (obs > 0) & (sim > 0)
    o = np.log(obs[mask] + eps)
    s = np.log(sim[mask] + eps)
    denom = np.sum((o - np.mean(o))**2)
    if denom == 0:
        return np.nan
    return 1.0 - np.sum((o - s)**2) / denom

# ── objective ────────────────────────────────────────────────────────

def objective(params, q_in, obs, metric='nse'):
    k_fast, k_slow, f = params

    # hard parameter bounds
    if (k_fast <= 0 or k_slow <= 0 or
        k_fast > 1 or k_slow > 1 or
        k_slow >= k_fast or          # slow must be slower than fast
        f <= 0 or f >= 1):
        return 1.0                   # infeasible — return worst possible score

    sim = two_reservoir_daily(q_in, k_fast, k_slow, f)

    metrics = {'nse': nse, 'kge': kge, 'log_nse': log_nse}
    score = metrics[metric](obs, sim)

    if not np.isfinite(score):
        return 1.0

    return -score   # minimise negative score

# ── optimizer ────────────────────────────────────────────────────────

def calibrate_reservoirs(q_in, obs, metric='nse',
                          n_starts=20, seed=42):
    """
    Calibrate two-reservoir model against observed daily streamflow.

    Parameters
    ----------
    q_in   : SUMMA simulated daily runoff array
    obs    : observed daily streamflow (same units, same length)
    metric : 'nse' | 'kge' | 'log_nse'
    n_starts : number of random starting points (multi-start)

    Returns
    -------
    dict with optimal parameters and diagnostic scores
    """
    rng = np.random.default_rng(seed)

    q_in = np.asarray(q_in, dtype=np.float64).ravel()
    obs  = np.asarray(obs,  dtype=np.float64).ravel()

    assert len(q_in) == len(obs), "q_in and obs must be the same length"

    best_result = None
    best_score  = np.inf

    # parameter bounds: [k_fast, k_slow, f]
    bounds = [(0.05, 0.99),   # k_fast: 1–20 day residence time
              (0.005, 0.10),  # k_slow: 10–200 day residence time
              (0.1,  0.9)]    # f: fast fraction

    for i in range(n_starts):
        # random starting point within bounds
        x0 = [rng.uniform(lo, hi) for lo, hi in bounds]
        # ensure k_slow < k_fast at initialisation
        x0[1] = min(x0[1], x0[0] * 0.5)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = minimize(
                objective,
                x0=x0,
                args=(q_in, obs, metric),
                method='Nelder-Mead',
                options={'xatol': 1e-6, 'fatol': 1e-6,
                         'maxiter': 10000, 'adaptive': True}
            )

        if result.fun < best_score:
            best_score  = result.fun
            best_result = result

    k_fast, k_slow, f = best_result.x
    sim = two_reservoir_daily(q_in, k_fast, k_slow, f)

    return {
        'k_fast':            k_fast,
        'k_slow':            k_slow,
        'f_fast':            f,
        'residence_fast_days': 1 / k_fast,
        'residence_slow_days': 1 / k_slow,
        'nse':               nse(obs, sim),
        'kge':               kge(obs, sim),
        'log_nse':           log_nse(obs, sim),
        'sim':               sim,
        'optimizer_success': best_result.success,
        'n_starts_used':     n_starts,
    }

# ── usage ─────────────────────────────────────────────────────────────

