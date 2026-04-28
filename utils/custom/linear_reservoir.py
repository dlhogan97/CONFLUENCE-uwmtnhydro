import numpy as np
from scipy.signal import lfilter
from scipy.optimize import minimize
import warnings
from types import SimpleNamespace

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

def log_kge(obs, sim, eps=0.01):
    """KGE on log-transformed flows — emphasises low-flow / recession fit."""
    mask = np.isfinite(obs) & np.isfinite(sim) & (obs > 0) & (sim > 0)
    o = np.log(obs[mask] + eps)
    s = np.log(sim[mask] + eps)
    if len(o) < 10 or np.std(o) == 0:
        return np.nan
    r = np.corrcoef(o, s)[0, 1]
    alpha = np.std(s) / np.std(o)
    beta  = np.mean(s) / np.mean(o)
    return 1.0 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

def balanced(obs, sim, w_kge=0.25, w_nse=0.25, w_kge_log=0.25, w_nse_log=0.25):
    """Composite metric balancing high-flow skill (KGE, NSE) and recession skill (log-KGE, log-NSE).

    Weights default to equal (0.25 each). Adjust via calibrate_reservoirs(metric_weights=...).
    Returns the weighted average; components that return NaN are excluded and weights renormalized.
    """
    components = [
        (kge,     w_kge),
        (nse,     w_nse),
        (log_kge, w_kge_log),
        (log_nse, w_nse_log),
    ]
    total_w, total_score = 0.0, 0.0
    for fn, w in components:
        v = fn(obs, sim)
        if np.isfinite(v):
            total_score += w * v
            total_w += w
    return total_score / total_w if total_w > 0 else np.nan

# ── objective ────────────────────────────────────────────────────────

def objective(params, q_in, obs, metric='nse', bounds=None, metric_weights=None):
    k_fast, k_slow, f = params

    # Hard-penalize infeasible values so even unconstrained methods reject them.
    if bounds is not None:
        for value, (lower, upper) in zip((k_fast, k_slow, f), bounds):
            if value < lower or value > upper:
                return 1e6
    if k_slow >= k_fast:  # slow reservoir must remain slower than fast reservoir
        return 1e6

    sim = two_reservoir_daily(q_in, k_fast, k_slow, f)

    if metric == 'balanced':
        w = metric_weights or {}
        score = balanced(obs, sim, **w)
    else:
        metrics = {'nse': nse, 'kge': kge, 'log_nse': log_nse, 'log_kge': log_kge}
        score = metrics[metric](obs, sim)

    if not np.isfinite(score):
        return 1e6

    return -score   # minimise negative score

# ── optimizer ────────────────────────────────────────────────────────

def _project_to_bounds(x, bounds):
    """Clip a parameter vector to box bounds."""
    return np.array([
        min(max(v, lo), hi) for v, (lo, hi) in zip(x, bounds)
    ], dtype=np.float64)


def _run_local_opt(x0, q_in, obs, metric, bounds, constraints, metric_weights=None):
    """Run one constrained local optimization from a given starting point."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return minimize(
            objective,
            x0=x0,
            args=(q_in, obs, metric, bounds, metric_weights),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'maxiter': 1000, 'disp': False}
        )


def _evaluate_point(x, q_in, obs, metric, bounds, metric_weights=None):
    """Evaluate objective at a fixed point without local optimization."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size != 3:
        raise ValueError('evaluation point must contain exactly 3 values')
    fun = objective(x, q_in, obs, metric, bounds, metric_weights)
    if not np.isfinite(fun):
        return None
    return SimpleNamespace(x=x, fun=fun, success=True)


def _coerce_start_points(start_points, bounds):
    """Normalize user-provided start point(s) to a list of bounded np arrays."""
    if start_points is None:
        return []

    def _as_array(point):
        if isinstance(point, dict):
            required = ('k_fast', 'k_slow', 'f_fast')
            missing = [k for k in required if k not in point]
            if missing:
                raise ValueError(f"start point dict is missing keys: {missing}")
            arr = np.array([point['k_fast'], point['k_slow'], point['f_fast']], dtype=np.float64)
        else:
            arr = np.asarray(point, dtype=np.float64).ravel()
            if arr.size != 3:
                raise ValueError('each start point must contain exactly 3 values: [k_fast, k_slow, f_fast]')

        arr = _project_to_bounds(arr, bounds)
        # Respect ordering at initialization; optimizer constraint still enforces this.
        arr[1] = min(arr[1], arr[0] - 1e-5)
        arr = _project_to_bounds(arr, bounds)
        return arr

    # A dict or a single 3-vector is interpreted as one starting point.
    if isinstance(start_points, dict):
        return [_as_array(start_points)]

    if isinstance(start_points, (list, tuple, np.ndarray)):
        # Single vector-like input, e.g. [k_fast, k_slow, f_fast]
        if len(start_points) == 3 and not isinstance(start_points[0], (dict, list, tuple, np.ndarray)):
            return [_as_array(start_points)]
        # Multiple start points
        return [_as_array(p) for p in start_points]

    raise ValueError('start_points must be None, a dict, a 3-value sequence, or a list of those')

def calibrate_reservoirs(q_in, obs, metric='nse',
                          n_starts=300, seed=42,
                          strategy='adaptive',
                          start_points=None,
                          metric_weights=None):
    """
    Calibrate two-reservoir model against observed daily streamflow.

    Parameters
    ----------
    q_in   : SUMMA simulated daily runoff array
    obs    : observed daily streamflow (same units, same length)
    metric : 'nse' | 'kge' | 'log_nse' | 'log_kge' | 'balanced'
        'balanced' is a composite of KGE + NSE + log-KGE + log-NSE.
    metric_weights : dict, optional
        Only used when metric='balanced'. Keys: w_kge, w_nse, w_kge_log, w_nse_log.
        Defaults to equal weights (0.25 each).
        Example: {'w_kge': 0.2, 'w_nse': 0.2, 'w_kge_log': 0.3, 'w_nse_log': 0.3}
    n_starts : number of local optimization starts
    strategy : 'adaptive' (learns from elite starts) or 'multistart' (uniform random)
    start_points : optional initial guess(es) for [k_fast, k_slow, f_fast]
        Accepted forms:
            - dict: {'k_fast': ..., 'k_slow': ..., 'f_fast': ...}
            - sequence: [k_fast, k_slow, f_fast]
            - list of dicts/sequences

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
    bounds = [(0.005, 3),   # k_fast: 1–20 day residence time
              (1e-6, 0.5),  # k_slow: 10–200 day residence time
              (0.4,  1.0)]    # f: fast fraction

    constraints = [
        {'type': 'ineq', 'fun': lambda x: x[0] - x[1] - 1e-8}  # enforce k_fast > k_slow
    ]

    if strategy not in {'adaptive', 'multistart', 'seed_only'}:
        raise ValueError("strategy must be 'adaptive', 'multistart', or 'seed_only'")

    all_results = []

    local_runs = 0

    # Evaluate user-provided starting point(s) first so trusted manual guesses are always tested.
    # Keep both the raw seeded scores and locally optimized seeded scores.
    seeded_starts = _coerce_start_points(start_points, bounds)
    for x0 in seeded_starts:
        raw_seed_result = _evaluate_point(x0, q_in, obs, metric, bounds, metric_weights)
        if raw_seed_result is not None:
            all_results.append(raw_seed_result)

        result = _run_local_opt(x0, q_in, obs, metric, bounds, constraints, metric_weights)
        local_runs += 1
        if np.isfinite(result.fun):
            all_results.append(result)

    # Baseline: independent random multistart
    if strategy == 'seed_only':
        if not seeded_starts:
            raise ValueError("strategy='seed_only' requires at least one start point")

    elif strategy == 'multistart':
        for _ in range(n_starts):
            x0 = np.array([rng.uniform(lo, hi) for lo, hi in bounds], dtype=np.float64)
            x0[1] = min(x0[1], x0[0] * 0.5)  # ensure k_slow < k_fast at initialization
            result = _run_local_opt(x0, q_in, obs, metric, bounds, constraints, metric_weights)
            local_runs += 1
            if np.isfinite(result.fun):
                all_results.append(result)

    # Adaptive: learn start distribution from elite solutions over rounds
    else:
        n_rounds = max(2, min(8, int(np.sqrt(max(n_starts, 1)))))
        starts_per_round = max(4, int(np.ceil(n_starts / n_rounds)))

        lower = np.array([b[0] for b in bounds], dtype=np.float64)
        upper = np.array([b[1] for b in bounds], dtype=np.float64)

        # round 0 uses broad uniform exploration
        center = (lower + upper) / 2.0
        spread = (upper - lower) / 2.0

        for round_idx in range(n_rounds):
            round_results = []

            for _ in range(starts_per_round):
                if round_idx == 0:
                    x0 = np.array([rng.uniform(lo, hi) for lo, hi in bounds], dtype=np.float64)
                else:
                    # gradually reduce exploration as rounds progress
                    explore_scale = 0.65 ** round_idx
                    x0 = rng.normal(loc=center, scale=np.maximum(1e-6, spread * explore_scale))
                    x0 = _project_to_bounds(x0, bounds)

                # enforce ordering at initialization (constraint still enforced in optimizer)
                x0[1] = min(x0[1], x0[0] - 1e-5)
                x0 = _project_to_bounds(x0, bounds)

                result = _run_local_opt(x0, q_in, obs, metric, bounds, constraints, metric_weights)
                local_runs += 1
                if np.isfinite(result.fun):
                    round_results.append(result)
                    all_results.append(result)

            if round_results:
                # update search distribution from elite fraction
                round_results.sort(key=lambda r: r.fun)
                elite_n = max(2, int(np.ceil(0.2 * len(round_results))))
                elite_x = np.array([r.x for r in round_results[:elite_n]], dtype=np.float64)
                center = elite_x.mean(axis=0)
                spread = np.maximum(elite_x.std(axis=0), (upper - lower) * 0.05)

    if all_results:
        best_result = min(all_results, key=lambda r: r.fun)
        best_score = best_result.fun

    if best_result is None:
        raise RuntimeError('Reservoir calibration failed: no valid optimization result found.')

    k_fast, k_slow, f = best_result.x

    # Final guardrail for numerical tolerance at solution boundaries.
    if not (bounds[0][0] <= k_fast <= bounds[0][1] and
            bounds[1][0] <= k_slow <= bounds[1][1] and
            bounds[2][0] <= f <= bounds[2][1] and
            k_slow < k_fast):
        raise RuntimeError(
            f'Reservoir calibration returned invalid parameters: '
            f'k_fast={k_fast}, k_slow={k_slow}, f={f}'
        )

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
        'log_kge':           log_kge(obs, sim),
        'balanced':          balanced(obs, sim, **(metric_weights or {})),
        'sim':               sim,
        'optimizer_success': best_result.success,
        'n_starts_used':     local_runs,
        'seeded_starts_used': len(seeded_starts),
        'strategy':          strategy,
        'objective':         best_score,
        'selected_from_seed_raw': any(np.allclose(best_result.x, s, atol=1e-12, rtol=0.0) for s in seeded_starts),
    }

# ── usage ─────────────────────────────────────────────────────────────

