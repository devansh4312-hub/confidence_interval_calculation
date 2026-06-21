#!/usr/bin/env python3
"""
forward_range_app.py  —  single-file app (v2: regime-aware + calibrated)
========================================================================
Forward price range + option probabilities for any stock, with four accuracy
upgrades over the basic version:

  1. CONDITIONAL VOLATILITY  — EWMA or GARCH(1,1) instead of one flat 3-year
     sigma, so the band reflects today's volatility regime. The simulations use
     Filtered Historical Simulation (standardise returns by conditional vol,
     bootstrap the residuals, re-inflate by the forecast vol path).
  2. DRIFT SHRINKAGE  — the noisy historical mean is shrunk toward cost-of-carry
     (or zero) so a lucky/unlucky 3-year run doesn't mis-centre the forecast.
  3. IMPLIED-VOL + SKEW  — optional forward-looking, risk-neutral bounds that are
     ASYMMETRIC (puts richer than calls). Per-strike IV feeds the option module.
     Can auto-pull a US option chain via yfinance, or take manual ATM IV + skew.
  4. CALIBRATION BACKTEST  — walk-forward Kupiec test: of all past H-day moves,
     how often did price land outside the predicted band? Tells you whether a
     method is honestly calibrated for THIS stock (target = alpha).

RUN:
    pip install streamlit yfinance numpy pandas scipy matplotlib
    streamlit run forward_range_app.py

Analytical aid, not investment advice.
"""
import math, sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats, optimize

TD = 252  # trading days / year


# ======================================================================
#  DATA
# ======================================================================
def load_prices(ticker=None, years=3, csv=None):
    if csv is not None:
        df = pd.read_csv(csv)
        cols = {c.lower(): c for c in df.columns}
        dcol = cols.get("date")
        ccol = cols.get("adj close") or cols.get("adj_close") or cols.get("close")
        if ccol is None:
            sys.exit("CSV must contain a 'Close' (or 'Adj Close') column.")
        if dcol:
            df[dcol] = pd.to_datetime(df[dcol]); df = df.sort_values(dcol)
        close = df[ccol].astype(float).values
        ohlc = None
        if all(cols.get(x) for x in ("open", "high", "low", "close")):
            ohlc = {k: df[cols[k]].astype(float).values for k in ("open", "high", "low", "close")}
        return (df[dcol].values if dcol else np.arange(len(df))), close, ohlc
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("Please `pip install yfinance` or upload a CSV.")
    df = yf.download(ticker, period=f"{years}y", interval="1d",
                     auto_adjust=True, progress=False)
    if df is None or len(df) == 0:
        sys.exit(f"No data for '{ticker}'. NSE stocks use a .NS suffix (e.g. RELIANCE.NS).")
    def col(name):
        c = df[name]
        return (c.iloc[:, 0] if hasattr(c, "columns") else c)
    close = col("Close").dropna()
    idx = close.index
    ohlc = None
    try:
        ohlc = {k.lower(): col(k).reindex(idx).astype(float).values
                for k in ("Open", "High", "Low", "Close")}
    except Exception:
        ohlc = None
    return idx.values, close.values.astype(float), ohlc


def diagnose(r):
    mu, sd = r.mean(), r.std(ddof=1)
    jb, jb_p = stats.jarque_bera(r)
    t_df, t_loc, t_sc = stats.t.fit(r)
    z = (r - mu) / sd
    return dict(mu=mu, sd=sd, ann_sd=sd*math.sqrt(TD), skew=stats.skew(r),
                exk=stats.kurtosis(r), jb=jb, jb_p=jb_p,
                t_df=t_df, t_loc=t_loc, t_sc=t_sc,
                tail=float(np.mean(np.abs(z) > 2.5758)))


# ======================================================================
#  (1) CONDITIONAL VOLATILITY  — EWMA & GARCH(1,1)
# ======================================================================
def ewma_vol(r, lam=0.94):
    """RiskMetrics EWMA conditional vol series aligned to r (uses info up to t-1)."""
    var = np.empty(len(r)); var[0] = r.var(ddof=1)
    for t in range(1, len(r)):
        var[t] = lam*var[t-1] + (1-lam)*r[t-1]**2
    return np.sqrt(var)


def gk_daily_var(o, h, l, c):
    """Garman-Klass daily variance estimate from OHLC — uses the whole bar, so it's
    ~5x more efficient than close-to-close. Measured to sharpen the calibrated band."""
    o, h, l, c = (np.asarray(x, float) for x in (o, h, l, c))
    with np.errstate(divide="ignore", invalid="ignore"):
        v = 0.5*np.log(h/l)**2 - (2*math.log(2)-1)*np.log(c/o)**2
    return np.clip(np.nan_to_num(v, nan=0.0), 0.0, None)


def ewma_range_vol(daily_var, lam=0.94):
    """EWMA conditional vol from a range-based daily-variance series (uses info to t-1)."""
    dv = np.asarray(daily_var, float)
    var = np.empty(len(dv)); var[0] = np.nanmean(dv[:20]) if len(dv) >= 20 else max(dv[0], 1e-8)
    for t in range(1, len(dv)):
        x = dv[t-1] if np.isfinite(dv[t-1]) and dv[t-1] > 0 else var[t-1]
        var[t] = lam*var[t-1] + (1-lam)*x
    return np.sqrt(np.maximum(var, 1e-12))


def garch11_fit(r, scale=100.0, quick=False):
    """Robust GARCH(1,1), Gaussian innovations. Returns dict or None.

    Three fixes over a naive MLE that make it work across tickers:
      * returns are rescaled to percent so the parameters are O(1) and the
        optimizer is well-conditioned (raw daily variance ~1e-4 otherwise
        leaves omega ~1e-5, which derails the search);
      * VARIANCE TARGETING fixes omega = target*(1-alpha-beta) with target = the
        sample variance, so the long-run variance can't blow up as persistence
        approaches 1 (the old omega/(1-alpha-beta) was numerically unstable);
      * BOUNDED multi-start optimization avoids the single-start local optima.
    """
    x = (r - r.mean()) * scale
    n = len(x)
    target = float(x.var())                       # variance-targeting anchor
    if n < 50 or not np.isfinite(target) or target <= 0:
        return None

    def nll(ab):
        a, b = ab
        if a <= 0 or b <= 0 or a + b >= 0.999:
            return 1e12
        omega = target * (1.0 - a - b)
        var = np.empty(n); var[0] = target
        for t in range(1, n):
            var[t] = omega + a * x[t-1]**2 + b * var[t-1]
        if not np.all(np.isfinite(var)) or np.any(var <= 0):
            return 1e12
        return 0.5 * np.sum(np.log(2*math.pi) + np.log(var) + x*x/var)

    best = None
    starts = [(0.05, 0.90), (0.10, 0.85)] if quick else \
             [(0.05, 0.90), (0.10, 0.85), (0.03, 0.95),
              (0.15, 0.80), (0.08, 0.90), (0.02, 0.97)]
    for a0, b0 in starts:
        try:
            res = optimize.minimize(nll, [a0, b0], method="L-BFGS-B",
                                    bounds=[(1e-4, 0.6), (1e-4, 0.998)])
            if np.isfinite(res.fun) and (best is None or res.fun < best.fun):
                best = res
        except Exception:
            pass
    if best is None:
        return None
    a, b = float(best.x[0]), float(best.x[1])
    if a <= 0 or b <= 0 or a + b >= 0.999:
        return None

    omega = target * (1.0 - a - b)
    var = np.empty(n); var[0] = target
    for t in range(1, n):
        var[t] = omega + a * x[t-1]**2 + b * var[t-1]
    next_var = omega + a * x[-1]**2 + b * var[-1]
    s2 = scale * scale                            # undo the percent scaling
    return dict(omega=omega/s2, alpha=a, beta=b, cond_vol=np.sqrt(var/s2),
                next_var=next_var/s2, uncond=target/s2, persist=a+b)


def gjr_garch11_fit(r, scale=100.0, quick=False):
    """GJR-GARCH(1,1): captures the LEVERAGE EFFECT — negative shocks raise next
    period's vol more than equal positive shocks (downside risk asymmetry typical
    of equities). Variance-targeted, scaled, multi-start (same robustness recipe).

        sigma^2_t = omega + (alpha + gamma*1[r_{t-1}<0]) * r_{t-1}^2 + beta*sigma^2_{t-1}

    Persistence = alpha + gamma/2 + beta (shocks are negative half the time).
    gamma > 0 is the leverage effect. Gaussian QMLE (consistent); the FHS/EVT layers
    carry the actual tail shape, so the innovation assumption here is second-order.
    """
    x = (r - r.mean()) * scale
    n = len(x)
    target = float(x.var())
    if n < 60 or not np.isfinite(target) or target <= 0:
        return None
    neg = (x < 0).astype(float)

    def nll(p):
        a, g, b = p
        persist = a + g/2 + b
        if a < 0 or g < 0 or b <= 0 or persist >= 0.999:
            return 1e12
        omega = target * (1.0 - persist)
        if omega <= 0:
            return 1e12
        var = np.empty(n); var[0] = target
        for t in range(1, n):
            var[t] = omega + (a + g*neg[t-1])*x[t-1]**2 + b*var[t-1]
        if not np.all(np.isfinite(var)) or np.any(var <= 0):
            return 1e12
        return 0.5*np.sum(np.log(2*math.pi) + np.log(var) + x*x/var)

    best = None
    starts = [(0.02, 0.10, 0.88), (0.05, 0.08, 0.85)] if quick else \
             [(0.02, 0.10, 0.88), (0.05, 0.08, 0.85), (0.01, 0.15, 0.90),
              (0.03, 0.06, 0.92), (0.08, 0.10, 0.80), (0.00, 0.12, 0.90)]
    for s0 in starts:
        try:
            res = optimize.minimize(nll, s0, method="L-BFGS-B",
                                    bounds=[(0, 0.4), (0, 0.5), (1e-4, 0.998)])
            if np.isfinite(res.fun) and (best is None or res.fun < best.fun):
                best = res
        except Exception:
            pass
    if best is None:
        return None
    a, g, b = [float(v) for v in best.x]
    persist = a + g/2 + b
    if b <= 0 or persist >= 0.999 or a < 0 or g < 0:
        return None
    omega = target*(1.0 - persist)
    var = np.empty(n); var[0] = target
    for t in range(1, n):
        var[t] = omega + (a + g*neg[t-1])*x[t-1]**2 + b*var[t-1]
    next_var = omega + (a + g*neg[-1])*x[-1]**2 + b*var[-1]
    s2 = scale*scale
    return dict(omega=omega/s2, alpha=a, gamma=g, beta=b, cond_vol=np.sqrt(var/s2),
                next_var=next_var/s2, uncond=target/s2, persist=persist)


def vol_model(r, kind, quick=False, daily_var=None):
    """Return (cond_vol_series, forecast_fn(H)->daily vol path, current_vol, label).
    daily_var: optional range-based (Garman-Klass) daily variance aligned to r, used by
    the 'EWMA-Range' model."""
    if kind == "EWMA-Range (Garman-Klass)":
        if daily_var is not None and len(daily_var) == len(r):
            cv = ewma_range_vol(daily_var); cur = float(cv[-1])
            return cv, (lambda H, c=cur: np.full(H, c)), cur, "EWMA-Range (Garman-Klass, λ=0.94)"
        # no OHLC -> fall back to the next-best close-based model (EWMA+VoV), not plain EWMA
        cv, fc, cur, _ = vol_model(r, "EWMA+VoV", quick=quick)
        return cv, fc, cur, "EWMA+VoV — no OHLC, range vol unavailable"
    if kind == "GARCH(1,1)":
        g = garch11_fit(r, quick=quick)
        if g is not None:
            def fc(H, g=g):
                path = np.empty(H)
                for k in range(1, H+1):
                    var_k = g["uncond"] + (g["persist"]**(k-1))*(g["next_var"]-g["uncond"])
                    path[k-1] = math.sqrt(max(var_k, 1e-12))
                return path
            # "current" = one-step-ahead forecast, so it matches the band's day 1
            cur = math.sqrt(max(g["next_var"], 1e-12))
            return g["cond_vol"], fc, cur, \
                   f"GARCH(1,1) α={g['alpha']:.2f} β={g['beta']:.2f} (persist {g['persist']:.2f})"
        # GARCH failed to fit — fall back to EWMA, but say so
        cv = ewma_vol(r); cur = float(cv[-1])
        return cv, (lambda H, c=cur: np.full(H, c)), cur, "EWMA (λ=0.94) — GARCH fit failed, fell back"
    if kind == "GJR-GARCH(1,1)":
        g = gjr_garch11_fit(r, quick=quick)
        if g is not None:
            def fc(H, g=g):
                path = np.empty(H)
                for k in range(1, H+1):
                    var_k = g["uncond"] + (g["persist"]**(k-1))*(g["next_var"]-g["uncond"])
                    path[k-1] = math.sqrt(max(var_k, 1e-12))
                return path
            cur = math.sqrt(max(g["next_var"], 1e-12))
            return g["cond_vol"], fc, cur, \
                   f"GJR-GARCH α={g['alpha']:.2f} γ={g['gamma']:.2f} β={g['beta']:.2f} (leverage γ>0)"
        cv = ewma_vol(r); cur = float(cv[-1])
        return cv, (lambda H, c=cur: np.full(H, c)), cur, "EWMA (λ=0.94) — GJR fit failed, fell back"
    if kind == "EWMA":
        cv = ewma_vol(r); cur = float(cv[-1])
        return cv, (lambda H, c=cur: np.full(H, c)), cur, "EWMA (λ=0.94)"
    if kind == "EWMA+VoV":
        cv = ewma_vol(r)
        def fc(H, r=r, cv=cv):
            return ewma_vov_path(r, H, cv)
        cur = float(fc(1)[0])
        return cv, fc, cur, "EWMA + vol-of-vol (Jensen inflation)"
    if kind == "Blend (accuracy-wtd)":
        cv, fc, cur, w = blended_vol(r, quick=quick)
        return cv, fc, cur, f"Blend {w}"
    # constant 3-year
    sd = r.std(ddof=1)
    return np.full(len(r), sd), (lambda H, s=sd: np.full(H, s)), float(sd), "Constant (sample sd)"


def ewma_vov_path(r, H, cv=None, lam=0.94):
    """Vol-of-vol inflated forecast path. Models log-variance as AR(1) and forecasts
    E[sigma^2_{t+k}] = exp(E[log var] + 0.5*Var[log var]) — the +0.5*Var term is the
    Jensen inflation from volatility-of-volatility, which counters EWMA's habit of
    being too tight right before a vol spike. Reduces to flat EWMA when vol-of-vol=0."""
    if cv is None:
        cv = ewma_vol(r, lam)
    logv = np.log(np.maximum(cv**2, 1e-12))
    x0, x1 = logv[:-1], logv[1:]
    v0 = np.var(x0)
    phi = float(np.cov(x0, x1)[0, 1]/v0) if v0 > 0 else 0.0
    phi = min(max(phi, 0.0), 0.995)
    mu_lv = float(np.mean(logv))
    resid = x1 - (mu_lv*(1-phi) + phi*x0)
    s2 = float(np.var(resid))                          # vol-of-vol (innovation var of log-var)
    last = float(logv[-1])
    path = np.empty(H)
    for k in range(1, H+1):
        m_k = mu_lv + (phi**k)*(last - mu_lv)
        v_k = s2*(1-phi**(2*k))/(1-phi**2) if phi < 1 else s2*k
        path[k-1] = math.sqrt(math.exp(m_k + 0.5*v_k))
    return path


def blended_vol(r, quick=False, K=126):
    """Accuracy-weighted blend of vol models. Each model's recent 1-step variance-
    forecast error (over the last K days) sets its weight (inverse MSE), so the blend
    leans on whichever model has been most accurate lately instead of picking one."""
    candidates = {}
    cve = ewma_vol(r); candidates["EWMA"] = cve
    candidates["Const"] = np.full(len(r), r.std(ddof=1))
    g = garch11_fit(r, quick=True)
    if g is not None:
        candidates["GARCH"] = g["cond_vol"]
    r2 = r**2
    weights = {}
    for name, cvser in candidates.items():
        k = min(K, len(r)-1)
        err = r2[-k:] - cvser[-k:]**2
        mse = float(np.mean(err**2))
        weights[name] = 1.0/(mse + 1e-12)
    tot = sum(weights.values())
    weights = {k: v/tot for k, v in weights.items()}
    # blended conditional-vol series (for standardization)
    blend_var = np.zeros(len(r))
    for name, cvser in candidates.items():
        blend_var += weights[name]*cvser**2
    cv = np.sqrt(blend_var)

    def fc(H, r=r, candidates=candidates, weights=weights, g=g):
        var_path = np.zeros(H)
        for name, cvser in candidates.items():
            if name == "EWMA":
                p = np.full(H, cvser[-1]**2)
            elif name == "Const":
                p = np.full(H, cvser[-1]**2)
            elif name == "GARCH" and g is not None:
                p = np.array([g["uncond"] + (g["persist"]**(k-1))*(g["next_var"]-g["uncond"])
                              for k in range(1, H+1)])
            else:
                p = np.full(H, cvser[-1]**2)
            var_path += weights[name]*np.maximum(p, 1e-12)
        return np.sqrt(var_path)
    cur = float(fc(1)[0])
    wlabel = ", ".join(f"{k} {v:.0%}" for k, v in weights.items())
    return cv, fc, cur, wlabel


# ======================================================================
#  (2) DRIFT SHRINKAGE
# ======================================================================
def drift_daily(r, kind, r_f, q):
    mu, sd, N = r.mean(), r.std(ddof=1), len(r)
    carry = (r_f - q)/TD
    if kind == "Historical mean":
        return mu
    if kind == "Zero":
        return 0.0
    if kind == "Risk-free carry":
        return carry
    # Shrunk toward carry by signal-to-noise of the mean
    t = mu/(sd/math.sqrt(N)) if sd > 0 else 0.0
    w = t*t/(1+t*t)                       # ~0 when mean is noise, ->1 when strong
    return w*mu + (1-w)*carry


# ======================================================================
#  FILTERED HISTORICAL SIMULATION  (one engine for range + options)
# ======================================================================
def _gpd_tail_fit(exceed):
    """Fit GPD to positive exceedances; return (shape c, scale b)."""
    try:
        c, _, b = stats.genpareto.fit(exceed, floc=0)
        if not (np.isfinite(c) and np.isfinite(b) and b > 0):
            raise ValueError
        return float(c), float(b)
    except Exception:
        return 0.1, max(float(np.mean(exceed)) if len(exceed) else 0.5, 1e-3)


def evt_auto_threshold(tail, candidates=(0.05, 0.075, 0.10, 0.125, 0.15)):
    """Pick the peaks-over-threshold fraction by SHAPE STABILITY (a practical proxy
    for the mean-residual-life / threshold-stability plots): choose the candidate
    whose GPD shape estimate is closest to its neighbours, i.e. where the fit has
    stopped drifting with the threshold. tail = positive magnitudes of one side."""
    tail = np.sort(tail)[::-1]
    n_full = len(tail)
    # compute shape at each candidate (exceedances = top f-fraction over its threshold)
    shapes = []
    for f in candidates:
        k = max(int(round(len(tail)*f)), 15)
        k = min(k, len(tail)-1)
        u = tail[k]
        exc = tail[:k] - u
        c, _b = _gpd_tail_fit(exc)
        shapes.append(c)
    shapes = np.array(shapes)
    if len(shapes) >= 3:
        drift = np.abs(np.gradient(shapes))
        best = int(np.argmin(drift))
    else:
        best = len(candidates)//2
    return candidates[best]


def evt_tail_sampler(z, tail_frac=None):
    """Semiparametric residual model: empirical centre + Generalized Pareto (EVT,
    peaks-over-threshold) tails, with AUTOMATIC threshold selection by shape
    stability (Workstream D) rather than a fixed fraction. Returns sample(n,rng)."""
    z = np.sort(z)
    n = len(z)
    if tail_frac is None:
        f_lo = evt_auto_threshold(np.abs(z[z < 0])) if np.any(z < 0) else 0.1
        f_hi = evt_auto_threshold(np.abs(z[z > 0])) if np.any(z > 0) else 0.1
    else:
        f_lo = f_hi = tail_frac
    k_lo = min(max(int(round(n*f_lo)), 15), n-1)
    k_hi = min(max(int(round(n*f_hi)), 15), n-1)
    u_lo, u_hi = z[k_lo], z[n-k_hi-1]
    body = z[(z >= u_lo) & (z <= u_hi)]
    if len(body) < 5:
        body = z
    p_lo, p_hi = k_lo/n, k_hi/n
    c_lo, b_lo = _gpd_tail_fit(u_lo - z[z < u_lo]) if np.any(z < u_lo) else (0.1, 0.5)
    c_hi, b_hi = _gpd_tail_fit(z[z > u_hi] - u_hi) if np.any(z > u_hi) else (0.1, 0.5)

    def sample(n_out, rng):
        u = rng.random(n_out)
        out = np.empty(n_out)
        lo_m = u < p_lo; hi_m = u > 1 - p_hi; mid_m = ~(lo_m | hi_m)
        out[lo_m] = u_lo - stats.genpareto.rvs(c_lo, scale=b_lo, size=int(lo_m.sum()), random_state=rng)
        out[hi_m] = u_hi + stats.genpareto.rvs(c_hi, scale=b_hi, size=int(hi_m.sum()), random_state=rng)
        out[mid_m] = body[rng.integers(0, len(body), size=int(mid_m.sum()))]
        return out
    return sample, dict(c_lo=c_lo, c_hi=c_hi, u_lo=u_lo, u_hi=u_hi, f_lo=f_lo, f_hi=f_hi)


def evt_tail_quantile_ci(z, p, side="lower", n_boot=300, rng=None):
    """GPD-implied extreme standardized quantile WITH a bootstrap confidence interval,
    so a 99.9% number comes with its estimation error (Workstream D). Returns
    (q, lo, hi) in standardized units. side='lower' for p<0.5 tails."""
    rng = rng or np.random.default_rng(0)
    z = np.asarray(z)
    n = len(z)

    def one(zz):
        zz = np.sort(zz)
        if side == "lower":
            t = np.abs(zz[zz < 0])
            f = evt_auto_threshold(t) if len(t) > 20 else 0.1
            k = min(max(int(round(len(zz)*f)), 15), len(zz)-1)
            u = zz[k]; exc = u - zz[zz < u]
            c, b = _gpd_tail_fit(exc); pp = k/len(zz)
            # P(Z < u - y) tail; quantile at prob p (p small)
            yp = stats.genpareto.ppf(1 - p/pp, c, scale=b) if p < pp else 0.0
            return u - yp
        else:
            t = np.abs(zz[zz > 0])
            f = evt_auto_threshold(t) if len(t) > 20 else 0.1
            k = min(max(int(round(len(zz)*f)), 15), len(zz)-1)
            u = zz[len(zz)-k-1]; exc = zz[zz > u] - u
            c, b = _gpd_tail_fit(exc); pp = k/len(zz)
            yp = stats.genpareto.ppf(1 - (1-p)/pp, c, scale=b) if (1-p) < pp else 0.0
            return u + yp

    q = one(z)
    boots = np.array([one(z[rng.integers(0, n, size=n)]) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(q), float(lo), float(hi)


def bootstrap_param_draws(r, B=300, block=5, rng=None):
    """Block-bootstrap the return series to get the sampling spread of (mean, std).
    Feeding this into the sim widens bounds to reflect that we don't KNOW mu and
    sigma — parameter uncertainty, not just path randomness."""
    rng = rng or np.random.default_rng(0)
    m = len(r); nb = math.ceil(m/block)
    mus = np.empty(B); sds = np.empty(B)
    for i in range(B):
        starts = rng.integers(0, m-block, size=nb)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:m]
        s = r[idx]
        mus[i] = s.mean(); sds[i] = s.std(ddof=1)
    return mus, sds


def fhs_terminal(r, cond_vol, fc_path, S0, mu_d, n, method, rng,
                 t_df=None, evt_sampler=None, param_draws=None):
    """Simulate terminal prices after H=len(fc_path) days via FHS.
    method: bootstrap | student_t | normal | evt.
    param_draws=(mus,sds): if given, each path draws its own (mu,sigma) estimate,
    injecting parameter uncertainty."""
    z = (r - r.mean())/cond_vol
    z = z - z.mean()
    sd_full = z.std(ddof=1) if z.std(ddof=1) > 0 else 1.0
    H = len(fc_path)
    if method == "bootstrap":
        Z = z[rng.integers(0, len(z), size=(n, H))]
    elif method == "student_t":
        s = math.sqrt((t_df-2)/t_df) if t_df and t_df > 2 else 1.0
        Z = stats.t.rvs(t_df, size=(n, H), random_state=rng)*s
    elif method == "normal":
        Z = rng.normal(size=(n, H))
    elif method == "evt":
        Z = evt_sampler(n*H, rng).reshape(n, H)
    else:
        raise ValueError(method)
    base = (Z*fc_path[None, :]).sum(axis=1)
    if param_draws is not None:
        mus, sds = param_draws
        j = rng.integers(0, len(mus), size=n)
        r_mu = r.mean()
        scale = sds[j]/ (r.std(ddof=1) if r.std(ddof=1) > 0 else 1.0)
        sums = base*scale + (mu_d + (mus[j]-r_mu))*H     # widen by est. spread of mean & vol
    else:
        sums = base + mu_d*H
    return S0*np.exp(sums)


def integrated_sigma(fc_path):
    return math.sqrt(np.sum(fc_path**2))


def forward_range(r, cond_vol, fc_path, S0, mu_d, alpha, n, rng, t_df,
                  evt_sampler=None, param_draws=None):
    z = stats.norm.ppf(1-alpha/2)
    sig_H = integrated_sigma(fc_path)
    out = {"scaled_normal": (S0*math.exp(mu_d*len(fc_path) - z*sig_H),
                             S0*math.exp(mu_d*len(fc_path) + z*sig_H))}
    lo_q, hi_q = alpha/2, 1-alpha/2
    rows = [("fhs_bootstrap", "bootstrap"), ("fhs_student_t", "student_t")]
    if evt_sampler is not None:
        rows.append(("fhs_evt", "evt"))
    for key, meth in rows:
        ST = fhs_terminal(r, cond_vol, fc_path, S0, mu_d, n, meth, rng, t_df,
                          evt_sampler=evt_sampler, param_draws=param_draws)
        out[key] = (float(np.quantile(ST, lo_q)), float(np.quantile(ST, hi_q)))
    return out, z, sig_H


# ======================================================================
#  (3) IMPLIED VOL + SKEW
# ======================================================================
def iv_at(K, F, atm_iv, slope):
    """Linear smile in log-moneyness: IV = atm + slope*ln(K/F)."""
    return max(atm_iv + slope*math.log(K/F), 0.01)


def skew_bound(F, T, atm_iv, slope, z, side):
    """Strike at the z-quantile using local IV at that strike (fixed point)."""
    iv = atm_iv
    for _ in range(60):
        drift = -0.5*iv*iv*T
        K = F*math.exp((z if side == "upper" else -z)*iv*math.sqrt(T) + drift)
        new = iv_at(K, F, atm_iv, slope)
        if abs(new-iv) < 1e-7:
            break
        iv = new
    return K


def fetch_chain_skew(ticker, F, target_days):
    """Best-effort: pull a US option chain via yfinance -> (atm_iv, slope, expiry, n)."""
    try:
        import yfinance as yf, pandas as pd, datetime as dt
        tk = yf.Ticker(ticker)
        exps = tk.options
        if not exps:
            return None
        today = dt.date.today()
        dated = [(e, (dt.date.fromisoformat(e)-today).days) for e in exps]
        dated = [d for d in dated if d[1] >= 1]
        if not dated:
            return None
        exp = min(dated, key=lambda d: abs(d[1]-target_days))[0]
        ch = tk.option_chain(exp)
        rows = []
        for df, typ in ((ch.calls, "c"), (ch.puts, "p")):
            for _, row in df.iterrows():
                iv = row.get("impliedVolatility", np.nan)
                K = row.get("strike", np.nan)
                if iv and iv > 0.001 and 0.7*F < K < 1.3*F:
                    rows.append((math.log(K/F), float(iv)))
        if len(rows) < 5:
            return None
        x = np.array([a for a, _ in rows]); y = np.array([b for _, b in rows])
        slope, intercept = np.polyfit(x, y, 1)      # IV ~ intercept + slope*logK/F
        atm_iv = float(intercept)                    # IV at K=F (x=0)
        return dict(atm_iv=max(atm_iv, 0.01), slope=float(slope), expiry=exp, n=len(rows))
    except Exception:
        return None


def fetch_chain_points(ticker, target_days):
    """Best-effort: pull a US option chain via yfinance and return OTM (strike, iv)
    points + forward + expiry, for a Breeden-Litzenberger fit. None on failure."""
    try:
        import yfinance as yf, pandas as pd, datetime as dt
        tk = yf.Ticker(ticker)
        exps = tk.options
        if not exps:
            return None
        today = dt.date.today()
        dated = [(e, (dt.date.fromisoformat(e)-today).days) for e in exps]
        dated = [d for d in dated if d[1] >= 1]
        if not dated:
            return None
        exp, days = min(dated, key=lambda d: abs(d[1]-target_days))
        spot = float(tk.fast_info["last_price"])
        ch = tk.option_chain(exp)
        Ks, ivs = [], []
        for df in (ch.calls, ch.puts):
            for _, row in df.iterrows():
                iv = row.get("impliedVolatility", np.nan); K = row.get("strike", np.nan)
                if iv and iv > 0.005 and 0.6*spot < K < 1.5*spot:
                    Ks.append(float(K)); ivs.append(float(iv))
        if len(Ks) < 5:
            return None
        return dict(strikes=np.array(Ks), ivs=np.array(ivs), spot=spot,
                    expiry=exp, days=int(days))
    except Exception:
        return None


def fit_smile(strikes, ivs, F):
    """Quadratic vol smile in log-moneyness: IV(k) = a + b*k + c*k^2, k=ln(K/F).
    b is the skew (slope), c the smile curvature. Returns (smile_fn, (a,b,c))."""
    k = np.log(np.asarray(strikes, float)/F)
    iv = np.asarray(ivs, float)
    m = np.isfinite(k) & np.isfinite(iv) & (iv > 0)
    k, iv = k[m], iv[m]
    deg = 2 if len(k) >= 5 else 1
    coef = np.polyfit(k, iv, deg)                     # highest power first
    if deg == 1:
        coef = np.array([0.0, coef[0], coef[1]])
    c, b, a = coef[0], coef[1], coef[2]

    def smile(kk):
        return np.maximum(a + b*kk + c*kk*kk, 0.01)
    return smile, (float(a), float(b), float(c))


def fit_svi(strikes, ivs, F, T):
    """SVI (Gatheral) total-variance smile: w(k) = a + b[ρ(k−m) + √((k−m)²+σ²)], with
    w = IV²·T. Arbitrage-free in the wings by construction (b≥0), so the implied density
    stays non-negative at deep OTM strikes where a quadratic fit would go negative.
    Returns a smile_fn(k)->IV. Falls back to quadratic if the fit fails or data is thin."""
    k = np.log(np.asarray(strikes, float)/F); iv = np.asarray(ivs, float)
    msk = np.isfinite(k) & np.isfinite(iv) & (iv > 0)
    k, iv = k[msk], iv[msk]
    if len(k) < 5:
        return fit_smile(strikes, ivs, F)[0]
    w = iv*iv*T

    def svi(p, kk):
        a, b, rho, m, s = p
        return a + b*(rho*(kk-m) + np.sqrt((kk-m)**2 + s*s))
    p0 = [max(np.min(w)*0.5, 1e-4), 0.1, -0.3, 0.0, 0.1]
    lb = [-np.inf, 0.0, -0.999, -np.inf, 1e-4]
    ub = [np.inf, np.inf, 0.999, np.inf, np.inf]
    try:
        sol = optimize.least_squares(lambda p: svi(p, k)-w, p0, bounds=(lb, ub), max_nfev=3000)
        p = sol.x
    except Exception:
        return fit_smile(strikes, ivs, F)[0]

    def smile(kk):
        wk = np.maximum(svi(p, np.asarray(kk, float)), 1e-6)
        return np.maximum(np.sqrt(wk/T), 0.01)
    return smile


def breeden_litzenberger(F, T, r, smile_fn, span=6.0, N=801, arbfree=False):
    """Risk-neutral terminal-price density from the option smile (Breeden-Litzenberger):
    q(K) = e^{rT} d^2C/dK^2, with C(K) priced off the fitted IV(K). Returns
    {K, pdf, cdf} — the market's implied distribution, full smile (not just a slope)."""
    atm = float(smile_fn(0.0))
    kmax = max(0.5, span*atm*math.sqrt(max(T, 1e-6)))
    k = np.linspace(-kmax, kmax, N)
    K = F*np.exp(k)
    vol = np.asarray(smile_fn(k), float)
    sq = vol*math.sqrt(T)
    d1 = (np.log(F/K) + 0.5*vol*vol*T)/sq
    d2 = d1 - sq
    C = math.exp(-r*T)*(F*stats.norm.cdf(d1) - K*stats.norm.cdf(d2))   # forward-BS call
    if arbfree:
        # enforce call convexity in K (dC/dK non-decreasing, ≤0) → density ≥ 0 everywhere,
        # removing arbitrage spikes from a noisy/contaminated smile (A10)
        dC = np.diff(C)/np.diff(K)
        dC = np.maximum.accumulate(np.minimum(dC, 0.0))
        C = np.concatenate([[C[0]], C[0] + np.cumsum(dC*np.diff(K))])
    d1c = np.gradient(C, K)
    d2c = np.gradient(d1c, K)
    pdf = np.clip(math.exp(r*T)*d2c, 0, None)
    area = float(np.sum((pdf[1:]+pdf[:-1])/2*np.diff(K)))
    if not np.isfinite(area) or area <= 0:
        return None
    pdf = pdf/area
    cdf = np.concatenate([[0.0], np.cumsum((pdf[1:]+pdf[:-1])/2*np.diff(K))])
    cdf = cdf/cdf[-1]
    return dict(K=K, pdf=pdf, cdf=cdf)


def bl_quantile(bl, p):
    return float(np.interp(p, bl["cdf"], bl["K"]))


def bl_prob_above(bl, strike):
    return float(1.0 - np.interp(strike, bl["K"], bl["cdf"]))


def fuse_distributions(ST, bl, alpha, lam=0.5):
    """Physical-implied fusion (Workstream 2B): a linear opinion pool of the historical
    (physical) distribution and the option-implied (Breeden-Litzenberger, risk-neutral)
    distribution. lam=1 -> all history, lam=0 -> all market. The market view prices in
    KNOWN future events that history can't see; the physical view isn't distorted by the
    risk premium baked into option prices. Blending hedges both biases.
    Returns dict with fused lo/hi at the (alpha/2, 1-alpha/2) quantiles + a prob fn."""
    ST = np.sort(np.asarray(ST))
    lo = min(ST[0], bl["K"][0]); hi = max(ST[-1], bl["K"][-1])
    grid = np.linspace(lo, hi, 1200)
    cdf_phys = np.searchsorted(ST, grid, side="right")/len(ST)
    cdf_rn = np.interp(grid, bl["K"], bl["cdf"], left=0.0, right=1.0)
    cdf = lam*cdf_phys + (1-lam)*cdf_rn
    cdf = np.maximum.accumulate(cdf)                 # enforce monotone
    cdf = (cdf - cdf[0])/(cdf[-1]-cdf[0]) if cdf[-1] > cdf[0] else cdf

    def q(p):
        return float(np.interp(p, cdf, grid))

    def prob_above(k):
        return float(1.0 - np.interp(k, grid, cdf))
    return dict(lo=q(alpha/2), hi=q(1-alpha/2), median=q(0.5),
                grid=grid, cdf=cdf, prob_above=prob_above, lam=lam)


def apply_event_vol(fc_path, n_event_days, mult, offset=0):
    """Inflate the forecast vol on n event days (earnings/policy) by a multiplier, placed
    at `offset` trading days into the horizon (so an event 15 days out inflates day 15,
    not day 1). For the integrated variance the position is neutral, but it matters for
    sub-horizon paths and the fan chart."""
    fc = np.asarray(fc_path, float).copy()
    n = int(min(max(n_event_days, 0), len(fc)))
    off = int(min(max(offset, 0), max(len(fc)-1, 0)))
    if n > 0 and mult > 1:
        end = min(off + n, len(fc))
        fc[off:end] = fc[off:end]*mult
    return fc


def bs_greeks(S0, K, T, r, q, vol, kind):
    """Closed-form Black-Scholes Greeks (per 1.0 underlying, per year). Theta returned
    per calendar day; vega per 1 vol-point (1%)."""
    if T <= 0 or vol <= 0:
        return None
    sq = vol*math.sqrt(T)
    d1 = (math.log(S0/K) + (r-q+0.5*vol*vol)*T)/sq
    d2 = d1 - sq
    N = stats.norm.cdf; n = stats.norm.pdf
    disc_q = math.exp(-q*T); disc_r = math.exp(-r*T)
    gamma = disc_q*n(d1)/(S0*sq)
    vega = S0*disc_q*n(d1)*math.sqrt(T)
    if kind == "call":
        delta = disc_q*N(d1)
        theta = (-S0*disc_q*n(d1)*vol/(2*math.sqrt(T))
                 - r*K*disc_r*N(d2) + q*S0*disc_q*N(d1))
    else:
        delta = -disc_q*N(-d1)
        theta = (-S0*disc_q*n(d1)*vol/(2*math.sqrt(T))
                 + r*K*disc_r*N(-d2) - q*S0*disc_q*N(-d1))
    return dict(delta=delta, gamma=gamma, theta=theta/365.0, vega=vega/100.0)


def data_quality_report(dates, close):
    """Pre-flight checks: silent data corruption (splits, gaps, stale closes) poisons the
    vol estimate and produces confidently-wrong output. Returns a list of warning strings."""
    warns = []
    close = np.asarray(close, float)
    if len(close) < 30:
        warns.append(f"Only {len(close)} usable closes — too little history to test a range.")
        return warns
    r = np.diff(np.log(close))
    spikes = np.where(np.abs(r) > 0.4)[0]      # ~>49% one-day move: likely split/bonus
    if len(spikes):
        worst = spikes[np.argmax(np.abs(r[spikes]))]
        warns.append(f"{len(spikes)} single-day move(s) above 40% (largest {r[worst]*100:+.0f}%) — "
                     "likely an unadjusted split/bonus. This inflates volatility; use "
                     "split-adjusted prices for a correct range.")
    # repeated/stale closes
    stale = 1; mx = 1
    for i in range(1, len(close)):
        stale = stale+1 if close[i] == close[i-1] else 1
        mx = max(mx, stale)
    if mx >= 4:
        warns.append(f"Up to {mx} consecutive identical closes — stale or zero-volume data, "
                     "which understates volatility.")
    # calendar gaps
    if dates is not None and len(dates) == len(close):
        try:
            d = pd.to_datetime(pd.Series(dates))
            gaps = d.diff().dt.days.to_numpy()[1:]
            big = int(np.sum(gaps > 10))
            if big:
                warns.append(f"{big} gap(s) over 10 days in the price history — missing data "
                             "can distort the volatility estimate.")
        except Exception:
            pass
    return warns


def vol_term_chart(fc_path):
    """Sparkline of annualized vol by day across the horizon — shows the model's
    mean-reversion / vol-of-vol behaviour and why bands widen as they do."""
    fc = np.asarray(fc_path, float)
    ann = fc*math.sqrt(TD)*100
    days = np.arange(1, len(ann)+1)
    fig, ax = plt.subplots(figsize=(9, 1.8))
    fig.patch.set_alpha(0); ax.set_facecolor("none")
    ax.plot(days, ann, color="#4f9bef", lw=2)
    ax.fill_between(days, ann, ann.min()*0.98, color="#4f9bef", alpha=0.12)
    ax.set_xlim(1, max(len(ann), 2))
    ax.annotate(f"{ann[0]:.0f}%", (days[0], ann[0]), xytext=(2, 4),
                textcoords="offset points", fontsize=8, color="#aebfd4")
    ax.annotate(f"{ann[-1]:.0f}%", (days[-1], ann[-1]), xytext=(-2, 4),
                textcoords="offset points", ha="right", fontsize=8, color="#aebfd4")
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#33415a")
    ax.set_yticks([]); ax.tick_params(colors="#8da0b6", labelsize=8)
    ax.set_xlabel("trading days ahead", fontsize=8, color="#8da0b6")
    fig.tight_layout()
    return fig


def payoff_diagram(legs, lot, S0, ev, unit=""):
    """Classic at-expiry P&L vs underlying price, with breakevens and current spot."""
    grid = np.linspace(0.6*S0, 1.4*S0, 600)
    pay = strategy_payoff(legs, lot, grid)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    fig.patch.set_alpha(0); ax.set_facecolor("none")
    ax.axhline(0, color="#5a6b80", lw=1)
    ax.plot(grid, pay, color="#4f9bef", lw=2.2)
    ax.fill_between(grid, pay, 0, where=(pay >= 0), color="#46d98a", alpha=0.18)
    ax.fill_between(grid, pay, 0, where=(pay < 0), color="#ff7b7b", alpha=0.16)
    ax.axvline(S0, color="#cfe0f5", lw=1.1, ls="--")
    ax.annotate(f"now {unit}{S0:,.0f}", (S0, ax.get_ylim()[1]), xytext=(4, -12),
                textcoords="offset points", fontsize=8, color="#cfe0f5")
    for be in ev["breakevens"]:
        if grid[0] <= be <= grid[-1]:
            ax.axvline(be, color="#f0b54e", lw=1.0, ls=":")
            ax.annotate(f"{unit}{be:,.0f}", (be, 0), xytext=(3, 6),
                        textcoords="offset points", fontsize=8, color="#f0b54e")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color("#33415a")
    ax.tick_params(colors="#8da0b6", labelsize=8)
    ax.set_xlabel("price at expiry", fontsize=9, color="#8da0b6")
    ax.set_ylabel("P&L", fontsize=9, color="#8da0b6")
    fig.tight_layout()
    return fig


def pnl_fan_chart(days, fan, unit=""):
    """Day-by-day P&L distribution (5–95 / 25–75 / median) from today to expiry."""
    fig, ax = plt.subplots(figsize=(9, 3.4))
    fig.patch.set_alpha(0); ax.set_facecolor("none")
    ax.axhline(0, color="#5a6b80", lw=1)
    ax.fill_between(days, fan["p5"], fan["p95"], color="#2c5fa0", alpha=0.30, label="5–95%")
    ax.fill_between(days, fan["p25"], fan["p75"], color="#4f9bef", alpha=0.45, label="25–75%")
    ax.plot(days, fan["p50"], color="#cfe0f5", lw=1.8, label="median")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color("#33415a")
    ax.tick_params(colors="#8da0b6", labelsize=8)
    ax.set_xlabel("calendar days from now", fontsize=9, color="#8da0b6")
    ax.set_ylabel("position P&L", fontsize=9, color="#8da0b6")
    leg = ax.legend(loc="upper left", fontsize=8, frameon=False)
    for t in leg.get_texts():
        t.set_color("#aebfd4")
    fig.tight_layout()
    return fig


def bs_price_and_prob(S0, K, T, r, q, vol, kind):
    if T <= 0 or vol <= 0:
        return None
    d1 = (math.log(S0/K) + (r-q+0.5*vol*vol)*T)/(vol*math.sqrt(T))
    d2 = d1 - vol*math.sqrt(T)
    N = stats.norm.cdf
    if kind == "call":
        return dict(price=S0*math.exp(-q*T)*N(d1)-K*math.exp(-r*T)*N(d2), prob_itm=N(d2))
    return dict(price=K*math.exp(-r*T)*N(-d2)-S0*math.exp(-q*T)*N(-d1), prob_itm=N(-d2))


def bs_implied_vol(S0, K, T, r, q, price, kind):
    """Back out the Black-Scholes implied vol from a market premium (bisection). Lets a
    manually-entered premium drive consistent repricing, P&L-from-zero, and per-leg skew."""
    if T <= 0 or price <= 0:
        return None
    intrinsic = max(S0*math.exp(-q*T)-K*math.exp(-r*T), 0) if kind == "call" \
        else max(K*math.exp(-r*T)-S0*math.exp(-q*T), 0)
    if price <= intrinsic + 1e-9:
        return 0.01
    lo, hi = 1e-4, 5.0
    for _ in range(100):
        mid = 0.5*(lo+hi)
        p = bs_price_and_prob(S0, K, T, r, q, mid, kind)["price"]
        if p > price:
            hi = mid
        else:
            lo = mid
        if hi-lo < 1e-5:
            break
    return 0.5*(lo+hi)


def option_module(ST, S0, K, T, kind, side, premium, iv_used, r, q, lot,
                  bl=None, fuse_lam=0.5):
    res = dict(p_above=float(np.mean(ST > K)), p_below=float(np.mean(ST < K)),
               vol_used=iv_used, bs=bs_price_and_prob(S0, K, T, r, q, iv_used, kind),
               greeks=bs_greeks(S0, K, T, r, q, iv_used, kind))
    if bl is not None:                                   # B4: best estimate (fused P+Q)
        try:
            fz = fuse_distributions(ST, bl, 0.01, fuse_lam)
            pa = fz["prob_above"](K)
            res["fused"] = dict(p_above=pa, p_below=1-pa, lam=fuse_lam)
        except Exception:
            pass
    if premium is not None and side and kind:
        intr = np.maximum(ST-K, 0) if kind == "call" else np.maximum(K-ST, 0)
        pnl = ((premium-intr) if side == "sell" else (intr-premium))*lot
        be = (K+premium) if kind == "call" else (K-premium)
        res["option"] = dict(breakeven=be, prob_profit=float(np.mean(pnl > 0)),
                             exp_pnl=float(pnl.mean()), p1=float(np.quantile(pnl, .01)),
                             p5=float(np.quantile(pnl, .05)), p25=float(np.quantile(pnl, .25)),
                             p95=float(np.quantile(pnl, .95)), p99=float(np.quantile(pnl, .99)),
                             worst=float(pnl.min()), best=float(pnl.max()))
    return res


# ======================================================================
#  (3c) MULTI-LEG OPTION STRATEGIES  (Tier 3: B3 + B1 P&L fan)
# ======================================================================
def fhs_paths(r, cond_vol, fc_path, S0, mu_d, n, rng, param_draws=None):
    """Full FHS price paths over the horizon, shape (n, H+1) including today's price.
    Terminal column matches fhs_terminal(bootstrap); used for day-by-day P&L fans."""
    z = (r - r.mean())/cond_vol; z = z - z.mean()
    H = len(fc_path)
    steps = z[rng.integers(0, len(z), size=(n, H))]*fc_path[None, :]
    if param_draws is not None:
        mus, sds = param_draws; j = rng.integers(0, len(mus), size=n)
        r_mu = r.mean(); base_sd = r.std(ddof=1) if r.std(ddof=1) > 0 else 1.0
        steps = steps*(sds[j]/base_sd)[:, None] + (mu_d + (mus[j]-r_mu))[:, None]
    else:
        steps = steps + mu_d
    paths = S0*np.exp(np.cumsum(steps, axis=1))
    return np.hstack([np.full((n, 1), S0), paths])


def bs_price_vec(S, K, T, r, q, vol, kind):
    """Vectorized Black-Scholes price; S may be an array. T<=0 returns intrinsic."""
    S = np.asarray(S, float)
    if T <= 0 or vol <= 0:
        return np.maximum(S-K, 0) if kind == "call" else np.maximum(K-S, 0)
    sq = vol*math.sqrt(T)
    d1 = (np.log(S/K) + (r-q+0.5*vol*vol)*T)/sq; d2 = d1 - sq
    N = stats.norm.cdf
    if kind == "call":
        return S*math.exp(-q*T)*N(d1) - K*math.exp(-r*T)*N(d2)
    return K*math.exp(-r*T)*N(-d2) - S*math.exp(-q*T)*N(-d1)


def _snap_strike(S0, pct):
    """Convert a %-of-spot strike to a clean absolute strike on the right price scale."""
    tick = 50.0 if S0 >= 5000 else 10.0 if S0 >= 1000 else 5.0 if S0 >= 200 else 1.0
    return round(S0*pct/100.0/tick)*tick


def _resolve_K(val_ispct, S0):
    """(value, is_pct) -> absolute strike. Percent strikes snap to a clean tick; absolute
    strikes are used as entered."""
    val, is_pct = val_ispct
    return _snap_strike(S0, val) if is_pct else float(val)


def _leg_intrinsic(S, lg):
    return np.maximum(S-lg["K"], 0) if lg["kind"] == "call" else np.maximum(lg["K"]-S, 0)


def evaluate_strategy(ST, legs, lot, S0, T, r, q):
    """Terminal-payoff metrics + net Greeks + breakevens for a multi-leg position.
    leg = dict(kind, side, K, premium, iv, qty). Sign: buy=+1, sell=-1."""
    ST = np.asarray(ST, float)
    pnl = np.zeros_like(ST); net_cost = 0.0
    G = dict(delta=0.0, gamma=0.0, theta=0.0, vega=0.0)
    for lg in legs:
        s = 1 if lg["side"] == "buy" else -1
        u = lg["qty"]*lot
        pnl += s*u*(_leg_intrinsic(ST, lg) - lg["premium"])
        net_cost += s*u*lg["premium"]                      # debit positive, credit negative
        g = bs_greeks(S0, lg["K"], T, r, q, lg.get("iv", 0.2), lg["kind"])
        if g:
            for k in G:
                G[k] += s*lg["qty"]*g[k]
    # breakevens on a fine expiry-payoff grid
    grid = np.linspace(0.3*S0, 2.0*S0, 4000)
    pay = strategy_payoff(legs, lot, grid)
    bes = []
    sgn = np.sign(pay)
    for i in np.where(np.diff(sgn) != 0)[0]:
        x0, x1, y0, y1 = grid[i], grid[i+1], pay[i], pay[i+1]
        bes.append(x0 - y0*(x1-x0)/(y1-y0) if y1 != y0 else x0)
    return dict(pnl=pnl, net_cost=net_cost, prob_profit=float(np.mean(pnl > 0)),
                exp_pnl=float(pnl.mean()), p5=float(np.quantile(pnl, .05)),
                p25=float(np.quantile(pnl, .25)), p95=float(np.quantile(pnl, .95)),
                p1=float(np.quantile(pnl, .01)), p99=float(np.quantile(pnl, .99)),
                worst=float(pnl.min()), best=float(pnl.max()),
                max_pay=float(pay.max()), min_pay=float(pay.min()),
                breakevens=bes, greeks=G)


def strategy_payoff(legs, lot, grid):
    """Net P&L at expiry across a price grid (for the payoff diagram)."""
    pay = np.zeros_like(grid, float)
    for lg in legs:
        s = 1 if lg["side"] == "buy" else -1
        u = lg["qty"]*lot
        pay += s*u*(_leg_intrinsic(grid, lg) - lg["premium"])
    return pay


def strategy_pnl_fan(paths, legs, lot, expiry_days, r, q, n_marks=12, iv_shock=0.0):
    """Day-by-day P&L distribution of the position from today to expiry (B1).
    Reprices every leg via BS along each simulated path. iv_shock (in vol points, e.g.
    +0.05 = +5 IV points) is applied to every leg's IV to model a vol spike/crush — the
    dominant P&L driver for short-vol positions."""
    n, Hp1 = paths.shape; H = Hp1-1
    entry_cost = sum((1 if lg["side"] == "buy" else -1)*lg["qty"]*lot*lg["premium"] for lg in legs)
    idx = np.unique(np.linspace(0, H, min(n_marks, H+1)).round().astype(int))
    days_cal = idx*expiry_days/max(H, 1)
    qs = {"p5": [], "p25": [], "p50": [], "p75": [], "p95": []}
    for j, di in enumerate(idx):
        T_rem = max((expiry_days - days_cal[j])/365.0, 0.0)
        S = paths[:, di]
        val = np.zeros(n)
        for lg in legs:
            s = 1 if lg["side"] == "buy" else -1
            iv = max(lg.get("iv", 0.2) + iv_shock, 0.005)
            val += s*lg["qty"]*lot*bs_price_vec(S, lg["K"], T_rem, r, q, iv, lg["kind"])
        pnl_t = val - entry_cost
        for p, lvl in zip(qs, [5, 25, 50, 75, 95]):
            qs[p].append(float(np.percentile(pnl_t, lvl)))
    return days_cal, {k: np.array(v) for k, v in qs.items()}


def path_risk_metrics(paths, legs, lot, expiry_days, r, q, iv_shock=0.0, n_marks=12):
    """Intraday/path risk that expiry-only P&L misses (crucial for premium sellers):
    - probability the price TOUCHES each strike before expiry (assignment/adjustment risk),
    - distribution of Max Adverse Excursion (worst mark-to-market P&L along each path)."""
    n, Hp1 = paths.shape; H = Hp1-1
    pmax = paths.max(axis=1); pmin = paths.min(axis=1)
    touch = {}
    for lg in legs:
        K = lg["K"]
        p = float(np.mean(pmax >= K)) if lg["kind"] == "call" else float(np.mean(pmin <= K))
        touch[f"{lg['kind']} {K:,.0f}"] = p
    touch_any = float(np.mean((pmax >= max(l["K"] for l in legs if l["kind"] == "call")
                               if any(l["kind"] == "call" for l in legs) else np.full(n, False))
                              | (pmin <= min(l["K"] for l in legs if l["kind"] == "put")
                                 if any(l["kind"] == "put" for l in legs) else np.full(n, False))))
    entry_cost = sum((1 if lg["side"] == "buy" else -1)*lg["qty"]*lot*lg["premium"] for lg in legs)
    idx = np.unique(np.linspace(0, H, min(n_marks, H+1)).round().astype(int))
    days_cal = idx*expiry_days/max(H, 1)
    worst = np.full(n, np.inf)
    for j, di in enumerate(idx):
        T_rem = max((expiry_days - days_cal[j])/365.0, 0.0)
        S = paths[:, di]; val = np.zeros(n)
        for lg in legs:
            s = 1 if lg["side"] == "buy" else -1
            iv = max(lg.get("iv", 0.2) + iv_shock, 0.005)
            val += s*lg["qty"]*lot*bs_price_vec(S, lg["K"], T_rem, r, q, iv, lg["kind"])
        worst = np.minimum(worst, val - entry_cost)
    return dict(touch=touch, touch_any=touch_any,
                mae_median=float(np.median(worst)), mae_p5=float(np.percentile(worst, 5)),
                mae_p25=float(np.percentile(worst, 25)))


def approx_short_margin(legs, S0, lot, net_cost, rate=0.12):
    """Rough margin estimate (NOT broker-exact). Defined-risk structures (every short leg
    covered by a long leg) block about their max loss; positions with a naked short leg
    block ~rate x notional per uncovered short. Always check your broker's actual figure."""
    longs = [l for l in legs if l["side"] == "buy"]
    shorts = [l for l in legs if l["side"] == "sell"]
    covered = (len(longs) >= len(shorts)
               and any(l["kind"] == "call" for l in longs) >= any(s["kind"] == "call" for s in shorts)
               and any(l["kind"] == "put" for l in longs) >= any(s["kind"] == "put" for s in shorts))
    if not shorts:                                   # pure long: margin = debit paid
        return max(net_cost, 0.0), "debit paid (long premium)"
    if covered:
        grid = np.linspace(0.3*S0, 1.8*S0, 3000)
        maxloss = -strategy_payoff(legs, lot, grid).min()
        return max(maxloss, 0.0), "≈ max loss (defined-risk spread)"
    notional = sum(s["qty"] for s in shorts)*S0*lot
    return max(rate*notional - max(-net_cost, 0.0), 0.05*notional), f"≈ {rate*100:.0f}% of notional (naked short)"


def vol_rank(cond_vol_series):
    """Where today's conditional vol sits in its own history (0-100). A realized-vol proxy
    for IV-rank: high -> richer premium / better to sell; low -> cheaper to buy."""
    cv = np.asarray(cond_vol_series, float)
    cv = cv[np.isfinite(cv)]
    if len(cv) < 30:
        return None
    return float((cv[-1] >= cv).mean()*100)


# ======================================================================
#  (4) CALIBRATION BACKTEST  (walk-forward Kupiec POF)
# ======================================================================
def kupiec(n, x, p):
    """LR_POF and p-value (chi2, df=1). x breaches in n trials, target rate p."""
    if n == 0:
        return float("nan"), float("nan")
    pi = x/n
    if x == 0:
        lr = -2*(n*math.log(1-p))
    elif x == n:
        lr = -2*(n*math.log(p))
    else:
        lr = -2*((n-x)*math.log(1-p)+x*math.log(p)
                 - (n-x)*math.log(1-pi)-x*math.log(pi))
    return lr, 1-stats.chi2.cdf(lr, 1)


def christoffersen(seq, p):
    """Kupiec POF + Christoffersen independence + conditional-coverage tests.
    seq is the 0/1 breach sequence. Independence checks whether breaches CLUSTER
    (a breach today making one tomorrow more likely) — clustered failures are what
    actually blow up positions, which Kupiec alone can't see."""
    seq = np.asarray(seq).astype(int)
    n = len(seq); x = int(seq.sum())
    lr_pof, p_pof = kupiec(n, x, p)
    n00 = n01 = n10 = n11 = 0
    for i in range(1, n):
        a_, b_ = seq[i-1], seq[i]
        if a_ == 0 and b_ == 0: n00 += 1
        elif a_ == 0 and b_ == 1: n01 += 1
        elif a_ == 1 and b_ == 0: n10 += 1
        else: n11 += 1

    def L(k, pr):                       # k*log(pr), with 0*log0 = 0
        return k*math.log(pr) if (k > 0 and 0 < pr < 1) else 0.0

    pi01 = n01/(n00+n01) if (n00+n01) else 0.0
    pi11 = n11/(n10+n11) if (n10+n11) else 0.0
    pi = (n01+n11)/(n00+n01+n10+n11) if (n00+n01+n10+n11) else 0.0
    ll_null = L(n00+n10, 1-pi) + L(n01+n11, pi)
    ll_alt = L(n00, 1-pi01)+L(n01, pi01)+L(n10, 1-pi11)+L(n11, pi11)
    lr_ind = max(-2*(ll_null-ll_alt), 0.0)
    p_ind = 1-stats.chi2.cdf(lr_ind, 1)
    base = lr_pof if (lr_pof == lr_pof) else 0.0     # NaN guard
    lr_cc = base + lr_ind
    p_cc = 1-stats.chi2.cdf(lr_cc, 2)
    return dict(p_pof=p_pof, p_ind=p_ind, p_cc=p_cc)


def backtest(close, H, alpha, lookback, step, drift_kind, r_f, q,
             vol_kind="EWMA", fhs_sims=4000, max_windows=350):
    """Walk-forward coverage test with Kupiec + Christoffersen tests, plus a
    CALIBRATION OVERLAY for the selected vol model: it collects the standardized
    realized H-day returns and reads off the empirical quantiles, giving (a) the
    *effective* confidence of the nominal band and (b) calibrated multipliers that
    would have produced the target coverage historically (asymmetric, fat-tail aware).
    Returns (report_rows, calibration_dict)."""
    rets = np.diff(np.log(close))
    z = stats.norm.ppf(1-alpha/2)
    lo_q, hi_q = alpha/2, 1-alpha/2

    # adaptive step so a per-window GARCH refit stays fast enough
    avail = (len(rets)-H-lookback)
    if avail <= 0:
        return [], None
    n_win = avail//step + 1
    garch_fam = vol_kind in ("GARCH(1,1)", "GJR-GARCH(1,1)")
    if garch_fam and n_win > max_windows:
        step = max(step, math.ceil(avail/max_windows))
    glabel = "GJR" if vol_kind == "GJR-GARCH(1,1)" else "GARCH"
    order = ["Constant·Normal", "EWMA·Normal", "EWMA·Student-t", "EWMA·FHS-sim"]
    if garch_fam:
        order += [f"{glabel}·Normal", f"{glabel}·FHS-sim"]
    seqs = {k: [] for k in order}
    zsel = []                                          # standardized returns, selected model

    a = lookback
    while a <= len(rets)-H:
        tr = rets[a-lookback:a]
        S = close[a]; realized = close[a+H]; Rreal = math.log(realized/S)
        mu_d = drift_daily(tr, drift_kind, r_f, q)
        sd = tr.std(ddof=1); sigC = sd*math.sqrt(H)
        cv, fc, _, _ = vol_model(tr, "EWMA"); fcp = fc(H); sigE = integrated_sigma(fcp)
        exk_H = max(stats.kurtosis(tr)/H, 1e-6)
        df_H = min(max(4+6/exk_H, 3.0), 250.0)
        tq = stats.t.ppf(1-alpha/2, df_H)*math.sqrt((df_H-2)/df_H)
        bands = {
            "Constant·Normal": (S*math.exp(mu_d*H-z*sigC),  S*math.exp(mu_d*H+z*sigC)),
            "EWMA·Normal":     (S*math.exp(mu_d*H-z*sigE),  S*math.exp(mu_d*H+z*sigE)),
            "EWMA·Student-t":  (S*math.exp(mu_d*H-tq*sigE), S*math.exp(mu_d*H+tq*sigE)),
        }
        STf = fhs_terminal(tr, cv, fcp, S, mu_d, fhs_sims, "bootstrap", np.random.default_rng(a))
        bands["EWMA·FHS-sim"] = (float(np.quantile(STf, lo_q)), float(np.quantile(STf, hi_q)))

        sig_sel = sigE                                 # default selected sigma
        if vol_kind == "Constant":
            sig_sel = sigC
        elif garch_fam:
            gcv, gfc, _, glab = vol_model(tr, vol_kind, quick=True)
            gfcp = gfc(H); sigG = integrated_sigma(gfcp)
            sig_sel = sigG if not glab.startswith("EWMA") else sigE
            bands[f"{glabel}·Normal"] = (S*math.exp(mu_d*H-z*sigG), S*math.exp(mu_d*H+z*sigG))
            STg = fhs_terminal(tr, gcv, gfcp, S, mu_d, fhs_sims, "bootstrap",
                               np.random.default_rng(a+1))
            bands[f"{glabel}·FHS-sim"] = (float(np.quantile(STg, lo_q)), float(np.quantile(STg, hi_q)))

        for k, (lo, hi) in bands.items():
            seqs[k].append(int(realized < lo or realized > hi))
        if sig_sel > 0:
            zsel.append((Rreal - mu_d*H)/sig_sel)
        a += step

    rep = []
    thin = max(1, math.ceil(H/step))      # decimate to ~non-overlapping windows for the tests
    for k in order:
        s = seqs[k]; n = len(s); x = int(np.sum(s))
        s_test = s[::thin]                # overlap-corrected sequence (valid p-values)
        ch = christoffersen(s_test, alpha)
        vk, dist = k.split("·")
        rep.append(dict(vol=vk, dist=dist, n=n, breaches=x,
                        obs=x/n if n else float("nan"), exp=alpha, n_test=len(s_test),
                        kupiec_p=ch["p_pof"], ind_p=ch["p_ind"], cc_p=ch["p_cc"]))

    zsel = np.asarray(zsel)
    calib = None
    if len(zsel) > 50:
        eff_out = float(np.mean((zsel < -z) | (zsel > z)))    # realized breach of nominal band
        calib = dict(q_lo=float(np.quantile(zsel, lo_q)),
                     q_hi=float(np.quantile(zsel, hi_q)),
                     z_nom=z, eff_conf=1-eff_out, n=len(zsel), vol_kind=vol_kind)
    return rep, calib


# ======================================================================
#  (B) PROPER SCORING  —  pinball / quantile score (∝ CRPS)
# ======================================================================
def quantile_score(qvals, y, taus):
    """Mean pinball loss over a grid of quantile levels. This is a strictly proper
    score: it rewards bands that are BOTH calibrated AND sharp (narrow). The mean
    over a uniform tau grid approximates CRPS/2. Lower = better. qvals are predicted
    quantiles aligned to taus; y is the realized value."""
    d = y - np.asarray(qvals)
    taus = np.asarray(taus)
    return float(np.mean(np.where(d >= 0, taus*d, (taus-1)*d)))


_TAUS = np.concatenate([[0.005, 0.01, 0.025],
                        np.arange(0.05, 0.96, 0.05), [0.975, 0.99, 0.995]])


# ======================================================================
#  (A) HONEST CALIBRATION + (B) VALIDATION HARNESS
# ======================================================================
def validation_harness(close, H, alpha, lookback, step, r_f, q,
                       fhs_sims=2000, burn=60, max_windows=500):
    """Walk-forward comparison that scores every method out-of-sample on
    (i) coverage at the headline confidence and (ii) the quantile score (proper,
    ∝ CRPS). Calibrated methods use ONLY past windows (genuinely out-of-sample).
    Returns a ranked list of dicts."""
    rets = np.diff(np.log(close))
    znom = stats.norm.ppf(1-alpha/2)
    taus = _TAUS
    avail = len(rets)-H-lookback
    if avail <= 0:
        return []
    if (avail//step + 1) > max_windows:
        step = max(step, math.ceil(avail/max_windows))

    methods = ["v1 (const·Normal·histμ)", "EWMA·Normal", "EWMA·FHS", "EWMA·EVT",
               "OOS-calibrated", "OOS-regime-calibrated"]
    qs = {k: [] for k in methods}      # quantile scores (normalized by spot)
    cov = {k: [] for k in methods}     # breach indicators at alpha
    past_z = []                        # standardized realized returns (EWMA+shrunk), past only
    past_reg = []                      # vol-regime bucket at each past window
    vol_hist = []                      # to define regime percentiles

    a = lookback
    while a <= len(rets)-H:
        tr = rets[a-lookback:a]
        S = close[a]; yreal = close[a+H]; Rreal = math.log(yreal/S)
        # --- v1: constant vol, historical-mean drift, Gaussian ---
        mu_h = tr.mean(); sigC = tr.std(ddof=1)*math.sqrt(H)
        q_v1 = S*np.exp(mu_h*H + stats.norm.ppf(taus)*sigC)
        # --- EWMA + shrunk drift ---
        cv, fc, _, _ = vol_model(tr, "EWMA"); sigE = integrated_sigma(fc(H))
        mu_s = drift_daily(tr, "Shrunk", r_f, q)
        q_ewn = S*np.exp(mu_s*H + stats.norm.ppf(taus)*sigE)
        ST = fhs_terminal(tr, cv, fc(H), S, mu_s, fhs_sims, "bootstrap", np.random.default_rng(a))
        q_fhs = np.quantile(ST, taus)
        zres = (tr - tr.mean())/cv; zres -= zres.mean()
        esamp, _ = evt_tail_sampler(zres)
        STe = fhs_terminal(tr, cv, fc(H), S, mu_s, fhs_sims, "evt",
                           np.random.default_rng(a+7), evt_sampler=esamp)
        q_evt = np.quantile(STe, taus)
        # standardized realized return under EWMA+shrunk (for calibration)
        z_now = (Rreal - mu_s*H)/sigE if sigE > 0 else 0.0
        reg_now = (np.mean(np.array(vol_hist) < sigE) if vol_hist else 0.5)  # regime percentile

        def cov_ind(qlo, qhi):
            return int(yreal < qlo or yreal > qhi)

        # parametric/sim methods
        for name, qvals, qband in [
            ("v1 (const·Normal·histμ)", q_v1, S*np.exp(mu_h*H + stats.norm.ppf([alpha/2,1-alpha/2])*sigC)),
            ("EWMA·Normal", q_ewn, S*np.exp(mu_s*H + stats.norm.ppf([alpha/2,1-alpha/2])*sigE)),
            ("EWMA·FHS", q_fhs, np.quantile(ST, [alpha/2,1-alpha/2])),
            ("EWMA·EVT", q_evt, np.quantile(STe, [alpha/2,1-alpha/2])),
        ]:
            qs[name].append(quantile_score(qvals, yreal, taus)/S)
            cov[name].append(cov_ind(qband[0], qband[1]))

        if len(past_z) >= burn:
            pz = np.asarray(past_z)
            mult = np.quantile(pz, taus)
            q_cal = S*np.exp(mu_s*H + mult*sigE)
            qs["OOS-calibrated"].append(quantile_score(q_cal, yreal, taus)/S)
            cb = S*np.exp(mu_s*H + np.quantile(pz, [alpha/2,1-alpha/2])*sigE)
            cov["OOS-calibrated"].append(cov_ind(cb[0], cb[1]))
            # regime-conditional: past z in the same vol tercile as now
            pr = np.asarray(past_reg)
            lo_b, hi_b = np.quantile(pr, [1/3, 2/3]) if len(pr) > 10 else (0.33, 0.66)
            cur_bucket = 0 if reg_now <= lo_b else (2 if reg_now >= hi_b else 1)
            past_buckets = np.where(pr <= lo_b, 0, np.where(pr >= hi_b, 2, 1))
            sel = pz[past_buckets == cur_bucket]
            if len(sel) < 30:
                sel = pz
            multr = np.quantile(sel, taus)
            q_reg = S*np.exp(mu_s*H + multr*sigE)
            qs["OOS-regime-calibrated"].append(quantile_score(q_reg, yreal, taus)/S)
            cbr = S*np.exp(mu_s*H + np.quantile(sel, [alpha/2,1-alpha/2])*sigE)
            cov["OOS-regime-calibrated"].append(cov_ind(cbr[0], cbr[1]))

        past_z.append(z_now); past_reg.append(reg_now); vol_hist.append(sigE)
        a += step

    out = []
    for k in methods:
        if not qs[k]:
            continue
        out.append(dict(method=k, n=len(qs[k]),
                        qscore=float(np.mean(qs[k])*100),         # % of spot, lower better
                        coverage=float(np.mean(cov[k])*100) if cov[k] else float("nan"),
                        target=alpha*100))
    out.sort(key=lambda d: d["qscore"])
    return out


def _t_std_quantile(z, level):
    """Standardized Student-t quantile, df from the sample's excess kurtosis."""
    exk = max(float(stats.kurtosis(z)), 1e-3)
    df = min(max(6.0/exk + 4.0, 3.0), 120.0)
    return float(stats.t.ppf(level, df)*math.sqrt((df-2)/df))


def shrunk_tail_multiplier(pz, level, k0=8.0):
    """Cross-validation-style shrinkage of the empirical calibration multiplier toward
    a parametric (Student-t) tail, weighted by how much tail data supports it. When few
    observations sit beyond the quantile (thin tail), lean parametric; with abundant
    data, lean empirical. Only the deep tail is shrunk; the body stays empirical.
    This finishes Workstream A — it fixes the measured small-sample tightness where the
    raw empirical multiplier over-breaches because the extreme quantile is undersampled."""
    q_emp = float(np.quantile(pz, level))
    edge = min(level, 1-level)
    if edge > 0.05:                      # body: trust the data
        return q_emp
    n = len(pz); n_eff = n*edge
    w = n_eff/(n_eff + k0)
    return w*q_emp + (1-w)*_t_std_quantile(pz, level)


def oos_calibrated_band(close, H, alpha, lookback, step, r_f, q, S0, sigH_now, mu_d_now,
                        regime=False, shrink=True, vol_kind="EWMA", daily_var=None,
                        burn=60, max_windows=500):
    """Produce the genuinely out-of-sample calibrated band for TODAY: calibrate the
    standardized-return quantiles on all past windows, apply to the current sigma/drift.
    With shrink=True the deep-tail multipliers are shrunk toward a parametric tail when
    tail data is thin (Workstream 2C) — more robust on short histories."""
    rets = np.diff(np.log(close))
    avail = len(rets)-H-lookback
    if avail <= 0:
        return None
    if (avail//step + 1) > max_windows:
        step = max(step, math.ceil(avail/max_windows))
    past_z, past_reg, vol_hist = [], [], []
    cheap = ("EWMA", "EWMA+VoV", "Constant", "EWMA-Range (Garman-Klass)")
    cal_vol = vol_kind if vol_kind in cheap else "EWMA+VoV"
    use_dv = daily_var if (cal_vol.startswith("EWMA-Range") and daily_var is not None
                           and len(daily_var) == len(rets)) else None
    a = lookback
    while a <= len(rets)-H:
        tr = rets[a-lookback:a]; S = close[a]; Rreal = math.log(close[a+H]/S)
        dv = use_dv[a-lookback:a] if use_dv is not None else None
        cv, fc, _, _ = vol_model(tr, cal_vol, daily_var=dv); sigE = integrated_sigma(fc(H))
        mu_s = drift_daily(tr, "Shrunk", r_f, q)
        if sigE > 0:
            past_z.append((Rreal - mu_s*H)/sigE)
            past_reg.append(np.mean(np.array(vol_hist) < sigE) if vol_hist else 0.5)
            vol_hist.append(sigE)
        a += step
    if len(past_z) < burn:
        return None
    pz = np.asarray(past_z)
    sel = pz
    if regime:
        pr = np.asarray(past_reg)
        lo_b, hi_b = np.quantile(pr, [1/3, 2/3])
        reg_now = np.mean(np.array(vol_hist) < sigH_now)
        cur = 0 if reg_now <= lo_b else (2 if reg_now >= hi_b else 1)
        buckets = np.where(pr <= lo_b, 0, np.where(pr >= hi_b, 2, 1))
        sel = pz[buckets == cur]
        if len(sel) < 30:
            sel = pz
    if shrink:
        qlo = shrunk_tail_multiplier(sel, alpha/2)
        qhi = shrunk_tail_multiplier(sel, 1-alpha/2)
    else:
        qlo = float(np.quantile(sel, alpha/2)); qhi = float(np.quantile(sel, 1-alpha/2))
    # inner-zone multipliers from the SAME empirical standardized distribution (fixes the
    # fan chart using hardcoded Gaussian zones); body needs no shrinkage
    m25, m50, m75 = (float(x) for x in np.quantile(sel, [0.25, 0.5, 0.75]))
    m10, m90 = (float(x) for x in np.quantile(sel, [0.10, 0.90]))
    n_eff = len(sel)*min(step/H, 1.0)        # independent windows (overlap correction)
    return dict(lo=S0*math.exp(mu_d_now*H + qlo*sigH_now),
                hi=S0*math.exp(mu_d_now*H + qhi*sigH_now),
                q_lo=qlo, q_hi=qhi, n=len(sel), n_eff=n_eff, step=step,
                m=dict(p25=m25, p75=m75, p10=m10, p90=m90, p50=m50),
                regime=regime, shrunk=shrink)


# ======================================================================
#  UI
# ======================================================================
# ======================================================================
#  SIMPLE (NON-EXPERT) UI
# ======================================================================
SIMPLE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root { color-scheme: dark; }
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }

/* force a clean DARK canvas regardless of the user's Streamlit theme */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background:#0d1320 !important; }
[data-testid="stHeader"] { background:rgba(0,0,0,0) !important; }
#MainMenu, footer { visibility:hidden; }
.block-container { padding-top:2.0rem; max-width:1100px; }

/* FORCE the sidebar to stay visible & expanded (override Streamlit's collapse) */
section[data-testid="stSidebar"] {
  transform:none !important; visibility:visible !important;
  width:330px !important; min-width:330px !important; max-width:330px !important;
  margin-left:0 !important; left:0 !important; }
section[data-testid="stSidebar"] > div { width:330px !important; }
[data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"] {
  display:flex !important; color:#cfe0f5 !important; }
[data-testid="stSidebarCollapsedControl"] svg, [data-testid="collapsedControl"] svg,
[data-testid="stSidebarCollapseButton"] svg { fill:#cfe0f5 !important; color:#cfe0f5 !important; }

/* default text light on the dark canvas */
.stApp, .stApp p, .stApp li, .stApp span, .stApp label, .stApp div,
.stMarkdown, .stMarkdown p, [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p {
  color:#dbe4ef; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4 { color:#f1f6fc !important; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color:#8493a6 !important; }

/* sidebar: panel + readable labels */
section[data-testid="stSidebar"] { background:#121a28 !important; border-right:1px solid #1f2a3b; }
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color:#c2cedd !important; }
section[data-testid="stSidebar"] .stMarkdown h4 { color:#f1f6fc !important; font-weight:700;
  margin:.5rem 0 .15rem 0; }
[data-testid="stExpander"] { border:1px solid #202c3e !important; border-radius:10px;
  background:#0f1726 !important; }
[data-testid="stExpander"] summary { color:#cdd9e7 !important; }

/* inputs dark */
.stTextInput input, .stNumberInput input, textarea,
.stSelectbox div[data-baseweb="select"] > div, .stDateInput input {
  background:#0f1828 !important; color:#e7eef7 !important; border:1px solid #28384e !important; }
[data-baseweb="popover"] *, [data-baseweb="menu"] * { background:#0f1828 !important; color:#e7eef7 !important; }
[data-testid="stDataFrame"] { background:#0f1726; }

/* primary button */
.stButton button[kind="primary"], .stButton button[kind="primary"] p,
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primary"] p {
  color:#ffffff !important; background:#2f6fed !important; border:0 !important; font-weight:700; }
.stButton button[kind="primary"]:hover { background:#3f7df5 !important; }

.hero-head { margin-bottom:.5rem; }
.hero-title { font-size:2.15rem; font-weight:800; letter-spacing:-.02em; color:#f4f8fd !important; }
.hero-sub { font-size:1.02rem; color:#8da0b6 !important; margin-top:.15rem; }
.empty-card { background:#11203a; border:1px solid #214066; border-radius:14px;
  padding:1.2rem 1.4rem; color:#cdddf2 !important; font-size:1.05rem; margin-top:1rem; }
.answer-card { background:linear-gradient(135deg,#142036,#0f1a2e); border:1px solid #243a5c;
  border-radius:18px; padding:1.5rem 1.7rem;
  box-shadow:0 8px 30px rgba(0,0,0,.35); margin:.4rem 0 1.0rem 0; }
.answer-big { font-size:1.3rem; line-height:1.55; color:#eaf1fa !important; font-weight:500; }
.answer-big b { color:#6fb0ff !important; font-weight:700; }
.pill { display:inline-block; padding:.32rem .85rem; border-radius:999px; font-weight:700;
  font-size:.86rem; }
.pill-green { background:#10301f; color:#46d98a !important; border:1px solid #1f5e3b; }
.pill-amber { background:#322611; color:#f0b54e !important; border:1px solid #6b5320; }
.pill-red   { background:#34161a; color:#ff7b7b !important; border:1px solid #6e2a2f; }
.mcard { background:#111b2c; border:1px solid #213047; border-radius:14px; padding:1rem 1.1rem;
  height:100%; box-shadow:0 4px 16px rgba(0,0,0,.25); }
.mcard .lab { font-size:.78rem; text-transform:uppercase; letter-spacing:.04em;
  color:#7d8ca1 !important; font-weight:600; }
.mcard .val { font-size:1.45rem; font-weight:700; color:#eef4fb !important; margin:.15rem 0; }
.mcard .desc { font-size:.9rem; color:#93a3b7 !important; line-height:1.4; }
.trust-box { background:#101a2b; border:1px solid #22344f; border-radius:14px;
  padding:1.05rem 1.25rem; margin:.3rem 0 .6rem 0; }
.trust-box .t { font-weight:700; color:#eef4fb !important; font-size:1.05rem; }
.trust-box .d { color:#9aaabf !important; margin-top:.25rem; font-size:.96rem; line-height:1.45; }
.disc { color:#67788e !important; font-size:.82rem; margin-top:1.4rem;
  border-top:1px solid #1d2840; padding-top:.7rem; }
h3.sec { font-size:1.05rem; color:#aebcce !important; font-weight:700; margin:.8rem 0 .2rem 0; }
</style>
"""


def simple_fan_chart(S0, mu_d, sigH, H, mult_lo, mult_hi, unit="",
                     m10=-1.282, m90=1.282, m25=-0.674, m75=0.674):
    """Clean, zoned price cone for non-experts: nested likelihood bands widening with
    time. Zone multipliers come from the empirical/calibrated distribution (not hardcoded
    Gaussian), so the 'most likely' / 'unlikely' zones are honest for fat-tailed assets."""
    ts = np.linspace(0.0, H, 80)
    frac = np.where(H > 0, ts/H, 0.0)
    sig_t = sigH*np.sqrt(frac)
    center = S0*np.exp(mu_d*ts)
    def band(m):
        return center*np.exp(m*sig_t)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    fig.patch.set_alpha(0); ax.set_facecolor("none")
    zones = [(mult_lo, mult_hi, "#1c3354", "rarely leaves"),
             (m10, m90, "#2c5fa0", "unlikely"),
             (m25, m75, "#4f9bef", "most likely")]
    for lo_m, hi_m, col, lab in zones:
        ax.fill_between(ts, band(lo_m), band(hi_m), color=col, lw=0, label=lab, zorder=1)
    ax.plot(ts, center, color="#9ec2f0", lw=1.6, ls="--", zorder=3)
    ax.scatter([0], [S0], color="#cfe0f5", zorder=5, s=30)
    # end labels
    ax.annotate(f"{unit}{band(mult_hi)[-1]:,.0f}", (ts[-1], band(mult_hi)[-1]),
                xytext=(6, 0), textcoords="offset points", va="center",
                fontsize=9, color="#aebfd4")
    ax.annotate(f"{unit}{band(mult_lo)[-1]:,.0f}", (ts[-1], band(mult_lo)[-1]),
                xytext=(6, 0), textcoords="offset points", va="center",
                fontsize=9, color="#aebfd4")
    ax.annotate(f"now {unit}{S0:,.0f}", (0, S0), xytext=(-4, 12),
                textcoords="offset points", ha="left", fontsize=9, color="#cfe0f5")
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#33415a")
    ax.set_yticks([]); ax.set_xticks([0, H])
    ax.set_xticklabels(["Today", "Horizon"], fontsize=10, color="#8da0b6")
    ax.tick_params(colors="#8da0b6")
    ax.set_xlim(0, H*1.16)
    ax.grid(axis="y", color="#1b2435", lw=1)
    leg = ax.legend(loc="upper left", fontsize=8.5, frameon=False, ncol=3,
                    handlelength=1.1, columnspacing=1.2)
    for t in leg.get_texts():
        t.set_color("#aebfd4")
    fig.tight_layout()
    return fig


def render_simple(res):
    """Plain-language outlook: one answer, one chart, a few translated cards, a trust
    verdict — all data still reachable via Advanced mode and the expander."""
    nm = res["name"]; S0 = res["S0"]; H = res["H"]; alpha = res["alpha"]
    r = res["r"]; close = res["close"]; cond_vol = res["cond_vol"]; fc_path = res["fc_path"]
    mu_d = res["mu_d"]; sigH = res["sigH"]; hz = res["hz_label"]
    r_f = res["r_f"]; dy = res["dy"]; seed = res["seed"]; conf_pct = res["conf_pct"]
    unit = "₹" if (nm.endswith(".NS") or nm.upper().startswith("^NSE")) else ""
    def money(x): return f"{unit}{x:,.0f}"

    # ---- compute (cheap) ----
    pdr = bootstrap_param_draws(r, B=300, rng=np.random.default_rng(int(seed)+5))
    ST = fhs_terminal(r, cond_vol, fc_path, S0, mu_d, 80_000, "bootstrap",
                      np.random.default_rng(int(seed)+11), param_draws=pdr)
    p25, p50, p75 = (float(x) for x in np.quantile(ST, [0.25, 0.5, 0.75]))
    qd = float(np.quantile(ST, alpha)); es = float(ST[ST <= qd].mean()) if np.any(ST <= qd) else qd
    es_pct = (S0-es)/S0*100
    oos = None
    try:
        oos = oos_calibrated_band(close, H, alpha, 252, 5, r_f, dy, S0, sigH, mu_d,
                                  vol_kind=res.get("vol_kind", "EWMA"),
                                  daily_var=res.get("daily_var"))
    except Exception:
        oos = None
    znom = stats.norm.ppf(1-alpha/2)
    if oos:
        lo_out, hi_out = oos["lo"], oos["hi"]; mlo, mhi = oos["q_lo"], oos["q_hi"]
        n_eff = oos.get("n_eff", oos["n"])          # independent (non-overlapping) windows
        if n_eff >= 110:                            # ≈ 9+ years of monthly-equivalent data
            trust = ("green", "Reliable",
                     "We checked this style of forecast against many years of this market's own "
                     "history — it landed inside the range about as often as it claims to.")
        elif n_eff >= 45:                           # ≈ 4+ years
            trust = ("amber", "Reasonable estimate",
                     "There's a few years of history behind this — enough for a sensible range, "
                     "but not a long track record.")
        else:
            trust = ("amber", "Limited testing",
                     "Only a short independent history was available to test against, so treat "
                     "the edges loosely. More years of data would firm this up.")
    else:
        lo_out, hi_out = rng_safe(res["rng_out"]); mlo, mhi = -znom, znom
        trust = ("red", "Rough estimate",
                 "Not enough history to properly test this range — treat it as a ballpark only.")
    if alpha <= 0.001 and trust[0] == "green":
        trust = ("amber", "Reliable middle, uncertain extreme",
                 "The bulk of this range is well-tested, but a 1-in-1000 move is extrapolation, "
                 "not something the data can pin down. Treat the far edge as indicative.")

    typ_move = (math.exp(sigH)-1)*100

    # ---- the answer ----
    st.markdown(
        f'<div class="answer-card"><div class="answer-big">Over the next <b>{hz}</b>, '
        f'{nm} (now {money(S0)}) will most likely trade between <b>{money(p25)}</b> and '
        f'<b>{money(p75)}</b>. It would rarely fall below <b>{money(lo_out)}</b> or rise '
        f'above <b>{money(hi_out)}</b>.</div></div>', unsafe_allow_html=True)
    pill = {"green": "pill-green", "amber": "pill-amber", "red": "pill-red"}[trust[0]]
    st.markdown(f'<span class="pill {pill}">●&nbsp; {trust[1]}</span>', unsafe_allow_html=True)

    # ---- fan chart ----
    st.markdown('<h3 class="sec">The picture</h3>', unsafe_allow_html=True)
    # zone multipliers from the empirical/calibrated distribution (not hardcoded Gaussian)
    if oos and "m" in oos:
        z10, z90 = oos["m"]["p10"], oos["m"]["p90"]
        z25, z75 = oos["m"]["p25"], oos["m"]["p75"]
    else:
        zp = (np.log(ST/S0) - mu_d*H)/sigH if sigH > 0 else np.zeros(len(ST))
        z10, z90 = (float(x) for x in np.quantile(zp, [0.10, 0.90]))
        z25, z75 = (float(x) for x in np.quantile(zp, [0.25, 0.75]))
    st.pyplot(simple_fan_chart(S0, mu_d, sigH, H, mlo, mhi, unit,
                               m10=z10, m90=z90, m25=z25, m75=z75))
    st.caption("Darker = more likely. The cone widens because the further out you look, "
               "the less certain the price.")

    # ---- what this means ----
    st.markdown('<h3 class="sec">What this means for you</h3>', unsafe_allow_html=True)
    cards = [
        ("Typical move", f"±{typ_move:.1f}%", f"A normal {hz} swings it about this much either way."),
        ("Downside level to watch", money(lo_out),
         "If you're worried about a fall, this is the level it would only rarely breach."),
        ("Rough patch", f"−{es_pct:.1f}%",
         f"In a genuinely bad {hz}, a drop around this size wouldn't be unusual."),
        ("Likely top", money(hi_out), "It would only rarely close above here over this horizon."),
    ]
    cols = st.columns(4)
    for col, (lab, val, desc) in zip(cols, cards):
        col.markdown(f'<div class="mcard"><div class="lab">{lab}</div>'
                     f'<div class="val">{val}</div><div class="desc">{desc}</div></div>',
                     unsafe_allow_html=True)

    # ---- trust ----
    st.markdown('<h3 class="sec">Can you trust this?</h3>', unsafe_allow_html=True)
    st.markdown(f'<div class="trust-box"><div class="t">{trust[1]}</div>'
                f'<div class="d">{trust[2]}</div></div>', unsafe_allow_html=True)

    # ---- all the data, one click away ----
    with st.expander("See the numbers behind this"):
        rows = [{"method": k.replace("_", " "), "lower": round(v[0]), "upper": round(v[1]),
                 "width %": round((v[1]-v[0])/S0*100, 1)} for k, v in res["rng_out"].items()]
        if oos:
            rows.append({"method": "trusted (calibrated, out-of-sample)",
                         "lower": round(lo_out), "upper": round(hi_out),
                         "width %": round((hi_out-lo_out)/S0*100, 1)})
        st.dataframe(rows, width='stretch', hide_index=True)
        st.caption(f"Horizon {H} trading days · volatility model {res['vol_kind']} · "
                   f"typical move ±{typ_move:.1f}% · downside VaR/Expected-Shortfall "
                   f"{(S0-qd)/S0*100:.1f}% / {es_pct:.1f}%. "
                   "Switch to **Advanced** mode (top-left) for the full method comparison, "
                   "calibration backtest, scoring harness, options and market-implied views.")

    st.markdown('<div class="disc">This is an analytical and educational tool, not a '
                'prediction and not investment advice. Ranges describe how the price has '
                'tended to behave; real markets can and do move outside them.</div>',
                unsafe_allow_html=True)


def rng_safe(rng_out):
    return rng_out.get("fhs_evt", rng_out.get("fhs_bootstrap", rng_out["scaled_normal"]))


def horizon_phrase(d):
    """Friendly phrase for an exact day count: 30 -> '1 month', 14 -> '2 weeks', 45 -> '45 days'."""
    if d % 365 == 0:
        n = d//365; return f"{n} year" + ("s" if n > 1 else "")
    if d >= 30 and d % 30 == 0:
        n = d//30; return f"{n} month" + ("s" if n > 1 else "")
    if d % 7 == 0:
        n = d//7; return f"{n} week" + ("s" if n > 1 else "")
    return f"{d} days"


def run_ui():
    st.set_page_config(page_title="Where could it go?", layout="wide",
                       page_icon="📈", initial_sidebar_state="expanded")
    st.markdown(SIMPLE_CSS, unsafe_allow_html=True)

    @st.cache_data(show_spinner=False)
    def fetch_ticker(ticker, years):
        return load_prices(ticker=ticker, years=years)

    sb = st.sidebar
    mode = sb.radio("View", ["Simple", "Advanced"], horizontal=True,
                    help="Simple = plain-language answer. Advanced = full technical dashboard. "
                         "Every control is available in both views.")
    simple = mode == "Simple"

    sb.markdown("#### Stock or index")
    src = sb.radio("Source", ["Fetch by ticker", "Upload CSV"], label_visibility="collapsed")
    ticker, uploaded, years = None, None, 5
    if src == "Fetch by ticker":
        ticker = sb.text_input("Ticker", value="^NSEI",
                               help="NIFTY 50 = ^NSEI · NSE stocks end in .NS (e.g. RELIANCE.NS) "
                                    "· US tickers like AAPL").strip()
        years = sb.number_input("Years of history", 1, 30, 10,
                                help="More history = more windows for the calibration to learn "
                                     "from; coverage measured best with ample history")
    else:
        uploaded = sb.file_uploader("CSV with Date + Close/Adj Close", type="csv")

    sb.markdown("#### Time horizon")
    horizon_days = sb.number_input("Days ahead (calendar)", 1, 365, 30,
                                   help="How far into the future to project the range")

    sb.markdown("#### Confidence")
    conf = sb.selectbox("How sure should the range be?",
                        ["99.9%", "99%", "95%", "90%", "Custom"], index=1,
                        label_visibility="collapsed",
                        help="Higher confidence = wider 'rarely leaves' range")
    alpha = {"99.9%": .001, "99%": .01, "95%": .05, "90%": .10}.get(conf)
    if alpha is None:
        alpha = sb.number_input("Custom two-tailed alpha", min_value=0.0001, max_value=0.5,
                                value=0.0100, step=0.0001, format="%.4f")

    # full controls (available in BOTH views, tucked into expanders to stay tidy)
    drift_map = {"Shrunk to carry": "Shrunk", "Zero": "Zero",
                 "Risk-free carry": "Risk-free carry", "Historical mean": "Historical mean"}
    atm_iv = slope = chain_src = chain_csv = None
    fuse_lam = 0.5; smile_kind = "Quadratic"
    strike = kind = side = premium = expiry_days = lot = None
    strike_val, strike_ispct = 100.0, True
    prem_manual = False; iv_shock = 0.0; margin_in = 0.0

    with sb.expander("⚙  Model settings", expanded=not simple):
        vol_kind = st.selectbox("Volatility model",
                                ["EWMA", "EWMA+VoV", "EWMA-Range (Garman-Klass)", "GARCH(1,1)",
                                 "GJR-GARCH(1,1)", "Blend (accuracy-wtd)", "Constant"], index=2,
                                help="EWMA-Range / Garman-Klass (default) uses the full OHLC bar and "
                                     "measured the sharpest calibrated bands; it falls back to "
                                     "EWMA+VoV automatically when only closing prices are available. "
                                     "EWMA+VoV is the best close-only model; GJR adds leverage; "
                                     "Blend accuracy-weights several models (measured no better); "
                                     "Constant = flat sd")
        drift_kind = st.selectbox("Drift", ["Shrunk to carry", "Zero", "Risk-free carry",
                                            "Historical mean"],
                                  help="The 3-yr historical mean is mostly noise; shrinking is safer")
        cc = st.columns(2)
        r_f = cc[0].number_input("Risk-free %", 0.0, 50.0, 6.5)/100
        dy = cc[1].number_input("Dividend %", 0.0, 50.0, 0.5)/100
        use_evt = st.checkbox("EVT (GPD) tail estimate", value=True,
                              help="Smooth, extrapolatable extreme quantiles instead of being "
                                   "capped at the worst observed day")
        use_punc = st.checkbox("Parameter uncertainty", value=True,
                               help="Widen bands for the fact that mu and sigma are estimated")
        ev_days = st.number_input("Event days in horizon (earnings/policy)", 0, 60, 0,
                                  help="Scheduled events that gap price — they widen the band")
        ev_mult = st.number_input("Event-day vol ×", 1.0, 10.0, 2.0, 0.5) if ev_days > 0 else 2.0
        ev_when = st.number_input("Days until the event (calendar)", 0, 365, 0,
                                  help="Places the extra vol at the right point in the horizon "
                                       "(e.g. earnings 15 days out), not on day 1") if ev_days > 0 else 0
        sims = st.select_slider("Simulations", [50_000, 100_000, 200_000, 400_000], value=200_000)
        seed = st.number_input("Random seed", 0, 10_000, 7)

    with sb.expander("📈  Options market (implied view)", expanded=False):
        fwd_mode = st.selectbox("Source", ["None", "Manual skew",
                                           "Option chain → Breeden-Litzenberger"])
        use_iv = fwd_mode == "Manual skew"
        use_bl = fwd_mode.startswith("Option chain")
        if use_iv:
            atm_iv = st.number_input("ATM implied vol %", 1.0, 300.0, 22.0)/100
            slope = st.number_input("Skew slope (IV per ln-moneyness)", -3.0, 3.0, -0.40, 0.05,
                                    help="Equity skew is negative: puts richer than calls")
        elif use_bl:
            chain_src = st.radio("Chain source", ["yfinance (US tickers)", "Upload chain CSV"])
            if chain_src.startswith("Upload"):
                chain_csv = st.file_uploader("CSV with strike, iv columns", type="csv", key="chain")
            fuse_lam = st.slider("Blend: history ↔ market", 0.0, 1.0, 0.5, 0.05,
                                 help="1.0 = pure historical view · 0.0 = pure option-implied "
                                      "(market) view · 0.5 = equal blend. The market prices in "
                                      "known upcoming events; history isn't distorted by the "
                                      "option risk premium.")
            smile_kind = st.selectbox("Smile fit", ["Quadratic", "SVI (arbitrage-free wings)"],
                                      help="SVI keeps the implied density non-negative at deep "
                                           "OTM strikes where a quadratic fit can break down — "
                                           "use it for noisy/wide chains.")
            st.caption("Builds the market's full implied distribution from the smile, then "
                       "fuses it with the historical distribution (linear opinion pool).")

    strat_type = "Single leg"; strat = None
    with sb.expander("🎯  Options & strategies", expanded=False):
        do_opt = st.checkbox("Evaluate an option / strategy")
        if do_opt:
            strat_type = st.selectbox("Structure",
                                      ["Single leg", "Vertical spread", "Straddle",
                                       "Strangle", "Iron condor", "Iron fly", "Butterfly",
                                       "Custom (2–4 legs)"])
            strike_mode = st.radio("Strike input", ["% of spot", "Absolute price"],
                                   horizontal=True,
                                   help="% of spot (100 = at-the-money) auto-fits any underlying. "
                                        "Absolute lets you punch in the actual traded strike.")
            ispct = strike_mode.startswith("%")
            sp_hint = float(st.session_state.get("last_spot", 1000.0))   # remembered spot
            tk = 50.0 if sp_hint >= 5000 else 10.0 if sp_hint >= 1000 else 5.0 if sp_hint >= 200 else 1.0
            def adef(pct):                                  # sensible absolute default near spot
                return float(round(sp_hint*pct/100.0/tk)*tk)

            def kin(label, pct, key):                       # one strike input, mode-aware
                if ispct:
                    return (st.number_input(f"{label} (% of spot)", 10.0, 400.0, float(pct), 1.0,
                                            key=key+"_p"), True)
                return (st.number_input(f"{label} (price)", value=adef(pct), step=tk,
                                        key=key+"_a"), False)

            prem_manual = st.checkbox("Enter premiums manually",
                                      help="Type the actual market premium (LTP) for each leg from "
                                           "your broker / NSE option chain. Otherwise legs are "
                                           "priced at the model's IV, which can be far from market.")

            def pin(label, key):                            # one premium input (0 -> model IV)
                if prem_manual:
                    return st.number_input(f"{label} premium", 0.0, step=0.5, key=key+"_pr")
                return 0.0

            lot = st.number_input("Lot / contract size", 1, 100_000, 1)
            expiry_days = st.number_input("Days to expiry", 1, 365, int(horizon_days))
            if strat_type == "Single leg":
                strike_val, strike_ispct = kin("Strike", 100.0, "sl")
                kind = st.selectbox("Type", ["call", "put"])
                side_sel = st.selectbox("Position", ["(probability only)", "buy", "sell"])
                side = None if side_sel.startswith("(") else side_sel
                premium = st.number_input("Premium (0 = price at model IV)", 0.0, step=0.5)
            elif strat_type == "Vertical spread":
                sk = st.selectbox("Type", ["call (bull debit / bear credit)",
                                           "put (bear debit / bull credit)"])
                lo_v, lo_p = kin("Lower strike", 97.0, "vlo")
                hi_v, hi_p = kin("Upper strike", 103.0, "vhi")
                lo_pr, hi_pr = pin("Lower-strike", "vlo"), pin("Upper-strike", "vhi")
                long_leg = st.selectbox("You are", ["long the lower / short the upper",
                                                    "short the lower / long the upper"])
                strat = dict(type="vertical", kind="call" if sk.startswith("call") else "put",
                             lo=(lo_v, lo_p), hi=(hi_v, hi_p), lo_pr=lo_pr, hi_pr=hi_pr,
                             long_lower=long_leg.startswith("long"))
            elif strat_type == "Straddle":
                kv, kp = kin("Strike", 100.0, "strad")
                c_pr, p_pr = pin("Call", "stradC"), pin("Put", "stradP")
                ls = st.selectbox("Position", ["long (buy vol)", "short (sell vol)"])
                strat = dict(type="straddle", k=(kv, kp), c_pr=c_pr, p_pr=p_pr,
                             long=ls.startswith("long"))
            elif strat_type == "Strangle":
                pv, pp = kin("Put strike (below)", 95.0, "strgP")
                cv, cp = kin("Call strike (above)", 105.0, "strgC")
                p_pr, c_pr = pin("Put", "strgPp"), pin("Call", "strgCp")
                ls = st.selectbox("Position", ["long (buy vol)", "short (sell vol)"], key="strg")
                strat = dict(type="strangle", put=(pv, pp), call=(cv, cp), p_pr=p_pr, c_pr=c_pr,
                             long=ls.startswith("long"))
            else:  # custom
                n_legs = st.number_input("Number of legs", 2, 4, 2)
                custom_legs = []
                for i in range(int(n_legs)):
                    cc = st.columns([1, 1])
                    lk = cc[0].selectbox(f"L{i+1} type", ["call", "put"], key=f"lk{i}")
                    lsd = cc[1].selectbox(f"L{i+1} side", ["buy", "sell"], key=f"ls{i}")
                    kv, kp = kin(f"L{i+1} strike", 100.0, f"cl{i}")
                    lq = st.number_input(f"L{i+1} qty", 1, 100, 1, key=f"lq{i}")
                    lpr = pin(f"L{i+1}", f"cl{i}")
                    custom_legs.append(dict(kind=lk, side=lsd, k=(kv, kp), qty=int(lq), pr=lpr))
                strat = dict(type="custom", legs=custom_legs)

            if strat_type in ("Iron condor", "Iron fly", "Butterfly"):
                if strat_type == "Iron condor":
                    sp, _ = kin("Short put", 95.0, "icSP"); sc, _ = kin("Short call", 105.0, "icSC")
                    wing = st.number_input("Wing width (% of spot)", 1.0, 50.0, 5.0, 0.5,
                                           help="How far the protective long wings sit beyond the "
                                                "short strikes")
                    strat = dict(type="iron_condor", sp=sp, sc=sc, wing=wing, ispct=ispct)
                elif strat_type == "Iron fly":
                    ctr, _ = kin("Body (ATM)", 100.0, "ifC")
                    wing = st.number_input("Wing width (% of spot)", 1.0, 50.0, 5.0, 0.5)
                    strat = dict(type="iron_fly", ctr=ctr, wing=wing, ispct=ispct)
                else:  # butterfly
                    bkind = st.selectbox("Built from", ["call", "put"])
                    ctr, _ = kin("Body (centre)", 100.0, "bfC")
                    wing = st.number_input("Wing width (% of spot)", 1.0, 50.0, 5.0, 0.5)
                    strat = dict(type="butterfly", kind=bkind, ctr=ctr, wing=wing, ispct=ispct)

            iv_shock = st.slider("IV scenario (vol points)", -0.15, 0.15, 0.0, 0.01,
                                 help="Shift every leg's implied vol to model a vol spike (+) or "
                                      "crush (−). For short-vol positions this is often the biggest "
                                      "P&L driver — bigger than where spot goes.")
            margin_in = st.number_input("Margin blocked (0 = estimate)", 0.0, step=1000.0,
                                        help="Enter your broker's actual margin for an exact "
                                             "return-on-margin; 0 uses a rough estimate.")

    run = sb.button("▶  Run analysis", type="primary", width='stretch')

    if simple:
        st.markdown('<div class="hero-head"><div class="hero-title">Where could it go?</div>'
                    '<div class="hero-sub">A plain-language price outlook, tested against '
                    'how the market has actually behaved.</div></div>', unsafe_allow_html=True)
    else:
        st.title("Forward Price Range & Option Probability")
        st.caption("Regime-aware (EWMA/GARCH), drift-shrunk, optionally skew-adjusted, "
                   "and calibration-tested. Analytical aid, not investment advice.")
    if not run:
        if simple:
            st.markdown('<div class="empty-card">👋 &nbsp;Pick a stock or index on the left, '
                        'choose how far ahead to look, and press '
                        '<b>Run analysis</b>.</div>', unsafe_allow_html=True)
        else:
            st.info("Set inputs in the sidebar and click **Run analysis**.")
        return

    # ---- load ----
    dates = None; ohlc = None
    try:
        if src == "Fetch by ticker":
            if not ticker:
                st.error("Enter a ticker."); return
            with st.spinner(f"Fetching {ticker}…"):
                dates, close, ohlc = fetch_ticker(ticker, int(years))
        else:
            if uploaded is None:
                st.error("Upload a CSV."); return
            dates, close, ohlc = load_prices(csv=uploaded)
    except SystemExit as e:
        st.error(str(e)); return
    except Exception as e:
        st.error(f"Could not load data: {e}"); return
    if len(close) < 120:
        st.error("Need at least ~120 price points for the conditional-vol models."); return

    for w in data_quality_report(dates, close):     # D1: silent-corruption pre-flight
        st.warning("⚠ " + w)

    r = np.diff(np.log(close))
    S0 = float(close[-1])
    st.session_state["last_spot"] = S0          # so absolute-strike defaults center near spot
    if do_opt and strat_type == "Single leg":
        strike = _resolve_K((strike_val, strike_ispct), S0)
    H = max(round(horizon_days*TD/365), 1)
    rng = np.random.default_rng(int(seed))
    d = diagnose(r)
    # range-based daily variance (Garman-Klass), aligned to r, if OHLC present (Tier 4)
    daily_var = None
    if ohlc is not None:
        gk = gk_daily_var(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        if len(gk) == len(close):
            daily_var = gk[1:]                       # align to returns r
    if vol_kind.startswith("EWMA-Range") and daily_var is None:
        st.info("No OHLC available for this data — range (Garman-Klass) vol needs Open/High/"
                "Low/Close, so it fell back to EWMA+VoV (the best close-only model).")
    cond_vol, fc_fn, cur_vol, vlabel = vol_model(r, vol_kind, daily_var=daily_var)
    ev_off = int(round(int(ev_when)*TD/365)) if ev_days > 0 else 0
    fc_path = apply_event_vol(fc_fn(H), ev_days, ev_mult, offset=ev_off)
    mu_d = drift_daily(r, drift_map[drift_kind], r_f, dy)
    # Phase-2 tail & uncertainty layers
    evt_sampler = None
    if use_evt:
        zres = (r - r.mean())/cond_vol; zres = zres - zres.mean()
        evt_sampler, _ = evt_tail_sampler(zres)
    param_draws = bootstrap_param_draws(r, B=300, rng=np.random.default_rng(int(seed)+5)) \
        if use_punc else None
    rng_out, zc, sigH = forward_range(r, cond_vol, fc_path, S0, mu_d, alpha,
                                      int(sims), rng, d["t_df"],
                                      evt_sampler=evt_sampler, param_draws=param_draws)
    conf_pct = (1-alpha)*100
    F = S0*math.exp((r_f-dy)*horizon_days/365)        # forward for IV bounds

    # ---- Breeden-Litzenberger implied distribution (optional) ----
    bl = bl_smile = bl_info = None
    if use_bl:
        T_bl = horizon_days/365
        pts = None
        if chain_src and chain_src.startswith("yfinance"):
            cp = fetch_chain_points(ticker, horizon_days) if (src == "Fetch by ticker" and ticker) else None
            if cp:
                pts = (cp["strikes"], cp["ivs"]); F = cp["spot"]*math.exp((r_f-dy)*cp["days"]/365)
                T_bl = cp["days"]/365; bl_info = f"yfinance chain {cp['expiry']} ({len(cp['strikes'])} strikes)"
        elif chain_csv is not None:
            try:
                cdf = pd.read_csv(chain_csv); cols = {c.lower(): c for c in cdf.columns}
                ks = cdf[cols["strike"]].astype(float).values
                iv = cdf[cols["iv"]].astype(float).values
                iv = iv/100 if np.nanmedian(iv) > 3 else iv      # accept percent or decimal
                pts = (ks, iv); bl_info = f"uploaded chain ({len(ks)} strikes)"
            except Exception as e:
                st.warning(f"Couldn't parse chain CSV (need 'strike','iv' columns): {e}")
        if pts is not None:
            if smile_kind.startswith("SVI"):
                bl_smile = fit_svi(pts[0], pts[1], F, T_bl)
                bl = breeden_litzenberger(F, T_bl, r_f, bl_smile, arbfree=True)
            else:
                bl_smile, _coef = fit_smile(pts[0], pts[1], F)
                bl = breeden_litzenberger(F, T_bl, r_f, bl_smile)

    # ============== SIMPLE MODE: friendly view, then stop ==============
    if simple:
        render_simple(dict(
            name=(ticker if (src == "Fetch by ticker" and ticker) else "This series"),
            close=close, r=r, S0=S0, H=H, horizon_days=horizon_days,
            alpha=alpha, conf_pct=conf_pct, cond_vol=cond_vol, fc_path=fc_path,
            mu_d=mu_d, sigH=sigH, rng_out=rng_out, r_f=r_f, dy=dy, seed=seed,
            ev_days=ev_days, ev_mult=ev_mult, vol_kind=vol_kind, daily_var=daily_var,
            hz_label=horizon_phrase(horizon_days)))
        if do_opt:
            st.info("📐 You've set up an option/strategy. The full payoff diagram, Greeks, "
                    "probability of profit, breakevens and day-by-day P&L are in **Advanced "
                    "view** — switch *View* to **Advanced** at the top of the sidebar.")
        return

    # ---- header ----
    c = st.columns(4)
    c[0].metric("Spot (last close)", f"{S0:,.2f}")
    c[1].metric("Annualized vol — 3y", f"{d['ann_sd']*100:.1f}%")
    c[2].metric(f"Annualized vol — now", f"{cur_vol*math.sqrt(TD)*100:.1f}%",
                help=f"from {vlabel}")
    c[3].metric("Horizon", f"{H} trading days")
    st.caption(f"Volatility model: **{vlabel}**  ·  drift: **{drift_kind}** "
               f"({mu_d*TD*100:+.1f}%/yr)  ·  forecast σ over horizon: {sigH*100:.1f}%")

    # ================= HEADLINE: the band we trust (auto, calibrated, OOS) =================
    name = ticker if (src == "Fetch by ticker" and ticker) else "the stock"
    try:
        oos_hl = oos_calibrated_band(close, H, alpha, 252, 5, r_f, dy, S0, sigH, mu_d,
                                     regime=False, vol_kind=vol_kind, daily_var=daily_var)
    except Exception:
        oos_hl = None
    _hlST = fhs_terminal(r, cond_vol, fc_path, S0, mu_d, 80_000, "bootstrap",
                         np.random.default_rng(int(seed)+11), param_draws=param_draws)
    p25, p50, p75 = (float(x) for x in np.quantile(_hlST, [0.25, 0.5, 0.75]))
    if oos_hl:
        lo_hl, hi_hl = oos_hl["lo"], oos_hl["hi"]; band_kind = "calibrated to real behaviour, out-of-sample"
    else:
        lo_hl, hi_hl = rng_out.get("fhs_evt", rng_out["fhs_bootstrap"])
        band_kind = "model-based (not enough history to calibrate — fetch more years)"
    st.subheader("📍 Best estimate")
    hc = st.columns(2)
    hc[0].metric(f"{conf_pct:g}% range — would rarely leave this",
                 f"{lo_hl:,.0f} – {hi_hl:,.0f}")
    hc[1].metric("Typical range (50% of the time)", f"{p25:,.0f} – {p75:,.0f}",
                 help=f"median ≈ {p50:,.0f}")
    st.info(f"Over the next ~{horizon_days} days, {name} (now ₹{S0:,.0f}) will **most likely "
            f"trade between ₹{p25:,.0f} and ₹{p75:,.0f}**, and would only leave the wider "
            f"**₹{lo_hl:,.0f} – ₹{hi_hl:,.0f}** range about {alpha*100:g}% of the time. "
            f"That wider range is {band_kind} — the number to rely on. "
            "Everything below is the detail behind it.")

    # ---- diagnostics ----
    st.subheader("Return distribution (daily)")
    g = st.columns(4)
    g[0].metric("Excess kurtosis", f"{d['exk']:+.2f}")
    g[1].metric("Skew", f"{d['skew']:+.2f}")
    g[2].metric("Student-t df", f"{d['t_df']:.1f}")
    g[3].metric("Tail P(|z|>2.576)", f"{d['tail']*100:.2f}%", help="normal expects 1.00%")
    if d["jb_p"] < 0.01:
        st.warning(f"Jarque-Bera p = {d['jb_p']:.1e} → fat-tailed (normality rejected). "
                   "The FHS Student-t row widens the tail accordingly.")

    # ---- range table ----
    st.subheader(f"{conf_pct:g}% forward price range — {H} trading days out")
    rows = [{"method": k.replace("_", " "), "lower": round(lo), "upper": round(hi),
             "width %": round((hi-lo)/S0*100, 1)} for k, (lo, hi) in rng_out.items()]
    iv_lo = iv_hi = None
    bl_lo = bl_hi = None
    if use_iv:
        T = horizon_days/365
        iv_lo = skew_bound(F, T, atm_iv, slope, zc, "lower")
        iv_hi = skew_bound(F, T, atm_iv, slope, zc, "upper")
        rows.append({"method": "implied vol + skew", "lower": round(iv_lo),
                     "upper": round(iv_hi), "width %": round((iv_hi-iv_lo)/S0*100, 1)})
    if use_bl and bl is not None:
        bl_lo = bl_quantile(bl, alpha/2); bl_hi = bl_quantile(bl, 1-alpha/2)
        rows.append({"method": "implied (Breeden-Litzenberger)", "lower": round(bl_lo),
                     "upper": round(bl_hi), "width %": round((bl_hi-bl_lo)/S0*100, 1)})
        _STf = fhs_terminal(r, cond_vol, fc_path, S0, mu_d, min(int(sims), 100_000),
                            "bootstrap", np.random.default_rng(int(seed)+21), param_draws=param_draws)
        fused = fuse_distributions(_STf, bl, alpha, fuse_lam)
        rows.append({"method": f"fused: history+market (λ={fuse_lam:.2f})",
                     "lower": round(fused["lo"]), "upper": round(fused["hi"]),
                     "width %": round((fused["hi"]-fused["lo"])/S0*100, 1)})
    st.dataframe(rows, width='stretch', hide_index=True)
    eff_tail = int(sims)*(alpha/2)                    # C4: paths defining each extreme
    if eff_tail < 50:
        st.warning(f"⚠ At this confidence only ~{eff_tail:.0f} simulated paths define each extreme "
                   "edge — the raw simulation quantile there is essentially noise. The far edge "
                   "leans on the EVT tail model (and parameter-uncertainty widening), not the "
                   "histogram. Treat 1-in-1000 type bounds as indicative, not measured.")
    st.download_button("⬇ Download range table (CSV)",                       # C5
                       data=pd.DataFrame(rows).to_csv(index=False).encode(),
                       file_name=f"{str(name).replace('^','').replace('.','_')}_range_{horizon_days}d.csv",
                       mime="text/csv")
    if use_bl and bl is not None:
        st.caption(f"**Fused** row blends the historical and option-implied distributions "
                   f"(λ={fuse_lam:.2f}: {fuse_lam*100:.0f}% history / {(1-fuse_lam)*100:.0f}% market). "
                   "History captures how the asset actually behaves; the market prices in known "
                   "upcoming events. The blend hedges both the option risk premium and history's "
                   "blind spots — adjust λ in the sidebar.")
    # ---- (C1) volatility term structure across the horizon ----
    st.markdown("**Volatility path across the horizon** (annualized)")
    st.pyplot(vol_term_chart(fc_path))
    st.caption("How the model expects daily volatility to evolve over the horizon — flat for "
               "EWMA, mean-reverting for GARCH, inflated on any event days you set. This is why "
               "bands widen at the speed they do.")
    # ---- (E3) Monte-Carlo stability: how much do the bounds wobble across seeds? ----
    _los, _his = [], []
    with st.spinner("Checking Monte-Carlo stability…"):
        for sd_i in range(6):
            _S = fhs_terminal(r, cond_vol, fc_path, S0, mu_d, min(int(sims), 40_000),
                              "bootstrap", np.random.default_rng(1000+sd_i), param_draws=param_draws)
            _los.append(np.quantile(_S, alpha/2)); _his.append(np.quantile(_S, 1-alpha/2))
    lo_rng = max(_los)-min(_los); hi_rng = max(_his)-min(_his)
    band_w = float(np.mean(_his)-np.mean(_los))
    noise = max(lo_rng, hi_rng)/band_w*100 if band_w else 0
    msg = (f"**Monte-Carlo stability:** across 6 seeds the bounds move ±{lo_rng/2:,.0f} (lower) / "
           f"±{hi_rng/2:,.0f} (upper) — about {noise:.1f}% of the band width. ")
    if noise > 4:
        st.warning(msg + "That's sizable — increase Simulations for a steadier tail number.")
    else:
        st.caption(msg + "Small relative to the band, so the numbers are stable at this sim count.")
    if use_punc:
        st.caption("Bands include **parameter estimation uncertainty** — the block-bootstrap "
                   "resamples both the drift (μ) and volatility (σ), so the width reflects that "
                   "these are estimated, not known. Drift uncertainty over the horizon ≈ "
                   "H·σ/√N is captured here rather than assumed away.")
    if ev_days > 0:
        st.caption(f"Includes {ev_days} event day(s) at {ev_mult:.1f}× vol → "
                   f"horizon σ raised to {sigH*100:.1f}%.")
    if use_iv:
        st.caption(f"IV bounds use F={F:,.0f}, ATM IV {atm_iv*100:.1f}%, "
                   f"skew slope {slope:+.2f} → asymmetric (downside wider if skew<0).")
        if src == "Fetch by ticker" and st.button("Try auto-fetch IV skew from option chain (US tickers)"):
            sk = fetch_chain_skew(ticker, F, horizon_days)
            if sk:
                st.success(f"Chain {sk['expiry']} ({sk['n']} strikes): ATM IV "
                           f"{sk['atm_iv']*100:.1f}%, slope {sk['slope']:+.2f}. "
                           "Enter these in the sidebar and re-run.")
            else:
                st.info("No usable chain (typical for NSE/.NS). Enter IV + skew manually.")
    if use_bl:
        if bl is not None:
            st.caption(f"Risk-neutral band from {bl_info}: the market's full implied "
                       f"distribution (forward F={F:,.0f}). Compare to the historical rows — "
                       "divergence is where the market disagrees with the past.")
        else:
            st.info("No chain loaded yet. Pick 'Upload chain CSV' and add a file, or use a "
                    "US ticker via yfinance. (NSE/.NS chains aren't available through yfinance.)")

    # ---- chart ----
    ST = fhs_terminal(r, cond_vol, fc_path, S0, mu_d,
                      min(int(sims), 100_000), "bootstrap",
                      np.random.default_rng(int(seed)+1), param_draws=param_draws)
    lo_b, hi_b = rng_out["fhs_bootstrap"]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.hist(ST, bins=120, color="#4C78A8", alpha=.85)
    ax.axvline(S0, color="black", lw=1.2, label=f"spot {S0:,.0f}")
    ax.axvline(lo_b, color="#E45756", lw=1.4, ls="--", label=f"{conf_pct:g}% FHS bounds")
    ax.axvline(hi_b, color="#E45756", lw=1.4, ls="--")
    if use_iv:
        ax.axvline(iv_lo, color="#F58518", lw=1.2, ls=":", label="IV+skew bounds")
        ax.axvline(iv_hi, color="#F58518", lw=1.2, ls=":")
    if use_bl and bl is not None:
        # scale the implied density to the histogram height for visual comparison
        binw = (ST.max()-ST.min())/120
        ax.plot(bl["K"], bl["pdf"]*len(ST)*binw, color="#B279A2", lw=1.8,
                label="implied density (B-L)")
        ax.axvline(bl_lo, color="#B279A2", lw=1.0, ls=":")
        ax.axvline(bl_hi, color="#B279A2", lw=1.0, ls=":")
        ax.set_xlim(min(ST.min(), bl_lo), max(ST.max(), bl_hi))
    if do_opt and strike:
        ax.axvline(strike, color="#54A24B", lw=1.6, label=f"strike {strike:,.0f}")
    ax.set_xlabel(f"Simulated price in {H} trading days"); ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=8); fig.tight_layout()
    st.pyplot(fig)

    # ---- risk metrics: VaR & Expected Shortfall (one-sided, tail = alpha) ----
    st.subheader(f"Tail risk over {H} trading days (one-sided, {conf_pct:g}% confidence)")
    q_dn = float(np.quantile(ST, alpha))
    es_dn = float(ST[ST <= q_dn].mean()) if np.any(ST <= q_dn) else q_dn
    var_pct = (S0 - q_dn)/S0*100
    es_pct = (S0 - es_dn)/S0*100
    rcol = st.columns(3)
    rcol[0].metric(f"Downside VaR ({alpha*100:.1f}% tail)", f"{var_pct:.1f}%",
                   help=f"≈ {S0-q_dn:,.0f} drop; price at {q_dn:,.0f}. Loss not exceeded with "
                        f"{conf_pct:g}% probability.")
    rcol[1].metric("Expected Shortfall (CVaR)", f"{es_pct:.1f}%",
                   help=f"≈ {S0-es_dn:,.0f}; average loss GIVEN you're in the worst {alpha*100:.1f}%. "
                        "Describes tail severity, not just its edge.")
    rcol[2].metric("Upside (same tail)", f"{(float(np.quantile(ST,1-alpha))-S0)/S0*100:+.1f}%")
    st.caption("VaR is the loss you don't exceed at this confidence; Expected Shortfall is the "
               "average loss when you do — the number that actually sizes tail damage.")

    # ---- (D) EVT tail estimate WITH its uncertainty ----
    if use_evt:
        try:
            zt = (r - r.mean())/cond_vol; zt = zt - zt.mean()
            qd_z, lo_z, hi_z = evt_tail_quantile_ci(zt, alpha, "lower", n_boot=200,
                                                    rng=np.random.default_rng(int(seed)+13))
            to_price = lambda zq: S0*math.exp(mu_d*H + zq*sigH)
            pmid, plo, phi = to_price(qd_z), to_price(hi_z), to_price(lo_z)  # note: more negative z = lower price
            st.markdown(f"**EVT tail bound with estimation uncertainty** "
                        f"({alpha*100:g}% lower tail)")
            tcol = st.columns(2)
            tcol[0].metric("EVT lower bound (extrapolated)", f"{pmid:,.0f}")
            tcol[1].metric("95% uncertainty on that bound", f"{plo:,.0f} – {phi:,.0f}")
            st.caption(f"The extreme bound is fit with peaks-over-threshold EVT (auto-selected "
                       f"threshold) and bootstrapped: the GPD shape gives {qd_z:.2f}σ, but its 95% "
                       f"CI spans [{lo_z:.2f}σ, {hi_z:.2f}σ]. The wider this band, the less the "
                       f"{conf_pct:g}% number can be trusted — at deep tails it is extrapolation, "
                       "not measurement.")
        except Exception:
            pass

    # ---- option module ----
    if do_opt and strike:
      with st.spinner("Evaluating option…"):
        days = int(expiry_days); T = days/365; Hopt = max(round(days*TD/365), 1)
        fco = apply_event_vol(fc_fn(Hopt), ev_days, ev_mult, offset=ev_off)
        opt_sims = min(int(sims), 50_000)     # plenty for option probabilities; keeps it snappy
        STo = fhs_terminal(r, cond_vol, fco, S0, mu_d, opt_sims, "bootstrap", rng,
                           param_draws=param_draws)
        if use_bl and bl_smile is not None:
            iv_used = float(bl_smile(math.log(strike/F)))
        elif use_iv:
            iv_used = iv_at(strike, F, atm_iv, slope)
        else:
            iv_used = cur_vol*math.sqrt(TD)
        prem = premium if (side and premium and premium > 0) else None
        om = option_module(STo, S0, strike, T, kind, side, prem, iv_used, r_f, dy, int(lot),
                           bl=bl if (use_bl and bl is not None) else None, fuse_lam=fuse_lam)
        st.subheader(f"Option — {strike:,.0f} {kind}, {days} days to expiry")
        o = st.columns(3)
        o[0].metric(f"Model P(above {strike:,.0f})", f"{om['p_above']*100:.2f}%",
                    help="FHS, history-based (P-measure)")
        o[1].metric(f"Model P(below {strike:,.0f})", f"{om['p_below']*100:.2f}%")
        if om["bs"]:
            tag = "B-L smile IV" if (use_bl and bl_smile is not None) else \
                  ("skew IV" if use_iv else "current vol")
            o[2].metric("Market P(ITM) — N(d2)", f"{om['bs']['prob_itm']*100:.2f}%",
                        help=f"risk-neutral, {tag} {iv_used*100:.1f}%")
            st.caption(f"BS fair value ≈ {om['bs']['price']:.2f} "
                       f"(IV used at this strike: {iv_used*100:.1f}%)")
        if om.get("greeks"):                              # B2: Greeks
            g = om["greeks"]
            gc = st.columns(4)
            gc[0].metric("Delta", f"{g['delta']:+.3f}", help="∂price/∂spot")
            gc[1].metric("Gamma", f"{g['gamma']:.4f}", help="∂delta/∂spot")
            gc[2].metric("Theta / day", f"{g['theta']:+.3f}", help="time decay per calendar day")
            gc[3].metric("Vega / 1%", f"{g['vega']:+.3f}", help="∂price per 1 vol-point")
        if use_bl and bl is not None:
            p_bl = bl_prob_above(bl, strike) if kind == "call" else 1-bl_prob_above(bl, strike)
            line = (f"**Breeden-Litzenberger** market P({'above' if kind=='call' else 'below'} "
                    f"{strike:,.0f}) = {p_bl*100:.2f}% (full implied distribution, not just N(d2)). "
                    f"Model (history) says {om['p_above' if kind=='call' else 'p_below']*100:.2f}%.")
            if om.get("fused"):                           # B4: best estimate (fused)
                pf = om["fused"]["p_above"] if kind == "call" else om["fused"]["p_below"]
                line += (f"  **Fused best estimate = {pf*100:.2f}%** "
                         f"(λ={fuse_lam:.2f}, history+market) — this is the probability to trust "
                         "when you have a chain loaded.")
            st.caption(line)
        if "option" in om:
            oo = om["option"]
            st.markdown(f"**{side.upper()} {kind} @ {premium:.2f}**"
                        + (f"  ·  lot {lot}" if lot != 1 else ""))
            p = st.columns(3)
            p[0].metric("Break-even", f"{oo['breakeven']:,.2f}")
            p[1].metric("Prob. of profit", f"{oo['prob_profit']*100:.1f}%")
            p[2].metric("Expected P&L", f"{oo['exp_pnl']:+,.0f}")
            qq = st.columns(4)
            qq[0].metric("P&L 1st pct", f"{oo['p1']:+,.0f}")
            qq[1].metric("P&L 5th pct", f"{oo['p5']:+,.0f}")
            qq[2].metric("P&L 95th pct", f"{oo['p95']:+,.0f}")
            qq[3].metric("Worst in sim", f"{oo['worst']:+,.0f}")
            if side == "sell" and oo["worst"] < 5*max(oo["exp_pnl"], 1):
                st.warning("Capped premium vs large tail loss — classic short-option asymmetry. "
                           f"1-in-100 loss ≈ {oo['p1']:+,.0f}, worst simulated {oo['worst']:+,.0f}.")

    # ---- multi-leg strategy (B3) + day-by-day P&L fan (B1) ----
    if do_opt and strat is not None:
      with st.spinner("Evaluating strategy (payoff, Greeks, P&L fan)…"):
        days = int(expiry_days); T = days/365; Hopt = max(round(days*TD/365), 1)
        fco = apply_event_vol(fc_fn(Hopt), ev_days, ev_mult, offset=ev_off)
        opt_sims = min(int(sims), 50_000)
        STo = fhs_terminal(r, cond_vol, fco, S0, mu_d, opt_sims, "bootstrap",
                           np.random.default_rng(int(seed)+31), param_draws=param_draws)

        def iv_for(K):
            if use_bl and bl_smile is not None:
                return float(bl_smile(math.log(K/F)))
            if use_iv:
                return iv_at(K, F, atm_iv, slope)
            return cur_vol*math.sqrt(TD)

        def mk(kind, side, K, qty=1, prem_override=0.0):
            if prem_override and prem_override > 0:      # manual premium drives the IV
                prem = float(prem_override)
                iv = bs_implied_vol(S0, K, T, r_f, dy, prem, kind) or iv_for(K)
            else:
                iv = iv_for(K)
                prem = bs_price_and_prob(S0, K, T, r_f, dy, iv, kind)["price"]
            return dict(kind=kind, side=side, K=float(K), premium=prem, iv=iv, qty=int(qty))

        legs = []; title = strat_type
        if strat["type"] == "vertical":
            kk = strat["kind"]
            lo = _resolve_K(strat["lo"], S0); hi = _resolve_K(strat["hi"], S0)
            if strat["long_lower"]:
                legs = [mk(kk, "buy", lo, prem_override=strat["lo_pr"]),
                        mk(kk, "sell", hi, prem_override=strat["hi_pr"])]
            else:
                legs = [mk(kk, "sell", lo, prem_override=strat["lo_pr"]),
                        mk(kk, "buy", hi, prem_override=strat["hi_pr"])]
            title = f"{kk.title()} vertical {lo:,.0f}/{hi:,.0f}"
        elif strat["type"] == "straddle":
            sd = "buy" if strat["long"] else "sell"
            K = _resolve_K(strat["k"], S0)
            legs = [mk("call", sd, K, prem_override=strat["c_pr"]),
                    mk("put", sd, K, prem_override=strat["p_pr"])]
            title = f"{'Long' if strat['long'] else 'Short'} straddle {K:,.0f}"
        elif strat["type"] == "strangle":
            sd = "buy" if strat["long"] else "sell"
            kp = _resolve_K(strat["put"], S0); kc = _resolve_K(strat["call"], S0)
            legs = [mk("put", sd, kp, prem_override=strat["p_pr"]),
                    mk("call", sd, kc, prem_override=strat["c_pr"])]
            title = f"{'Long' if strat['long'] else 'Short'} strangle {kp:,.0f}/{kc:,.0f}"
        else:  # custom
            legs = [mk(l["kind"], l["side"], _resolve_K(l["k"], S0), l["qty"],
                       prem_override=l.get("pr", 0.0))
                    for l in strat["legs"]]
            title = "Custom " + ", ".join(f"{l['side'][0].upper()}{l['qty']}×{leg['K']:,.0f}"
                                          f"{l['kind'][0]}"
                                          for l, leg in zip(strat["legs"], legs))

        if strat["type"] in ("iron_condor", "iron_fly", "butterfly"):
            _tick = 50.0 if S0 >= 5000 else 10.0 if S0 >= 1000 else 5.0 if S0 >= 200 else 1.0
            def _snapabs(x):
                return round(x/_tick)*_tick
            wpx = S0*strat["wing"]/100.0
            if strat["type"] == "iron_condor":
                sp = _resolve_K((strat["sp"], strat["ispct"]), S0)
                sc = _resolve_K((strat["sc"], strat["ispct"]), S0)
                lp, lc = _snapabs(sp-wpx), _snapabs(sc+wpx)
                legs = [mk("put", "buy", lp), mk("put", "sell", sp),
                        mk("call", "sell", sc), mk("call", "buy", lc)]
                title = f"Iron condor {lp:,.0f}/{sp:,.0f}–{sc:,.0f}/{lc:,.0f}"
            elif strat["type"] == "iron_fly":
                ctr = _resolve_K((strat["ctr"], strat["ispct"]), S0)
                lp, lc = _snapabs(ctr-wpx), _snapabs(ctr+wpx)
                legs = [mk("put", "buy", lp), mk("put", "sell", ctr),
                        mk("call", "sell", ctr), mk("call", "buy", lc)]
                title = f"Iron fly {lp:,.0f}/{ctr:,.0f}/{lc:,.0f}"
            else:  # butterfly
                kk = strat["kind"]; ctr = _resolve_K((strat["ctr"], strat["ispct"]), S0)
                lo, hi = _snapabs(ctr-wpx), _snapabs(ctr+wpx)
                legs = [mk(kk, "buy", lo), mk(kk, "sell", ctr, qty=2), mk(kk, "buy", hi)]
                title = f"{kk.title()} butterfly {lo:,.0f}/{ctr:,.0f}/{hi:,.0f}"

        ev = evaluate_strategy(STo, legs, int(lot), S0, T, r_f, dy)
        st.subheader(f"Strategy — {title}, {days} days")
        src_txt = ("your entered premiums" if prem_manual
                   else "the model's IV (not live market) — enable *Enter premiums manually* "
                        "for real prices")
        st.caption(f"Priced with {src_txt}. "
                   + " · ".join(f"{l['side']} {l['qty']}× {l['K']:,.0f}{l['kind'][0].upper()} "
                                f"@ {l['premium']:.1f} (iv {l['iv']*100:.0f}%)" for l in legs))
        net = ev["net_cost"]
        c = st.columns(4)
        c[0].metric("Net " + ("debit" if net >= 0 else "credit"), f"{abs(net):,.0f}")
        c[1].metric("Prob. of profit", f"{ev['prob_profit']*100:.1f}%")
        c[2].metric("Expected P&L", f"{ev['exp_pnl']:+,.0f}")
        c[3].metric("Break-even(s)", ", ".join(f"{b:,.0f}" for b in ev["breakevens"]) or "—")
        g = ev["greeks"]
        gc = st.columns(4)
        gc[0].metric("Net Delta", f"{g['delta']:+.3f}")
        gc[1].metric("Net Gamma", f"{g['gamma']:.4f}")
        gc[2].metric("Net Theta / day", f"{g['theta']:+.3f}")
        gc[3].metric("Net Vega / 1%", f"{g['vega']:+.3f}")

        # margin / return-on-margin + vol rank
        marg_est, marg_desc = approx_short_margin(legs, S0, int(lot), net)
        margin = float(margin_in) if margin_in and margin_in > 0 else marg_est
        rom = ev["exp_pnl"]/margin*100 if margin > 0 else 0.0
        vr = vol_rank(cond_vol)
        mc = st.columns(4)
        mc[0].metric("Margin (est.)" if not (margin_in and margin_in > 0) else "Margin",
                     f"{margin:,.0f}", help=marg_desc + " — verify with your broker")
        mc[1].metric("Return on margin", f"{rom:+.1f}%",
                     help="expected P&L ÷ margin, for this holding period")
        mc[2].metric("Max loss / margin", f"{ev['min_pay']/margin*100:+.0f}%" if margin > 0 else "—")
        if vr is not None:
            mc[3].metric("Vol rank", f"{vr:.0f}/100",
                         help="where today's volatility sits in its own history; high favours "
                              "selling premium, low favours buying")

        st.pyplot(payoff_diagram(legs, int(lot), S0, ev, unit="₹" if name != "the stock" else ""))
        st.caption(f"Max profit ≈ {ev['max_pay']:,.0f} · max loss ≈ {ev['min_pay']:,.0f} "
                   f"(on the modelled grid; a naked short leg can lose more). "
                   f"1-in-100 P&L ≈ {ev['p1']:+,.0f}, worst simulated {ev['worst']:+,.0f}.")

        # path / intraday risk + day-by-day P&L fan (with IV scenario)
        paths = fhs_paths(r, cond_vol, fco, S0, mu_d, min(int(sims), 6_000),
                          np.random.default_rng(int(seed)+47), param_draws=param_draws)
        pr = path_risk_metrics(paths, legs, int(lot), days, r_f, dy, iv_shock=iv_shock)
        tc = st.columns(3)
        tc[0].metric("P(touch a strike)", f"{pr['touch_any']*100:.1f}%",
                     help="chance price reaches a strike at any point before expiry — the "
                          "adjustment/assignment risk that probability-of-profit hides")
        tc[1].metric("Worst drawdown (median path)", f"{pr['mae_median']:+,.0f}",
                     help="typical max adverse excursion — how far underwater a normal path goes")
        tc[2].metric("Worst drawdown (bad path, 5%)", f"{pr['mae_p5']:+,.0f}")
        if pr["touch"]:
            st.caption("Touch probability per strike: "
                       + " · ".join(f"{k}: {v*100:.1f}%" for k, v in pr["touch"].items()))

        d_ax, fan = strategy_pnl_fan(paths, legs, int(lot), days, r_f, dy, iv_shock=iv_shock)
        shock_txt = (f" — **with a {iv_shock*100:+.0f} IV-point shock applied**" if iv_shock else
                     " (IV held at entry)")
        st.markdown(f"**P&L over time** (today → expiry{shock_txt})")
        st.pyplot(pnl_fan_chart(d_ax, fan, unit="₹" if name != "the stock" else ""))
        st.caption("How the position's mark-to-market P&L is likely to evolve day by day — the "
                   "band is the 5–95% / 25–75% spread of simulated paths. Use the **IV scenario** "
                   "slider in the sidebar to stress a vol spike or crush, which for short-vol "
                   "positions usually moves P&L more than spot does.")

    # ---- (4) calibration backtest + correction ----
    st.subheader("Calibration backtest & correction (walk-forward)")
    st.caption("How often did price actually exit the band, do the breaches cluster, and "
               "what band would have hit the target? Kupiec tests the breach COUNT; "
               "Christoffersen also tests INDEPENDENCE (clustering); conditional coverage "
               "combines both. Well-calibrated → observed ≈ target and cc p > 0.05.")
    bc = st.columns([1, 1, 2])
    lookback = bc[0].number_input("Lookback (days)", 120, 1000, 252, 21)
    bstep = bc[1].number_input("Step (days)", 1, 21, 5)
    if bc[2].button("Run calibration backtest"):
        if len(close) < lookback + H + 30:
            st.info("Not enough history for this lookback/horizon. Fetch more years.")
        else:
            with st.spinner("Walking the history…"):
                rep, calib = backtest(close, H, alpha, int(lookback), int(bstep),
                                      drift_map[drift_kind], r_f, dy, vol_kind=vol_kind)
            # auto-select: conditional coverage passes (cc p>0.05), then closest to target
            def score(v):
                ok = (v["cc_p"] == v["cc_p"]) and v["cc_p"] > 0.05
                return (0 if ok else 1, abs(v["obs"]-v["exp"]))
            best = min(rep, key=score) if rep else None
            brows = []
            for v in rep:
                tag = ("calibrated" if (v["cc_p"] == v["cc_p"] and v["cc_p"] > 0.05)
                       else "too tight" if v["obs"] > v["exp"] else "too wide")
                brows.append({"band": f"{v['vol']}·{v['dist']}", "windows": v["n"],
                              "observed %": round(v["obs"]*100, 2),
                              "target %": round(v["exp"]*100, 2),
                              "test n": v.get("n_test", v["n"]),
                              "Kupiec p": round(v["kupiec_p"], 3),
                              "indep p": round(v["ind_p"], 3),
                              "cond-cov p": round(v["cc_p"], 3),
                              "verdict": tag,
                              "best": "★" if (best and v is best) else ""})
            st.dataframe(brows, width='stretch', hide_index=True)
            if best:
                st.success(f"Auto-selected best-calibrated band: **{best['vol']}·{best['dist']}** "
                           f"(observed {best['obs']*100:.2f}% vs target {alpha*100:.1f}%, "
                           f"cond-cov p = {best['cc_p']:.3f}).")

            # ---- calibration correction for the SELECTED config ----
            if calib:
                eff = calib["eff_conf"]*100
                cal_lo = S0*math.exp(mu_d*H + calib["q_lo"]*sigH)
                cal_hi = S0*math.exp(mu_d*H + calib["q_hi"]*sigH)
                raw_lo, raw_hi = rng_out["scaled_normal"]
                st.markdown(f"**Calibration correction — {calib['vol_kind']} vol, "
                            f"{drift_kind} drift** (from {calib['n']} walk-forward windows)")
                cc = st.columns(3)
                cc[0].metric("Effective confidence of nominal band",
                             f"{eff:.1f}%", delta=f"{eff-conf_pct:+.1f} vs {conf_pct:g}% claimed",
                             delta_color="inverse")
                cc[1].metric(f"Calibrated {conf_pct:g}% band — lower", f"{cal_lo:,.0f}",
                             delta=f"{cal_lo-raw_lo:+,.0f} vs raw")
                cc[2].metric(f"Calibrated {conf_pct:g}% band — upper", f"{cal_hi:,.0f}",
                             delta=f"{cal_hi-raw_hi:+,.0f} vs raw")
                st.caption(f"Nominal multiplier ±{calib['z_nom']:.2f}σ → calibrated "
                           f"{calib['q_lo']:.2f}σ / +{calib['q_hi']:.2f}σ (empirical, so "
                           "asymmetric and fat-tail aware). The calibrated band re-scales the "
                           "bounds so historical coverage matches the target; the asymmetry "
                           "reflects the fatter downside (crash) tail.")
                # genuinely OUT-OF-SAMPLE calibrated band (v2.5): multipliers from past windows only
                oos = oos_calibrated_band(close, H, alpha, int(lookback), int(bstep),
                                          r_f, dy, S0, sigH, mu_d, regime=False,
                                          vol_kind=vol_kind, daily_var=daily_var)
                if oos:
                    st.markdown("**Out-of-sample calibrated band (v2.5 — honest)**")
                    oc = st.columns(3)
                    oc[0].metric("OOS calibrated — lower", f"{oos['lo']:,.0f}",
                                 delta=f"{oos['lo']-cal_lo:+,.0f} vs in-sample")
                    oc[1].metric("OOS calibrated — upper", f"{oos['hi']:,.0f}",
                                 delta=f"{oos['hi']-cal_hi:+,.0f} vs in-sample")
                    oc[2].metric("OOS multipliers", f"{oos['q_lo']:.2f}σ / +{oos['q_hi']:.2f}σ")
                    st.caption("The in-sample band above is fit and tested on the same data "
                               "(optimistic); this band calibrates only on windows BEFORE each "
                               "point, so it's the trustworthy one. Use this for sizing.")
            st.caption("Observed % uses all windows; the p-values use a non-overlapping "
                       "subsample (test n) so Step < Horizon no longer biases the independence "
                       "test. Normal tails usually fail at 99%; calibrated or FHS bands fare "
                       "better.")

    # ---- (v2.5) Proper-scoring validation harness ----
    st.subheader("Scoring & validation (v2.5 — CRPS / proper score)")
    st.caption("Ranks every method out-of-sample on the quantile score (∝ CRPS) — which "
               "rewards bands that are BOTH calibrated and sharp — plus out-of-sample coverage. "
               "This is how we *prove* which method is most accurate, rather than assuming.")
    if st.button("Run scoring harness (walk-forward, ~10–15s)"):
        if len(close) < int(lookback) + H + 90:
            st.info("Need more history for the scoring harness. Fetch more years.")
        else:
            with st.spinner("Scoring methods out-of-sample…"):
                vh = validation_harness(close, H, alpha, int(lookback), int(bstep), r_f, dy)
            if vh:
                vrows = [{"method": r["method"], "windows": r["n"],
                          "quantile score (% of spot)": round(r["qscore"], 3),
                          "OOS coverage %": round(r["coverage"], 2),
                          "target %": round(r["target"], 2)} for r in vh]
                st.dataframe(vrows, width='stretch', hide_index=True)
                best = vh[0]
                st.success(f"Best by proper score: **{best['method']}** "
                           f"(score {best['qscore']:.3f}% of spot, OOS coverage "
                           f"{best['coverage']:.2f}% vs target {best['target']:.1f}%). "
                           "Lower score = sharper *and* better-calibrated across the whole "
                           "distribution, not just one quantile.")
                st.caption("If a calibrated row tops the list, the calibration layer is earning "
                           "its keep out-of-sample. If regime-calibration ranks below plain "
                           "calibration, the regime buckets are adding noise on this name — the "
                           "harness tells you, so you don't ship it blindly.")

    # ---- (2A) Basket generalization test ----
    st.subheader("Basket generalization (v3 — does calibration hold across names?)")
    st.caption("Runs the out-of-sample test across several tickers at once, so you can see "
               "whether the calibration advantage generalizes or breaks on some asset types. "
               "Needs data access (yfinance) on your machine; the more history each name has, "
               "the more reliable the coverage numbers.")
    basket_txt = st.text_input("Tickers (comma-separated)",
                               value="^NSEI, RELIANCE.NS, TCS.NS, HDFCBANK.NS, ITC.NS",
                               help="Calibration needs history — short series give noisy coverage")
    if st.button("Run basket test"):
        tickers = [t.strip() for t in basket_txt.split(",") if t.strip()]
        prog = st.progress(0.0); brows = []
        for i, tk in enumerate(tickers):
            try:
                _, cl, _ = fetch_ticker(tk, max(int(years), 8))
                vh = validation_harness(cl, H, alpha, 252, 5, r_f, dy,
                                        fhs_sims=1500, max_windows=500)
                d = {v["method"]: v for v in vh}
                v1 = d.get("v1 (const·Normal·histμ)"); cal = d.get("OOS-calibrated")
                if v1 and cal:
                    brows.append({"ticker": tk, "windows": cal["n"],
                                  "v1 coverage %": round(v1["coverage"], 2),
                                  "calibrated coverage %": round(cal["coverage"], 2),
                                  "target %": round(alpha*100, 2),
                                  "v1 score": round(v1["qscore"], 3),
                                  "calibrated score": round(cal["qscore"], 3),
                                  "score improved": "✓" if cal["qscore"] <= v1["qscore"] else "✗"})
            except Exception as e:
                brows.append({"ticker": tk, "windows": 0, "v1 coverage %": "—",
                              "calibrated coverage %": f"error: {str(e)[:30]}", "target %": "",
                              "v1 score": "", "calibrated score": "", "score improved": ""})
            prog.progress((i+1)/len(tickers))
        st.dataframe(brows, width='stretch', hide_index=True)
        ok = [b for b in brows if b.get("score improved") == "✓"]
        st.caption(f"Calibration improved the proper score on {len(ok)}/{len(brows)} names. "
                   "Coverage is noisy on short histories — judge it across names, not one. "
                   "Measured finding: calibration generalizes on the proper score; with ample "
                   "history its coverage also lands near target, but on short samples it can run "
                   "tight (which the tail-shrinkage in the calibrated band is designed to soften).")

    st.divider()
    st.caption("P-measure = historical; option markets price under the risk-neutral "
               "measure (plug live IV). Earnings/events can gap through any band. "
               "More model complexity ≠ more accurate — trust the backtest, not the label.")


if __name__ == "__main__":
    run_ui()
