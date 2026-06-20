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
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats, optimize

TD = 252  # trading days / year


# ======================================================================
#  DATA
# ======================================================================
def load_prices(ticker=None, years=3, csv=None):
    import pandas as pd
    if csv is not None:
        df = pd.read_csv(csv)
        cols = {c.lower(): c for c in df.columns}
        dcol = cols.get("date")
        ccol = cols.get("adj close") or cols.get("adj_close") or cols.get("close")
        if ccol is None:
            sys.exit("CSV must contain a 'Close' (or 'Adj Close') column.")
        if dcol:
            df[dcol] = pd.to_datetime(df[dcol]); df = df.sort_values(dcol)
        return (df[dcol].values if dcol else np.arange(len(df))), df[ccol].astype(float).values
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("Please `pip install yfinance` or upload a CSV.")
    df = yf.download(ticker, period=f"{years}y", interval="1d",
                     auto_adjust=True, progress=False)
    if df is None or len(df) == 0:
        sys.exit(f"No data for '{ticker}'. NSE stocks use a .NS suffix (e.g. RELIANCE.NS).")
    close = df["Close"]
    if hasattr(close, "columns"):
        close = close.iloc[:, 0]
    close = close.dropna()
    return close.index.values, close.values.astype(float)


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


def vol_model(r, kind, quick=False):
    """Return (cond_vol_series, forecast_fn(H)->daily vol path, current_vol, label)."""
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


def breeden_litzenberger(F, T, r, smile_fn, span=6.0, N=801):
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


def apply_event_vol(fc_path, n_event_days, mult):
    """Inflate the forecast vol on n event days (earnings/policy) by a multiplier.
    Which days they fall on doesn't change the horizon-integrated variance, so we
    simply scale the first n entries. Raises the band over event-bearing horizons."""
    fc = np.asarray(fc_path, float).copy()
    n = int(min(max(n_event_days, 0), len(fc)))
    if n > 0 and mult > 1:
        fc[:n] = fc[:n]*mult
    return fc


def bs_price_and_prob(S0, K, T, r, q, vol, kind):
    if T <= 0 or vol <= 0:
        return None
    d1 = (math.log(S0/K) + (r-q+0.5*vol*vol)*T)/(vol*math.sqrt(T))
    d2 = d1 - vol*math.sqrt(T)
    N = stats.norm.cdf
    if kind == "call":
        return dict(price=S0*math.exp(-q*T)*N(d1)-K*math.exp(-r*T)*N(d2), prob_itm=N(d2))
    return dict(price=K*math.exp(-r*T)*N(-d2)-S0*math.exp(-q*T)*N(-d1), prob_itm=N(-d2))


def option_module(ST, S0, K, T, kind, side, premium, iv_used, r, q, lot):
    res = dict(p_above=float(np.mean(ST > K)), p_below=float(np.mean(ST < K)),
               vol_used=iv_used, bs=bs_price_and_prob(S0, K, T, r, q, iv_used, kind))
    if premium is not None and side and kind:
        intr = np.maximum(ST-K, 0) if kind == "call" else np.maximum(K-ST, 0)
        pnl = ((premium-intr) if side == "sell" else (intr-premium))*lot
        be = (K+premium) if kind == "call" else (K-premium)
        res["option"] = dict(breakeven=be, prob_profit=float(np.mean(pnl > 0)),
                             exp_pnl=float(pnl.mean()), p5=float(np.quantile(pnl, .05)),
                             p95=float(np.quantile(pnl, .95)),
                             worst=float(pnl.min()), best=float(pnl.max()))
    return res


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


def oos_calibrated_band(close, H, alpha, lookback, step, r_f, q, S0, sigH_now, mu_d_now,
                        regime=False, burn=60, max_windows=500):
    """Produce the genuinely out-of-sample calibrated band for TODAY: calibrate the
    standardized-return quantiles on all past windows, apply to the current sigma/drift."""
    rets = np.diff(np.log(close))
    avail = len(rets)-H-lookback
    if avail <= 0:
        return None
    if (avail//step + 1) > max_windows:
        step = max(step, math.ceil(avail/max_windows))
    past_z, past_reg, vol_hist = [], [], []
    a = lookback
    while a <= len(rets)-H:
        tr = rets[a-lookback:a]; S = close[a]; Rreal = math.log(close[a+H]/S)
        cv, fc, _, _ = vol_model(tr, "EWMA"); sigE = integrated_sigma(fc(H))
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
    qlo = float(np.quantile(sel, alpha/2)); qhi = float(np.quantile(sel, 1-alpha/2))
    return dict(lo=S0*math.exp(mu_d_now*H + qlo*sigH_now),
                hi=S0*math.exp(mu_d_now*H + qhi*sigH_now),
                q_lo=qlo, q_hi=qhi, n=len(sel), regime=regime)


# ======================================================================
#  UI
# ======================================================================
# ======================================================================
#  SIMPLE (NON-EXPERT) UI
# ======================================================================
SIMPLE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
#MainMenu, footer, header [data-testid="stToolbar"] { visibility: hidden; }
.block-container { padding-top: 2.2rem; max-width: 1080px; }
section[data-testid="stSidebar"] { background: #f7f9fc; border-right: 1px solid #eaeef3; }
.hero-head { margin-bottom: .4rem; }
.hero-title { font-size: 2.1rem; font-weight: 800; letter-spacing: -.02em; color: #0f1c2e; }
.hero-sub { font-size: 1.02rem; color: #5b6b7f; margin-top: .15rem; }
.empty-card { background:#f1f6fc; border:1px solid #dce7f3; border-radius:14px;
  padding:1.2rem 1.4rem; color:#33475b; font-size:1.05rem; margin-top:1rem; }
.answer-card { background: linear-gradient(135deg,#ffffff,#f5f9ff); border:1px solid #e4ecf6;
  border-radius:18px; padding:1.5rem 1.7rem; box-shadow:0 6px 24px rgba(20,50,90,.06);
  margin:.4rem 0 1.1rem 0; }
.answer-big { font-size:1.32rem; line-height:1.55; color:#13243a; font-weight:500; }
.answer-big b { color:#0a58ca; font-weight:700; }
.pill { display:inline-block; padding:.32rem .8rem; border-radius:999px; font-weight:700;
  font-size:.86rem; letter-spacing:.01em; }
.pill-green { background:#e3f6ec; color:#0f7a43; border:1px solid #b6e7c9; }
.pill-amber { background:#fff4e0; color:#9a6400; border:1px solid #ffe0a8; }
.pill-red   { background:#fde7e7; color:#b42323; border:1px solid #f6c5c5; }
.mcard { background:#ffffff; border:1px solid #e9eef4; border-radius:14px; padding:1rem 1.1rem;
  height:100%; box-shadow:0 2px 10px rgba(20,50,90,.04); }
.mcard .lab { font-size:.8rem; text-transform:uppercase; letter-spacing:.04em; color:#7a8aa0;
  font-weight:600; }
.mcard .val { font-size:1.5rem; font-weight:700; color:#13243a; margin:.15rem 0; }
.mcard .desc { font-size:.92rem; color:#5b6b7f; line-height:1.4; }
.trust-box { background:#f7fbff; border:1px solid #e1ecf8; border-radius:14px;
  padding:1.05rem 1.25rem; margin:.3rem 0 .6rem 0; }
.trust-box .t { font-weight:700; color:#13243a; font-size:1.05rem; }
.trust-box .d { color:#52627a; margin-top:.25rem; font-size:.96rem; line-height:1.45; }
.disc { color:#90a0b4; font-size:.83rem; margin-top:1.4rem; border-top:1px solid #edf1f6;
  padding-top:.7rem; }
h3.sec { font-size:1.05rem; color:#33475b; font-weight:700; margin:.6rem 0 .2rem 0; }
</style>
"""


def simple_fan_chart(S0, mu_d, sigH, H, mult_lo, mult_hi, unit=""):
    """Clean, zoned price cone for non-experts: nested likelihood bands widening with
    time. No jargon on the axes."""
    ts = np.linspace(0.0, H, 80)
    frac = np.where(H > 0, ts/H, 0.0)
    sig_t = sigH*np.sqrt(frac)
    center = S0*np.exp(mu_d*ts)
    def band(m):
        return center*np.exp(m*sig_t)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    fig.patch.set_alpha(0)
    zones = [(mult_lo, mult_hi, "#dbe9fb", "rarely leaves"),
             (-1.282, 1.282, "#a9ccf2", "unlikely"),
             (-0.674, 0.674, "#5b9bd5", "most likely")]
    for lo_m, hi_m, col, lab in zones:
        ax.fill_between(ts, band(lo_m), band(hi_m), color=col, lw=0, label=lab, zorder=1)
    ax.plot(ts, center, color="#0a3d62", lw=1.6, ls="--", zorder=3)
    ax.scatter([0], [S0], color="#0a3d62", zorder=5, s=30)
    # end labels
    ax.annotate(f"{unit}{band(mult_hi)[-1]:,.0f}", (ts[-1], band(mult_hi)[-1]),
                xytext=(6, 0), textcoords="offset points", va="center",
                fontsize=9, color="#33475b")
    ax.annotate(f"{unit}{band(mult_lo)[-1]:,.0f}", (ts[-1], band(mult_lo)[-1]),
                xytext=(6, 0), textcoords="offset points", va="center",
                fontsize=9, color="#33475b")
    ax.annotate(f"now {unit}{S0:,.0f}", (0, S0), xytext=(-4, 12),
                textcoords="offset points", ha="left", fontsize=9, color="#0a3d62")
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.set_yticks([]); ax.set_xticks([0, H])
    ax.set_xticklabels(["Today", "Horizon"], fontsize=10, color="#5b6b7f")
    ax.set_xlim(0, H*1.16)
    ax.grid(axis="y", color="#eef2f7", lw=1)
    leg = ax.legend(loc="upper left", fontsize=8.5, frameon=False, ncol=3,
                    handlelength=1.1, columnspacing=1.2)
    for t in leg.get_texts():
        t.set_color("#5b6b7f")
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
        oos = oos_calibrated_band(close, H, alpha, 252, 5, r_f, dy, S0, sigH, mu_d)
    except Exception:
        oos = None
    znom = stats.norm.ppf(1-alpha/2)
    if oos:
        lo_out, hi_out = oos["lo"], oos["hi"]; mlo, mhi = oos["q_lo"], oos["q_hi"]; n = oos["n"]
        if n >= 120:
            trust = ("green", "Reliable",
                     "We checked this style of forecast against several years of this market's "
                     "own history — it landed inside the range about as often as it claims to.")
        elif n >= 60:
            trust = ("amber", "Reasonable estimate",
                     "There's a few years of history behind this — enough for a sensible range, "
                     "but not a long track record.")
        else:
            trust = ("amber", "Limited testing",
                     "Only a short history was available to test against, so treat the edges loosely.")
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
    st.pyplot(simple_fan_chart(S0, mu_d, sigH, H, mlo, mhi, unit))
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
        st.dataframe(rows, use_container_width=True, hide_index=True)
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


def run_ui():
    st.set_page_config(page_title="Where could it go?", layout="wide",
                       page_icon="📈")
    st.markdown(SIMPLE_CSS, unsafe_allow_html=True)

    @st.cache_data(show_spinner=False)
    def fetch_ticker(ticker, years):
        return load_prices(ticker=ticker, years=years)

    sb = st.sidebar
    mode = sb.radio("Mode", ["Simple", "Advanced"], horizontal=True,
                    help="Simple = plain-language answer. Advanced = every number and diagnostic.")
    simple = mode == "Simple"

    sb.markdown("#### Stock or index")
    src = sb.radio("Source", ["Fetch by ticker", "Upload CSV"],
                   label_visibility="collapsed")
    ticker, years, uploaded = None, 5, None
    if src == "Fetch by ticker":
        ticker = sb.text_input("Ticker", value="^NSEI",
                               help="NIFTY 50 = ^NSEI · NSE stocks end in .NS (e.g. RELIANCE.NS) "
                                    "· US tickers like AAPL").strip()
        years = sb.number_input("Years of history", 1, 20, 5) if not simple else 5
    else:
        uploaded = sb.file_uploader("CSV with Date + Close/Adj Close", type="csv")

    sb.markdown("#### Look ahead")
    hz_label = sb.select_slider("Horizon", ["1 week", "2 weeks", "1 month", "3 months", "6 months"],
                                value="1 month", label_visibility="collapsed")
    horizon_days = {"1 week": 7, "2 weeks": 14, "1 month": 30,
                    "3 months": 91, "6 months": 182}[hz_label]

    # ---------- friendly defaults (used in Simple, overridable in Advanced) ----------
    vol_kind = "EWMA"; drift_kind = "Shrunk to carry"; r_f = 0.065; dy = 0.005
    use_evt = True; use_punc = True; ev_days = 0; ev_mult = 2.0
    sims = 200_000; seed = 7
    fwd_mode = "None"; use_iv = use_bl = False; atm_iv = slope = None
    chain_src = chain_csv = None
    do_opt = False; strike = kind = side = premium = expiry_days = lot = None
    drift_map = {"Shrunk to carry": "Shrunk", "Zero": "Zero",
                 "Risk-free carry": "Risk-free carry", "Historical mean": "Historical mean"}

    if simple:
        sb.markdown("#### How cautious?")
        cau = sb.select_slider("Caution", ["Balanced", "Cautious", "Very cautious"],
                               value="Cautious", label_visibility="collapsed",
                               help="How wide the 'rarely leaves' range is")
        alpha = {"Balanced": 0.05, "Cautious": 0.01, "Very cautious": 0.001}[cau]
        conf = {0.05: "95%", 0.01: "99%", 0.001: "99.9%"}[alpha]
        with sb.expander("Assumptions (optional)"):
            r_f = st.number_input("Risk-free %", 0.0, 50.0, 6.5)/100
            dy = st.number_input("Dividend yield %", 0.0, 50.0, 0.5)/100
            ev_days = st.number_input("Known event days ahead (earnings etc.)", 0, 60, 0)
            ev_mult = st.number_input("Event-day jumpiness ×", 1.0, 10.0, 2.0, 0.5) if ev_days > 0 else 2.0
    else:
        sb.header("2 · Range settings")
        conf = sb.selectbox("Confidence", ["99.9%", "99%", "95%", "90%", "Custom"], index=1)
        alpha = {"99.9%": .001, "99%": .01, "95%": .05, "90%": .10}.get(conf)
        if alpha is None:
            alpha = sb.number_input("alpha (two-tailed)", min_value=0.0001, max_value=0.5,
                                    value=0.0100, step=0.0001, format="%.4f")
        horizon_days = sb.number_input("Horizon (calendar days)", 1, 365, int(horizon_days))
        vol_kind = sb.selectbox("Volatility model",
                                ["EWMA", "EWMA+VoV", "GARCH(1,1)", "GJR-GARCH(1,1)",
                                 "Blend (accuracy-wtd)", "Constant"],
                                help="EWMA+VoV adds a vol-of-vol inflation that measured slightly "
                                     "better out-of-sample; GJR adds the leverage effect; "
                                     "Blend accuracy-weights several models; Constant = flat sample sd")
        drift_kind = sb.selectbox("Drift", ["Shrunk to carry", "Zero", "Risk-free carry",
                                            "Historical mean"],
                                  help="3-yr historical mean is mostly noise; shrinking is safer")
        r_f = sb.number_input("Risk-free %", 0.0, 50.0, 6.5)/100
        dy = sb.number_input("Dividend yield %", 0.0, 50.0, 0.5)/100
        use_evt = sb.checkbox("EVT (GPD) tail row", value=True,
                              help="Adds a peaks-over-threshold tail row — smooth, extrapolatable "
                                   "extreme quantiles instead of being capped at the worst observed day")
        use_punc = sb.checkbox("Propagate parameter uncertainty", value=True,
                               help="Block-bootstrap the window to widen bands for the fact that "
                                    "mu and sigma are estimated, not known")
        ev_days = sb.number_input("Event days in horizon (earnings/policy)", 0, 60, 0,
                                  help="Scheduled events that gap price — they widen the band")
        ev_mult = sb.number_input("Event-day vol ×", 1.0, 10.0, 2.0, 0.5) if ev_days > 0 else 2.0
        sims = sb.select_slider("Simulations", [50_000, 100_000, 200_000, 400_000], value=200_000)
        seed = sb.number_input("Random seed", 0, 10_000, 7)

        sb.header("3 · Forward-looking (optional)")
        fwd_mode = sb.selectbox("Source", ["None", "Manual skew",
                                           "Option chain → Breeden-Litzenberger"])
        use_iv = fwd_mode == "Manual skew"
        use_bl = fwd_mode.startswith("Option chain")
        if use_iv:
            atm_iv = sb.number_input("ATM implied vol %", 1.0, 300.0, 22.0)/100
            slope = sb.number_input("Skew slope (IV per ln-moneyness)", -3.0, 3.0, -0.40, 0.05,
                                    help="Equity skew is negative: puts richer than calls")
        elif use_bl:
            chain_src = sb.radio("Chain source", ["yfinance (US tickers)", "Upload chain CSV"])
            if chain_src.startswith("Upload"):
                chain_csv = sb.file_uploader("CSV with strike, iv columns", type="csv", key="chain")
            sb.caption("Builds the market's full implied distribution from the smile "
                       "(not just a slope). NSE/.NS: export the chain and upload as CSV.")

        sb.header("4 · Option (optional)")
        do_opt = sb.checkbox("Evaluate an option / strike")
        if do_opt:
            strike = sb.number_input("Strike", value=25000.0, step=50.0)
            kind = sb.selectbox("Type", ["call", "put"])
            side_sel = sb.selectbox("Position", ["(probability only)", "buy", "sell"])
            side = None if side_sel.startswith("(") else side_sel
            premium = sb.number_input("Premium (0 = skip P&L)", 0.0, step=0.5)
            expiry_days = sb.number_input("Days to expiry", 1, 365, int(horizon_days))
            lot = sb.number_input("Lot / contract size", 1, 100_000, 1)

    run = sb.button("▶  Run analysis", type="primary", use_container_width=True)

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
    try:
        if src == "Fetch by ticker":
            if not ticker:
                st.error("Enter a ticker."); return
            with st.spinner(f"Fetching {ticker}…"):
                _, close = fetch_ticker(ticker, int(years))
        else:
            if uploaded is None:
                st.error("Upload a CSV."); return
            _, close = load_prices(csv=uploaded)
    except SystemExit as e:
        st.error(str(e)); return
    except Exception as e:
        st.error(f"Could not load data: {e}"); return
    if len(close) < 120:
        st.error("Need at least ~120 price points for the conditional-vol models."); return

    r = np.diff(np.log(close))
    S0 = float(close[-1])
    H = max(round(horizon_days*TD/365), 1)
    rng = np.random.default_rng(int(seed))
    d = diagnose(r)
    cond_vol, fc_fn, cur_vol, vlabel = vol_model(r, vol_kind)
    fc_path = apply_event_vol(fc_fn(H), ev_days, ev_mult)
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
                import pandas as pd
                cdf = pd.read_csv(chain_csv); cols = {c.lower(): c for c in cdf.columns}
                ks = cdf[cols["strike"]].astype(float).values
                iv = cdf[cols["iv"]].astype(float).values
                iv = iv/100 if np.nanmedian(iv) > 3 else iv      # accept percent or decimal
                pts = (ks, iv); bl_info = f"uploaded chain ({len(ks)} strikes)"
            except Exception as e:
                st.warning(f"Couldn't parse chain CSV (need 'strike','iv' columns): {e}")
        if pts is not None:
            bl_smile, _coef = fit_smile(pts[0], pts[1], F)
            bl = breeden_litzenberger(F, T_bl, r_f, bl_smile)

    # ============== SIMPLE MODE: friendly view, then stop ==============
    if simple:
        render_simple(dict(
            name=(ticker if (src == "Fetch by ticker" and ticker) else "This series"),
            close=close, r=r, S0=S0, H=H, horizon_days=horizon_days, hz_label=hz_label,
            alpha=alpha, conf_pct=conf_pct, cond_vol=cond_vol, fc_path=fc_path,
            mu_d=mu_d, sigH=sigH, rng_out=rng_out, r_f=r_f, dy=dy, seed=seed,
            ev_days=ev_days, ev_mult=ev_mult, vol_kind=vol_kind))
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
        oos_hl = oos_calibrated_band(close, H, alpha, 252, 5, r_f, dy, S0, sigH, mu_d, regime=False)
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
    st.dataframe(rows, use_container_width=True, hide_index=True)
    # ---- (E3) Monte-Carlo stability: how much do the bounds wobble across seeds? ----
    _los, _his = [], []
    for sd_i in range(6):
        _S = fhs_terminal(r, cond_vol, fc_path, S0, mu_d, min(int(sims), 100_000),
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
                      min(int(sims), 200_000), "bootstrap",
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
        days = int(expiry_days); T = days/365; Hopt = max(round(days*TD/365), 1)
        fco = apply_event_vol(fc_fn(Hopt), ev_days, ev_mult)
        STo = fhs_terminal(r, cond_vol, fco, S0, mu_d, int(sims), "bootstrap", rng,
                           param_draws=param_draws)
        if use_bl and bl_smile is not None:
            iv_used = float(bl_smile(math.log(strike/F)))
        elif use_iv:
            iv_used = iv_at(strike, F, atm_iv, slope)
        else:
            iv_used = cur_vol*math.sqrt(TD)
        prem = premium if (side and premium and premium > 0) else None
        om = option_module(STo, S0, strike, T, kind, side, prem, iv_used, r_f, dy, int(lot))
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
        if use_bl and bl is not None:
            p_bl = bl_prob_above(bl, strike) if kind == "call" else 1-bl_prob_above(bl, strike)
            st.caption(f"**Breeden-Litzenberger** market P({'above' if kind=='call' else 'below'} "
                       f"{strike:,.0f}) = {p_bl*100:.2f}% (full implied distribution, not just N(d2)). "
                       f"Model (history) says {om['p_above' if kind=='call' else 'p_below']*100:.2f}% — "
                       "the gap is the market-vs-history disagreement that signals edge.")
        if "option" in om:
            oo = om["option"]
            st.markdown(f"**{side.upper()} {kind} @ {premium:.2f}**"
                        + (f"  ·  lot {lot}" if lot != 1 else ""))
            p = st.columns(3)
            p[0].metric("Break-even", f"{oo['breakeven']:,.2f}")
            p[1].metric("Prob. of profit", f"{oo['prob_profit']*100:.1f}%")
            p[2].metric("Expected P&L", f"{oo['exp_pnl']:+,.0f}")
            qq = st.columns(3)
            qq[0].metric("P&L 5th pct", f"{oo['p5']:+,.0f}")
            qq[1].metric("P&L 95th pct", f"{oo['p95']:+,.0f}")
            qq[2].metric("Worst in sim", f"{oo['worst']:+,.0f}")
            if side == "sell" and oo["worst"] < 5*max(oo["exp_pnl"], 1):
                st.warning("Capped premium vs large tail loss — classic short-option asymmetry.")

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
            st.dataframe(brows, use_container_width=True, hide_index=True)
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
                                          r_f, dy, S0, sigH, mu_d, regime=False)
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
                st.dataframe(vrows, use_container_width=True, hide_index=True)
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

    st.divider()
    st.caption("P-measure = historical; option markets price under the risk-neutral "
               "measure (plug live IV). Earnings/events can gap through any band. "
               "More model complexity ≠ more accurate — trust the backtest, not the label.")


if __name__ == "__main__":
    run_ui()
