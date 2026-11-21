"""Standardized drought indices on monthly ``(time, lat, lon)`` arrays, fitted per calendar month.

* SPI: two-parameter gamma distribution (Thom maximum-likelihood approximation) with a zero-precipitation mass.
* SPEI: three-parameter log-logistic distribution fitted by unbiased probability-weighted moments
  (Vicente-Serrano et al., 2010).
* SMI: empirical (Weibull plotting position) percentile mapped to a standard normal.

All fits use the baseline period only; values are clipped to [-3.09, 3.09].
"""
import numpy as np
import xarray as xr
from scipy import special, stats

__all__ = ["spi", "spei", "smi"]

CLIP = 3.09


def _accumulate(x, scale):
    return x.rolling(time=scale, min_periods=scale).sum() if scale > 1 else x


def _by_month(acc, baseline, fit_apply):
    out = xr.full_like(acc, np.nan)
    ref = acc.sel(time=slice(str(baseline[0]), str(baseline[1])))
    for m in range(1, 13):
        sel = acc.time.dt.month == m
        r = ref.sel(time=ref.time.dt.month == m).values
        r = r[np.isfinite(r).any(axis=(1, 2))]   # drop years still inside the accumulation spin-up
        x = acc.sel(time=sel).values
        z = fit_apply(r, x)
        valid = np.isfinite(x) & (np.isfinite(r).sum(axis=0) >= 10)  # NaN input / ocean stay NaN
        out[dict(time=np.where(sel)[0])] = np.where(valid, z, np.nan)
    return out.clip(-CLIP, CLIP)


def _spi_month(ref, x):
    with np.errstate(all="ignore"):
        q = np.mean(ref <= 0, axis=0)
        pos = np.where(ref > 0, ref, np.nan)
        mean = np.nanmean(pos, axis=0)
        a = np.log(mean) - np.nanmean(np.log(pos), axis=0)
        alpha = (1 + np.sqrt(1 + 4 * a / 3)) / (4 * a)
        beta = mean / alpha
        g = special.gammainc(alpha, np.where(x > 0, x, 0) / beta)
        p = np.where(x > 0, q + (1 - q) * g, q)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return stats.norm.ppf(p)


def _pwm(sorted_x, s):
    n = sorted_x.shape[0]
    i = np.arange(1, n + 1)[:, None, None]
    w = special.comb(n - i, s) / special.comb(n - 1, s)
    return np.nansum(w * sorted_x, axis=0) / n


def _spei_month(ref, x):
    """Log-logistic is fitted on D where its PWM shape beta > 1; for left-skewed samples
    (beta <= 1 or undefined, common in humid-tropical dry seasons) it is fitted on -D and
    F = 1 - F(-x) (reflected log-logistic)."""
    f = _loglogistic_cdf(ref, x)
    fr = 1.0 - _loglogistic_cdf(-ref, -x)
    ok, okr = _beta_ok(ref), _beta_ok(-ref)
    F = np.where(ok, f, np.where(okr, fr, np.nan))
    return stats.norm.ppf(np.clip(F, 1e-6, 1 - 1e-6))


def _beta_ok(ref):
    with np.errstate(all="ignore"):
        srt = np.sort(ref, axis=0)
        w0, w1, w2 = (_pwm(srt, s) for s in (0, 1, 2))
        beta = (2 * w1 - w0) / (6 * w1 - w0 - 6 * w2)
        return np.isfinite(beta) & (beta > 1)


def _loglogistic_cdf(ref, x):
    with np.errstate(all="ignore"):
        srt = np.sort(ref, axis=0)
        w0, w1, w2 = (_pwm(srt, s) for s in (0, 1, 2))
        beta = (2 * w1 - w0) / (6 * w1 - w0 - 6 * w2)
        gb1 = special.gamma(1 + 1 / beta)
        gb2 = special.gamma(1 - 1 / beta)
        alpha = (w0 - 2 * w1) * beta / (gb1 * gb2)
        gam = w0 - alpha * gb1 * gb2
        f = 1.0 / (1 + (alpha / (x - gam)) ** beta)
        return np.where(x - gam <= 0, 1e-6, f)


def _smi_month(ref, x):
    n = np.sum(np.isfinite(ref), axis=0)
    rank = np.sum(ref[None, ...] <= x[:, None, ...], axis=1)
    p = (rank + 0.5) / (n + 1)
    return stats.norm.ppf(np.clip(p, 1e-6, 1 - 1e-6))


def spi(precip, scale, baseline):
    """Standardized Precipitation Index (McKee et al., 1993), fitted per calendar month.

    Parameters
    ----------
    precip : xarray.DataArray
        Monthly precipitation with dimensions ``(time, lat, lon)``.
    scale : int
        Accumulation period in months (e.g. 3 for SPI-3).
    baseline : (int, int)
        First and last year of the calibration period, e.g. ``(1981, 2020)``.

    Returns
    -------
    xarray.DataArray
        SPI clipped to [-3.09, 3.09]; NaN where the input is NaN.
    """
    return _by_month(_accumulate(precip, scale), baseline, _spi_month).rename(f"spi{scale}")


def spei(precip, pet, scale, baseline):
    """Standardized Precipitation Evapotranspiration Index (Vicente-Serrano et al., 2010).

    Three-parameter log-logistic distribution fitted by probability-weighted moments; left-skewed samples are fitted
    on the reflected water balance. Parameters as in :func:`spi`, with ``pet`` the potential evapotranspiration.
    """
    return _by_month(_accumulate(precip - pet, scale), baseline, _spei_month).rename(f"spei{scale}")


def smi(sm, scale, baseline):
    """Soil-moisture index: empirical percentile of ``sm`` (mean over ``scale`` months) mapped to a standard normal."""
    acc = sm.rolling(time=scale, min_periods=scale).mean() if scale > 1 else sm
    return _by_month(acc, baseline, _smi_month).rename(f"smi{scale}")
