"""Event-level significance against phase-randomized surrogate fields.

H0(e): event ``e`` is a fluctuation of a stationary field with the same space–time power spectrum and marginal
distribution as the analysed field. Surrogates are 3-D Fourier phase randomizations (Theiler et al., 1992) with the
marginal restored by rank remapping. Each surrogate is processed with exactly the same DRIFT configuration.

Event score (volume-weighted deficit contrast): ``S_e = (max D_e - floor) (V_e / v_min)^gamma``.

* ``p_fw``: family-wise p-value from the maximum score of every surrogate;
* ``p_pool``: per-event p-value from all surrogate events, to be combined with Benjamini–Hochberg (:func:`bh`).

The test is conservative because the surrogates keep part of the power of the observed events.
"""
import numpy as np
import pandas as pd

from .core import detect

__all__ = ["phase_surrogate", "event_scores", "pvalues", "bh", "surrogate_test"]


def phase_surrogate(x, rng):
    """Phase-randomized surrogate of ``x`` (NaN cells stay NaN; marginal distribution kept)."""
    valid = np.isfinite(x)
    z = np.where(valid, x, 0.0).astype(np.float64)
    f = np.fft.rfftn(z)
    ph = np.exp(2j * np.pi * rng.random(f.shape))
    ph.flat[0] = 1.0
    y = np.fft.irfftn(np.abs(f) * ph, s=z.shape, axes=tuple(range(z.ndim)))
    out = np.full(x.shape, np.nan, np.float32)
    out[valid] = np.sort(x[valid])[np.argsort(np.argsort(y[valid]))]
    return out


def event_scores(labels, deficit, cell_area, floor, v_min, gamma=1.0):
    """Score ``S_e`` of every event. Returns ``{event: (score, peak_deficit, volume)}``."""
    w3 = np.broadcast_to(np.asarray(cell_area, float), labels.shape[1:])
    w3 = np.broadcast_to(w3, labels.shape)
    m = labels > 0
    ids = labels[m]
    if ids.size == 0:
        return {}
    n = int(ids.max()) + 1
    vol = np.bincount(ids, weights=w3[m], minlength=n)
    peak = np.full(n, -np.inf)
    np.maximum.at(peak, ids, np.nan_to_num(deficit[m], nan=-np.inf))
    return {int(e): (float(max(peak[e] - floor, 0.0) * (vol[e] / v_min) ** gamma), float(peak[e]), float(vol[e]))
            for e in np.unique(ids)}


def pvalues(obs, null_scores):
    """Family-wise and pooled p-values of observed scores against lists of surrogate event scores."""
    N = len(null_scores)
    mx = np.array([s.max() if len(s) else 0.0 for s in null_scores])
    pool = np.sort(np.concatenate([np.asarray(s, float) for s in null_scores])) if N else np.array([])
    return {e: (float((1 + (mx >= s).sum()) / (N + 1)),
                float((1 + (pool.size - np.searchsorted(pool, s, side="left"))) / (1 + pool.size)))
            for e, s in obs.items()}


def bh(p):
    """Benjamini–Hochberg adjusted p-values (q-values)."""
    p = np.asarray(p, float)
    if p.size == 0:
        return p
    o = np.argsort(p)
    q = np.minimum.accumulate((p[o] * p.size / np.arange(1, p.size + 1))[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1.0)
    return out


def surrogate_test(index, cell_area, v_min, floor=0.5, n_surrogates=29, seed=0, gamma=1.0, **detect_kwargs):
    """Detect events and test each against ``n_surrogates`` phase-randomized surrogates.

    Parameters
    ----------
    index : numpy.ndarray
        Smoothed index used for detection.
    n_surrogates : int, default 29
        Number of surrogates (29 gives a smallest family-wise p of 0.033).

    Returns
    -------
    labels : numpy.ndarray
    table : pandas.DataFrame
        ``event, score, p_fw, p_pool, q_pool``.
    """
    rng = np.random.default_rng(seed)
    labels = detect(index, cell_area, v_min, floor, return_info=False, **detect_kwargs)
    obs = {e: v[0] for e, v in event_scores(labels, -index, cell_area, floor, v_min, gamma).items()}
    null = []
    for _ in range(n_surrogates):
        s = phase_surrogate(index, rng)
        ls = detect(s, cell_area, v_min, floor, return_info=False, **detect_kwargs)
        null.append(np.array([v[0] for v in event_scores(ls, -s, cell_area, floor, v_min, gamma).values()]))
    pv = pvalues(obs, null)
    t = pd.DataFrame([dict(event=e, score=obs[e], p_fw=pv[e][0], p_pool=pv[e][1]) for e in obs])
    if len(t):
        t["q_pool"] = bh(t.p_pool.values)
    return labels, t
