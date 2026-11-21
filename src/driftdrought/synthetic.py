"""Synthetic drought cubes with known events (the DRIFT benchmark).

Gaussian scenarios (:func:`generate`), shape ``(60, 60, 80)`` = (months, rows, columns):

* ``S1_isolated`` — stationary, well-separated events;
* ``S2_advection`` — events that migrate 0.3–1.2 cells per month;
* ``S3_splitmerge`` — one event splits into two diverging lobes (one true split), two others converge and touch;
* ``S4_fragmented`` — weaker events on strongly correlated (red) noise.

Irregular scenarios (:func:`generate_irregular`) place real drought shapes from a library built with
:func:`build_library` from any label field and index (for example, a DRIFT catalogue of your own region), so the
method can be tuned and tested on events with the geometry of the target region.

Truth: a cell belongs to event ``k`` if its clean deficit is ``>= support`` and is the largest contribution.
"""
import numpy as np
from scipy.ndimage import find_objects, gaussian_filter, zoom

__all__ = ["SHAPE", "SCENARIOS", "generate", "build_library", "example_library", "generate_irregular"]

SHAPE = (60, 60, 80)
SCENARIOS = ("S1_isolated", "S2_advection", "S3_splitmerge", "S4_fragmented")


def _blob(shape, t0, y0, x0, st, sy, sx, amp, vy=0.0, vx=0.0):
    T, Y, X = shape
    t = np.arange(T)[:, None, None]
    y = np.arange(Y)[None, :, None]
    x = np.arange(X)[None, None, :]
    return amp * np.exp(-((t - t0) / st) ** 2 - ((y - (y0 + vy * (t - t0))) / sy) ** 2 - ((x - (x0 + vx * (t - t0))) / sx) ** 2)


def _finish(fields, noise, rng, red, support):
    fields = np.asarray(fields)
    clean = fields.max(axis=0)
    raw = np.where(clean >= support, fields.argmax(axis=0) + 1, 0)
    keep = np.unique(raw[raw > 0])
    lut = np.zeros(len(fields) + 1, np.int32)
    lut[keep] = np.arange(1, len(keep) + 1)
    r = gaussian_filter(rng.standard_normal(SHAPE), red)
    r /= r.std()
    return (-clean + noise * r).astype(np.float32), lut[raw], len(keep)


def generate(scenario, noise, seed, support=0.5):
    """Gaussian benchmark cube.

    Parameters
    ----------
    scenario : str
        One of :data:`SCENARIOS`.
    noise : float
        Noise standard deviation (the paper uses 0.3 and 0.6).
    seed : int
        Random seed (the paper uses 0–19).
    support : float, default 0.5
        Clean-deficit level that defines the true event extent.

    Returns
    -------
    index : numpy.ndarray of float32
        Synthetic standardized index ``(60, 60, 80)`` (drought negative).
    truth : numpy.ndarray of int32
        True event labels.
    meta : dict
        ``n_events`` and ``true_splits``.
    """
    rng = np.random.default_rng(seed)
    T, Y, X = SHAPE
    comps, splits = [], 0
    if scenario in ("S1_isolated", "S2_advection", "S4_fragmented"):
        for _ in range(int(rng.integers(3, 7))):
            mv = scenario == "S2_advection"
            amp = rng.uniform(1.2, 2.5) * (0.75 if scenario == "S4_fragmented" else 1.0)
            comps.append([_blob(SHAPE, rng.uniform(8, T - 8), rng.uniform(10, Y - 10), rng.uniform(10, X - 10),
                                rng.uniform(3, 9), rng.uniform(4, 9), rng.uniform(5, 12), amp,
                                rng.uniform(-1, 1) * mv * rng.uniform(0.3, 1.2), rng.uniform(-1, 1) * mv * rng.uniform(0.3, 1.2))])
    elif scenario == "S3_splitmerge":
        t0 = rng.uniform(15, 25)
        y0, x0 = rng.uniform(20, 40), rng.uniform(15, 30)
        parent = _blob(SHAPE, t0 - 4, y0, x0, 5, 7, 8, rng.uniform(1.6, 2.3))
        lobes = [_blob(SHAPE, t0 + 8, y0, x0, 6, 5, 6, rng.uniform(1.6, 2.3), vy=s * 0.9, vx=0.4) for s in (-1, 1)]
        comps.append([parent] + lobes)
        splits = 1
        tb = rng.uniform(25, 40)
        comps.append([_blob(SHAPE, tb, rng.uniform(15, 45), 55, 7, 6, 7, rng.uniform(1.5, 2.2), vx=0.8)])
        comps.append([_blob(SHAPE, tb, rng.uniform(15, 45), 75, 7, 6, 7, rng.uniform(1.5, 2.2), vx=-0.8)])
    else:
        raise ValueError(f"unknown scenario {scenario!r}; choose from {SCENARIOS}")
    fields = [np.max(np.stack(c), axis=0) for c in comps]
    clean = np.max(fields, axis=0)
    truth = np.where(clean >= support, np.argmax(fields, axis=0) + 1, 0).astype(np.int32)
    red = gaussian_filter(rng.standard_normal(SHAPE), (1.5, 2, 2) if scenario == "S4_fragmented" else (0.5, 1, 1))
    red /= red.std()
    return (-clean + noise * red).astype(np.float32), truth, dict(n_events=len(comps), true_splits=splits)


def build_library(labels, index, min_cells=200, sigma=1.0):
    """Event-shape library from a label field and its index.

    Each entry is the space–time deficit (``-index`` smoothed with ``sigma``, clipped at 0) inside one event,
    cropped to its bounding box.

    Parameters
    ----------
    labels : numpy.ndarray of int
        Event labels ``(time, y, x)`` (e.g. a DRIFT catalogue).
    index : numpy.ndarray
        Index field of the same shape.
    min_cells : int, default 200
        Smallest event (in voxels) kept in the library.

    Returns
    -------
    list of numpy.ndarray
    """
    d = np.clip(-gaussian_filter(np.where(np.isfinite(index), index, 0.0), sigma), 0, None).astype(np.float32)
    lib = []
    for k, sl in enumerate(find_objects(labels), 1):
        if sl is None:
            continue
        m = labels[sl] == k
        if m.sum() >= min_cells:
            lib.append(np.where(m, d[sl], 0.0).astype(np.float32))
    return lib


def example_library():
    """Library of 99 real drought shapes from the Amazon and Orinoco basins.

    The space–time deficits of the DRIFT pulses detected on SPEI-3, SPI-3 and soil-moisture index fields
    (0.25°, monthly, 1981–2026; CHIRPS v3 and ERA5-Land), each cropped to its bounding box and reduced to at most
    24 months × 32 × 40 cells. Use it with :func:`generate_irregular` when no catalogue of the target region exists.

    Returns
    -------
    list of numpy.ndarray of float32
    """
    from pathlib import Path
    z = np.load(Path(__file__).with_name("data") / "amazon_orinoco_pulses.npz")
    return [z[k].astype(np.float32) for k in sorted(z.files, key=lambda s: int(s[1:]))]


def generate_irregular(library, noise, seed, support=0.5, red_noise=False, n_events=(3, 6),
                       box=((10, 28), (15, 36), (20, 48))):
    """Benchmark cube with irregular, multi-core events drawn from ``library``.

    Each event is a library shape rescaled to a random box, mirrored with probability 0.5, normalized to a peak
    amplitude of 1.2–2.5 and placed at random.

    Parameters
    ----------
    library : list of numpy.ndarray
        Output of :func:`build_library`.
    noise : float
        Noise standard deviation.
    seed : int
        Random seed.
    red_noise : bool, default False
        Strongly correlated noise (as scenario S4) instead of weakly correlated noise.
    n_events : (int, int)
        Minimum and maximum number of events.
    box : three (min, max) pairs
        Ranges of the rescaled event size in months, rows and columns.

    Returns
    -------
    index, truth, meta
        As in :func:`generate`.
    """
    rng = np.random.default_rng(50_000 + seed)
    T, Y, X = SHAPE
    fields = []
    for _ in range(int(rng.integers(n_events[0], n_events[1] + 1))):
        ev = library[int(rng.integers(len(library)))]
        dims = [int(rng.integers(lo, hi + 1)) for lo, hi in box]
        if rng.random() < 0.5:
            ev = ev[:, :, ::-1]
        e = np.clip(zoom(ev, [n / s for n, s in zip(dims, ev.shape)], order=1), 0, None)
        if e.max() <= 0:
            continue
        e = e / e.max() * rng.uniform(1.2, 2.5)
        t0, y0, x0 = (int(rng.integers(0, n - m + 1)) for n, m in zip((T, Y, X), e.shape))
        f = np.zeros(SHAPE, np.float32)
        f[t0:t0 + e.shape[0], y0:y0 + e.shape[1], x0:x0 + e.shape[2]] = e
        fields.append(f)
    idx, truth, n = _finish(fields, noise, rng, (1.5, 2, 2) if red_noise else (0.5, 1, 1), support)
    return idx, truth, dict(n_events=n, true_splits=0)
