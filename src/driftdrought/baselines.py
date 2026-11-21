"""Reference detectors used in the DRIFT benchmark.

* :func:`ccl2d_overlap` — monthly 8-connected patches linked across months by overlap (Andreadis et al., 2005).
* :func:`ccl3d` — 26-connected space–time components with a minimum volume (Lloyd-Hughes, 2012; Diaz et al., 2020).
* :func:`dbscan3d` — density-based clustering of drought voxels (Ester et al., 1996; Cammalleri & Toreti, 2023).

All take a standardized index ``(time, y, x)`` (drought negative, NaN outside the domain) and a threshold, and return
an integer label cube (0 = no event).
"""
import numpy as np
import pandas as pd
from scipy import ndimage

from .core import _relabel

__all__ = ["ccl2d_overlap", "ccl3d", "dbscan3d"]


def ccl2d_overlap(index, thr, cell_area, min_frac=0.016, overlap_frac=0.016):
    """Two-dimensional connected components linked through time by overlap.

    Parameters
    ----------
    index : numpy.ndarray
        Index ``(time, y, x)``.
    thr : float
        Drought threshold (cells with ``index <= thr`` are dry).
    cell_area : numpy.ndarray
        Cell areas ``(y, x)``.
    min_frac : float
        Minimum patch area as a share of the domain area.
    overlap_frac : float
        Minimum overlap between patches of consecutive months, as a share of the domain area.
    """
    dry = np.nan_to_num(index, nan=0.0) <= thr
    land = np.isfinite(index[0])
    area = np.broadcast_to(np.asarray(cell_area, float), index.shape[1:])
    total = area[land].sum()
    labels = np.zeros(index.shape, dtype=np.int64)
    parent = {}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    nxt = 1
    s8 = np.ones((3, 3), bool)
    for t in range(index.shape[0]):
        lab, n = ndimage.label(dry[t], structure=s8)
        if n == 0:
            continue
        a = ndimage.sum(area, lab, index=np.arange(1, n + 1))
        for k in np.where(a >= min_frac * total)[0] + 1:
            labels[t][lab == k] = nxt
            parent[nxt] = nxt
            nxt += 1
        if t == 0:
            continue
        prev, cur = labels[t - 1], labels[t]
        both = (prev > 0) & (cur > 0)
        if not both.any():
            continue
        ov = pd.DataFrame({"p": prev[both], "c": cur[both], "a": area[both]}).groupby(["p", "c"]).a.sum()
        for (p, c), a in ov.items():
            if a >= overlap_frac * total:
                rp, rc = find(p), find(c)
                if rp != rc:
                    parent[rc] = rp
    if nxt > 1:
        lut = np.zeros(nxt, dtype=np.int64)
        for k in range(1, nxt):
            lut[k] = find(k)
        labels = lut[labels]
    return _relabel(labels)


def ccl3d(index, thr, cell_area, min_volume_frac=0.0):
    """26-connected space–time components with a minimum volume.

    Parameters
    ----------
    min_volume_frac : float
        Minimum event volume as a share of (domain area × one time step).
    """
    dry = np.nan_to_num(index, nan=0.0) <= thr
    lab, n = ndimage.label(dry, structure=np.ones((3, 3, 3), bool))
    if n == 0 or min_volume_frac <= 0:
        return lab.astype(np.int32)
    land = np.isfinite(index[0])
    area = np.broadcast_to(np.asarray(cell_area, float), index.shape[1:])
    vol = ndimage.sum(np.broadcast_to(area, index.shape), lab, index=np.arange(1, n + 1))
    keep = np.zeros(n + 1, bool)
    keep[1:] = vol >= min_volume_frac * area[land].sum()
    return _relabel(np.where(keep[lab], lab, 0))


def dbscan3d(index, thr, s_xy=3, s_t=3, fr_mp=0.25, c_min=50):
    """Density-based clustering (DBSCAN) of drought voxels in a scaled space–time window.

    Parameters
    ----------
    s_xy, s_t : int
        Half-widths of the search window in cells and time steps.
    fr_mp : float
        Minimum share of the window that must be dry for a core point.
    c_min : int
        Minimum cluster size in voxels.
    """
    from sklearn.cluster import DBSCAN
    dry = np.nan_to_num(index, nan=0.0) <= thr
    pts = np.argwhere(dry)
    if len(pts) == 0:
        return np.zeros(index.shape, np.int32)
    scaled = pts / np.array([s_t, s_xy, s_xy], float)
    win = (2 * s_t + 1) * (2 * s_xy + 1) ** 2
    db = DBSCAN(eps=1.0, min_samples=max(1, int(round(fr_mp * win))), metric="chebyshev", algorithm="ball_tree").fit(scaled)
    lab = np.zeros(index.shape, np.int64)
    lab[tuple(pts.T)] = db.labels_ + 1
    sizes = np.bincount(lab.ravel())
    lab[np.isin(lab, np.where(sizes < c_min)[0])] = 0
    return _relabel(lab)
