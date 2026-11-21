"""DRIFT: Drought Regions Identified From the space–time component Tree.

A connected-component catalogue at one threshold is one horizontal cut through the max-tree of the deficit
field ``d = -index``. DRIFT builds the whole max-tree on the (time, y, x) grid, keeps the nodes that are large
and deep enough (volume ``>= v_min`` and level ``>= floor``), condenses chains of such nodes into clusters and
selects events inside the hierarchy with volume-weighted topological persistence and lineage inheritance.
"""
import numba
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.ndimage import gaussian_filter

__all__ = ["detect", "smooth", "cell_area_km2", "noise_persistence", "merge_branches", "event_table", "select_floor"]

EARTH_R = 6371.0
_NB26 = np.array([[a, b, c] for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) != (0, 0, 0)])
_NB6 = np.array([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]])


def cell_area_km2(lat, res):
    """Area of regular latitude–longitude cells.

    Parameters
    ----------
    lat : array_like
        Cell-centre latitudes in degrees.
    res : float
        Grid spacing in degrees.

    Returns
    -------
    numpy.ndarray
        Cell areas in km² (same shape as ``lat``).
    """
    lat = np.asarray(lat, float)
    return (np.deg2rad(res) * EARTH_R) ** 2 * np.cos(np.deg2rad(lat))


def smooth(index, sigma):
    """Three-dimensional Gaussian smoothing that keeps invalid (NaN) cells invalid.

    Parameters
    ----------
    index : numpy.ndarray
        Standardized index, shape ``(time, y, x)``; NaN outside the domain.
    sigma : float or sequence of float
        Gaussian width in time steps and cells (``0`` returns the input).

    Returns
    -------
    numpy.ndarray
        Smoothed field with NaN at the same cells as ``index``.
    """
    if not np.any(sigma):
        return index
    x = gaussian_filter(np.nan_to_num(index, nan=0.0).astype(np.float64), sigma)
    x[~np.isfinite(index)] = np.nan
    return x.astype(np.float32)


def noise_persistence(deficit, spec):
    """Noise-adaptive persistence threshold ``auto:k``.

    ``pi = k * 1.4826 * MAD(d - G_2(d))``: ``k`` times a robust estimate of the standard deviation of the
    high-frequency residual of the deficit field (Equation 2 of the DRIFT paper).

    Parameters
    ----------
    deficit : numpy.ndarray
        Deficit field ``-index``; invalid cells are ``<= -9``.
    spec : str
        ``"auto:k"``, e.g. ``"auto:3"``.

    Returns
    -------
    float
        Persistence threshold in index units.
    """
    k = float(spec.split(":")[1])
    x = np.where(deficit > -9, deficit, 0.0)
    r = (x - gaussian_filter(x, 2.0))[deficit > -9]
    return k * 1.4826 * np.median(np.abs(r - np.median(r)))


@numba.njit(cache=True)
def _condense(parents, sig, n_leaves):
    n = parents.size
    nsig_child = np.zeros(n, np.int64)
    for v in range(n - 1):
        if sig[v]:
            nsig_child[parents[v]] += 1
    cid = -np.ones(n, np.int64)
    cparent = -np.ones(n, np.int64)
    nc = 0
    for v in range(n - 1, n_leaves - 1, -1):
        if not sig[v]:
            continue
        p = parents[v]
        if v == n - 1 or not sig[p]:
            cid[v] = nc
            nc += 1
        elif nsig_child[p] >= 2:
            cid[v] = nc
            cparent[nc] = cid[p]
            nc += 1
        else:
            cid[v] = cid[p]
    return cid, cparent[:nc], nc


@numba.njit(cache=True)
def _propagate_labels(parents, sel_node_label, n_leaves):
    n = parents.size
    lab = np.zeros(n, np.int64)
    for v in range(n - 1, -1, -1):
        if sel_node_label[v] > 0:
            lab[v] = sel_node_label[v]
        elif v < n - 1:
            lab[v] = lab[parents[v]]
    return lab[:n_leaves]


def merge_branches(labels, inherit=0.75):
    """Lineage inheritance: give one identity to the branches of a splitting drought.

    Event ``j`` takes the identity of an older event ``i`` when at least a share ``inherit`` of the footprint of
    ``j`` in its first month was occupied by ``i`` in the previous month (Equation 4 of the paper). Droughts that only
    touch, or converge, inherit no cells and keep separate identities. The rule is transitive (union–find).

    Parameters
    ----------
    labels : numpy.ndarray of int
        Event labels ``(time, y, x)``, 0 = no event.
    inherit : float, default 0.75
        Minimum inherited share ``q`` of the onset footprint.

    Returns
    -------
    numpy.ndarray of int
        Labels after merging.
    """
    ids = np.unique(labels[labels > 0])
    if len(ids) < 2:
        return labels
    tt = {i: np.where((labels == i).any(axis=(1, 2)))[0] for i in ids}
    onset = {i: t[0] for i, t in tt.items()}
    parent = {i: i for i in ids}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    order = sorted(ids, key=lambda i: onset[i])
    for j in order:
        t = onset[j]
        bj = labels[t] == j
        for i in order:
            if i == j or onset[i] >= t:
                continue
            if t - 1 in set(tt[i]) and ((labels[t - 1] == i) & bj).sum() >= inherit * bj.sum():
                parent[find(j)] = find(i)
                break
    out = labels.copy()
    for i in ids:
        r = find(i)
        if r != i:
            out[labels == i] = r
    return out


def detect(index, cell_area, v_min, floor=0.5, *, selection="persistence_vl", persistence="auto:3", pers_gamma=1.0,
           inherit=0.75, footprint="birth", ext_floor=None, connectivity=26, max_frac=None, return_info=True):
    """Detect drought events in a space–time index field with DRIFT.

    Parameters
    ----------
    index : numpy.ndarray
        Standardized drought index (drought negative, e.g. SPI or SPEI), shape ``(time, y, x)``, NaN outside the
        domain. Smooth it first with :func:`smooth` (the paper uses ``sigma = 2.5``).
    cell_area : numpy.ndarray
        Cell areas ``(y, x)`` (km², or 1 for unit cells); 0 outside the domain.
    v_min : float
        Minimum space–time volume of a significant node, in area × time units (e.g. km² month).
    floor : float, default 0.5
        Weakest deficit that may belong to an event (``index <= -floor``). :func:`select_floor` chooses it by the
        event-count stability rule.
    selection : {"persistence_vl", "persistence_v", "persistence", "leaf", "eom"}, default "persistence_vl"
        Event-selection rule. ``"persistence_vl"`` is the final DRIFT (volume-weighted persistence + lineage
        inheritance); ``"persistence_v"`` omits inheritance; ``"persistence"`` uses unweighted persistence
        (DRIFT-P); ``"leaf"`` takes the leaves of the condensed tree; ``"eom"`` uses excess-of-mass stability as in
        HDBSCAN.
    persistence : float or str, default "auto:3"
        Persistence threshold ``pi`` in index units, or ``"auto:k"`` (see :func:`noise_persistence`).
    pers_gamma : float, default 1.0
        Exponent ``gamma`` of the volume weighting ``P_V = P (V / v_min)^gamma``; 0 gives unweighted persistence.
    inherit : float, default 0.75
        Inherited share ``q`` for lineage inheritance (only with ``"persistence_vl"``).
    footprint : {"birth", "watershed"}, default "birth"
        ``"birth"``: the selected component itself; ``"watershed"``: events grown by a marker-controlled watershed
        into all cells with deficit ``>= ext_floor`` (hysteresis footprint).
    ext_floor : float, optional
        Extension floor of the watershed footprint (default: ``floor``).
    connectivity : {26, 6}, default 26
        Space–time neighbourhood.
    max_frac : float, optional
        Only for ``selection="eom"``: clusters larger than this share of the domain volume cannot be selected.
    return_info : bool, default True
        Also return the per-event table.

    Returns
    -------
    labels : numpy.ndarray of int32
        Event labels ``(time, y, x)``, 0 = no event.
    info : pandas.DataFrame
        One row per event: ``event``, ``birth_threshold``, ``peak_index``, ``volume``, ``persistence``,
        ``n_cores`` (number of significant sub-clusters, i.e. nested pulses).

    Examples
    --------
    >>> from driftdrought import synthetic, detect, smooth
    >>> idx, truth, meta = synthetic.generate("S1_isolated", noise=0.3, seed=0)
    >>> labels, info = detect(smooth(idx, 2.5), cell_area=1.0 + 0 * idx[0], v_min=50)
    """
    import higra as hg
    if selection == "persistence_vl":
        lab, info = detect(index, cell_area, v_min, floor, selection="persistence_v", persistence=persistence,
                           pers_gamma=pers_gamma, footprint=footprint, ext_floor=ext_floor, connectivity=connectivity)
        merged = merge_branches(lab, inherit=inherit)
        if len(info):
            info = info.copy()
            info["lineage_event"] = [int(np.bincount(merged[lab == e]).argmax()) if (lab == e).any() else e
                                     for e in info.event]
        return (merged, info) if return_info else merged
    shape = index.shape
    d = -np.nan_to_num(index, nan=np.inf).astype(np.float64)
    d[~np.isfinite(d)] = -10.0
    g = hg.get_nd_regular_graph(shape, _NB26 if connectivity == 26 else _NB6)
    tree, level = hg.component_tree_max_tree(g, d.ravel())
    w = np.broadcast_to(np.asarray(cell_area, float), shape).ravel()
    vol = hg.attribute_area(tree, vertex_area=w)
    parents = tree.parents().astype(np.int64)
    nl = tree.num_leaves()
    sig = (vol >= v_min) & (level >= floor)
    sig[:nl] = False
    cid, cpar, nc = _condense(parents, sig, nl)
    if nc == 0:
        empty = np.zeros(shape, np.int32)
        return (empty, pd.DataFrame()) if return_info else empty

    nodes = np.where(cid >= 0)[0]
    dlev = level[nodes] - np.maximum(level[parents[nodes]], floor)
    stab = np.bincount(cid[nodes], weights=vol[nodes] * np.clip(dlev, 0, None), minlength=nc)
    birth_node = np.full(nc, -1, np.int64)
    for v in nodes[np.argsort(-nodes)]:
        if birth_node[cid[v]] < 0:
            birth_node[cid[v]] = v
    peak = np.zeros(nc)
    np.maximum.at(peak, cid[nodes], level[nodes])
    peak_sub = peak.copy()
    for c in range(nc - 1, -1, -1):
        if cpar[c] >= 0:
            peak_sub[cpar[c]] = max(peak_sub[cpar[c]], peak_sub[c])
    saddle = np.where(cpar >= 0, level[parents[birth_node]], floor)
    pers = peak_sub - saddle

    selected = np.zeros(nc, bool)
    n_child = np.bincount(cpar[cpar >= 0], minlength=nc)
    if selection == "leaf":
        selected = n_child == 0
    elif selection in ("persistence", "persistence_v"):
        pi = noise_persistence(d, persistence) if isinstance(persistence, str) else float(persistence)
        pw = pers * (vol[birth_node] / v_min) ** pers_gamma if (selection == "persistence_v" and pers_gamma) else pers
        alive = np.ones(nc, bool)
        transparent = np.zeros(nc, bool)
        alive[(cpar < 0) & (peak_sub - floor < pi)] = False
        for pc in np.unique(cpar[cpar >= 0]):
            ch = np.where(cpar == pc)[0]
            elder = ch[np.argmax(peak_sub[ch])]
            young = ch[ch != elder]
            ok = pw[young] >= pi
            alive[young[~ok]] = False
            transparent[elder] = not ok.any()
        for c in range(nc):
            if cpar[c] >= 0 and not alive[cpar[c]]:
                alive[c] = False
        has_split_below = np.zeros(nc, bool)
        for c in range(nc - 1, -1, -1):
            p_ = cpar[c]
            if p_ >= 0 and alive[c] and (has_split_below[c] or not transparent[c]):
                has_split_below[p_] = True
        selected = alive & ~transparent & ~has_split_below
    elif selection == "eom":
        cap = max_frac * w.sum() if max_frac else np.inf
        best = stab.copy()
        child_sum = np.zeros(nc)
        for c in range(nc - 1, -1, -1):
            has_children = child_sum[c] > 0
            if vol[birth_node[c]] > cap and has_children:
                best[c] = child_sum[c]
            elif not has_children or stab[c] >= child_sum[c]:
                selected[c] = True
                best[c] = stab[c]
            else:
                best[c] = child_sum[c]
            if cpar[c] >= 0:
                child_sum[cpar[c]] += best[c]
    else:
        raise ValueError(f"unknown selection {selection!r}")
    for c in range(nc):
        p = cpar[c]
        while p >= 0:
            if selected[p]:
                selected[c] = False
                break
            p = cpar[p]

    sel_ids = np.where(selected)[0]
    node_lab = np.zeros(parents.size, np.int64)
    node_lab[birth_node[sel_ids]] = np.arange(1, sel_ids.size + 1)
    labels = _propagate_labels(parents, node_lab, nl).reshape(shape).astype(np.int32)
    if footprint == "watershed":
        from skimage.segmentation import watershed
        ext = floor if ext_floor is None else ext_floor
        labels = watershed(-d, markers=labels, mask=d >= ext,
                           connectivity=np.ones((3, 3, 3), bool) if connectivity == 26 else 1).astype(np.int32)
    elif footprint != "birth":
        raise ValueError(f"unknown footprint {footprint!r}")
    if not return_info:
        return labels
    n_cores = np.zeros(nc, np.int64)
    for c in range(nc):
        p = cpar[c]
        while p >= 0:
            if selected[p]:
                n_cores[p] += 1
                break
            p = cpar[p]
    info = pd.DataFrame({"event": np.arange(1, sel_ids.size + 1), "birth_threshold": -level[birth_node[sel_ids]],
                         "peak_index": -peak_sub[sel_ids], "volume": vol[birth_node[sel_ids]],
                         "persistence": pers[sel_ids], "n_cores": n_cores[sel_ids]})
    return labels, info


def event_table(labels, index, cell_area, times=None):
    """Summary of a label field: one row per event.

    Parameters
    ----------
    labels : numpy.ndarray of int
        Event labels ``(time, y, x)``.
    index : numpy.ndarray
        Index field used for the severity statistics.
    cell_area : numpy.ndarray or float
        Cell areas ``(y, x)``.
    times : sequence, optional
        Time stamps of the first axis (default: integer steps).

    Returns
    -------
    pandas.DataFrame
        ``event, start, end, duration, volume, max_area, peak_time, mean_index``.
    """
    w = np.broadcast_to(np.asarray(cell_area, float), labels.shape[1:])
    times = np.arange(labels.shape[0]) if times is None else np.asarray(times)
    rows = []
    for e in np.unique(labels[labels > 0]):
        m = labels == e
        area_t = (m * w).sum(axis=(1, 2))
        tt = np.where(area_t > 0)[0]
        rows.append(dict(event=int(e), start=times[tt[0]], end=times[tt[-1]], duration=len(tt),
                         volume=float(area_t.sum()), max_area=float(area_t.max()), peak_time=times[int(area_t.argmax())],
                         mean_index=float(np.nanmean(index[m]))))
    return pd.DataFrame(rows)


def select_floor(index, cell_area, v_min, floors=np.arange(0.4, 1.41, 0.1), min_events=5, max_duration_frac=1 / 3,
                 **detect_kwargs):
    """Choose the deficit floor by event-count stability (Equation 5 of the paper).

    For each candidate floor the number of events ``N`` is computed; the instability of floor ``j`` is the mean
    relative change of ``N`` to its neighbours. Floors that give fewer than ``min_events`` events, or an event longer
    than ``max_duration_frac`` of the record, are excluded. No impact data are used.

    Parameters
    ----------
    index, cell_area, v_min
        As in :func:`detect` (``index`` already smoothed).
    floors : sequence of float
        Candidate floors in increasing order.
    min_events : int, default 5
    max_duration_frac : float, default 1/3
    **detect_kwargs
        Passed to :func:`detect`.

    Returns
    -------
    best : float
        Selected floor.
    scan : pandas.DataFrame
        ``floor, n_events, max_duration, instability, eligible`` for every candidate.
    """
    rows = []
    for f in floors:
        lab = detect(index, cell_area, v_min, floor=float(f), return_info=False, **detect_kwargs)
        ids = np.unique(lab[lab > 0])
        dur = max((int((lab == e).any(axis=(1, 2)).sum()) for e in ids), default=0)
        rows.append(dict(floor=float(f), n_events=len(ids), max_duration=dur))
    s = pd.DataFrame(rows)
    n = s.n_events.values.astype(float)
    rel = np.abs(np.diff(n)) / np.maximum(n[:-1], 1)
    s["instability"] = np.r_[rel[0], (rel[:-1] + rel[1:]) / 2, rel[-1]] if len(n) > 1 else 0.0
    s["eligible"] = (s.n_events >= min_events) & (s.max_duration < max_duration_frac * index.shape[0])
    cand = s[s.eligible] if s.eligible.any() else s
    return float(cand.loc[cand.instability.idxmin(), "floor"]), s


def _relabel(lab):
    u = np.unique(lab[lab > 0])
    lut = np.zeros(int(lab.max()) + 1, dtype=np.int32)
    lut[u] = np.arange(1, u.size + 1)
    return lut[lab]


def patches(labels):
    """Monthly 8-connected patches of every event (helper for lineage diagnostics).

    Returns
    -------
    numpy.ndarray of int
        Patch labels ``(time, y, x)``, unique over the whole cube.
    """
    out = np.zeros(labels.shape, np.int64)
    nxt = 0
    s8 = np.ones((3, 3), bool)
    for t in range(labels.shape[0]):
        for e in np.unique(labels[t][labels[t] > 0]):
            lab, n = ndimage.label(labels[t] == e, structure=s8)
            out[t][lab > 0] = lab[lab > 0] + nxt
            nxt += n
    return out
