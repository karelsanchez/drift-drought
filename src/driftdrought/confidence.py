"""Event confidence: how robust each event is to parameters or to the input product.

The confidence of event ``e`` is the mean, over ``K`` alternative catalogues ``L_k``, of the best space–time IoU between
``e`` and any event of ``L_k`` (Equation 9 of the DRIFT paper). With catalogues from perturbed parameters it measures
parameter robustness; with a catalogue from another precipitation product it is the *product confidence*.
"""
import numpy as np

from .core import detect

__all__ = ["best_iou", "event_confidence", "parameter_confidence", "consensus"]


def best_iou(ref, other):
    """Best IoU of every event of ``ref`` with any event of ``other``.

    Parameters
    ----------
    ref, other : numpy.ndarray of int
        Label cubes of equal shape.

    Returns
    -------
    numpy.ndarray
        Array indexed by the event id of ``ref`` (entry 0 unused).
    """
    r, m = ref.ravel().astype(np.int64), other.ravel().astype(np.int64)
    n_r, n_m = int(r.max()) + 1, int(m.max()) + 1
    both = (r > 0) & (m > 0)
    inter = np.bincount(r[both] * n_m + m[both], minlength=n_r * n_m).reshape(n_r, n_m)
    union = np.bincount(r, minlength=n_r)[:, None] + np.bincount(m, minlength=n_m)[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    iou[:, 0] = 0.0
    return iou.max(axis=1)


def event_confidence(ref, members):
    """Mean best IoU of each event of ``ref`` over the catalogues ``members``.

    Returns
    -------
    dict
        ``{event_id: confidence}`` with values in [0, 1].
    """
    conf = np.mean([best_iou(ref, m) for m in members], axis=0)
    return {int(e): float(conf[e]) for e in np.unique(ref[ref > 0])}


def parameter_confidence(index, cell_area, v_min, floor, floors_delta=(-0.1, 0.1), sigma=None, smooth_deltas=(-0.5, 0.5),
                         **detect_kwargs):
    """Confidence of the events of a reference run against runs with perturbed floor (and smoothing).

    Parameters
    ----------
    index : numpy.ndarray
        Raw (unsmoothed) index when ``sigma`` is given, otherwise the already smoothed index.
    sigma : float, optional
        Smoothing width of the reference run; when given, runs with ``sigma + smooth_deltas`` are added.

    Returns
    -------
    labels : numpy.ndarray
        Reference labels.
    confidence : dict
        ``{event_id: confidence}``.
    """
    from .core import smooth
    base = smooth(index, sigma) if sigma else index
    ref = detect(base, cell_area, v_min, floor, return_info=False, **detect_kwargs)
    members = []
    sigmas = [sigma] + ([sigma + d for d in smooth_deltas] if sigma else [])
    for s in sigmas:
        x = smooth(index, s) if s else index
        for df in (0.0,) + tuple(floors_delta):
            if s == sigma and df == 0.0:
                continue
            members.append(detect(x, cell_area, v_min, floor + df, return_info=False, **detect_kwargs))
    return ref, event_confidence(ref, members)


def consensus(ref, members, q=0.5):
    """Consensus footprint: cells of ``ref`` labelled as drought in at least a share ``q`` of ``members``.

    Returns
    -------
    labels : numpy.ndarray
        Consensus labels (events emptied by the rule are dropped).
    confidence : dict
        Event confidence of the remaining events.
    frequency : numpy.ndarray
        Share of members that label each cell.
    """
    freq = np.mean([m > 0 for m in members], axis=0)
    lab = np.where(freq >= q, ref, 0).astype(np.int32)
    conf = event_confidence(ref, members)
    return lab, {e: c for e, c in conf.items() if (lab == e).any()}, freq
