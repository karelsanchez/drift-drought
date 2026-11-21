"""Evaluation metrics for space–time event catalogues.

Against a known truth (synthetic benchmark):

* object matching on 3-D volumes (area-weighted IoU, Hungarian assignment): precision, recall, F1, mean IoU of
  matches, relative volume error;
* HOTA (Luiten et al., 2021) with DetA and AssA, averaged over localization thresholds 0.05–0.95; a detection is the
  slice of an event in one time step;
* partition agreement: adjusted Rand index and variation of information.

Between catalogues without truth (real data) the partition and matching metrics are used symmetrically.
"""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

__all__ = ["object_match", "hota", "partition_agreement", "cv"]


def _overlap_table(gt, pr, w):
    m = (gt > 0) | (pr > 0)
    df = pd.DataFrame({"g": gt[m].ravel(), "p": pr[m].ravel(), "w": w[m].ravel()})
    inter = df[(df.g > 0) & (df.p > 0)].groupby(["g", "p"]).w.sum()
    vg = df[df.g > 0].groupby("g").w.sum()
    vp = df[df.p > 0].groupby("p").w.sum()
    return inter, vg, vp


def object_match(gt, pr, w, iou_thr=0.5):
    """Match true and detected events by space–time IoU (Hungarian assignment).

    Parameters
    ----------
    gt, pr : numpy.ndarray of int
        True and detected label cubes ``(time, y, x)``.
    w : numpy.ndarray or float
        Cell weights (areas) broadcastable to the cube.
    iou_thr : float, default 0.5
        IoU needed for a match to count as a true positive.

    Returns
    -------
    dict
        ``precision, recall, f1`` (at ``iou_thr``), ``f1_iou25``, ``mean_iou``, ``vol_err``, ``n_gt``, ``n_pred``.
    """
    w = np.broadcast_to(w, gt.shape)
    inter, vg, vp = _overlap_table(gt, pr, w)
    G, P = vg.index.values, vp.index.values
    if len(G) == 0 or len(P) == 0:
        return dict(precision=0.0, recall=0.0, f1=0.0, f1_iou25=0.0, mean_iou=0.0,
                    mean_iou_assigned=0.0, vol_err=np.nan, n_gt=len(G), n_pred=len(P))
    gi = {g: i for i, g in enumerate(G)}
    pi = {p: i for i, p in enumerate(P)}
    iou = np.zeros((len(G), len(P)))
    for (g, p), v in inter.items():
        iou[gi[g], pi[p]] = v / (vg[g] + vp[p] - v)
    r, c = linear_sum_assignment(-iou)
    ok = iou[r, c] >= iou_thr
    tp = int(ok.sum())
    prec, rec = tp / len(P), tp / len(G)
    f1 = 2 * prec * rec / (prec + rec) if tp else 0.0
    verr = np.mean(np.abs(vp.values[c[ok]] - vg.values[r[ok]]) / vg.values[r[ok]]) if tp else np.nan
    ok25 = iou[r, c] >= 0.25
    f1_25 = 2 * ok25.sum() / (len(P) + len(G))
    return dict(precision=prec, recall=rec, f1=f1, f1_iou25=float(f1_25),
                mean_iou=float(iou[r, c][ok].mean()) if tp else 0.0,
                mean_iou_assigned=float(iou[r, c].sum() / max(len(G), len(P))),
                vol_err=verr, n_gt=len(G), n_pred=len(P))


def hota(gt, pr, w, alphas=np.arange(0.05, 0.96, 0.05)):
    """Higher Order Tracking Accuracy (Luiten et al., 2021) for label cubes.

    Each time step is a frame and each event slice a detection; the similarity is the IoU. HOTA, DetA and AssA are
    averaged over the localization thresholds ``alphas``.

    Returns
    -------
    dict
        ``HOTA, DetA, AssA, HOTA_05``.
    """
    w = np.broadcast_to(w, gt.shape)
    T = gt.shape[0]
    frames = []
    for t in range(T):
        inter, vg, vp = _overlap_table(gt[t], pr[t], w[t])
        frames.append((inter, vg, vp))
    res = []
    for a in alphas:
        tp_pairs, n_tp, n_fn, n_fp = {}, 0, 0, 0
        gt_count, pr_count = {}, {}
        matches = []
        for inter, vg, vp in frames:
            for g in vg.index:
                gt_count[g] = gt_count.get(g, 0) + 1
            for p in vp.index:
                pr_count[p] = pr_count.get(p, 0) + 1
            if len(vg) == 0 or len(vp) == 0:
                n_fn += len(vg)
                n_fp += len(vp)
                continue
            G, P = vg.index.values, vp.index.values
            S = np.zeros((len(G), len(P)))
            gi = {g: i for i, g in enumerate(G)}
            pi = {p: i for i, p in enumerate(P)}
            for (g, p), v in inter.items():
                S[gi[g], pi[p]] = v / (vg[g] + vp[p] - v)
            r, c = linear_sum_assignment(-S)
            ok = S[r, c] >= a
            for i, j in zip(r[ok], c[ok]):
                key = (G[i], P[j])
                tp_pairs[key] = tp_pairs.get(key, 0) + 1
                matches.append(key)
            k = int(ok.sum())
            n_tp += k
            n_fn += len(G) - k
            n_fp += len(P) - k
        det = n_tp / max(n_tp + n_fn + n_fp, 1)
        if n_tp == 0:
            res.append((a, 0.0, det, 0.0))
            continue
        ass = 0.0
        for (g, p) in matches:
            tpa = tp_pairs[(g, p)]
            ass += tpa / (gt_count[g] + pr_count[p] - tpa)
        ass /= n_tp
        res.append((a, np.sqrt(det * ass), det, ass))
    r = np.array(res)
    return dict(HOTA=float(r[:, 1].mean()), DetA=float(r[:, 2].mean()), AssA=float(r[:, 3].mean()),
                HOTA_05=float(r[np.argmin(np.abs(r[:, 0] - 0.5)), 1]))


def partition_agreement(a, b, mask=None):
    """Adjusted Rand index and variation of information (bits) between two label cubes.

    Computed over the cells where either cube labels an event (or over ``mask``).
    """
    m = (a > 0) | (b > 0) if mask is None else mask
    x, y = a[m].ravel(), b[m].ravel()
    ari = adjusted_rand_score(x, y)
    ct = pd.crosstab(x, y).values.astype(float)
    pxy = ct / ct.sum()
    px, py = pxy.sum(1), pxy.sum(0)
    nz = pxy > 0
    hx = -np.sum(px[px > 0] * np.log2(px[px > 0]))
    hy = -np.sum(py[py > 0] * np.log2(py[py > 0]))
    mi = np.sum(pxy[nz] * np.log2(pxy[nz] / np.outer(px, py)[nz]))
    return dict(ARI=ari, VI=hx + hy - 2 * mi)


def cv(values):
    """Coefficient of variation (e.g. of event counts across thresholds)."""
    v = np.asarray(values, float)
    return float(np.std(v, ddof=1) / np.mean(v)) if len(v) > 1 and np.mean(v) else np.nan
