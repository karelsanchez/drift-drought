"""Quickstart: detect drought events in a synthetic cube with known truth and compare DRIFT with a threshold detector.

Run:  python examples/quickstart.py
Output: the scores in the terminal and docs/_static/quickstart.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import driftdrought as dd

# 1. A benchmark cube: one drought splits into two lobes, two others converge (60 months x 60 x 80 cells)
index, truth, meta = dd.synthetic.generate("S3_splitmerge", noise=0.6, seed=3)
area = np.ones(index.shape[1:])          # unit cells; use dd.cell_area_km2(lat, res) on a lat-lon grid

# 2. DRIFT with the configuration selected in the benchmark (smoothing 2.5, auto:3, gamma 1, inheritance)
x = dd.smooth(index, 2.5)
labels, info = dd.detect(x, area, v_min=50, floor=0.5, footprint="watershed", ext_floor=0.5)
print(info.round(2).to_string(index=False))

# 3. A reference detector: 3-D connected components at one threshold
ccl = dd.baselines.ccl3d(x, thr=-0.5, cell_area=area, min_volume_frac=0.1)

# 4. Scores against the known truth
for name, lab in (("DRIFT", labels), ("CCL-3D", ccl)):
    h = dd.metrics.hota(truth, lab, area)
    m = dd.metrics.object_match(truth, lab, area, iou_thr=0.25)
    print(f"{name:7s} HOTA {h['HOTA']:.3f}  DetA {h['DetA']:.3f}  AssA {h['AssA']:.3f}  "
          f"F1(IoU>=0.25) {m['f1_iou25']:.2f}  events {m['n_pred']} (true {meta['n_events']})")

# 5. Event table and figure: footprints at four months
print(dd.event_table(labels, index, area).to_string(index=False))
# colour each DRIFT event like the true event it overlaps most (unmatched events get new colours)
show = np.zeros_like(labels)
for k, e in enumerate(np.unique(labels[labels > 0])):
    ov = np.bincount(truth[(labels == e) & (truth > 0)], minlength=truth.max() + 1)
    show[labels == e] = ov.argmax() if ov.max() > 0 else truth.max() + 1 + k
months = [12, 20, 28, 36]
fig, ax = plt.subplots(3, 4, figsize=(9, 6.2), sharex=True, sharey=True)
rows = (("index", index, "RdBu"), ("truth", truth, "tab10"), ("DRIFT", show, "tab10"))
for i, (name, field, cmap) in enumerate(rows):
    for j, t in enumerate(months):
        a = ax[i, j]
        if name == "index":
            a.imshow(field[t], cmap=cmap, vmin=-2.5, vmax=2.5)
        else:
            a.imshow(np.ma.masked_equal(field[t], 0), cmap=cmap, vmin=1, vmax=10, interpolation="nearest")
        a.set_xticks([])
        a.set_yticks([])
        if i == 0:
            a.set_title(f"month {t}", fontsize=9)
        if j == 0:
            a.set_ylabel(name, fontsize=9)
fig.suptitle("Synthetic split-merge cube: index, true events and DRIFT events", fontsize=10)
fig.tight_layout()
out = Path(__file__).resolve().parents[1] / "docs" / "_static" / "quickstart.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150)
print("figure:", out)
