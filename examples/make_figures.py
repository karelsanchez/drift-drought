"""Figures and the mini-benchmark table used in the README and the documentation.

Run:  python examples/make_figures.py      (about 5 minutes on a laptop)
Writes docs/_static/{concept_tree,scenarios,mini_benchmark,floor_scan,pulses_episodes}.png and mini_benchmark.csv
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import driftdrought as dd  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "_static"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
AREA = np.ones(dd.synthetic.SHAPE[1:])
COL = {"DRIFT": "#1b7f79", "CCL-2D + overlap": "#8c8c8c", "CCL-3D": "#5a5a5a", "3D-DBSCAN": "#b5b5b5"}


def concept_tree():
    """1-D deficit profile: one threshold vs the whole tree and persistence."""
    x = np.linspace(0, 10, 1000)
    d = (1.3 * np.exp(-((x - 2) / 0.7) ** 2) + 1.7 * np.exp(-((x - 4.2) / 0.6) ** 2)
         + 0.9 * np.exp(-((x - 7.5) / 0.5) ** 2) + 0.55 * np.exp(-((x - 8.6) / 0.25) ** 2) + 0.1)
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.0), sharey=True)
    for a in ax:
        a.fill_between(x, 0, d, color="#e8eef3")
        a.plot(x, d, color="#1f3b57", lw=1.4)
        a.set_xlabel("space or time")
    for lam, c in ((0.4, "#999999"), (0.9, "#e08a2b"), (1.4, "#b23a48")):
        ax[0].axhline(lam, ls="--", lw=1, color=c)
        ax[0].text(9.95, lam + 0.03, f"λ = {lam}", ha="right", color=c, fontsize=8)
    ax[0].set_title("(a) one threshold = one cut", loc="left")
    ax[0].set_ylabel("deficit d = −index")
    from scipy.signal import argrelextrema
    pk = argrelextrema(d, np.greater)[0]
    mn = argrelextrema(d, np.less)[0]
    top = pk[np.argmax(d[pk])]
    for p in pk:
        if p == top:
            continue
        # saddle: the highest minimum between this peak and any higher peak
        higher = pk[d[pk] > d[p]]
        sad = max(d[mn[(mn > min(p, h)) & (mn < max(p, h))]].min() for h in higher)
        ax[1].plot([x[p], x[p]], [sad, d[p]], color="#1b7f79", lw=2.2)
        ax[1].text(x[p] + 0.12, d[p] - 0.05, f"P = {d[p] - sad:.2f}", fontsize=8, color="#1b7f79", va="top")
    ax[1].axhline(0.5, ls=":", color="#b23a48")
    ax[1].text(0.1, 0.53, "floor", color="#b23a48", fontsize=8)
    ax[1].set_title("(b) the tree: persistence P = peak − saddle", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "concept_tree.png", dpi=150)
    plt.close(fig)


def scenarios():
    """Time-collapsed true events of the four Gaussian scenarios and an irregular cube."""
    cubes = [(sc, dd.synthetic.generate(sc, 0.3, 1)) for sc in dd.synthetic.SCENARIOS]
    cubes.append(("irregular (real shapes)", dd.synthetic.generate_irregular(dd.synthetic.example_library(), 0.3, 2)))
    fig, ax = plt.subplots(2, 5, figsize=(11, 4.4))
    for j, (name, (i, t, m)) in enumerate(cubes):
        tm = np.where(t > 0, np.arange(t.shape[0])[:, None, None], np.nan)
        first = np.nanmin(np.where(np.isfinite(tm), tm, np.inf), axis=0)
        ax[0, j].imshow(np.ma.masked_invalid(np.where(np.isfinite(first), first, np.nan)), cmap="viridis",
                        vmin=0, vmax=60)
        ax[0, j].set_title(f"{name}\n{m['n_events']} events", fontsize=8.5)
        ax[1, j].imshow(i[i.shape[0] // 2], cmap="RdBu", vmin=-2.5, vmax=2.5)
        for a in ax[:, j]:
            a.set_xticks([])
            a.set_yticks([])
    ax[0, 0].set_ylabel("true events\n(colour: first month)")
    ax[1, 0].set_ylabel("index, month 30")
    fig.tight_layout()
    fig.savefig(OUT / "scenarios.png", dpi=150)
    plt.close(fig)


def mini_benchmark(seeds=range(10)):
    """DRIFT and three reference detectors on the Gaussian scenarios (configurations selected in the paper)."""
    methods = {
        "DRIFT": lambda x: dd.detect(x, AREA, 50, 0.5, footprint="watershed", ext_floor=0.5, inherit=0.9,
                                     return_info=False),
        "CCL-2D + overlap": lambda x: dd.baselines.ccl2d_overlap(x, -0.5, AREA, 0.005, 0.005),
        "CCL-3D": lambda x: dd.baselines.ccl3d(x, -0.5, AREA, 0.1),
        "3D-DBSCAN": lambda x: dd.baselines.dbscan3d(x, -0.5, s_xy=5, s_t=1, fr_mp=0.4, c_min=10),
    }
    rows = []
    for sc in dd.synthetic.SCENARIOS:
        for nz in (0.3, 0.6):
            for s in seeds:
                idx, truth, meta = dd.synthetic.generate(sc, nz, s)
                x = dd.smooth(idx, 2.5)
                for name, f in methods.items():
                    lab = f(x)
                    h = dd.metrics.hota(truth, lab, AREA)
                    m = dd.metrics.object_match(truth, lab, AREA)
                    rows.append(dict(method=name, scenario=sc, noise=nz, seed=s, HOTA=h["HOTA"], DetA=h["DetA"],
                                     AssA=h["AssA"], F1_iou25=m["f1_iou25"], events=m["n_pred"], true_events=m["n_gt"]))
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "mini_benchmark_cubes.csv", index=False)
    S = D.groupby(["method", "noise"])[["HOTA", "DetA", "AssA", "F1_iou25", "events", "true_events"]].mean().round(3)
    S.to_csv(OUT / "mini_benchmark.csv")
    order = list(methods)
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4))
    for k, nz in enumerate((0.3, 0.6)):
        g = D[D.noise == nz].groupby(["scenario", "method"]).HOTA.mean().unstack()[order]
        xs = np.arange(len(g))
        for j, mth in enumerate(order):
            ax[k].bar(xs + (j - 1.5) * 0.2, g[mth], 0.2, color=COL[mth], label=mth)
        ax[k].set_xticks(xs)
        ax[k].set_xticklabels([s.split("_")[0] for s in g.index])
        ax[k].set_title(f"noise {nz}", loc="left")
        ax[k].set_ylim(0, 0.8)
    ax[0].set_ylabel("HOTA (mean of 10 seeds)")
    ax[1].legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "mini_benchmark.png", dpi=150)
    plt.close(fig)
    return S


def floor_scan():
    idx, truth, _ = dd.synthetic.generate("S2_advection", 0.3, 4)
    best, scan = dd.select_floor(dd.smooth(idx, 2.5), AREA, v_min=50, floors=np.arange(0.3, 1.21, 0.1), min_events=1)
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.plot(scan.floor, scan.n_events, "o-", color="#1f3b57", label="events")
    ax2 = ax.twinx()
    ax2.plot(scan.floor, scan.instability, "s--", color="#b23a48", label="instability")
    ax.axvline(best, color="#1b7f79", lw=2, alpha=0.6)
    ax.text(best + 0.02, 0.4, f"selected floor {best:.1f}", color="#1b7f79", fontsize=9, rotation=90, va="bottom")
    ax.set_xlabel("floor λ_floor")
    ax.set_ylabel("number of events")
    ax2.set_ylabel("instability", color="#b23a48")
    ax2.spines["right"].set_visible(True)
    fig.tight_layout()
    fig.savefig(OUT / "floor_scan.png", dpi=150)
    plt.close(fig)


def pulses_episodes():
    """Episodes and the pulses nested in them, on a cube with neighbouring events."""
    idx, truth, _ = dd.synthetic.generate("S2_advection", 0.3, 0)
    x = dd.smooth(idx, 2.5)
    ep = dd.detect(x, AREA, 50, 0.5, return_info=False)
    pu = dd.detect(x, AREA, 50, 0.5, selection="leaf", return_info=False)
    fig, ax = plt.subplots(1, 3, figsize=(9.6, 3.2))
    for a, (name, lab) in zip(ax, (("true events", truth), ("DRIFT episodes", ep), ("DRIFT pulses", pu))):
        proj = np.zeros(lab.shape[1:], int)
        for e in np.unique(lab[lab > 0]):
            proj[(lab == e).any(axis=0) & (proj == 0)] = e
        a.imshow(np.ma.masked_equal(proj, 0), cmap="tab20", interpolation="nearest")
        a.set_title(f"{name}: {len(np.unique(lab[lab > 0]))}", fontsize=9)
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle("Footprints collapsed over time: two neighbouring droughts form one episode with two pulses", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "pulses_episodes.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    concept_tree()
    scenarios()
    floor_scan()
    pulses_episodes()
    print(mini_benchmark())
