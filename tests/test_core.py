import numpy as np
import pytest

import driftdrought as dd

AREA = np.ones(dd.synthetic.SHAPE[1:])


@pytest.fixture(scope="module")
def cube():
    return dd.synthetic.generate("S1_isolated", noise=0.3, seed=0)


def test_generator_is_deterministic():
    a = dd.synthetic.generate("S3_splitmerge", 0.6, 3)
    b = dd.synthetic.generate("S3_splitmerge", 0.6, 3)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert a[2]["true_splits"] == 1


def test_detect_recovers_isolated_events(cube):
    idx, truth, meta = cube
    labels, info = dd.detect(dd.smooth(idx, 2.5), AREA, v_min=50, floor=0.5, footprint="watershed", ext_floor=0.5)
    assert labels.shape == idx.shape and labels.dtype == np.int32
    assert len(info) >= 1
    h = dd.metrics.hota(truth, labels, AREA)
    assert h["HOTA"] > 0.4
    m = dd.metrics.object_match(truth, labels, AREA, iou_thr=0.25)
    assert m["f1_iou25"] > 0.6


@pytest.mark.parametrize("selection", ["persistence_vl", "persistence_v", "persistence", "leaf", "eom"])
def test_all_selection_rules_run(cube, selection):
    labels, _ = dd.detect(dd.smooth(cube[0], 2.0), AREA, v_min=50, floor=0.5, selection=selection)
    assert labels.max() >= 1


def test_nan_cells_are_never_events(cube):
    idx = cube[0].copy()
    idx[:, :10, :10] = np.nan
    labels = dd.detect(dd.smooth(idx, 2.5), AREA, v_min=50, return_info=False)
    assert (labels[:, :10, :10] == 0).all()


def test_merge_branches_joins_a_split():
    lab = np.zeros((3, 5, 5), np.int32)
    lab[0, 1:4, 1:4] = 1
    lab[1, 1:4, 1:4] = 1
    lab[2, 1:3, 1:4] = 2          # born inside event 1 in the previous month
    out = dd.merge_branches(lab, inherit=0.75)
    assert set(np.unique(out[out > 0])) == {1}


def test_baselines_and_confidence(cube):
    idx, truth, _ = cube
    x = dd.smooth(idx, 2.5)
    for lab in (dd.baselines.ccl3d(x, -0.5, AREA, 0.05), dd.baselines.ccl2d_overlap(x, -0.5, AREA)):
        assert lab.max() >= 1
    conf = dd.confidence.event_confidence(truth, [truth])
    assert all(abs(c - 1) < 1e-12 for c in conf.values())


def test_select_floor_returns_candidate(cube):
    best, scan = dd.select_floor(dd.smooth(cube[0], 2.5), AREA, v_min=50, floors=[0.4, 0.5, 0.6], min_events=1)
    assert best in (0.4, 0.5, 0.6) and len(scan) == 3


def test_surrogate_test_small(cube):
    labels, table = dd.significance.surrogate_test(dd.smooth(cube[0], 2.5), AREA, v_min=50, n_surrogates=3)
    assert set(table.columns) >= {"event", "score", "p_fw", "p_pool", "q_pool"}
    assert ((table.p_fw > 0) & (table.p_fw <= 1)).all()


def test_example_library():
    lib = dd.synthetic.example_library()
    assert len(lib) == 99 and all(e.ndim == 3 and e.max() > 0 for e in lib)
    idx, truth, meta = dd.synthetic.generate_irregular(lib, noise=0.3, seed=0)
    assert idx.shape == dd.synthetic.SHAPE and meta["n_events"] == len(np.unique(truth[truth > 0]))
