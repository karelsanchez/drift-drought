# DRIFT — drought events from the space–time component tree

**DRIFT** (*Drought Regions Identified From the space–time component Tree*) detects drought events in gridded
standardized indices (SPI, SPEI, soil-moisture percentiles) without fixing a single threshold. It builds the max-tree
of the deficit field over time, latitude and longitude and selects events inside that hierarchy with
volume-weighted topological persistence, a noise-adaptive persistence threshold and lineage inheritance. One tree gives
two nested levels: drought **episodes** and the **pulses** they contain.

The package also provides what is needed to test a drought-event detector: a synthetic benchmark with known events
(Gaussian and irregular shapes from real droughts), three reference detectors, tracking metrics (HOTA, object F1),
event confidence and a surrogate significance test.

![Quickstart](docs/_static/quickstart.png)

## Installation

```bash
pip install git+https://github.com/karelsanchez/drift-drought
```

Python ≥ 3.10. Dependencies: NumPy, SciPy, pandas, Higra, Numba, scikit-image, scikit-learn, xarray.

## Quickstart

```python
import numpy as np
import driftdrought as dd

index, truth, meta = dd.synthetic.generate("S3_splitmerge", noise=0.6, seed=3)   # cube with known events
area = np.ones(index.shape[1:])

labels, info = dd.detect(dd.smooth(index, 2.5), area, v_min=50, floor=0.5,
                         footprint="watershed", ext_floor=0.5)
print(dd.metrics.hota(truth, labels, area))          # {'HOTA': 0.574, 'DetA': ..., 'AssA': ...}
print(dd.event_table(labels, index, area))
```

On real data, pass the cell areas (`dd.cell_area_km2`), choose the floor with `dd.select_floor` and use
`selection="leaf"` for pulses. See `examples/quickstart.py` and the documentation.

## What is in the package

| Module | Contents |
|---|---|
| `driftdrought.core` | `detect`, `smooth`, `select_floor`, `event_table`, `merge_branches` |
| `driftdrought.indices` | SPI, SPEI and soil-moisture index fitted per calendar month |
| `driftdrought.synthetic` | benchmark cubes (S1–S4) and irregular events from a library of real drought shapes |
| `driftdrought.baselines` | CCL-2D with overlap tracking, CCL-3D, 3D-DBSCAN |
| `driftdrought.metrics` | HOTA, object matching (F1 at an IoU threshold), partition agreement |
| `driftdrought.confidence` | event confidence against perturbed parameters or another input product |
| `driftdrought.significance` | phase-randomized surrogate test with family-wise and FDR p-values |

## Documentation

https://drift-drought.readthedocs.io (built from `docs/` with Sphinx).

## Citation

If you use DRIFT, please cite the software (Zenodo DOI of the version) and the article (see `CITATION.cff`):

> Sanchez H., K. A. (2026). DRIFT for detecting and tracking spatiotemporal drought events with component trees.
> *Water Resources Research* (submitted).

## License

MIT (see `LICENSE`).
