Installation
============

DRIFT needs Python 3.10 or newer.

.. code-block:: bash

   pip install drift-drought            # once published on PyPI
   pip install git+https://github.com/karelsanchez/drift-drought   # development version

Optional extras: ``[plot]`` (matplotlib, for the examples), ``[test]`` (pytest), ``[docs]`` (Sphinx).

Main dependencies: NumPy, SciPy, pandas, `Higra <https://higra.readthedocs.io>`_ (component trees), Numba,
scikit-image (watershed footprints), scikit-learn (DBSCAN baseline) and xarray (drought indices).
