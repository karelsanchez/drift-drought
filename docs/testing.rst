Tests and validation
====================

Automated tests
---------------

The test suite (``tests/``, run with ``pytest``) is executed by GitHub Actions on every push and pull request with
Python 3.10, 3.11 and 3.12.

=====================================  ===================================================================
Test                                   What it checks
=====================================  ===================================================================
``test_generator_is_deterministic``    the same scenario, noise and seed always give the same cube and truth
``test_detect_recovers_isolated``      DRIFT recovers isolated events: HOTA > 0.4 and F1 (IoU ≥ 0.25) > 0.6
``test_all_selection_rules_run``       the five selection rules (``persistence_vl``, ``persistence_v``,
                                       ``persistence``, ``leaf``, ``eom``) run and return events
``test_nan_cells_are_never_events``    cells outside the domain (NaN) never belong to an event
``test_merge_branches_joins_a_split``  lineage inheritance gives one identity to a lobe born inside an event
``test_baselines_and_confidence``      the reference detectors run; a catalogue has confidence 1 against itself
``test_select_floor``                  the floor rule returns one of the candidate floors and the full scan
``test_surrogate_test_small``          the surrogate test returns valid family-wise and FDR p-values
``test_example_library``               the bundled library of real drought shapes loads and builds cubes
=====================================  ===================================================================

Run them locally:

.. code-block:: bash

   pip install -e ".[test]"
   pytest -v

Equivalence with the article
----------------------------

Before release, :func:`driftdrought.detect` was compared with the research implementation used in the DRIFT article
on the four benchmark scenarios (noise 0.6, seed 3, final configuration): the label cubes were **identical** voxel by
voxel, and so were the synthetic cubes and their truth.

Reproducing the benchmark
-------------------------

``examples/make_figures.py`` runs a reduced benchmark (four scenarios, two noise levels, ten seeds, configurations
selected in the article) and writes ``docs/_static/mini_benchmark.csv``:

=====================  =========  =========  =========  =============  ==============
Method                 HOTA 0.3   HOTA 0.6   AssA 0.6   F1 0.25 (0.6)  events (0.6)
=====================  =========  =========  =========  =============  ==============
DRIFT                  **0.575**  **0.461**  **0.496**  **0.78**       6.8
CCL-2D + overlap       0.520      0.424      0.460      0.67           10.8
CCL-3D                 0.506      0.409      0.430      0.69           4.0
3D-DBSCAN              0.496      0.411      0.447      0.68           5.3
=====================  =========  =========  =========  =============  ==============

The cubes contain 4.6 true events on average. With 20 seeds and held-out selection the article reports HOTA 0.579 and
0.459 for DRIFT, against at most 0.530 and 0.428 for the reference detectors.
