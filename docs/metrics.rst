Evaluation metrics
==================

All metrics compare a *detected* label cube :math:`P` with a *true* label cube :math:`G` (shape time × y × x, 0 = no
event). Cells are weighted by their area :math:`w`, so on a latitude–longitude grid the scores are area-weighted.

Space–time overlap (IoU)
------------------------

For a true event :math:`g` and a detected event :math:`p`,

.. math:: \mathrm{IoU}(g,p) = \frac{\sum_{v \in g \cap p} w_v}{\sum_{v \in g \cup p} w_v}.

Object matching — :func:`driftdrought.metrics.object_match`
-----------------------------------------------------------

True and detected events are paired one-to-one by the Hungarian algorithm on the IoU matrix. A pair is a true
positive when its IoU reaches the threshold (0.5 by default; ``f1_iou25`` uses 0.25):

.. math:: \mathrm{precision} = \frac{TP}{N_P}, \qquad \mathrm{recall} = \frac{TP}{N_G}, \qquad
          F_1 = \frac{2\,\mathrm{precision}\cdot\mathrm{recall}}{\mathrm{precision}+\mathrm{recall}}.

It also returns the mean IoU of the matches and the relative volume error of matched events,
:math:`|V_p - V_g| / V_g`. The number of detected events (``n_pred``) against true events (``n_gt``) exposes
fragmentation (too many events) and chaining (too few) that a single score can hide.

HOTA — :func:`driftdrought.metrics.hota`
----------------------------------------

Higher Order Tracking Accuracy (Luiten et al., 2021) balances *detection* and *association*. Every time step is a
frame and every event slice a detection. For a localization threshold :math:`\alpha`, slices are matched per frame
(Hungarian, IoU :math:`\ge \alpha`) into true positives (TP), false negatives (FN) and false positives (FP):

.. math:: \mathrm{DetA}_\alpha = \frac{|TP|}{|TP| + |FN| + |FP|}, \qquad
          \mathrm{AssA}_\alpha = \frac{1}{|TP|}\sum_{c \in TP} \frac{|TPA(c)|}{|TPA(c)| + |FNA(c)| + |FPA(c)|},

where, for a matched pair :math:`c=(g,p)`, :math:`TPA(c)` counts the frames in which :math:`g` and :math:`p` are
matched, :math:`FNA(c)` the frames in which :math:`g` exists but is not matched to :math:`p`, and :math:`FPA(c)`
the frames in which :math:`p` exists but is not matched to :math:`g`. Then

.. math:: \mathrm{HOTA}_\alpha = \sqrt{\mathrm{DetA}_\alpha\,\mathrm{AssA}_\alpha}, \qquad
          \mathrm{HOTA} = \frac{1}{19}\sum_{\alpha=0.05}^{0.95} \mathrm{HOTA}_\alpha .

**Reading the scores.** DetA is high when drought cells are found in the right months; AssA is high when each true
drought keeps one identity through time (no fragmentation, no chaining of separate droughts). A detector that merges
two droughts into one long object can have a high DetA and a low AssA; one that breaks a drought into monthly pieces
has the opposite.

Partition agreement — :func:`driftdrought.metrics.partition_agreement`
----------------------------------------------------------------------

Over the cells labelled by either cube, the adjusted Rand index (ARI; 1 = identical partitions, about 0 = random) and
the variation of information,

.. math:: \mathrm{VI}(G,P) = H(G) + H(P) - 2\,I(G;P) \quad \text{(bits; 0 = identical)}.

Event confidence and significance
---------------------------------

These are attributes of each detected event, not scores against a truth; see :doc:`concepts` and the
:mod:`driftdrought.confidence` and :mod:`driftdrought.significance` modules.
