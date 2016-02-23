.. _extensions:
Standard Extension implementations (experimental)
=================================================

.. module:: chainer.trainer.extensions

Chainer provides basic :class:`chainer.trainer.Extension` implementations in
the :mod:`chainer.trainer.extensions` package.

.. note::
   These features are experimental.
   The interface may be changed in the future releases.

.. autoclass:: ComputationalGraph
.. autoclass:: Evaluator
.. autoclass:: ExponentialDecay
.. autoclass:: LinearShift
.. autoclass:: LogResult
   :members:
.. autoclass:: PrintResult
   :members:
.. autoclass:: Snapshot
