Dataset
=======

.. currentmodule:: chainer

:class:`Dataset` provides a reusable interface of dataset implementations.
There are two types of datasets: basic datasets and dataset adapters.
Chainer provides some useful datasets and dataset adapters under the :mod:`chainer.datasets` module.

Dataset class
-------------
.. autoclass:: Dataset
   :members:

Batch iterator implementation
-----------------------------
.. autoclass:: chainer.dataset.BatchIterator
   :members:
.. autofunction:: chainer.dataset.build_minibatch

Canonical paths to datasets
---------------------------
.. autofunction:: chainer.dataset.set_dataset_root
.. autofunction:: chainer.dataset.get_dataset_path
