Standard Dataset implementations
================================

.. module:: chainer.datasets

Chainer provides basic :class:`~chainer.Dataset` implementations in the
mod:`chainer.datasets` package. They are divided into three groups:
general datasets, specific datasets, and dataset loaders.


General datasets
----------------
.. autoclass:: ImageDataset
   :members:
.. autoclass:: ImageListDataset

.. autoclass:: SimpleDataset
   :members:


Specific datasets
-----------------


Dataset loaders
---------------
.. autoclass:: CrossValidationTrainingDataset
.. autoclass:: CrossValidationTestDataset
.. autofunction:: get_cross_validation_datasets
