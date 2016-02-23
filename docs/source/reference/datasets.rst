Standard Dataset implementations (experimental)
===============================================

.. module:: chainer.datasets

Chainer provides basic :class:`~chainer.Dataset` implementations in the
mod:`chainer.datasets` package. They are divided into three groups:
general datasets, specific datasets, and dataset loaders.

.. note::
   These features are experimental.
   The interface may be changed in the future releases.


General datasets
----------------

Array datasets
~~~~~~~~~~~~~~
.. autoclass:: ArrayDataset
   :members:

Image datasets
~~~~~~~~~~~~~~
.. autoclass:: ImageDataset
   :members:
.. autoclass:: ImageListDataset


Specific datasets
-----------------


Dataset loaders
---------------

Splitting dataset
~~~~~~~~~~~~~~~~~
.. autoclass:: CrossValidationTrainingDataset
.. autoclass:: CrossValidationTestDataset
.. autofunction:: get_cross_validation_datasets
.. autoclass:: SubDataset
.. autofunction:: split_dataset
.. autofunction:: split_dataset_random

Parallel loading
~~~~~~~~~~~~~~~~
.. autoclass:: MultiprocessLoader
