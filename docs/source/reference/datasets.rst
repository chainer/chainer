.. module:: chainer.datasets

.. _datasets:

Dataset examples
================

The most basic :mod:`~chainer.dataset` implementation is an array.
Both NumPy and CuPy arrays can be used directly as datasets.

In many cases, though, the simple arrays are not enough to write the training procedure.
In order to cover most of such cases, Chainer provides many built-in implementations of datasets.

These built-in datasets are divided into two groups.
One is a group of general datasets.
Most of them are wrapper of other datasets to introduce some structures (e.g., tuple or dict) to each data point.
The other one is a group of concrete, popular datasets.
These concrete examples use the downloading utilities in the :mod:`chainer.dataset` module to cache downloaded and converted datasets.

General datasets
----------------

General datasets are further divided into four types.

The first one is :class:`DictDataset` and :class:`TupleDataset`, both of which combine other datasets and introduce some structures on them.

The second one is :class:`ConcatenatedDataset` and :class:`SubDataset`.
:class:`ConcatenatedDataset` represents a concatenation of existing datasets. It can be used to merge datasets and make a larger dataset.
:class:`SubDataset` represents a subset of an existing dataset. It can be used to separate a dataset for hold-out validation or cross validation. Convenient functions to make random splits are also provided.

The third one is :class:`TransformDataset`, which wraps around a dataset by applying a function to data indexed from the underlying dataset.
It can be used to modify behavior of a dataset that is already prepared.

The last one is a group of domain-specific datasets. Currently, :class:`ImageDataset` and :class:`LabeledImageDataset` are provided for datasets of images.


DictDataset
~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.datasets.DictDataset

TupleDataset
~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.datasets.TupleDataset

ConcatenatedDataset
~~~~~~~~~~~~~~~~~~~
.. autoclass:: ConcatenatedDataset
   :members:

SubDataset
~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.datasets.SubDataset
   chainer.datasets.split_dataset
   chainer.datasets.split_dataset_random
   chainer.datasets.get_cross_validation_datasets
   chainer.datasets.get_cross_validation_datasets_random

TransformDataset
~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.datasets.TransformDataset

ImageDataset
~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.datasets.ImageDataset

LabeledImageDataset
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.datasets.LabeledImageDataset

Concrete datasets
-----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.datasets.get_mnist
   chainer.datasets.get_fashion_mnist
   chainer.datasets.get_cifar10
   chainer.datasets.get_cifar100
   chainer.datasets.get_ptb_words
   chainer.datasets.get_ptb_words_vocabulary
   chainer.datasets.get_svhn
