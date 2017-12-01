.. module:: chainer.dataset

Dataset abstraction
===================

Chainer has a support of common interface of training and validation datasets. The dataset support consists of three components: datasets, iterators, and batch conversion functions.

**Dataset** represents a set of examples. The interface is only determined by combination with iterators you want to use on it. The built-in iterators of Chainer requires the dataset to support ``__getitem__`` and ``__len__`` method. In particular, the ``__getitem__`` method should support indexing by both an integer and a slice. We can easily support slice indexing by inheriting :class:`DatasetMixin`, in which case users only have to implement :meth:`~DatasetMixin.get_example` method for indexing. Some iterators also restrict the type of each example. Basically, datasets are considered as `stateless` objects, so that we do not need to save the dataset as a checkpoint of the training procedure.

**Iterator** iterates over the dataset, and at each iteration, it yields a mini batch of examples as a list. Iterators should support the :class:`Iterator` interface, which includes the standard iterator protocol of Python. Iterators manage where to read next, which means they are `stateful`.

**Batch conversion function** converts the mini batch into arrays to feed to the neural nets. They are also responsible to send each array to an appropriate device. Chainer currently provides :func:`concat_examples` as the only example of batch conversion functions.

These components are all customizable, and designed to have a minimum interface to restrict the types of datasets and ways to handle them. In most cases, though, implementations provided by Chainer itself are enough to cover the usages.

Chainer also has a light system to download, manage, and cache concrete examples of datasets. All datasets managed through the system are saved under `the dataset root directory`, which is determined by the ``CHAINER_DATASET_ROOT`` environment variable, and can also be set by the :func:`set_dataset_root` function.


Dataset representation
~~~~~~~~~~~~~~~~~~~~~~
See :ref:`datasets` for dataset implementations.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.dataset.DatasetMixin

Iterator interface
~~~~~~~~~~~~~~~~~~
See :ref:`iterators` for dataset iterator implementations.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.dataset.Iterator

Batch conversion function
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.dataset.concat_examples
   chainer.dataset.ConcatWithAsyncTransfer
   chainer.dataset.to_device

Dataset management
~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.dataset.get_dataset_root
   chainer.dataset.set_dataset_root
   chainer.dataset.cached_download
   chainer.dataset.cache_or_load_file
.. module:: chainer.datasets

.. _datasets:

Examples
========

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
