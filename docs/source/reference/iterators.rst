.. module:: chainer.iterators

.. _iterators:

Iterator examples
=================

Chainer provides some iterators that implement typical strategies to create mini-batches by iterating over datasets.
:class:`SerialIterator` is the simplest one, which extract mini batches in the main thread.
:class:`MultiprocessIterator` is a parallelized version of :class:`SerialIterator`. It maintains worker subprocesses to load the next mini-batch in parallel.


.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.iterators.SerialIterator
   chainer.iterators.MultiprocessIterator


Order sampler examples
======================

An Iterator iterate over a dataset according to ``order``, which is 1-D array of the indices of a dataset.
Order samplers are functions that are called inside of an iterator to generate an order.


.. autosummary::
    :toctree generated/
    :nosignatures

    chainer.iterators.no_shuffle_order_sampler
    chainer.iterators.shuffle_order_sampler
