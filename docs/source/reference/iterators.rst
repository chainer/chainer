.. module:: chainer.iterators

.. _iterators:

Iterator
========

Chainer provides some iterators that implement typical strategies to create mini-batches by iterating over datasets.
:class:`SerialIterator` is the simplest one, which extract mini-batches in the main thread.
:class:`MultiprocessIterator` and :class:`MultithreadIterator` are a parallelized version of :class:`SerialIterator`. It maintains worker subprocesses and subthreads to load the next mini-batch in parallel.


.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.iterators.SerialIterator
   chainer.iterators.MultiprocessIterator
   chainer.iterators.MultithreadIterator


Order sampler examples
======================

An Iterator iterate over a dataset according to ``order``, which is 1-D array of the indices of a dataset.
Order samplers are functions that are called inside of an iterator to generate an order.


.. autosummary::
    :toctree generated/
    :nosignatures

    chainer.iterators.NoShuffleOrderSampler
    chainer.iterators.ShuffleOrderSampler
