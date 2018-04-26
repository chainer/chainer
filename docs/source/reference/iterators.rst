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

An Iterator iterates over a dataset according to an order represented by a 1-D array of indices.
Order samplers are callables that are used by those iterators to generate this array.


.. autosummary::
    :toctree generated/
    :nosignatures:

    chainer.iterators.OrderSampler
    chainer.iterators.ShuffleOrderSampler
