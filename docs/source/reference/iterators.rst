.. module:: chainer.iterators

.. _iterators:

Iterator
========

Chainer provides some iterators that implement typical strategies to create mini-batches by iterating over datasets.
:class:`~chainer.iterators.SerialIterator` is the simplest one, which extracts mini-batches in the main thread.
:class:`~chainer.iterators.MultiprocessIterator` and :class:`~chainer.iterators.MultithreadIterator` are parallelized versions of :class:`~chainer.iterators.SerialIterator`. They maintain worker subprocesses and subthreads, respectively, to load the next mini-batch in parallel.


.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.iterators.SerialIterator
   chainer.iterators.MultiprocessIterator
   chainer.iterators.MultithreadIterator
   chainer.iterators.DaliIterator


Order sampler examples
----------------------

An Iterator iterates over a dataset according to an order represented by a 1-D array of indices.
Order samplers are callables that are used by those iterators to generate this array.


.. autosummary::
    :toctree: generated/
    :nosignatures:

    chainer.iterators.OrderSampler
    chainer.iterators.ShuffleOrderSampler
