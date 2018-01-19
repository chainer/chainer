.. module:: chainer.iterators

.. _iterators:

Iterator examples
=================

Chainer provides some iterators that implement typical strategies to create mini-batches by iterating over datasets.
:class:`SerialIterator` is the simplest one, which extract mini batches in the main thread.
:class:`MultiprocessIterator` and :class:`MultithreadIterator` are a parallelized version of :class:`SerialIterator`. It maintains worker subprocesses and subthreads to load the next mini-batch in parallel.


.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.iterators.SerialIterator
   chainer.iterators.MultiprocessIterator
   chainer.iterators.MultithreadIterator


Chainer also provides a handy way to use properly initialized :class:`numpy.random.RandomState` instances at iteration.


.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.iterators.get_random_state
