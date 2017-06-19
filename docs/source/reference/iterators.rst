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
