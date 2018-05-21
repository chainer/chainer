Sparse utilities
-----------------------------------

A :class:`chainer.Variable` can be converted into a sparse matrix in e.g.
COO (Coordinate list) format.
A sparse matrix stores the same data as the original object but with a
different internal representation, optimized for efficient operations on
sparse data, i.e. data with many zero elements.

Following are a list of supported sparse matrix formats and utilities for
converting between a :class:`chainer.Variable` and these representations.

.. note::

  Please be aware that only certain functions accept sparse matrices as
  inputs, such as :func:`chainer.functions.sparse_matmul`.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.utils.CooMatrix
   chainer.utils.to_coo
