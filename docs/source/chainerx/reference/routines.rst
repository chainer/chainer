Array Operations
================

.. _chainerx_routines:

.. module:: chainerx

Array creation routines
-----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainerx.empty
   chainerx.empty_like
   chainerx.eye
   chainerx.identity
   chainerx.ones
   chainerx.ones_like
   chainerx.zeros
   chainerx.zeros_like
   chainerx.full
   chainerx.full_like
   chainerx.array
   chainerx.asarray
   chainerx.asanyarray
   chainerx.ascontiguousarray
   chainerx.copy
   chainerx.frombuffer
   chainerx.fromfile
   chainerx.fromfunction
   chainerx.fromiter
   chainerx.fromstring
   chainerx.loadtxt
   chainerx.arange
   chainerx.linspace
   chainerx.diag
   chainerx.diagflat
   chainerx.tri
   chainerx.tril
   chainerx.triu

Activation functions
--------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainerx.log_softmax
   chainerx.tanh
   chainerx.relu
   chainerx.sigmoid

Array manipulation routines
---------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainerx.reshape
   chainerx.ravel
   chainerx.transpose
   chainerx.broadcast_to
   chainerx.squeeze
   chainerx.asarray
   chainerx.ascontiguousarray
   chainerx.concatenate
   chainerx.stack
   chainerx.hstack
   chainerx.vstack
   chainerx.dstack
   chainerx.atleast_2d
   chainerx.atleast_3d
   chainerx.split
   chainerx.dsplit
   chainerx.swapaxes
   chainerx.repeat
   chainerx.expand_dims
   chainerx.flip
   chainerx.fliplr
   chainerx.flipud
   chainerx.moveaxis

Indexing routines
-----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainerx.take
   chainerx.where

Linear algebra
--------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainerx.dot

   chainerx.linalg.svd
   chainerx.linalg.solve
   chainerx.linalg.inv
   chainerx.linalg.pinv

Logic functions
---------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainerx.all
   chainerx.any

   chainerx.isinf
   chainerx.isnan

   chainerx.logical_and
   chainerx.logical_or
   chainerx.logical_xor
   chainerx.logical_not

   chainerx.greater
   chainerx.greater_equal
   chainerx.less
   chainerx.less_equal
   chainerx.equal
   chainerx.not_equal

Loss functions
--------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainerx.absolute_error
   chainerx.squared_error
   chainerx.huber_loss
   chainerx.gaussian_kl_divergence

Mathematical functions
----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainerx.negative
   chainerx.add
   chainerx.subtract
   chainerx.multiply
   chainerx.divide
   chainerx.sum
   chainerx.maximum
   chainerx.exp
   chainerx.log
   chainerx.log10
   chainerx.log2
   chainerx.log1p
   chainerx.logsumexp
   chainerx.log_softmax
   chainerx.sqrt
   chainerx.sin
   chainerx.cos
   chainerx.tan
   chainerx.arcsin
   chainerx.arccos
   chainerx.arctan
   chainerx.arctan2
   chainerx.sinh
   chainerx.cosh
   chainerx.tanh
   chainerx.arcsinh
   chainerx.arccosh
   chainerx.square
   chainerx.clip
   chainerx.fabs
   chainerx.sign
   chainerx.ceil
   chainerx.floor
   chainerx.bitwise_and
   chainerx.bitwise_or
   chainerx.bitwise_xor
   chainerx.left_shift
   chainerx.right_shift

Random sampling
---------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainerx.random.normal
   chainerx.random.uniform

Sorting, searching, and counting
--------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainerx.argmax
   chainerx.argmin

Statistics
----------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainerx.amax
   chainerx.mean
   chainerx.var

Connection
----------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainerx.conv
   chainerx.conv_transpose
   chainerx.linear

Normalization
-------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainerx.batch_norm
   chainerx.fixed_batch_norm

Pooling
-------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainerx.max_pool
   chainerx.average_pool
