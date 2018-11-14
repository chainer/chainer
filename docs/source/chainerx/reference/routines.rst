Array Operations
================

.. module:: chainerx

Array creation routines
-----------------------

Ones and zeros
^^^^^^^^^^^^^^

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

From existing data
^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :nosignatures:

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

Numerical ranges
^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainerx.arange
   chainerx.linspace

Building matrices
^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainerx.diag
   chainerx.diagflat

Array manipulation routines
---------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainerx.reshape
   chainerx.transpose
   chainerx.broadcast_to
   chainerx.squeeze
   chainerx.asarray
   chainerx.ascontiguousarray
   chainerx.asscalar
   chainerx.concatenate
   chainerx.stack
   chainerx.split

Indexing routines
-----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainerx.take

Linear algebra
--------------

Matrix and vector products
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainerx.dot

Logic functions
---------------

Array contents
^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainerx.isinf
   chainerx.isnan

Logical operations
^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainerx.logical_not

Comparison
^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainerx.greater
   chainerx.greater_equal
   chainerx.less
   chainerx.less_equal
   chainerx.equal
   chainerx.not_equal

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
   chainerx.logsumexp
   chainerx.log_softmax
   chainerx.sqrt
   chainerx.tanh

Sorting routines
----------------

Statistics
----------

Order statistics
^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainerx.amax

Connection
----------

Convolution
^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainerx.conv
   chainerx.conv_transpose

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
