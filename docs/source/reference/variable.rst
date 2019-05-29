Variable and Parameter
======================

Variable classes and utilities
------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.Variable
   chainer.as_array
   chainer.as_variable
   chainer.Parameter
   chainer.variable.VariableNode


.. _ndarray:

N-dimensional array
-------------------

:class:`chainer.Variable` holds its value as an n-dimensional array (ndarray).
Chainer supports the following classes:

* :class:`numpy.ndarray`, including ``ideep4py.mdarray``
* :class:`cupy.ndarray`
* :class:`chainerx.ndarray`

.. note::
    Python scalars (``float``, etc.) and NumPy scalars (``numpy.float16``, ``numpy.float32``, etc.) cannot be used as :attr:`chainer.Variable.array`.
    See also :func:`chainer.utils.force_array`.
