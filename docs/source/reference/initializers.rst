.. _initializer:

Weight Initializers
===================

Weight initializers are used to initialize arrays.
They destructively modify the content of :class:`numpy.ndarray`
or :class:`cupy.ndarray`.
Typically, weight initializers are passed to :class:`~chainer.Link`\ s
to initialize their weights and biases.

A weight initializer can be any of the following objects.

* :class:`chainer.Initializer` class instance.
* Python or NumPy scalar or :class:`numpy.ndarray`.
* A callable that takes an array (:class:`numpy.ndarray` or :class:`cupy.ndarray`)
  and feeds the initial data into it.
* ``None``, in which case *the default initializer* is used.
  Unless explicitly documented, it is :class:`~chainer.initializers.LeCunNormal`
  with scale value 1.

If an initializer object has the ``dtype`` attribute, the initializer can assume that the array to feed the data into has that dtype. If the required dtype, depending on the context where the initializer is used, does not match the ``dtype`` attribute, Chainer will report an error.

Base class
----------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.Initializer

.. module:: chainer.initializers

..
   This currentmodule directive is to avoid the reference error due to
   initializers/__init__.py importing chainer.
.. currentmodule:: chainer

Concrete initializers
---------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.initializers.Identity
   chainer.initializers.Constant
   chainer.initializers.Zero
   chainer.initializers.One
   chainer.initializers.NaN
   chainer.initializers.Normal
   chainer.initializers.LeCunNormal
   chainer.initializers.GlorotNormal
   chainer.initializers.HeNormal
   chainer.initializers.Orthogonal
   chainer.initializers.Uniform
   chainer.initializers.LeCunUniform
   chainer.initializers.GlorotUniform
   chainer.initializers.HeUniform
   chainer.initializers.UpsamplingDeconvFilter
   chainer.initializers.DownsamplingConvFilter

Helper function
---------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.initializers.generate_array
