Weight Initializers
===================

Weight initializer is an instance of :class:`~chainer.Initializer` that
destructively edits the contents of :class:`numpy.ndarray` or :class:`cupy.ndarray`.
Typically, weight initializers are passed to ``__init__`` of :class:`~chainer.Link`
and initializes its the weights and biases.

Base class
----------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.Initializer

.. module:: chainer.initializers

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

Helper function
---------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.initializers.generate_array
