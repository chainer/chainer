Function
--------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.Function
   chainer.FunctionAdapter
   chainer.FunctionNode
   chainer.force_backprop_mode
   chainer.no_backprop_mode
   chainer.grad

Wrapper Functions
=================

Wrapper functions are backward-able functions (probably differentiable) plain Python functions.

Argument inputs are tuples of input :class:`~chainer.Variable`, such as :class:`~numpy.ndarray` or :class:`~cupy.ndarray` objects. If the input is a :class:`~numpy.ndarray` or a :class:`~cupy.ndarray`, it is automatically wrapped with :class:`~chainer.Variable`.

Wrapper functions return a :class:`~chainer.Variable` object or a tuple of multiple :class:`~chainer.Variable` objects.

Wrapper functions should not have learnable parameters when used in models and are usually not members of the :class:`~chainer.Chain`. Even if they do have learnable parameters, these are ignored by Chainer training, to prevent the functions from changing output due to adjustment of learned parameters during training.
