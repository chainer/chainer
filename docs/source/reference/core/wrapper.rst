Wrapper Functions
=================

Wrapper functions are backward-able functions (probably differentiable), that take an array or a Chainer :class:`~chainer.Variable` as an argument and return a Chainer :class:`~chainer.Variable`, or a tuple of a Chainer :class:`~chainer.Variable`.
Wrapper functions should not have learnable parameters when used in models and are usually not members of the :class:`~chainer.Chain`. Even if they do have learnable parameters, these are ignored by Chainer training, to prevent the functions from changing output due to adjustment of learned parameters during training.

Argument inputs are tuples of input :class:`~chainer.Variable`, :class:`~numpy.ndarray` or :class:`~cupy.ndarray` objects. If the input is a :class:`~numpy.ndarray` or a :class:`~cupy.ndarray`, it is automatically wrapped with :class:`~chainer.Variable`.

Wrapper functions returns a :class:`~chainer.Variable` object or a tuple of multiple :class:`~chainer.Variable` objects.

