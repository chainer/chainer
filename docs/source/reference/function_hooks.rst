Function hooks
==============

Chainer provides a function-hook mechanism that enriches
the behavior of forward and backward propagation of :class:`~chainer.Function`.

Base class
----------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.FunctionHook

.. module:: chainer.function_hooks

Concrete function hooks
-----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.function_hooks.PrintHook
   chainer.function_hooks.TimerHook
