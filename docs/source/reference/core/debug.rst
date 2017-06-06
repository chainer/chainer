.. _debug:

Debug mode
==========

In debug mode, Chainer checks values of variables on runtime and shows more
detailed error messages.
It helps you to debug your programs.
Instead it requires additional overhead time.

In debug mode, Chainer checks all results of forward and backward computation, and if it founds a NaN value, it raises :class:`RuntimeError`.
Some functions and links also check validity of input values.

As of v2.0.0, it is recommended to turn on the debug mode using ``chainer.config.debug``.
See :ref:`configuration` for the way to use the config object.
We leave the reference of the conventional ways (which have been available since Chainer v1) as follows.


.. currentmodule:: chainer

.. autofunction:: is_debug
.. autofunction:: set_debug
.. autoclass:: DebugMode
