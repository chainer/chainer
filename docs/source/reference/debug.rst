.. _debug:

Debug Mode
==========

In debug mode, Chainer checks values of variables on runtime and shows more detailed error messages.
It helps you to debug your programs.
However, it requires some additional overhead time.

If you want to enable debug mode for the entire code, you can set ``CHAINER_DEBUG`` environment variable to ``1``.

You can also enable or disable debug mode for the specific scope of code with :func:`chainer.using_config` or by changing ``chainer.config.debug`` configuration.

.. testcode::

    with chainer.using_config('debug', True):
       ...

See :ref:`configuration` for the details of Chainer's configuration mechanism.

In debug mode, Chainer checks all results of forward and backward computation, and if it finds a NaN value, it raises :class:`RuntimeError`.
Some functions and links also check validity of input values more strictly.

You can check if debug mode is enabled with :func:`chainer.is_debug` function.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.is_debug
   chainer.set_debug


Deprecated interface
--------------------

It is recommended to turn on the debug mode using ``chainer.config.debug``.
See :ref:`configuration` for the way to use the config object.
