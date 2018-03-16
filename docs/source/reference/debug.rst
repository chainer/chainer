.. _debug:

Debug mode
==========

In debug mode, Chainer checks values of variables on runtime and shows more
detailed error messages.
It helps you to debug your programs.
However, it requires some additional overhead time.

You can enable debug mode with :func:`chainer.using_config`:

.. testcode::

    with chainer.using_config('debug', True):
       ...

See :ref:`configuration` for Chainer's configuration mechanism.

You can also set ``CHAINER_DEBUG`` environment variable to ``1`` to enable this mode.

In debug mode, Chainer checks all results of forward and backward computation, and if it finds a NaN value, it raises :class:`RuntimeError`.
Some functions and links also check validity of input values.

You can check if debug mode is enabled with :func:`chainer.is_debug` function.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.is_debug
   chainer.set_debug


Deprecated interface
--------------------

As of v2.0.0, it is recommended to turn on the debug mode using ``chainer.config.debug``.
See :ref:`configuration` for the way to use the config object.
We leave the reference of the conventional way (which has been available since Chainer v1) as follows.


.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.DebugMode
