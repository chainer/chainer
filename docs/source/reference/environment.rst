Environment variables
=====================

Here are the environment variables Chainer uses.


+------------------------+--------------------------------------------------------------------------+
| ``CHAINER_CUDNN``      | Set ``0`` to disable cuDNN in Chainer.                                   |
|                        | Otherwise cuDNN is enabled automatically.                                |
+------------------------+--------------------------------------------------------------------------+
| ``CHAINER_SEED``       | Default seed value of random number generators for CUDA.                 |
|                        | If it is not set, the seed value is generated from Python random module. |
|                        | Set an integer value in decimal format.                                  |
+------------------------+--------------------------------------------------------------------------+
| ``CHAINER_TYPE_CHECK`` | Set ``0`` to disable type checking.                                      |
|                        | Otherwise type checking is enabled automatically.                        |
|                        | See :ref:`type-check-utils` for details.                                 |
+------------------------+--------------------------------------------------------------------------+
| ``CHAINER_DEBUG``      | Set ``1`` to enable debug mode. It is disabled by default.               |
|                        | In debug mode, Chainer performs various runtime checks that can help     |
|                        | debug user's code at the cost of some overhead.                          |
|                        | For details, see :ref:`debug`.                                           |
+------------------------+--------------------------------------------------------------------------+
