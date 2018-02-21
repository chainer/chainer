Environment variables
=====================

Here are the environment variables Chainer uses.

+-------------------------------------------+-------------------------------------------------------------------------------------------------------+
| ``CHAINER_SEED``                          | Default seed value of random number generators for CUDA.                                              |
|                                           | If it is not set, the seed value is generated from Python random module.                              |
|                                           | Set an integer value in decimal format.                                                               |
+-------------------------------------------+-------------------------------------------------------------------------------------------------------+
| ``CHAINER_DATASET_ROOT``                  | Default directory path to store datasets downloaded by :doc:`core/dataset`.                           |
+-------------------------------------------+-------------------------------------------------------------------------------------------------------+
| ``CHAINER_CUDNN``                         | Set ``0`` to completely disable cuDNN in Chainer.                                                     |
|                                           | In this case, cuDNN will not be used regardless of ``CHAINER_USE_CUDNN`` and                          |
|                                           | ``chainer.config.use_cudnn`` configuration.                                                           |
|                                           | Otherwise cuDNN is enabled automatically.                                                             |
+-------------------------------------------+-------------------------------------------------------------------------------------------------------+
| ``CHAINER_USE_CUDNN``                     | Used as the default value for ``chainer.config.use_cudnn`` configuration.                             |
|                                           | The value must be any of ``'always'``, ``'auto'`` or ``'never'``.                                     |
|                                           | If ``CHAINER_CUDNN`` is set to ``0``, this environment variable has no effect.                        |
|                                           | See :ref:`configuration` for details.                                                                 |
+-------------------------------------------+-------------------------------------------------------------------------------------------------------+
| ``CHAINER_USE_IDEEP``                     | Used as the default value for ``chainer.config.use_ideep`` configuration.                             |
|                                           | The value must be any of ``'always'``, ``'auto'`` or ``'never'``.                                     |
|                                           | See :ref:`configuration` for details.                                                                 |
+-------------------------------------------+-------------------------------------------------------------------------------------------------------+
| ``CHAINER_TYPE_CHECK``                    | Used as the default value for ``chainer.config.type_check`` configuration.                            |
|                                           | Set ``0`` to disable type checking.                                                                   |
|                                           | Otherwise type checking is enabled automatically.                                                     |
|                                           | See :ref:`configuration` and :ref:`type-check-utils` for details.                                     |
+-------------------------------------------+-------------------------------------------------------------------------------------------------------+
| ``CHAINER_DEBUG``                         | Used as the default value for ``chainer.config.debug`` configuration.                                 |
|                                           | Set ``1`` to enable debug mode. It is disabled by default.                                            |
|                                           | In debug mode, Chainer performs various runtime checks that can help                                  |
|                                           | debug user's code at the cost of some overhead.                                                       |
|                                           | See :ref:`configuration` and :ref:`debug` for details.                                                |
+-------------------------------------------+-------------------------------------------------------------------------------------------------------+
| ``CHAINER_KEEP_GRAPH_ON_REPORT``          | Used as the default value for ``chainer.config.keep_graph_on_report`` configuration.                  |
|                                           | Set ``1`` to let :func:`report` keep the computational graph.                                         |
|                                           | See :ref:`configuration` for details.                                                                 |
+-------------------------------------------+-------------------------------------------------------------------------------------------------------+
| ``CHAINER_PYTHON_350_FORCE``              | Set ``1`` to force using Chainer with Python 3.5.0.                                                   |
|                                           | Note that Chainer does not work with Python 3.5.0.                                                    |
|                                           | Use Python 3.5.1+ or other supported versions (see :ref:`install-guide`).                             |
+-------------------------------------------+-------------------------------------------------------------------------------------------------------+

The following environment variables are only effective when running unit tests.

+------------------------------------------+----------------------------------------------------------------------------------------------------------------+
| ``CHAINER_TEST_GPU_LIMIT``               | Number of GPUs available for unit tests.                                                                       |
|                                          | When running unit test, test cases that require more GPUs than the specified value will be skipped.            |
|                                          | Set ``0`` to skip all test cases that require GPU.                                                             |
|                                          | See :ref:`testing-guide` for details.                                                                          |
+------------------------------------------+----------------------------------------------------------------------------------------------------------------+
| ``CHAINER_TEST_RANDOM_NONDETERMINISTIC`` | Set ``1`` to use non-fixed seed for random number generators, even for test cases annotated with `fix_random`. |
+------------------------------------------+----------------------------------------------------------------------------------------------------------------+
