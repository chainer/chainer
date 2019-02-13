.. _configuration:

Configuring Chainer
===================

..  currentmodule:: chainer

Chainer provides some global settings that affect the behavior of some functionalities.
Such settings can be configured using the *unified configuration system*.
The system provides a transparent way to manage the configuration for each process and for each thread.

The configuration is managed by two global objects: :data:`chainer.global_config` and :data:`chainer.config`.

- The :data:`global_config` object maintains the configuration shared in the Python process.
  This is an instance of the :class:`~chainer.configuration.GlobalConfig` class.
  It can be used just as a plain object, and users can freely set any attributes on it.
- The :data:`config` object, on the other hand, maintains the configuration for the current thread.
  This is an instance of the :class:`~chainer.configuration.LocalConfig` class.
  It behaves like a thread-local object, and any attribute modifications are only visible to the current thread.

If no value is set to :data:`config` for a given key, :data:`global_config` is transparently referred.
Thanks to this transparent lookup, users can always use :data:`config` to read any configuration so that the thread-local configuration is used if available and otherwise the default global setting is used.

The following entries of the configuration are currently provided by Chainer.
Some entries support environment variables to set the default values.
Note that the default values are set in the global config.

Configuration Keys
------------------

* ``cudnn_deterministic`` (default: ``False``)
   Flag to configure deterministic computations in cuDNN APIs.

   If it is ``True``, convolution functions that use cuDNN use the deterministic mode (i.e, the computation is reproducible).
   Otherwise, the results of convolution functions using cuDNN may be non-deterministic in exchange for better performance.

* ``debug`` (default: ``False``)
   Debug mode flag.

   If it is ``True``, Chainer runs in debug mode.
   Enabling debug mode may introduce some performance overhead.
   See :ref:`debug` for more information of the debug mode.

   You can change the default value to ``True`` by setting ``CHAINER_DEBUG`` environment variable to ``1``.

* ``dtype`` (default: ``numpy.float32``)
   Default floating point data type.

   Chainer uses this dtype to construct arrays when the dtype is not specified (e.g. initializers).

   You can change the default value by setting ``CHAINER_DTYPE`` environment variable to ``float16``, ``float32`` or ``float64``.

* ``enable_backprop`` (default: ``True``)
   Flag to enable backpropagation support.

   If it is ``True``, computational graphs are created during forward passes by :class:`FunctionNode`\ s, allowing backpropagation to start from any :class:`Variable` in the graph.
   Otherwise, computational graphs are not created but memory consumptions are reduced.
   So calling :func:`~chainer.Variable.backward` on the results of a function will not compute any gradients of any input.

* ``keep_graph_on_report`` (default: ``False``)
   Flag to configure whether or not to let :func:`report` keep the computational graph.

   If it is ``False``, :func:`report` does not keep the computational graph when a :class:`Variable` object is reported.
   It means that :func:`report` stores a copy of the :class:`Variable` object which is purged from the computational graph.
   If it is ``True``, :func:`report` just stores the :class:`Variable` object as is with the computational graph left attached.

   You can change the default value to ``True`` by setting ``CHAINER_KEEP_GRAPH_ON_REPORT`` environment variable to ``1``.

* ``warn_nondeterministic`` (default: ``False``)
   Flag to give warning when a non-deterministic function is used. This function is experimental.

   If it is true, then functions that use non-deterministic functions and cannot be given a seed, such as atomicAdd, will
   give a warning when executed. For functions that can take a seed argument, such as
   :func:`~chainer.datasets.split_dataset_random`, setting the seed should be done when the function is called and will not
   be flagged by this setting.
   
   Note that this feature is provided as best-effort. It cannot assure that every nondeterministic function can be detected.  For example, SSE computations in CPU mode may cause non-deterministic behavior that would not raise a warning.

   Also, determinisitic outputs may still result, even if this flag produces a non-deterministic warning. For example, reduction on 1-dim axis should always be deterministic, but it may raise a warning.

* ``train`` (default: ``True``)
   Training mode flag.

   If it is ``True``, Chainer runs in training mode.
   Otherwise, it runs in the testing (evaluation) mode.

   This configuration is used by Functions and Links that need to behave differently between training phase and evaluation (inference) phase.
   One example is :class:`chainer.links.BatchNormalization` updates statistics using input data only when ``train`` is set to ``True``.
   The other example is :func:`chainer.functions.dropout`, which does nothing when ``train`` is set to ``False``.

   Generally, you are responsible to change the configuration to ``False`` during evaluation.
   If you are using :class:`~chainer.training.Trainer` with :class:`~chainer.training.extensions.Evaluator` extension, ``train`` configuration will automatically be switched to ``False`` during evaluation in the training loop.

   Note that this parameter does not reduce memory consumption or affect the creation of computational graphs required in order to compute gradients.

* ``type_check`` (default: ``True``)
   Type checking mode flag.

   If it is ``True``, Chainer checks the types (data types and shapes) of inputs on :class:`Function` applications.
   Otherwise, it skips type checking.

   You can change the default value to ``False`` by setting ``CHAINER_TYPE_CHECK`` environment variable to ``0``.

* ``use_cudnn`` (default: ``'auto'``)
   Flag to configure whether or not to use cuDNN.

   This is a ternary flag with ``'always'``, ``'auto'``, and ``'never'`` as its allowed values.
   The meaning of each flag is as follows.

       - If it is ``'always'``, Chainer will try to use cuDNN everywhere if possible.
       - If it is ``'auto'``, Chainer will use cuDNN only if it is known that the usage does not degrade the performance.
       - If it is ``'never'``, Chainer will never use cuDNN anywhere.

   You can change the default value by setting ``CHAINER_USE_CUDNN`` environment variable to any of ``'always'``, ``'auto'`` or ``'never'``.

* ``use_ideep`` (default: ``'never'``)
   Flag to configure whether or not to use iDeep.

   This is a ternary flag with ``'always'``, ``'auto'``, and ``'never'`` as its allowed values.
   The meaning of each flag is as follows.

       - If it is ``'always'``, Chainer will try to use iDeep everywhere if possible.
       - If it is ``'auto'``, Chainer will use iDeep only if it is known that the usage does not degrade the performance.
       - If it is ``'never'``, Chainer will never use iDeep anywhere.

   You can change the default value by setting ``CHAINER_USE_IDEEP`` environment variable to any of ``'always'``, ``'auto'`` or ``'never'``.

   Note that in spite of the configuration, optimizers will use iDeep if and only if the link is converted manually to iDeep (e.g., ``model.to_intel64()``).

* ``lazy_grad_sum`` (default: ``False``)
   Flag to control the behavior of gradient accumulation.

   If it is ``True``, gradients are accumulated in batch for performance.
   Otherwise gradients are accumulated one by one.

   You can change the default value to ``True`` by setting ``CHAINER_LAZY_GRAD_SUM`` environment variable to ``1``.

* ``use_cudnn_tensor_core`` (default: ``'auto'``)
   Flag to configure whether or not to enable Tensor Core operatons in cuDNN.

   This is a ternary flag with ``'always'``, ``'auto'``, and ``'never'`` as its allowed values.
   The meaning of each flag is as follows.

       - If it is ``always``, Chainer uses cuDNN's Tensor Core operations.
       - If it is ``never``, Chainer does not use cuDNN's Tensor Core operations.
       - If it is ``auto``, Chainer checks cuDNN version, the data type of input, the compute capability of the GPU used, and configures whether or not to use cuDNN's Tensor Core operations.

* ``autotune`` (default: ``False``)
   Autotune for convolutional networks flag.

   If it is ``True``, Chainer uses the cuDNN autotune feature to find the fastest calculation process for :class:`chainer.links.Convolution2D`, :class:`ConvolutionND`, :class:`Deconvolution2D`, or :class:`DeconvolutionND` links.

* ``cudnn_fast_batch_normalization`` (default: ``False``)
   Flag to configure whether or not to enable use of fast implementation for batch normalization in cuDNN.

   If ``True``, Chainer will try to use the fast implementation for batch normalization in cuDNN by setting cuDNN's batch normalization mode to ``CUDNN_BATCHNORM_SPATIAL_PERSISTENT``. You can change the default value to ``True`` by setting ``CHAINER_CUDNN_FAST_BATCH_NORMALIZATION`` environment variable to ``1``.

* ``in_recomputing`` (default: ``False``)
   This flag is automatically set by :func:`chainer.functions.forget` and not intended to be changed by users.
   You can use this flag when implementing your own Link to avoid updating the internal states during recomputation done by :func:`chainer.functions.forget`.
   See the documentation of :func:`chainer.functions.forget` for details.


User-defined Keys
-----------------

Users can also define their own configurations.
There are two ways:

1. Use Chainer's configuration objects.
   In this case, **it is strongly recommended to prefix the name by "user_"** to avoid name conflicts with configurations introduced to Chainer in the future.
2. Use your own configuration objects.
   Users can define their own configuration objects using :class:`chainer.configuration.GlobalConfig` and :class:`chainer.configuration.LocalConfig`.
   In this case, there is no need to take care of the name conflicts.


Changing Configuration
----------------------

If you want to share a setting within the process, set an attribute to the global configuration.
This value is automatically extracted by referring to the local config.

.. doctest::

  >>> chainer.global_config.train
  True
  >>> chainer.config.train
  True

  >>> chainer.global_config.train = False

  >>> chainer.global_config.train
  False
  >>> chainer.config.train
  False

.. doctest::
  :hide:

  >>> chainer.global_config.train = True

If you set an attribute to the local configuration, the value is only visible to the current thread.

.. doctest::

  >>> chainer.global_config.train
  True
  >>> chainer.config.train
  True

  >>> chainer.config.train = False

  >>> chainer.global_config.train
  True
  >>> chainer.config.train
  False

.. doctest::
  :hide:

  >>> chainer.config.train = True

If you want to temporarily modify the configuration for the specific scope, you can use :func:`using_config`.
For example, if you only want to enable debug mode in a fragment of code, write as follows.

.. doctest::

  >>> with chainer.using_config('debug', True):
  ...     pass  # code running in debug mode

If you want to switch to the test mode for an evaluation, you can do that in the same way.

.. doctest::

  >>> # Do training here
  >>> with chainer.using_config('train', False):
  ...     pass  # Perform evaluation here

Note that :class:`~chainer.training.extensions.Evaluator` automatically switches to the test mode, and thus you do not need to manually switch in the loss function for the evaluation.

You can also make your own code behave differently in training and test modes as follows.

.. code-block:: python

  if chainer.config.train:
      pass  # code only running in the training mode
  else:
      pass  # code only running in the test mode


.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.global_config
   chainer.config
   chainer.using_config
   chainer.configuration.GlobalConfig
   chainer.configuration.LocalConfig

Related functions
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.get_dtype


Environment Variables
---------------------

Here are the environment variables Chainer uses.


+-------------------------------------------+-------------------------------------------------------------------------------------------------------+
| ``CHAINER_SEED``                          | Default seed value of random number generators for CUDA.                                              |
|                                           | If it is not set, the seed value is generated from Python random module.                              |
|                                           | Set an integer value in decimal format.                                                               |
+-------------------------------------------+-------------------------------------------------------------------------------------------------------+
| ``CHAINER_DATASET_ROOT``                  | Default directory path to store the downloaded datasets.                                              |
|                                           | See :doc:`datasets` for details.                                                                      |
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
| ``CHAINER_CUDNN_FAST_BATCH_NORMALIZATION``| Used as the default value for ``chainer.config.cudnn_fast_batch_normalization`` configuration.        |
|                                           | Set ``1`` to enable use of fast implementation for batch normalization in cuDNN.                      |
|                                           | See :ref:`configuration` for details.                                                                 |
+-------------------------------------------+-------------------------------------------------------------------------------------------------------+
| ``CHAINER_USE_IDEEP``                     | Used as the default value for ``chainer.config.use_ideep`` configuration.                             |
|                                           | The value must be any of ``'always'``, ``'auto'`` or ``'never'``.                                     |
|                                           | See :ref:`configuration` for details.                                                                 |
+-------------------------------------------+-------------------------------------------------------------------------------------------------------+
| ``CHAINER_LAZY_GRAD_SUM``                 | Used as the default value for ``chainer.config.lazy_grad_sum`` configuration.                         |
|                                           | Set ``1`` to enable batch accumulation of gradients.                                                  |
|                                           | See :ref:`configuration` for details.                                                                 |
+-------------------------------------------+-------------------------------------------------------------------------------------------------------+
| ``CHAINER_DTYPE``                         | Used as the default value for ``chainer.config.dtype`` configuration.                                 |
|                                           | The value must be any of ``'float16'``, ``'float32'`` or ``'float64'``.                               |
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
