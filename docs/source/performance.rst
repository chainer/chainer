Performance Best Practices
==========================

This guide explains some tips and advice for maximizing the performance of Chainer.


Use the Latest Version
----------------------

It is generally recommended to use the latest version of Chainer and its dependent libraries (CUDA, cuDNN, iDeep, etc.).
Some of the new features and performance optimizations introduced in newer versions of dependent libraries may not be available in older versions of Chainer.
Also, Chainer itself is incrementally being improved to provide better performance.

If you are using Chainer v4 or later, you can check the version configuration by:

.. testcode::

    chainer.print_runtime_info()

.. testoutput::
    :hide:

    ...

.. code::

    Chainer: 4.0.0
    NumPy: 1.14.3
    CuPy:
      CuPy Version          : 4.0.0
      CUDA Root             : /usr/local/cuda
      CUDA Build Version    : 9000
      CUDA Driver Version   : 9000
      CUDA Runtime Version  : 9000
      cuDNN Build Version   : 7100
      cuDNN Version         : 7100
      NCCL Build Version    : 2102

Generally, the Chainer team is maintaining the API between minor updates (e.g., v4.0 to v4.1) so that users can upgrade Chainer without modifying their code (see :doc:`compatibility` for our policy).
As for major updates, please refer to the :doc:`upgrade` to understand what should be done for migration.

Enable Hardware Accelerations
-----------------------------

Using GPU
~~~~~~~~~

In most cases, running on GPU will give you better performance than on CPU.
When using GPU, also make sure to install cuDNN, which is a library to accelerate deep neural network computations.

.. note::

    You don't have to manually install cuDNN if you are using `CuPy wheels <https://docs-cupy.chainer.org/en/latest/install.html#install-cupy-from-source>`_, which includes the latest version of cuDNN.
    Check the output of ``chainer.print_runtime_info()``; if you see the cuDNN version number, it is installed properly and will be used by Chainer automatically.

.. note::

    If you wish, you can manually disable use of cuDNN using ``chainer.config.use_cudnn`` configuration option.
    See :doc:`reference/configuration` for details.

Using CPU
~~~~~~~~~

If you are running Chainer on CPU, you can use `iDeep <https://github.com/intel/ideep>`__ to utilize vector instructions of CPU.
See :doc:`tips` for steps to run your model with iDeep.

You can also improve performance by building NumPy linked to `Intel MKL <https://software.intel.com/en-us/mkl>`__.
See `Numpy/Scipy with Intel® MKL and Intel® Compilers <https://software.intel.com/en-us/articles/numpyscipy-with-intel-mkl>`__ for the detailed instructions.

.. note::

    If you installed `numpy <https://anaconda.org/anaconda/numpy>`__ package using Anaconda, you may already have MKL-linked NumPy.
    Check the output of ``numpy.show_config()`` to see what linear algebra library is linked.

.. note::

    Use of iDeep and MKL-linked NumPy are orthogonal.
    You can use both of them at once to maximize the performance.

Migrate Data Preprocessing Code from NumPy to CuPy
--------------------------------------------------

If you are preprocessing your dataset or running data augmentation using NumPy, you may be able to use CuPy as a substitution to improve performance.

.. note::

    It is **not always** efficient to use CuPy instead of NumPy, especially when the computation is not very heavy, or it cannot be done in batch.

Avoid Data Transfer
-------------------

If you are using GPU, be aware of data transfer between CPU and GPU.
For example, ``print``\ing :class:`chainer.Variable` on GPU (e.g., for debugging) will cause memory transfer from GPU to CPU, which will incur synchronization overhead.

You can use `NVIDIA Visual Profiler <https://docs.nvidia.com/cuda/profiler-users-guide/>`__ to diagnose this kind of issue.

Optimize cuDNN Convolution
--------------------------

Workspace Size
~~~~~~~~~~~~~~

Some convolution algorithms in cuDNN use additional GPU memory as a temporary buffer.
This is called "workspace," and users can adjust the upper limit of its size.
By increasing the limit of workspace size, cuDNN may be able to use better (i.e., memory consuming but faster) algorithm.

The default size (in bytes) is:

.. doctest::

    >>> chainer.backends.cuda.get_max_workspace_size()
    8388608

and can be adjusted using :func:`chainer.backends.cuda.set_max_workspace_size`.

Maximum required workspace size may vary depending on various conditions such as GPU hardware and batch size of inputs.

Auto-Tuner
~~~~~~~~~~

Some convolution algorithms in cuDNN support the auto-tuner feature that finds the fastest convolution algorithm for given inputs.
You can turn on this feature by setting ``autotune`` configuration to ``True``.

See :doc:`reference/configuration` for detailed descriptions.

.. note::

    Auto-tuner tries to find the best algorithm for every first observation of the input shape combination.
    Therefore, the first batch will become slower when auto-tuner is enabled.
    The result of auto-tuner is cached on memory so that it can be reused for data with the same input shape combination.
    In other words, algorithm selected in the first batch will be reused for the second and later batches, as long as the input shape combination is the same.

    If you set ``autotune`` configuration to ``False``, the default convolution algorithm will always be selected, regardless of the previous auto-tuner results.

.. note::

    Auto-tuner always use the maximum workspace size.

Fine-Tune Configuration
-----------------------

There are some Chainer configuration values that affect performance.
Although the default values work well in most cases, you can adjust the following configurations for better performance.

* ``enable_backprop``

  If you are running your model for inference (i.e., you don't have to use back propagation because you are not training the model), you can set this configuration to ``False`` to improve performance and reduce memory consumption.

* ``type_check``

  By default, Chainer checks the integrity between input data and functions.
  This makes possible to display friendly message when, for example, data with invalid dtype or shape is given to a function.
  By setting this configuration to ``False``, you can let Chainer skip such check to improve performance.
  It is recommended to turn off the check only for well-tested code and input data.

See :doc:`reference/configuration` for detailed descriptions.

Load Datasets Concurrently
--------------------------

If loading process of your dataset is I/O-bound or CPU-bound, consider using :class:`chainer.iterators.MultithreadIterator` or :class:`chainer.iterators.MultiprocessIterator` to load dataset concurrently using multiple threads or processes, instead of :class:`chainer.iterators.SerialIterator` which works in a single thread in a single process.

Use Multiple GPUs
-----------------

You can utilize multiple GPUs to make the training process faster.

For data parallelism, you can use :class:`chainer.training.updaters.ParallelUpdater` or :class:`chainer.training.updaters.MultiprocessParallelUpdater` instead of :class:`chainer.training.updaters.StandardUpdater`.
For model parallelism, you need to manually transfer each :class:`chainer.Link` in your model to each device.

See :doc:`guides/gpu` for the working examples of each case.

Use Multiple Nodes
------------------

You can scale-out the training process of your Chainer model to multiple-node cluster by using `ChainerMN <http://github.com/chainer/chainermn>`__, an additional package for Chainer which enables distributed deep learning.
See `ChainerMN Official Documentation <http://chainermn.readthedocs.io/en/latest/>`_ for details.
