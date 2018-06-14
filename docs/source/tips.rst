Tips and FAQs
=============

It takes too long time to compile a computational graph. Can I skip it?
-----------------------------------------------------------------------

Chainer does not compile computational graphs, so you cannot skip it, or, I mean, you have already skipped it :).

It seems you have actually seen on-the-fly compilations of CUDA kernels.
CuPy compiles kernels on demand to make kernels optimized to the number of dimensions and element types of input arguments.
Pre-compilation is not available, because we have to compile an exponential number of kernels to support all CuPy functionalities.
This restriction is unavoidable because Python cannot call CUDA/C++ template functions in generic way.
Note that every framework using CUDA require compilation at some point; the difference between other statically-compiled frameworks (such as cutorch) and Chainer is whether a kernel is compiled at installation or at the first use.

These compilations should run only at the first use of the kernels.
The compiled binaries are cached to the ``$(HOME)/.cupy/kernel_cache`` directory by default.
If you see that compilations run every time you run the same script, then the caching is failed.
Please check that the directory is kept as is between multiple executions of the script.
If your home directory is not suited to caching the kernels (e.g. in case that it uses NFS), change the kernel caching directory by setting the ``CUPY_CACHE_DIR`` environment variable to an appropriate path.
See `CuPy Overview <https://docs-cupy.chainer.org/en/stable/overview.html>`_ for more details.


MNIST example does not converge in CPU mode on Mac OS X
-------------------------------------------------------

.. note::

   Mac OS X is not officially supported.
   Please use it at your own risk.

Many users have reported that MNIST example does not work correctly
when using vecLib as NumPy backend on Mac OS X.
vecLib is the default BLAS library installed on Mac OS X.

We recommend using other BLAS libraries such as `OpenBLAS <http://www.openblas.net/>`_.

To use an alternative BLAS library, it is necessary to reinstall NumPy.
Here is an instruction to install NumPy with OpenBLAS using `Homebrew <https://brew.sh/>`_.

::

   $ brew tap homebrew/science
   $ brew install openblas
   $ brew install numpy --with-openblas

If you want to install NumPy with pip, use `site.cfg <https://github.com/numpy/numpy/blob/master/site.cfg.example>`_ file.

For details of this problem, see `issue #704 <https://github.com/chainer/chainer/issues/704>`_.


How do I fix InvalidType error?
-------------------------------

Chainer raises an ``InvalidType`` exception when invalid inputs are given to :doc:`reference/functions`.
If you got ``InvalidType``, generally you need to check if ``dtype`` and/or ``shape`` of inputs are valid for the function.

Here are some examples of ``InvalidType`` errors:

.. testcode::

    import chainer.functions as F
    import numpy as np

    x = np.arange(10) - 5
    F.relu(x)

.. testoutput::

   Traceback (most recent call last):
   ...
   chainer.utils.type_check.InvalidType:
   Invalid operation is performed in: ReLU (Forward)

   Expect: in_types[0].dtype.kind == f
   Actual: i != f

In this case, :class:`~numpy.dtype.kind` of ``in_types[0]`` (which means the first input to the function, ``x``) is expected to be ``f`` (floating-point), whereas the input was ``i`` (signed integer).
You need to cast the input appropriately before passing to the function (e.g., ``x.astype(np.float32)``).

.. testcode::

    import chainer.functions as F
    import numpy as np

    x = np.ones((4, 4))
    y = np.ones((3, 3))
    F.concat([x, y])

.. testoutput::

   Traceback (most recent call last):
   ...
   chainer.utils.type_check.InvalidType:
   Invalid operation is performed in: Concat (Forward)

   Expect: in_types[0].shape[0] == in_types[1].shape[0]
   Actual: 4 != 3

In this case, the function expects that ``x.shape[0]`` is equal to ``y.shape[0]``, but actually it was ``4`` and ``3``, respectively.

See :doc:`guides/type_checks` for the detailed behavior of type checking system in Chainer.

How do I accelerate my model using iDeep on Intel CPU?
------------------------------------------------------

Follow these steps to utilize iDeep in your model.

Install iDeep
~~~~~~~~~~~~~

The following environments are recommended by `iDeep <https://github.com/intel/ideep>`_.

* Ubuntu 14.04 / 16.04 LTS (64-bit) and CentOS 7 (64-bit)
* Python 2.7.5+, 3.5.2+, and 3.6.0+

On recommended systems, you can install iDeep wheel (binary distribution) by:

.. code-block:: console

    $ pip install ideep4py

Enable iDeep Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~

Currently iDeep is disabled by default because it is an experimental feature.
You need to manually enable iDeep by changing ``chainer.config.use_ideep`` configuration to ``'auto'``.
See :ref:`configuration` for details.

The easiest way to change the configuration is to set environment variable as follows:

.. code-block:: console

    export CHAINER_USE_IDEEP="auto"

You can also use :func:`chainer.using_config` to change the configuration.

.. testcode::

    x = np.ones((3, 3), dtype='f')
    with chainer.using_config('use_ideep', 'auto'):
        y = chainer.functions.relu(x)
    print(type(y.data))

.. testoutput::

    <class 'ideep4py.mdarray'>

Convert Your Model to iDeep
~~~~~~~~~~~~~~~~~~~~~~~~~~~

You need to call ``model.to_intel64()`` (in the same way you call ``model.to_gpu()`` to transfer your link to GPU) to convert the link to iDeep.

Run Your Model
~~~~~~~~~~~~~~

Now your model is accelerated by iDeep!

Please note that not all functions and optimizers support iDeep acceleration.
Also note that iDeep will not be used depending on the shape and data type of the input data.

My training process gets stuck when using MultiprocessIterator
--------------------------------------------------------------

When you are using OpenCV somewhere in your code and the :class:`~chainer.iterators.MultiprocessIterator` is used in the
training code, the training loop may get stuck at some point. In such situation, there are several workarounds to
prevent the process got stuck.

1. Set the environment variable as follows: ``OMP_NUM_THREADS=1``
2. Add ``cv2.setNumThreads(0)`` right after ``import cv2`` in your training script.
3. Use :class:`~chainer.iterators.MultithreadIterator` instead of :class:`~chainer.iterators.MultiprocessIterator`.

This problem is originally reported here: `A training loop got stuck in a certain condition with multi-processing updater and opencv <https://github.com/chainer/chainer/issues/2903>`_
for Chainer and the discussion on related problems is still going here: `OpenCV + Python multiprocessing breaks on OSX <https://github.com/opencv/opencv/issues/5150>`_.
