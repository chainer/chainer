.. currentmodule:: chainer

=============
Upgrade Guide
=============

This is a list of changes introduced in each release that users should be aware of when migrating from older versions.
Most changes are carefully designed not to break existing code; however changes that may possibly break them are highlighted with a box.


Chainer v4
==========

Introduction of Backend Namespace
---------------------------------

We introduced ``chainer.backends`` subpackage for future support of various backend libraries other than NumPy and CuPy.
By this change, ``chainer.cuda`` module is now moved to ``chainer.backends.cuda``.

This does not break the existing code; you can safely continue to use ``chainer.cuda`` (e.g., ``from chainer import cuda``) but it is now encouraged to use ``from chainer.backends import cuda`` instead.

Namespace Changes for Updaters
------------------------------

:class:`chainer.training.StandardUpdater` and :class:`chainer.training.ParallelUpdater` are now moved to :class:`chainer.training.updaters.StandardUpdater` and :class:`chainer.training.updaters.ParallelUpdater` respectively, to align with the namespace convention of other subpackages.
See the discussion in `#2982 <https://github.com/chainer/chainer/pull/2982>`_ for more details.

This change does not break the existing code; you can safely continue to use updater classes directly under ``chainer.training`` but it is now encouraged to use ``chainer.training.updaters`` instead.

Prohibition of Mixed Use of Arrays on Different Devices in Function Arguments
-----------------------------------------------------------------------------

Argument validation of functions is now strictened to check device consistency of argument variables to provide better error messages to users.
Suppose the following code:

.. code-block:: py

   v1 = chainer.Variable(np.arange(10, dtype=np.float32))      # CPU
   v2 = chainer.Variable(cupy.arange(10, dtype=cupy.float32))  # GPU

   # The line below raises an exception, because arguments are on different device.
   F.maximum(v1, v2)

Prior to v4, the above code raises an exception like ``ValueError: object __array__ method not producing an array``, which was difficult to understand.
In v4, the error message would become ``ValueError: incompatible array types are mixed in the forward input (Maximum)``.
This kind of error usually occurs by mistake (for example, not performing ``to_gpu`` for some variables).

.. attention::

   As the argument validation is strictened, call of functions intentionally mixing NumPy/CuPy arrays in arguments will not work in Chainer v4.
   Please transfer all arrays to the same device before calling functions.

References to Function Nodes Not Retained in TimerHook and CupyMemoryProfilerHook
---------------------------------------------------------------------------------

To reduce memory consumption, references to the function nodes will no longer be retained in the :class:`chainer.function_hooks.CupyMemoryProfileHook` and :class:`chainer.function_hooks.TimerHook`.
See the discussion in `#4300 <https://github.com/chainer/chainer/pull/4300>`_ for more details.

.. attention::

   The existing code using function nodes retained in ``call_history`` attribute of these hooks will not work.
   The first element of ``call_history`` became the name of the function, instead of the function node instance itself.
   You can define your own function hook if you need to access the function node instances.

Update of Docker Images
-----------------------

Chainer official Docker images (see :doc:`install` for details) are now updated to use CUDA 8.0 and cuDNN 6.0.
This change was introduced because CUDA 7.5 does not support NVIDIA Pascal GPUs.

To use these images, you may need to upgrade the NVIDIA driver on your host.
See `Requirements of nvidia-docker <https://github.com/NVIDIA/nvidia-docker/wiki/CUDA#requirements>`_ for details.

Chainer v3
==========

Introduction of New-style Functions
-----------------------------------

This release introduces new-style functions (classes inheriting from :class:`FunctionNode`) that support double backward (gradient of gradient).
See the `Release Note for v3.0.0 <https://github.com/chainer/chainer/releases/tag/v3.0.0>`_ for the usage of this feature.

Many of :doc:`reference/functions` are already migrated to new-style, although some of functions are still old-style (classes inheriting from :class:`Function`).
We are going to migrate more old-style functions to new-style in upcoming minor releases.

This does not break the existing code.
Old-style functions (classes inheriting from :class:`Function`) are still supported in v3 and future versions of Chainer.

If you are going to write new functions, it is encouraged to use :class:`FunctionNode` to support double backward.

.. attention::

   Users relying on undocumented function APIs (directly instantiating old-style classes) may experience an error like ``TypeError: 'SomeFunction' object is not callable`` after upgrading to v3.
   Please use the function APIs documented in :doc:`reference/functions`.

Changed Behavior of matmul Function
-----------------------------------

The behavior of :func:`chainer.functions.matmul` has been changed to behave like the corresponding NumPy function (:func:`numpy.matmul`).
See the discussion in `#2426 <https://github.com/chainer/chainer/pull/2426>`_ for more details.

.. attention::

   The existing code using :func:`chainer.functions.matmul` may require modification to work with Chainer v3.

Also note that :func:`chainer.functions.batch_matmul` is now deprecated by this change.
You can rewrite it using :func:`chainer.functions.matmul`.

Removed use_cudnn Argument in spatial_transformer_grid and spatial_transformer_sampler Functions
------------------------------------------------------------------------------------------------

``use_cudnn`` argument has been removed from :func:`chainer.functions.spatial_transformer_grid` and :func:`chainer.functions.spatial_transformer_sampler`.
See the discussion in `#2955 <https://github.com/chainer/chainer/pull/2955>`_ for more details.

.. attention::

   The existing code using ``use_cudnn`` argument of :func:`chainer.functions.spatial_transformer_grid` and :func:`chainer.functions.spatial_transformer_sampler` require modification to work with Chainer v3.
   Please use the configuration context (e.g., ``with chainer.using_config('use_cudnn', 'auto'):``) to enable or disable use of cuDNN.
   See :ref:`configuration` for details.


Chainer v2
==========

See :doc:`upgrade_v2` for the changes introduced in Chainer v2.

.. toctree::
   :maxdepth: 1
   :hidden:

   upgrade_v2
