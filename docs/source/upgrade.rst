.. currentmodule:: chainer

=============
Upgrade Guide
=============

This is a list of changes introduced in each release that users should be aware of when migrating from older versions.
Most changes are carefully designed not to break existing code; however changes that may possibly break them are highlighted with a box.

Chainer v6
==========

CuPy Needs To Be Manually Updated
---------------------------------

Prior to Chainer v6, CuPy is automatically updated to the appropriate version when updating Chainer (i.e., ``pip install -U chainer`` updates CuPy package).
In Chainer v6, Chainer does not perform this automatic update.
You need to manually update CuPy package when updating Chainer package.

This is because the automatic update made users difficult to switch between CuPy packages (e.g. ``cupy-cuda90`` and ``cupy-cuda92`` etc).
See `#5425 <https://github.com/chainer/chainer/pull/5425>`__ for details.

CuPy v6
-------

Chainer v6 requires CuPy v6 if you need GPU support.
Please see the `Upgrade Guide for CuPy v6 <https://docs-cupy.chainer.org/en/latest/upgrade.html#cupy-v6>`_ for details.


Chainer v5
==========

ChainerMN Became Part of Chainer
--------------------------------

ChainerMN, which enables multi-node distributed deep learning using Chainer, has been merged to Chainer v5.

Prior to Chainer v4, ChainerMN was provided as a separate ``chainermn`` package.
In Chainer v5, ChainerMN now became a part of Chainer; ChainerMN will be installed just by installing ``chainer`` package.
If you are using ``chainermn`` package, make sure to remove it by ``pip uninstall chainermn`` before upgrading to Chainer v5 or later.

For documentation of ChainerMN, see :doc:`chainermn/index`.

Use ``forward`` Instead of ``__call__`` in Links
------------------------------------------------

Prior to Chainer v5, ``__call__`` method is used to define the behavior of :class:`~chainer.Link`.
In Chainer v5, ``forward`` method has been introduced, and is now recommended to use it instead of ``__call__``.
The base class (:class:`~chainer.Link`) provides ``__call__`` method implementation that invokes ``forward`` method defined in the subclass; the only thing you need to do is to rename the method name (replace ``def __call__(...)`` with ``def forward(...)``).

For backward compatibility, you can still use ``__call__`` to define your own link.
However, new features introduced in Chainer v5 (e.g., :class:`~chainer.LinkHook`) may not be available for such links.

FunctionNode Classes are Hidden from ``chainer.functions``
----------------------------------------------------------

Prior to Chainer v5, :class:`~chainer.FunctionNode` classes (e.g., ``chainer.functions.MaxPooling2D``) are exposed under :mod:`chainer.functions`.
In Chainer v5, these classes are hidden from :mod:`chainer.functions`.
Use the equivalent wrapper functions listed in :doc:`reference/functions` (e.g., :func:`chainer.functions.max_pooling_2d`) instead.

Some wrapper functions now provide options to access internal states to avoid directly using :class:`~chainer.FunctionNode` classes.

* :func:`chainer.functions.max_pooling_2d`: ``return_indices``
* :func:`chainer.functions.max_pooling_nd`: ``return_indices``
* :func:`chainer.functions.dropout`: ``mask``, ``return_mask``
* :func:`chainer.functions.gaussian`: ``eps``, ``return_eps``

For example, suppose your existing code needs to access ``MaxPooling2D.indexes`` to later perform upsampling::

    p = F.MaxPooling2D(2, 2)
    h = p.apply((x,))[0]
    ...
    y = F.upsampling_2d(h, p.indexes, ksize=2)

The above code may raise this error in Chainer v5::

    AttributeError: module 'chainer.functions' has no attribute 'MaxPooling2D'

You can rewrite the above code using ``return_indices`` option of :func:`chainer.functions.max_pooling_2d`::

    h, indices = F.max_pooling_2d(x, 2, 2, return_indices=True)
    ...
    y = F.upsampling_2d(h, indices, ksize=2)

Persistent Values are Copied in ``Link.copyparams``
---------------------------------------------------

:meth:`chainer.Link.copyparams` is a method to copy all parameters of the link to another link.
This method can be used, for example, to copy parameters between two chains that partially share the same network structure to reuse pretrained weights.

Prior to Chainer v5, only parameters are copied between links.
In Chainer v5, in addition to parameters, persistent values (see :doc:`guides/serializers` for details) are also copied between links.
This is especially beneficial when copying parameters of :class:`~chainer.links.BatchNormalization`, as it uses persistent values to record running statistics.

You can skip copying persistent values by passing newly introduced ``copy_persistent=False`` option to :meth:`~chainer.Link.copyparams` so that it behaves as in Chainer v4.

Updaters Automatically Call ``Optimizer.new_epoch``
---------------------------------------------------

This change should affect only a minority of users (who call :meth:`~chainer.Optimizer.new_epoch` while using a trainer, or who implement their own :class:`~chainer.training.Updater` class).

Optimizers provide :meth:`~chainer.Optimizer.new_epoch` method, which can be used to change the behavior of optimizers depending on the current epoch number.
Prior to Chainer v5, this method was expected to be called by users.
In Chainer v5, updaters have been changed to call :meth:`~chainer.Optimizer.new_epoch` automatically.
If you have been calling :meth:`~chainer.Optimizer.new_epoch` method manually while using a trainer (or an updater), you may need any of the following fixes:

* Pass ``auto_new_epoch=False`` to the constructor of the updater (e.g., :class:`~chainer.training.updaters.StandardUpdater`) to stop :meth:`~chainer.Optimizer.new_epoch` from being called automatically by the updater.
* Avoid calling :meth:`~chainer.Optimizer.new_epoch` method manually.

If you implement your own :class:`~chainer.training.Updater` class, you may need to update your code to automatically call :meth:`~chainer.Optimizer.new_epoch` (you can refer to the changes introduced in `#4608 <https://github.com/chainer/chainer/pull/4608>`__ to understand how to fix your updater).

Extending the Backend Namespace
-------------------------------

In addition to ``chainer.backends``, we introduced ``chainer.backend``. This subpackage contains utility functions that span several backends. For instance, it includes ``chainer.backend.get_array_module`` which used to be defined in ``chainer.backends.cuda.get_array_module``. Both can be used but the latter will be deprecated.

``get_device_from_array`` Returns Actual Device for Empty Arrays
----------------------------------------------------------------

Prior to Chainer v5, :func:`chainer.backends.cuda.get_device_from_array` returned :class:`chainer.backends.cuda.DummyDeviceType` if the array is empty.
In Chainer v5, it has been changed to return the actual :class:`cupy.cuda.Device` object::

    >>> x = cupy.array([])
    >>> chainer.backends.cuda.get_device_from_array(x)
    <CUDA Device 0>

Update of Docker Images
-----------------------

Chainer official Docker images (see :doc:`install` for details) are now updated to use CUDA 9.2 and cuDNN 7.

To use these images, you may need to upgrade the NVIDIA driver on your host.
See `Requirements of nvidia-docker <https://github.com/NVIDIA/nvidia-docker/wiki/CUDA#requirements>`_ for details.

CuPy v5
-------

Chainer v5 requires CuPy v5 if you need GPU support.
Please see the `Upgrade Guide for CuPy v5 <https://docs-cupy.chainer.org/en/latest/upgrade.html#cupy-v5>`_ for details.


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

Namespace Changes for Optimizer Hooks
-------------------------------------

:doc:`Optimizer hook functions <reference/optimizers>` are moved from ``chainer.optimizer.*`` to ``chainer.optimizer_hooks.*``.
For example, ``chainer.optimizer.WeightDecay`` is now located :class:`chainer.optimizer_hooks.WeightDecay`.

If the existing code is using hooks directly under ``chainer.optimizer``, ``DeprecationWarning`` will be shown.
You are now encouraged to use ``chainer.optimizer_hooks`` instead.

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
In v4, the error message would become ``TypeError: incompatible array types are mixed in the forward input (Maximum)``.
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

CuPy v4
-------

Chainer v4 requires CuPy v4 if you need GPU support.
Please see the `Upgrade Guide for CuPy v4 <https://docs-cupy.chainer.org/en/latest/upgrade.html#cupy-v4>`_ for details.


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

CuPy v2
-------

Chainer v3 requires CuPy v2 if you need GPU support.
Please see the `Upgrade Guide for CuPy v2 <https://docs-cupy.chainer.org/en/latest/upgrade.html#cupy-v2>`_ for details.


Chainer v2
==========

See :doc:`upgrade_v2` for the changes introduced in Chainer v2.

.. toctree::
   :maxdepth: 1
   :hidden:

   upgrade_v2
