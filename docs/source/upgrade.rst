.. currentmodule:: chainer

=============
Upgrade Guide
=============

This is a list of changes introduced in each release that users should be aware of when migrating from older versions.
Most changes are carefully designed not to break existing code; however changes that may possibly break them are highlighted with a box.


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
