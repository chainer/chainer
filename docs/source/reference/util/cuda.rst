CUDA and Backend Utilities
==========================

Utilities across backends
-------------------------

.. module:: chainer.backend
.. currentmodule:: /

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.backend.copyto

CUDA
----

.. automodule:: chainer.backends.cuda

.. currentmodule:: /

Devices
~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.backends.cuda.get_device
   chainer.backends.cuda.get_device_from_id
   chainer.backends.cuda.get_device_from_array

CuPy array allocation and copy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.backends.cuda.copy
   chainer.backends.cuda.to_cpu
   chainer.backends.cuda.to_gpu

Kernel definition utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.backends.cuda.memoize
   chainer.backends.cuda.clear_memo
   chainer.backends.cuda.elementwise
   chainer.backends.cuda.raw
   chainer.backends.cuda.reduce

CPU/GPU generic code support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.backends.cuda.get_array_module

cuDNN support
~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.backends.cuda.set_max_workspace_size
   chainer.backends.cuda.get_max_workspace_size

iDeep
-----

`iDeep <https://github.com/intel/ideep>`__ is a module that provides NumPy-like API and DNN acceleration using MKL-DNN for Intel CPUs.
See :doc:`../../tips` and :doc:`../../performance` for details.

.. module:: chainer.backends.intel64
.. currentmodule:: chainer

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.backends.intel64.is_ideep_available
