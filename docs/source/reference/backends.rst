Backends and Devices
====================

.. module:: chainer.backend

Common Classes and Utilities
----------------------------

.. currentmodule:: chainer

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.backend.Device
   chainer.get_device
   chainer.using_device
   chainer.backend.get_device_from_array
   chainer.backend.get_array_module
   chainer.DeviceResident
   chainer.device_resident.DeviceResidentsVisitor
   chainer.backend.copyto


Concrete Device Classes
-----------------------

.. currentmodule:: chainer

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.backend.CpuDevice
   chainer.backend.GpuDevice
   chainer.backend.Intel64Device
   chainer.backend.ChainerxDevice

GPU (CuPy)
----------

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

Intel64 (iDeep)
---------------

`iDeep <https://github.com/intel/ideep>`__ is a module that provides NumPy-like API and DNN acceleration using MKL-DNN for Intel CPUs.
See :doc:`../../tips` and :doc:`../../performance` for details.

.. module:: chainer.backends.intel64
.. currentmodule:: chainer

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.backends.intel64.is_ideep_available
   
ChainerX
--------

.. currentmodule:: chainer

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.backend.from_chx
   chainer.backend.to_chx
