CUDA utilities
--------------
.. automodule:: chainer.cuda

.. currentmodule:: /

Devices
~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.cuda.get_device
   chainer.cuda.get_device_from_id
   chainer.cuda.get_device_from_array

CuPy array allocation and copy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.cuda.copy
   chainer.cuda.to_cpu
   chainer.cuda.to_gpu

Kernel definition utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.cuda.memoize
   chainer.cuda.clear_memo
   chainer.cuda.elementwise
   chainer.cuda.reduce

CPU/GPU generic code support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.cuda.get_array_module

cuDNN support
~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.cuda.set_max_workspace_size
   chainer.cuda.get_max_workspace_size
