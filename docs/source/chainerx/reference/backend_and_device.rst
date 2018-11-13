Backend and Device
==================

.. module:: chainerx

ChainerX adds a level of abstraction between the higher level array operations and the lower level computes and resource management.
This abstraction is managed by the :class:`~chainerx.Backend` and the :class:`~chainerx.Device` classes.
Native (CPU) and CUDA backends are two concrete implementations and are both a part of ChainerX but the abstraction allows you to plug any backend into the framework.

Backend
-------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainerx.Backend
   chainerx.get_backend

Device
------
.. _chainerx_device:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainerx.Device
   chainerx.get_device
   chainerx.get_default_device
   chainerx.set_default_device
   chainerx.device_scope

