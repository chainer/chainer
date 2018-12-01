ChainerX Documentation
======================

.. Warning::

   This feature is still in the earliest stage of its development. The behavior and interface are subject to change.

ChainerX is an ndarray implementation with Define-by-Run automatic differentiation capability.
It roughly corresponds to "NumPy/CuPy + Chainer Variable", while some additional features follow:

- **Speed**: The whole ndarray and autograd implementation is written in C++, with a thin Python binding. It lowers the overhead existing in the pure Python implementation of Chainer.
- **Extensibility**: The backend is pluggable so that it is much easier to add a support of new devices.

The speed is best achieved by directly using ChainerX APIs,
while it also provides a compatibility layer through the conventional :class:`chainer.Variable` interface for easier adoption of ChainerX in existing projects.
See :ref:`chainerx_tutorial` for more details.


.. toctree::
   :maxdepth: 2

   install/index
   tutorial/index
   limitations
   reference/index
   contribution
