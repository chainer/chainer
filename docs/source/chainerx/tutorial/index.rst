.. _chainerx_tutorial:

ChainerX Tutorial
=================

ChainerX, or :data:`chainerx`, is meant to be a drop-in replacement for `NumPy <https://docs.scipy.org/doc/>`_ and `CuPy <https://docs-cupy.chainer.org/en/stable/>`_, with additional operations specific to neural networks.
As its core is implemented in C++, you can reduce the Python overhead for both the forward and backward passes compared to Chainer, speeding up your training and inference.
This section will guide you through the essential APIs of Chainer to utilize ChainerX, but also how to use ChainerX on its own.

Introduction to ChainerX
------------------------

The module :data:`chainerx` aims to support a NumPy compatible interface with additional operations specific to neural networks.
It for instance provides :func:`chainerx.conv` for N-dimensional convolutions and :func:`chainerx.batch_norm` for batch normalization.
Additionally, and most importantly, the array in ChainerX :class:`chainerx.ndarray`, distinguishes itself from NumPy and CuPy arrays in the following two aspects.

Automatic differentiation
    Graph construction and backpropagation is built into the array, meaning that any function, including the NumPy-like functions, can be backpropagated through.
    In Chainer terms, it is a NumPy/CuPy array with :class:`chainer.Variable` properties.

Device agnostic
    Arrays can be allocated on any device belonging to any backend, in contrast to NumPy/CuPy arrays which are implemented for specific computing platforms (i.e. CPUs/GPUs respectively).

These differences are explained more in details by the sections further down.

The array :class:`chainerx.ndarray`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following example demonstrates how you can create an array and access its most basic attributes.
Note that the APIs are identical to that of NumPy and CuPy.
Other array creation routines including :func:`chainerx.ones`, :func:`chainerx.ones_like` and :func:`chainerx.random.normal` are all listed in :ref:`here <chainerx_routines>`.

.. code-block:: python

    import chainerx as chx

    x = chx.array([[0, 1, 2], [3, 4, 5]], dtype=chx.float32)

    x.shape  # (2, 3)
    x.dtype  # dtype('float32')
    x.size  # 6
    x.ndim  # 2

Backends and devices
""""""""""""""""""""

Chainer distinguishes between CPU and GPU arrays using NumPy and CuPy but ChainerX arrays may be allocated on any device on any backend.
You can specify the device during instantiation or transfer the array to a different device after it has been created.

.. code-block:: python

    x = chx.array([1, 2, 3])
    x.device  # native:0

    x = chx.array([1, 2, 3], device='cuda:0')
    x.device  # cuda:0

    x = x.to_device('cuda:1')
    x.device  # cuda:1

The left-hand-side of the colon shows the name of the backend to which the device belongs.
``native`` in this case refers to the CPU and ``cuda`` to CUDA GPUs.
The integer on the right-hand-side shows the device index.
Together, they uniquely identify a physical device on which an array is allocated.

If you do not want to specify the device each time you create an array, it is possible to change the default device with :func:`chainerx.using_device`.

.. code-block:: python

    with chx.using_device('cuda:0')
        x = chx.array([1, 2, 3])
    x.device  # cuda:0

.. note::

    Currently, two backends are built into ChainerX.

    1. The ``native`` backend, which is built by default.
    2. The ``cuda`` backend which is optional (See :ref:`installation <chainerx_install>`).

    This backend abstraction allows developers to implement their own backends and plug them into ChainerX to perform computations on basically any other platform.

Array operations and backpropagation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Arrays support basic arithmetics and can be passed to functions just as you would expect.
By marking an array to require gradients with :meth:`chainerx.ndarray.require_grad`, further computations involving that array will construct a computational graph allowing backpropagation directly from the array.
The following code shows how you could implement an affine transformation and backpropgate through it to compute the gradient of the output w.r.t. the input weight and bias.

.. code-block:: python

    x = chx.ones(784, dtype=chx.float32)
    W = chx.random.normal(size=(784, 1000)).astype(chx.float32).require_grad()
    b = chx.random.normal(size=(1000)).astype(chx.float32).require_grad()

    y = x.dot(W) + b

    y.grad = chx.ones_like(y)  # Initial upstream gradients, i.e. `grad_outputs`.
    y.backward()

    assert type(W.grad) is chx.ndarray
    assert type(b.grad) is chx.ndarray

.. note::

    The code above is device agnostic, meaning that you can execute it on any backend by simply wrapping the code with a :func:`chainerx.using_device`.

Relation to Chainer
-------------------

A :class:`chainerx.ndarray` can be wrapped in a :class:`chainer.Variable` and passed to any existing Chainer code.

.. code-block:: python

    var = ch.Variable(x)  # x is a chainerx.ndarray.

    # Your Chainer code...

When further applying functions to the ``var``, the computational graph is recorded in the underlying ndarray in C++ implementation, not in the :class:`chainer.Variable` or the :class:`chainer.FunctionNode`, as in the conventional Chainer.
This eliminates the heavy Python overhead of the graph construction.
Similarly, calling :meth:`chainer.Variable.backward` on any resulting variable will delegate the work to C++ by calling :meth:`chainerx.ndarray.backward` spending no time in the Python world.

.. _chainerx-tutorial-numpy-cupy-fallback:

NumPy/CuPy fallback
^^^^^^^^^^^^^^^^^^^

As the features above require ChainerX to provide an implementation corresponding to every :class:`chainer.FunctionNode` implementation in Chainer, ChainerX utilizes a fallback mechanism while gradually extending the support.
This approach is taken because the integration with Chainer takes time and we do not want existing Chainer users to have to make severe changes to their code bases in order to try ChainerX.
The fallback logic simply casts the :class:`chainerx.ndarray`\ s inside the :class:`chainer.Variable` to :class:`numpy.ndarray`\ s or :class:`cupy.ndarray`\ s (without copy) and calls the forward and backward methods respectively.

Run your Chainer code with ChainerX
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to utilize :data:`chainerx`, you first need to transfer your model to a ChainerX device using :meth:`chainer.Link.to_device`.
This is a new method that has been introduced to replace :meth:`chainer.Link.to_cpu` and :meth:`chainer.Link.to_gpu`, extending device transfer to arbitrary devices.
Similarly, you have to transfer the data (:class:`chainer.Variable`\ s) to the same device before feeding them to the model.

Will my FunctionNode work with ChainerX?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Our expectation is that it should work because of the fallback mechanism explained above, but in practice you may need some occasional fixes, depending on how the function was implemented.
Also, you will not see any performance improvements from the fallback (but most likely a degradation because of the additional conversions).

To support ChainerX with your :class:`chainer.FunctionNode`, you need to implement :meth:`chainer.FunctionNode.forward_chainerx` with the same signature as :meth:`chainer.FunctionNode.forward`, but where given inputs are of type :class:`chainerx.ndarray`.
It is expected to return a ``tuple`` just like :meth:`chainer.FunctionNode.forward`.

The example below shows how :func:`chainer.functions.matmul` is extended to support ChainerX. Note that :class:`chainer.Fallback` can be returned in case the function cannot be implemented using ChainerX functions.
This is also the default behavior in case the method is not implemented at all.

.. code-block:: python

    class MatMul(function_node.FunctionNode):

        def forward_chainerx(self, x):
            a, b = x
            if self.transa or self.transb or self.transc:
                return chainer.Fallback
            if a.dtype != b.dtype:
                return chainer.Fallback
            if a.ndim != 2 or b.ndim != 2:
                return chainer.Fallback
            if self.dtype is not None and self.dtype != a.dtype:
                return chainer.Fallback
            return chainerx.dot(a, b),  # Fast C++ implementation
