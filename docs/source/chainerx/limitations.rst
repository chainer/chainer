Limitations
===========

There are some non-obvious limitations in ChainerX:

* ChainerX only supports a limited set of dtypes:
  :data:`~chainerx.bool_` :data:`~chainerx.int8` :data:`~chainerx.int16` :data:`~chainerx.int32` :data:`~chainerx.int64` :data:`~chainerx.uint8` :data:`~chainerx.float32` :data:`~chainerx.float64`.

* Operations with mixed dtypes are not supported. You need to explicitly convert dtypes using either :func:`chainerx.astype` or :func:`F.cast() <chainer.functions.cast>`.

* Only a limited set of Chainer :mod:`function <chainer.functions>`\ s are well tested with the ChainerX integration.

* ChainerX CUDA backend requires cuDNN. See :ref:`installation <chainerx_install>` for details.

* As ChainerX :class:`array <chainerx.ndarray>`\ s have a computational graph in their own, some operations are prohibited for safety:

  * Unless an array is free from the computational graph, in-place modification of its data is prohibited.

    .. code-block:: py

       a = chainerx.zeros((2,), numpy.float32)
       a.require_grad()  # install the computational graph on `a`.

       a += 1  # ! error

    The reason of this limitation is that, as backward operations may depend on the value of ``a``,
    the backward gradients might be unexpectedly affected if it would be altered.

    You may circumvent this limitation by making a disconnected view:

    .. code-block:: py

       # A memory-shared view of `a` which is disconnected from the computational graph of `a`.
       b = a.as_grad_stopped()

       b += 1

    Note however that this operation is inherently dangerous.
    You should be super careful to ensure that that does not affect backward computations.

    Note also that we may restrict further in the future so that even in-place modification on a disconnected view is only allowed if it is actually safe.


  * If an array is wrapped with a :class:`~chainer.Variable` with ``requires_grad=True`` (which is default), you won't be able to re-assign the array::

       a = chainerx.zeros((2,), numpy.float32)
       b = chainerx.zeros((2,), numpy.float32)
       var = chainer.Variable(a)

       a.array = b  # ! error

    You may circumvent this by using in-place assignment on ``a.array``::

       a.array[:] = b

    This workaround may also be dangerous just as in the previous limitation.

