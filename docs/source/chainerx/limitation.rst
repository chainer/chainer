Limitation
==========

There are some non-obvious limitations in ChainerX:

* ChainerX only supports a limited set of dtypes:
  :data:`~chainerx.bool_` :data:`~chainerx.int8` :data:`~chainerx.int16` :data:`~chainerx.int32` :data:`~chainerx.int64` :data:`~chainerx.uint8` :data:`~chainerx.float32` :data:`~chainerx.float64`

* Operations with mixed dtypes are not supported. You need to explicitly convert dtypes using either :func:`chainerx.astype` or :func:`F.cast() <chainer.functions.cast>`.

* Only a limited set of Chainer :mod:`function <chainer.functions>`\ s are well tested with ChainerX integration.

* ChainerX CUDA backend requires cuDNN. See :ref:`installation <chainerx_install>` for details.

