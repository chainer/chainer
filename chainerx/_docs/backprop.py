import chainerx
from chainerx import _docs


def set_docs():
    _docs.set_doc(
        chainerx.no_backprop_mode,
        """no_backprop_mode()
Creates a context manager which temporarily disables backpropagation.

Within this context, no computational graph will be formed unless
:meth:`~chainerx.force_backprop_mode` is used.

Arrays resulted from operations enclosed with the context will be disconnected
from the computational graph. Trying to perform backpropagation from such
arrays would result in an error.

.. code-block:: py

    x = chainerx.array([4, 3], numpy.float32)
    x.require_grad()

    with chainerx.no_backprop_mode():
        y = 2 * x + 1

    y.backward()  # ! error

Benefits of ``no_backprop_mode`` include reduced CPU overhead for building
computational graphs, and reduced consumption of device memory that
would be otherwise retained for backward propagation.

.. seealso::
    * :func:`chainerx.force_backprop_mode`
    * :func:`chainer.no_backprop_mode`
""")

    _docs.set_doc(
        chainerx.force_backprop_mode,
        """force_backprop_mode()
Creates a context manager which temporarily enables backpropagation.

This context re-enables backpropagation which is disabled by
surrounding :func:`~chainerx.no_backprop_mode` context.

.. code-block:: py

    x = chainerx.array([4, 3], numpy.float32)
    x.require_grad()

    with chainerx.no_backprop_mode():
        with chainerx.force_backprop_mode():
            y = 2 * x + 1

    y.backward()
    x.grad
    # array([2., 2.], shape=(2,), dtype=float32, device='native:0')

.. seealso::
    * :func:`chainerx.no_backprop_mode`
    * :func:`chainer.force_backprop_mode`
""")
