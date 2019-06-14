import chainerx
from chainerx import _docs


def set_docs():
    _docs.set_doc(
        chainerx.backward,
        """backward(outputs, *, enable_double_backprop=False)
Runs backpropagation.

On backpropagation (a.k.a. backprop),
the computational graph is traversed backward starting from the output arrays,
up until the root arrays on which :func:`ndarray.require_grad()` have been
called.

Backpropagation uses :data:`ndarray.grad <chainerx.ndarray.grad>` held by
the output arrays as the initial gradients.
You can manually assign them before calling this function.
Otherwise, they are assumed to be 1.

To enable higher order differentiation, pass ``enable_double_backprop=True``
so that you can further run backpropagation from the resulting gradient arrays.
Note that enabling it results in larger memory consumption needed to store the
gradients w.r.t intermediate arrays that are required for the second gradient
computation.

Note:
    The whole process of backpropagation is executed in C++, except those
    operations whose backward computation falls back to the corresponding
    Python implementation. Currently this function does not release the GIL at
    all.

Args:
    outputs (~chainerx.ndarray or list of ndarrays):
        Output arrays from which backpropagation starts.
    enable_double_backprop (bool): If ``True``,
        a computational trace of the whole backpropagation procedure is
        recorded to the computational graph so that one can further do
        backpropagation from the resulting gradients.

.. seealso::
    * :meth:`chainerx.ndarray.backward`
""")

    _docs.set_doc(
        chainerx.grad,
        """grad(outputs, inputs, *, enable_double_backprop=False)
Computes and returns the gradients of the outputs w.r.t. the inputs.

This function differs from :func:`chainerx.backward` in the sense that
gradients are returned instead of being added to the gradients held by the
inputs. Gradients held by the inputs are not modified. Also, instead of
traversing through the whole graph starting from the outputs, a sub-graph is
extracted for computation. This means that is is more efficient, especially
for larger computational graphs.

Args:
    outputs (list of ndarrays):
        Output arrays from which backpropagation starts.
    inputs (list of ndarrays):
        Input arrays of which this function computes the gradients w.r.t.
    enable_double_backprop (bool): If ``True``,
        a computational trace of the whole backpropagation procedure is
        recorded to the computational graph so that one can further do
        backpropagation from the resulting gradients.

Returns:
    list of :class:`~chainerx.ndarray`\\ s:
        A list of gradients. The list always has the same length as the number
        of inputs.

.. seealso::
    * :func:`chainerx.backward`
    * :func:`chainer.grad`
""")

    _docs.set_doc(
        chainerx.no_backprop_mode,
        """no_backprop_mode()
Creates a context manager which temporarily disables backpropagation.

Within this context, no computational graph will be formed unless
:meth:`~chainerx.force_backprop_mode` is used.

Arrays resulting from operations enclosed with this context will be
disconnected from the computational graph. Trying to perform backpropagation
from such arrays would result in an error.

.. code-block:: py

    x = chainerx.array([4, 3], numpy.float32)
    x.require_grad()

    with chainerx.no_backprop_mode():
        y = 2 * x + 1

    y.backward()  # ! error

Benefits of ``no_backprop_mode`` include reduced CPU overhead of building
computational graphs, and reduced consumption of device memory that
would be otherwise retained for backward propagation.

.. seealso::
    * :func:`chainerx.force_backprop_mode`
    * :func:`chainerx.is_backprop_required`
    * :func:`chainer.no_backprop_mode`
""")

    _docs.set_doc(
        chainerx.force_backprop_mode,
        """force_backprop_mode()
Creates a context manager which temporarily enables backpropagation.

This context re-enables backpropagation that is disabled by
any surrounding :func:`~chainerx.no_backprop_mode` context.

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
    * :func:`chainerx.is_backprop_required`
    * :func:`chainer.force_backprop_mode`
""")

    _docs.set_doc(
        chainerx.is_backprop_required,
        """is_backprop_required()
Returns whether the backpropagation is enabled in the current thread.

The result is affect by :func:`chainerx.no_backprop_mode` and
:func:`chainerx.force_backprop_mode`.

.. seealso::
    * :func:`chainerx.no_backprop_mode`
    * :func:`chainerx.force_backprop_mode`
""")
