Variables and Derivatives
=========================

.. include:: ../imports.rst

As described previously, Chainer uses the "Define-by-Run" scheme, so forward computation itself *defines* the network.
In order to start forward computation, we have to set the input array to a :class:`chainer.Variable` object.
Here we start with a simple :class:`~numpy.ndarray` with only one element:

.. doctest::

   >>> x_data = np.array([5], dtype=np.float32)
   >>> x = Variable(x_data)

A Variable object supports basic arithmetic operators.
In order to compute :math:`y = x^2 - 2x + 1`, just write:

.. doctest::

   >>> y = x**2 - 2 * x + 1

The resulting ``y`` is also a Variable object, whose value can be extracted by accessing the :attr:`~chainer.Variable.array` attribute:

.. doctest::

   >>> y.array
   array([16.], dtype=float32)

.. note::

   :attr:`~chainer.Variable` has two attributes to represent the underlying array: :attr:`~chainer.Variable.array` and :attr:`~chainer.Variable.data`.
   There is no difference between the two; both refer to exactly the same object.
   However it is not recommended to use ``.data`` because it might be confused with :attr:`numpy.ndarray.data` attribute.

What ``y`` holds is not only the result value.
It also holds the history of computation (or computational graph), which enables us to compute its derivative.
This is done by calling its :meth:`~Variable.backward` method:

.. doctest::

   >>> y.backward()

This runs *error backpropagation* (a.k.a. *backprop* or *reverse-mode automatic differentiation*).
Then, the gradient is computed and stored in the :attr:`~chainer.Variable.grad` attribute of the input variable ``x``:

.. doctest::

   >>> x.grad
   array([8.], dtype=float32)

Also we can compute gradients of intermediate variables.
Note that Chainer, by default, releases the gradient arrays of intermediate variables for memory efficiency.
In order to preserve gradient information, pass the ``retain_grad`` argument to the backward method:

.. doctest::

   >>> z = 2*x
   >>> y = x**2 - z + 1
   >>> y.backward(retain_grad=True)
   >>> z.grad
   array([-1.], dtype=float32)

All these computations are can be generalized to a multi-element array input.
While single-element arrays are automatically initialized to ``[1]``, to start backward computation from a variable holding a multi-element array, we must set the *initial error* manually.
This is done simply by setting the :attr:`~chainer.Variable.grad` attribute of the output variable:

.. doctest::

   >>> x = Variable(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
   >>> y = x**2 - 2*x + 1
   >>> y.grad = np.ones((2, 3), dtype=np.float32)
   >>> y.backward()
   >>> x.grad
   array([[ 0.,  2.,  4.],
          [ 6.,  8., 10.]], dtype=float32)

.. note::

   Many functions taking :class:`~chainer.Variable` object(s) are defined in the :mod:`chainer.functions` module.
   You can combine them to realize complicated functions with automatic backward computation.

.. note::

   Instead of using :func:`~chainer.Variable.backward`, you can also calculate gradients of any variables in a computational graph w.r.t. any other variables in the graph using the :func:`chainer.grad` function.


Higher-Order Derivatives
------------------------

:class:`~chainer.Variable` also supports higher-order derivatives (a.k.a. double backpropagation).

Let's see a simple example.
First calculate the first-order derivative.
Note that ``enable_double_backprop=True`` is passed to ``y.backward()``.

.. doctest::

    >>> x = chainer.Variable(np.array([[0, 2, 3], [4, 5, 6]], dtype=np.float32))
    >>> y = x ** 3
    >>> y.grad = np.ones((2, 3), dtype=np.float32)
    >>> y.backward(enable_double_backprop=True)
    >>> x.grad_var
    variable([[  0.,  12.,  27.],
              [ 48.,  75., 108.]])
    >>> assert x.grad_var.array is x.grad
    >>> assert (x.grad == (3 * x**2).array).all()

:attr:`chainer.Variable.grad_var` is a :class:`~chainer.Variable` for :attr:`chainer.Variable.grad` (which is an :class:`~numpy.ndarray`).
By passing ``enable_double_backprop=True`` to ``backward()``, a computational graph for the backward calculation is recorded.
So, you can start backpropagation from ``x.grad_var`` to calculate the second-order derivative.

.. doctest::

    >>> gx = x.grad_var
    >>> x.cleargrad()
    >>> gx.grad = np.ones((2, 3), dtype=np.float32)
    >>> gx.backward()
    >>> x.grad
    array([[ 0., 12., 18.],
           [24., 30., 36.]], dtype=float32)
    >>> assert (x.grad == (6 * x).array).all()
