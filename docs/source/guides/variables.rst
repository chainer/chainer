Variables and Derivatives
~~~~~~~~~~~~~~~~~~~~~~~~~

.. include:: ../imports.rst

As described previously, Chainer uses the "Define-by-Run" scheme, so forward computation itself *defines* the network.
In order to start forward computation, we have to set the input array to a :class:`Variable` object.
Here we start with a simple :class:`~numpy.ndarray` with only one element:

.. doctest::

   >>> x_data = np.array([5], dtype=np.float32)
   >>> x = Variable(x_data)

A Variable object has basic arithmetic operators.
In order to compute :math:`y = x^2 - 2x + 1`, just write:

.. doctest::

   >>> y = x**2 - 2 * x + 1

The resulting ``y`` is also a Variable object, whose value can be extracted by accessing the :attr:`~Variable.data` attribute:

.. doctest::

   >>> y.data
   array([16.], dtype=float32)

What ``y`` holds is not only the result value.
It also holds the history of computation (or computational graph), which enables us to compute its derivative.
This is done by calling its :meth:`~Variable.backward` method:

.. doctest::

   >>> y.backward()

This runs *error backpropagation* (a.k.a. *backprop* or *reverse-mode automatic differentiation*).
Then, the gradient is computed and stored in the :attr:`~Variable.grad` attribute of the input variable ``x``:

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
This is done simply by setting the :attr:`~Variable.grad` attribute of the output variable:

.. doctest::

   >>> x = Variable(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
   >>> y = x**2 - 2*x + 1
   >>> y.grad = np.ones((2, 3), dtype=np.float32)
   >>> y.backward()
   >>> x.grad
   array([[ 0.,  2.,  4.],
          [ 6.,  8., 10.]], dtype=float32)

.. note::

   Many functions taking :class:`Variable` object(s) are defined in the :mod:`~chainer.functions` module.
   You can combine them to realize complicated functions with automatic backward computation.

