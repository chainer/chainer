Links
~~~~~

In order to write neural networks, we have to combine functions with *parameters* and optimize the parameters.
You can use the class :class:`Link` to do this.
A :class:`Link` is an object that holds parameters (i.e. optimization targets).

The most fundamental ones are links that behave like regular functions while replacing some arguments by their parameters.
We will introduce higher level links, but here think of links as simply functions with parameters.

One of the most frequently used links is the :class:`~functions.connection.linear.Linear` link (a.k.a. *fully-connected layer* or *affine transformation*).
It represents a mathematical function :math:`f(x) = Wx + b`, where the matrix :math:`W` and the vector :math:`b` are parameters.
This link corresponds to its pure counterpart :func:`~functions.linear`, which accepts :math:`x, W, b` as arguments.
A linear link from three-dimensional space to two-dimensional space is defined by the following line:

.. doctest::

   >>> f = L.Linear(3, 2)

.. note::

   Most functions and links only accept mini-batch input, where the first dimension of the input array is considered as the *batch dimension*.
   In the above Linear link case, input must have shape of :math:`(N, 3)`, where :math:`N` is the mini-batch size.

The parameters of a link are stored as attributes.
Each parameter is an instance of :class:`~chainer.Variable`.
In the case of the Linear link, two parameters, ``W`` and ``b``, are stored.
By default, the matrix ``W`` is initialized randomly, while the vector ``b`` is initialized with zeros.
This is the preferred way to initialize these parameters.

.. doctest::

   >>> f.W.data
   array([[ 1.0184761 ,  0.23103087,  0.5650746 ],
          [ 1.2937803 ,  1.0782351 , -0.56423163]], dtype=float32)
   >>> f.b.data
   array([0., 0.], dtype=float32)

An instance of the Linear link acts like a usual function:

.. doctest::

   >>> x = Variable(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
   >>> y = f(x)
   >>> y.data
   array([[3.1757617, 1.7575557],
          [8.619507 , 7.1809077]], dtype=float32)

.. note::

  Sometimes it is cumbersome to compute the dimension of the input space.
  The linear link and some of (de)convolution links can omit the input dimension
  in their instantiation and infer it from the first mini-batch.

  For example, the following line creates a linear link whose output dimension
  is two::

  >>> f = L.Linear(2)

  If we feed a mini-batch of shape :math:`(2, M)`, the input dimension will be inferred as ``M``,
  which means ``l.W`` will be a 2 x M matrix.
  Note that its parameters are initialized in a lazy manner at the first mini-batch.
  Therefore, ``l`` does not have ``W`` attribute if no data is put to the link.

Gradients of parameters are computed by the :meth:`~Variable.backward` method.
Note that gradients are **accumulated** by the method rather than overwritten.
So first you must clear the gradients to renew the computation.
It can be done by calling the :meth:`~Link.cleargrads` method.

.. doctest::

   >>> f.cleargrads()

Now we can compute the gradients of parameters by simply calling the backward method and access them via the ``grad`` property.

.. doctest::

   >>> y.grad = np.ones((2, 2), dtype=np.float32)
   >>> y.backward()
   >>> f.W.grad
   array([[5., 7., 9.],
          [5., 7., 9.]], dtype=float32)
   >>> f.b.grad
   array([2., 2.], dtype=float32)

