.. _creating_models:

Creating Models
~~~~~~~~~~~~~~~

.. include:: ../imports.rst

Most neural network architectures contain multiple links.
For example, a multi-layer perceptron consists of multiple linear layers.
We can write complex procedures with parameters by combining multiple links like this:

.. doctest::

   >>> l1 = L.Linear(4, 3)
   >>> l2 = L.Linear(3, 2)

   >>> def my_forward(x):
   ...     h = l1(x)
   ...     return l2(h)

Here the ``L`` indicates the :mod:`~chainer.links` module.
A procedure with parameters defined in this way is hard to reuse.
More Pythonic way is combining the links and procedures into a class:

.. doctest::

   >>> class MyProc(object):
   ...     def __init__(self):
   ...         self.l1 = L.Linear(4, 3)
   ...         self.l2 = L.Linear(3, 2)
   ...
   ...     def forward(self, x):
   ...         h = self.l1(x)
   ...         return self.l2(h)

In order to make it more reusable, we want to support parameter management, CPU/GPU migration, robust and flexible save/load features, etc.
These features are all supported by the :class:`Chain` class in Chainer.
Then, what we have to do here is just define the above class as a subclass of Chain:

.. doctest::

   >>> class MyChain(Chain):
   ...     def __init__(self):
   ...         super(MyChain, self).__init__()
   ...         with self.init_scope():
   ...             self.l1 = L.Linear(4, 3)
   ...             self.l2 = L.Linear(3, 2)
   ...
   ...     def forward(self, x):
   ...         h = self.l1(x)
   ...         return self.l2(h)

It shows how a complex chain is constructed by simpler links.
Links like ``l1`` and ``l2`` are called *child links* of ``MyChain``.
**Note that Chain itself inherits Link**.
It means we can define more complex chains that hold ``MyChain`` objects as their child links.

.. note::

   We often define a single forward method of a link by the ``forward`` operator.
   Such links and chains are callable and behave like regular functions of Variables.

.. note::

    In Chainer v1, we could also register the trainable layers
    (i.e., :class:`~chainer.Link` s) to the model by putting them to the
    :meth:`~chainer.Chain.__init__` of :class:`~chainer.Chain`
    or registering them via :meth:`~chainer.Chain.add_link`.
    But as these ways are deprecated in Chainer v2, users are recommended
    to use the way explained above.

Another way to define a chain is using the :class:`ChainList` class, which behaves like a list of links:

.. doctest::

   >>> class MyChain2(ChainList):
   ...     def __init__(self):
   ...         super(MyChain2, self).__init__(
   ...             L.Linear(4, 3),
   ...             L.Linear(3, 2),
   ...         )
   ...
   ...     def forward(self, x):
   ...         h = self[0](x)
   ...         return self[1](h)

ChainList can conveniently use an arbitrary number of links, however if the number of links is fixed like in the above case, the Chain class is recommended as a base class.

