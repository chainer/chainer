Optimizer
~~~~~~~~~

.. include:: ../imports.rst

From the previous guide on :doc:`models`, let's use the ``MyChain`` class:

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

To tune parameters values to minimize loss, etc., we have to optimize them by the :class:`Optimizer` class.
It runs a numerical optimization algorithm on a given link.
Many algorithms are implemented in the :mod:`~chainer.optimizers` module.
Here we use the simplest one, called Stochastic Gradient Descent (SGD):

.. doctest::

   >>> model = MyChain()
   >>> optimizer = optimizers.SGD().setup(model)

The method :meth:`~Optimizer.setup` prepares for the optimization given a link.

Some parameter/gradient manipulations, e.g. weight decay and gradient clipping, can be done by setting *hook functions* to the optimizer.
Hook functions are called after the gradient computation and right before the actual update of parameters.
For example, we can set weight decay regularization by running the next line beforehand:

.. doctest::

   >>> optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(0.0005))

Of course, you can write your own hook functions.
It should be a function or a callable object.

There are two ways to use the optimizer.
One is using it via :class:`~chainer.training.Trainer`, which we will see in the following sections.
The other way is using it directly.
We here review the latter case.
To use the optimizer in an automated fashion, see the :doc:`trainer` guide.

There are two further ways to use the optimizer directly.
One is manually computing gradients and then calling the :meth:`~Optimizer.update` method with no arguments.
Do not forget to clear the gradients beforehand!

.. doctest::

   >>> x = np.random.uniform(-1, 1, (2, 4)).astype(np.float32)
   >>> model.cleargrads()
   >>> # compute gradient here...
   >>> loss = F.sum(model(chainer.Variable(x)))
   >>> loss.backward()
   >>> optimizer.update()

The other way is just passing a loss function to the :meth:`~Optimizer.update` method.
In this case, :meth:`~Link.cleargrads` is automatically called by the update method, so the user does not have to call it manually.

.. doctest::

   >>> def lossfun(arg1, arg2):
   ...     # calculate loss
   ...     loss = F.sum(model(arg1 - arg2))
   ...     return loss

   >>> arg1 = np.random.uniform(-1, 1, (2, 4)).astype(np.float32)
   >>> arg2 = np.random.uniform(-1, 1, (2, 4)).astype(np.float32)
   >>> optimizer.update(lossfun, chainer.Variable(arg1), chainer.Variable(arg2))

See :meth:`Optimizer.update` for the full specification.

