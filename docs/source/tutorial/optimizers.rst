Optimizer
~~~~~~~~~

In order to get good values for parameters, we have to optimize them by the :class:`Optimizer` class.
It runs a numerical optimization algorithm on a given link.
Many algorithms are implemented in the :mod:`~chainer.optimizers` module.
Here we use the simplest one, called Stochastic Gradient Descent (SGD):

.. doctest::

   >>> model = MyChain()
   >>> optimizer = optimizers.SGD()
   >>> optimizer.setup(model)

The method :meth:`~Optimizer.setup` prepares for the optimization given a link.

Some parameter/gradient manipulations, e.g. weight decay and gradient clipping, can be done by setting *hook functions* to the optimizer.
Hook functions are called after the gradient computation and right before the actual update of parameters.
For example, we can set weight decay regularization by running the next line beforehand:

   >>> optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))

Of course, you can write your own hook functions.
It should be a function or a callable object, taking the optimizer as the argument.

There are two ways to use the optimizer.
One is using it via :class:`~chainer.training.Trainer`, which we will see in the following sections.
The other way is using it directly.
We here review the latter case.
*If you are interested in getting able to use the optimizer in a simple way, skip this section and go to the next one.*

There are two further ways to use the optimizer directly.
One is manually computing gradients and then calling the :meth:`~Optimizer.update` method with no arguments.
Do not forget to clear the gradients beforehand!

   >>> x = np.random.uniform(-1, 1, (2, 4)).astype('f')
   >>> model.cleargrads()
   >>> # compute gradient here...
   >>> loss = F.sum(model(chainer.Variable(x)))
   >>> loss.backward()
   >>> optimizer.update()

The other way is just passing a loss function to the :meth:`~Optimizer.update` method.
In this case, :meth:`~Link.cleargrads` is automatically called by the update method, so the user does not have to call it manually.

   >>> def lossfun(arg1, arg2):
   ...     # calculate loss
   ...     loss = F.sum(model(arg1 - arg2))
   ...     return loss
   >>> arg1 = np.random.uniform(-1, 1, (2, 4)).astype('f')
   >>> arg2 = np.random.uniform(-1, 1, (2, 4)).astype('f')
   >>> optimizer.update(lossfun, chainer.Variable(arg1), chainer.Variable(arg2))

See :meth:`Optimizer.update` for the full specification.



