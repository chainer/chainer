from chainer.functions.activation import swish
from chainer import initializers
from chainer import link
from chainer import variable


class Swish(link.Link):

    """Swish activation function as a link.

    Args:
        beta_shape (tuple of ints or None): Shape of the parameter variable
            :math:`\\beta`. If ``None``, parameter initialization will be
            deferred until the first forward data pass at which time the shape
            will be determined.
        beta_init (float): Initial value of the parameter variable
            :math:`\\beta`.

    See the paper for details: `Searching for Activation Functions
    <https://arxiv.org/abs/1710.05941>`_

    To try Swish instead of ReLU, replace ``F.relu`` with individual ``Swish``
    links registered to the model. For example, the model defined in the
    `MNIST example
    <https://github.com/chainer/chainer/tree/master/examples/mnist/train_mnist.py>`_
    can be rewritten as follows.

    ReLU version (original)::

        class MLP(chainer.Chain):

            def __init__(self, n_units, n_out):
                super(MLP, self).__init__()
                with self.init_scope():
                    self.l1 = L.Linear(None, n_units)
                    self.l2 = L.Linear(None, n_units)
                    self.l3 = L.Linear(None, n_out)

            def forward(self, x):
                h1 = F.relu(self.l1(x))
                h2 = F.relu(self.l2(h1))
                return self.l3(h2)

    Swish version::

        class MLP(chainer.Chain):

            def __init__(self, n_units, n_out):
                super(MLP, self).__init__()
                with self.init_scope():
                    self.l1 = L.Linear(None, n_units)
                    self.s1 = L.Swish(None)
                    self.l2 = L.Linear(None, n_units)
                    self.s2 = L.Swish(None)
                    self.l3 = L.Linear(None, n_out)

            def forward(self, x):
                h1 = self.s1(self.l1(x))
                h2 = self.s2(self.l2(h1))
                return self.l3(h2)

    .. seealso::
        See :func:`chainer.functions.swish` for the definition of Swish
        activation function.

    Attributes:
        beta (~chainer.Parameter): Parameter variable :math:`\\beta`.

    """

    def __init__(self, beta_shape, beta_init=1.0):
        super(Swish, self).__init__()

        with self.init_scope():
            if beta_shape is not None:
                self.beta = variable.Parameter(beta_init, beta_shape)
            else:
                beta_init = initializers.Constant(beta_init)
                self.beta = variable.Parameter(beta_init)

    def forward(self, x):
        """Applies the Swish activation function.

        Args:
            x (~chainer.Variable): Input variable.

        Returns:
            ~chainer.Variable: Output of the Swish activation function.

        """
        if self.beta.array is None:
            self.beta.initialize(x.shape[1:])

        return swish.swish(x, self.beta)
