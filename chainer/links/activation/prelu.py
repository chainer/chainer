from chainer.functions.activation import prelu
from chainer import link
from chainer import variable


class PReLU(link.Link):

    """Parametric ReLU function as a link.

    Args:
        shape (tuple of ints): Shape of the parameter array.
        init (float): Initial parameter value.

    See the paper for details: `Delving Deep into Rectifiers: Surpassing
    Human-Level Performance on ImageNet Classification
    <https://arxiv.org/abs/1502.01852>`_.

    To try PReLU instead of ReLU, replace ``F.relu`` with individual ``PReLU``
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

    PReLU version::

        class MLP(chainer.Chain):

            def __init__(self, n_units, n_out):
                super(MLP, self).__init__()
                with self.init_scope():
                    self.l1 = L.Linear(None, n_units)
                    self.a1 = L.PReLU()
                    self.l2 = L.Linear(None, n_units)
                    self.a2 = L.PReLU()
                    self.l3 = L.Linear(None, n_out)

            def forward(self, x):
                h1 = self.a1(self.l1(x))
                h2 = self.a2(self.l2(h1))
                return self.l3(h2)

    .. seealso:: :func:`chainer.functions.prelu`

    Attributes:
        W (~chainer.Parameter): Coefficient of parametric ReLU.

    """

    def __init__(self, shape=(), init=0.25):
        super(PReLU, self).__init__()
        with self.init_scope():
            self.W = variable.Parameter(init, shape)

    def forward(self, x):
        """Applies the parametric ReLU activation function.

        Args:
            x (~chainer.Variable): Input variable.

        Returns:
            ~chainer.Variable: Output of the parametric ReLU function.

        """
        return prelu.prelu(x, self.W)
