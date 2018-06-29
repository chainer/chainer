from chainer.functions.activation import relu
from chainer.functions.connection import linear
from chainer import initializers
from chainer import link
from chainer import variable


class MADE(link.Link):

    """Masked Autoencoder for Distribution Estimation

    This is a link to use a network structure of MADE (Masked Autoencoder for \
    Distribution Estimation).

    Args:
        in_size (int): Dimension of input vectors.
        hidden_num (int): Number of hidden layers.
        hidden_size (int): Number of units in a hidden layer.

    Attributes:
        W{N} (~chainer.Variable): Weight parameter.
        b{N} (~chainer.Variable): Bias parameter.
        m{N} (:class:`numpy.ndarray` or :class:`cupy.ndarray`): Number \
            related with each unit.
        M{N} (:class:`numpy.ndarray` or :class:`cupy.ndarray`): Mask \
            persistent values.
    """

    def __init__(self, in_size, hidden_num, hidden_size):
        super(MADE, self).__init__()
        self.hidden_num = hidden_num

        m0 = self.xp.random.permutation(in_size)
        self.add_persistent('m0', m0)
        for i in range(hidden_num):
            min_mm1 = self.get_var('m', i).min()
            m_ = self.xp.random.randint(min_mm1, in_size-1, hidden_size)
            self.add_persistent('m%d' % (i + 1), m_)

        for i in range(hidden_num):
            M_ = self.get_var('m', i + 1).reshape(-1, 1) \
                >= self.get_var('m', i)
            self.add_persistent('M%d' % i, M_)
        M_ = self.m0.reshape(-1, 1) > self.get_var('m', hidden_num)
        self.add_persistent('M%d' % hidden_num, M_)

        with self.init_scope():
            W_initializer = initializers._get_initializer(None)
            bias_initializer = initializers._get_initializer(0)
            for i in range(hidden_num + 1):
                W_ = variable.Parameter(W_initializer)
                self.__setattr__('W%d' % i, W_)
                if i == 0:
                    W_.initialize((hidden_size, in_size))
                elif i == hidden_num:
                    W_.initialize((in_size, hidden_size))
                else:
                    W_.initialize((hidden_size, hidden_size))

                if i == hidden_num:
                    b_ = variable.Parameter(bias_initializer, in_size)
                    self.__setattr__('b%d' % i, b_)
                else:
                    b_ = variable.Parameter(bias_initializer, hidden_size)
                    self.__setattr__('b%d' % i, b_)

    def get_var(self, var, idx):
        return self.__dict__['%s%d' % (var, idx)]

    def __call__(self, x):
        h = x
        for i in range(self.hidden_num):
            h = linear.linear(
                h, self.get_var('M', i) * self.get_var('W', i),
                self.get_var('b', i))
            h = relu.relu(h)
        return linear.linear(
            h, self.get_var('M', self.hidden_num)
            * self.get_var('W', self.hidden_num),
            self.get_var('b', self.hidden_num))
