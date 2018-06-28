from chainer.functions.activation import sigmoid
from chainer.functions.connection import linear
from chainer import initializers
from chainer import link
from chainer import variable


class MADE(link.Link):

    """Masked Autoencoder for Distribution Estimation

    Args:
        in_size (int): Dimension of input vectors.
        hidden_num (int): Number of hidden layers.
        hidden_size (int): Number of units in a hidden layer.
    """

    def __init__(self, in_size, hidden_num, hidden_size):
        super(MADE, self).__init__()
        self.hidden_num = hidden_num

        with self.init_scope():
            self.m = []
            self.m0 = self.xp.random.permutation(in_size)
            self.m.append(self.m0)
            for i in range(hidden_num):
                min_mm1 = self.m[-1].min()
                m_ = self.xp.random.randint(min_mm1, in_size-1, hidden_size)
                self.m.append(m_)

            self.M = []
            for i in range(hidden_num):
                M_ = self.m[i+1].reshape(-1, 1) >= self.m[i]
                self.__setattr__('M_%d' % i, M_)
                self.M.append(M_)
            M_ = self.m[0].reshape(-1, 1) > self.m[-1]
            self.M.append(M_)

            self.W = []
            self.b = []
            W_initializer = initializers._get_initializer(None)
            bias_initializer = initializers._get_initializer(0)
            for i in range(hidden_num + 1):
                W_ = variable.Parameter(W_initializer)
                self.__setattr__('W_%d' % i, W_)
                if i == 0:
                    W_.initialize((hidden_size, in_size))
                elif i == hidden_num:
                    W_.initialize((in_size, hidden_size))
                else:
                    W_.initialize((hidden_size, hidden_size))

                if i == hidden_num:
                    b_ = variable.Parameter(bias_initializer, in_size)
                    self.__setattr__('b_%d' % i, b_)
                else:
                    b_ = variable.Parameter(bias_initializer, hidden_size)
                    self.__setattr__('b_%d' % i, b_)

                self.W.append(W_)
                self.b.append(b_)

    def __call__(self, x):
        h = x
        for i in range(self.hidden_num):
            self.M[i] = self.xp.asarray(self.M[i])
            h = linear.linear(h, self.M[i]*self.W[i], self.b[i])
            h = sigmoid.sigmoid(h)
        self.M[self.hidden_num] = self.xp.asarray(self.M[self.hidden_num])
        return linear.linear(
            h, self.M[self.hidden_num]*self.W[self.hidden_num],
            self.b[self.hidden_num])
