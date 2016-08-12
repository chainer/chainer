import unittest

from chainer import link
from chainer import testing
from chainer.utils import rnn


class SimpleStatelessRNN(link.Chain):

    state_names = ('s',)

    def __init__(self, state_size):
        self.state_shapes = ((state_size,),)
        super(SimpleStatelessRNN, self).__init__()

    def __call__(self, s, x):
        return s + x

SimpleStatefulRNN = rnn.create_stateful_rnn(
    SimpleStatelessRNN, 'SimpleRNN')


class TestCreateStatefulRNN(unittest.TestCase):

    def setUp(self):
        state_size = 10
        self.l = SimpleStatefulRNN(state_size)

    def test_classs(self):
        self.assertIsInstance(self.l, SimpleStatefulRNN)
        self.assertIsInstance(self.l, link.Chain)

    def test_methods_exist(self):
        self.assertTrue(hasattr(self.l, '__init__'))
        self.assertTrue(hasattr(self.l, 'to_cpu'))
        self.assertTrue(hasattr(self.l, 'to_gpu'))
        self.assertTrue(hasattr(self.l, 'set_state'))
        self.assertTrue(hasattr(self.l, 'reset_state'))
        self.assertTrue(hasattr(self.l, '__getattr__'))
        self.assertTrue(hasattr(self.l, '__call__'))


class InvalidRNN1(link.Chain):

    def __init__(self, state_size):
        self.state_shapes = ((state_size,),)
        super(InvalidRNN1, self).__init__()

    def __call__(self, s, x):
        return s + x


class InvalidRNN2(link.Chain):

    state_names = ('s',)

    def __init__(self, state_size):
        super(InvalidRNN2, self).__init__()


class InvalidRNN3(link.Chain):

    state_names = ('s',)

    def __init__(self, state_size):
        self.state_shapes = ((state_size,),)
        super(InvalidRNN3, self).__init__()


@testing.parameterize(
    *testing.product(
        {'stateless_rnn': [InvalidRNN1, InvalidRNN2, InvalidRNN3]}
        )
)
class TestInvalidCreation(unittest.TestCase):

    def setUp(self):
        self.stateful_class = rnn.create_stateful_rnn(
            self.stateless_rnn, 'Invalid')

    def test_invalid_stateless_rnn(self):
        self.assertRaises(RuntimeError,
                          self.stateful_class, 10)


testing.run_module(__name__, __file__)
