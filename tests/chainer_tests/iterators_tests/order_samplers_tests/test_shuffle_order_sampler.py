import unittest

import numpy

from chainer.iterators.order_samplers.shuffle_order_sampler import ShuffleOrderSampler  # NOQA
from chainer import serializer
from chainer import testing


class DummySerializer(serializer.Serializer):

    def __init__(self, target):
        super(DummySerializer, self).__init__()
        self.target = target

    def __getitem__(self, key):
        raise NotImplementedError

    def __call__(self, key, value):
        self.target[key] = value
        return self.target[key]


class DummyDeserializer(serializer.Deserializer):

    def __init__(self, target):
        super(DummyDeserializer, self).__init__()
        self.target = target

    def __getitem__(self, key):
        raise NotImplementedError

    def __call__(self, key, value):
        if value is None:
            value = self.target[key]
        elif isinstance(value, numpy.ndarray):
            numpy.copyto(value, self.target[key])
        else:
            value = type(value)(numpy.asarray(self.target[key]))
        return value


@testing.parameterize(*testing.product({
    'seed': [None, 0],
}))
class TestShuffleOrderSampler(unittest.TestCase):

    def setUp(self):
        if self.seed is None:
            self.order_sampler = ShuffleOrderSampler()
        else:
            self.order_sampler = ShuffleOrderSampler(
                numpy.random.RandomState(self.seed))

    def test_serialize(self):
        # Test just to call serialize
        target = dict()
        self.order_sampler.serialize(DummySerializer(target))

    def test_call(self):
        # Just call
        current_order = numpy.arange(3)
        current_position = 1
        order = self.order_sampler(current_order, current_position)
        print('order', order)
        self.assertEqual(len(order), 3)
        self.assertTrue(0 in order)
        self.assertTrue(1 in order)
        self.assertTrue(2 in order)


testing.run_module(__name__, __file__)
