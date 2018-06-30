import unittest

import numpy

from chainer import testing, serializer
from chainer.iterators.order_samplers.balanced_order_sampler import BalancedOrderSampler  # NOQA


class DummySerializer(serializer.Serializer):

    def __init__(self, target):
        super(DummySerializer, self).__init__()
        self.target = target

    def __getitem__(self, key):
        target_child = dict()
        self.target[key] = target_child
        return DummySerializer(target_child)

    def __call__(self, key, value):
        self.target[key] = value
        return self.target[key]


class DummyDeserializer(serializer.Deserializer):

    def __init__(self, target):
        super(DummyDeserializer, self).__init__()
        self.target = target

    def __getitem__(self, key):
        target_child = self.target[key]
        return DummyDeserializer(target_child)

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
    'x': [numpy.arange(8)],
    't': [numpy.asarray([0, 0, -1, 1, 1, 2, -1, 1])],
    'batch_balancing': [False, True],
    'ignore_labels': [-1]
}))
class TestBalancedOrderSampler(unittest.TestCase):

    def setUp(self):
        if self.seed is None:
            self.random_state = None
        else:
            self.random_state = numpy.random.RandomState(self.seed)
        self.order_sampler = BalancedOrderSampler(
            self.t, random_state=self.random_state,
            shuffle=True, batch_balancing=self.batch_balancing,
            ignore_labels=self.ignore_labels)

    def test_serialize(self):
        self.order_sampler(numpy.arange(8), 7)

        target = dict()
        # serialize
        self.order_sampler.serialize(DummySerializer(target))

        current_index_list_orig = dict()
        current_pos_orig = dict()
        for label, index_iterator in \
                self.order_sampler.labels_iterator_dict.items():
            ii_label = 'index_iterator_{}'.format(label)
            current_index_list_orig[
                ii_label] = index_iterator.current_index_list
            current_pos_orig[ii_label] = index_iterator.current_pos

        # deserialize
        sampler = BalancedOrderSampler(
            self.t, random_state=self.random_state,
            shuffle=True, batch_balancing=self.batch_balancing,
            ignore_labels=self.ignore_labels)
        sampler.serialize(DummyDeserializer(target))
        for label, index_iterator in sampler.labels_iterator_dict.items():
            ii_label = 'index_iterator_{}'.format(label)
            self.assertTrue(numpy.array_equal(
                index_iterator.current_index_list,
                current_index_list_orig[ii_label]))
            self.assertEqual(index_iterator.current_pos, current_pos_orig[ii_label])

    def test_call(self):
        # In this case, we have 3 examples of label=1.
        # When BalancedSerialIterator runs, all label examples are sampled 3 times
        # in one epoch.
        # Therefore, number of data is "augmented" as 9
        # 3 (number of label types) * 3 (number of maximum examples in one label)
        expect_N_augmented = 9
        order = self.order_sampler(numpy.arange(8), 7)
        self.assertEqual(len(order), expect_N_augmented)
        labels_batch = numpy.array([self.t[index] for index in order])

        self.assertEqual(numpy.sum(labels_batch == 0), 3)
        self.assertEqual(numpy.sum(labels_batch == 1), 3)
        self.assertEqual(numpy.sum(labels_batch == 2), 3)

        if self.batch_balancing:
            batch_size = 3
            for i in range(0, 9, batch_size):
                labels_batch = numpy.array([self.t[index] for index in
                                            order[i:i+batch_size]])
                assert numpy.sum(labels_batch == 0) == 1
                assert numpy.sum(labels_batch == 1) == 1
                assert numpy.sum(labels_batch == 2) == 1


testing.run_module(__name__, __file__)
