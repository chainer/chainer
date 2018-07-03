import numpy

from chainer import serializer
from chainer import testing
from chainer.iterators._index_iterator import IndexIterator  # NOQA


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


def test_index_iterator():
    _test_index_iterator_no_shuffle()
    _test_index_iterator_with_shuffle()


def _test_index_iterator_no_shuffle():
    index_list = [1, 3, 5, 10]
    ii = IndexIterator(index_list, shuffle=False, num=2)

    indices1 = ii.get_next_indices(3)
    indices2 = ii.get_next_indices(6)
    indices3 = ii.__next__()

    assert isinstance(indices1, numpy.ndarray)
    assert len(indices1) == 3
    assert isinstance(indices2, numpy.ndarray)
    assert len(indices2) == 6
    assert isinstance(indices3, numpy.ndarray)
    assert len(indices3) == 2
    assert indices1[0] == index_list[0]
    assert indices1[1] == index_list[1]
    assert indices1[2] == index_list[2]
    assert indices2[0] == index_list[3]
    assert indices2[1] == index_list[0]
    assert indices2[2] == index_list[1]
    assert indices2[3] == index_list[2]
    assert indices2[4] == index_list[3]
    assert indices2[5] == index_list[0]
    assert indices3[0] == index_list[1]
    assert indices3[1] == index_list[2]


def _test_index_iterator_with_shuffle():
    index_list = [1, 3, 5, 10]
    ii = IndexIterator(index_list, shuffle=True, num=2)

    indices1 = ii.get_next_indices(3)
    indices2 = ii.get_next_indices(6)
    indices3 = ii.__next__()

    assert isinstance(indices1, numpy.ndarray)
    assert len(indices1) == 3
    assert isinstance(indices2, numpy.ndarray)
    assert len(indices2) == 6
    assert isinstance(indices3, numpy.ndarray)
    assert len(indices3) == 2
    for indices in [indices1, indices2, indices3]:
        for index in indices:
            assert index in index_list


def test_index_iterator_serialization():
    _test_index_iterator_serialization_no_shuffle()
    _test_index_iterator_serialization_with_shuffle()


def _test_index_iterator_serialization_no_shuffle():
    index_list = [1, 3, 5, 10]
    ii = IndexIterator(index_list, shuffle=False, num=2)

    ii.get_next_indices(3)
    ii.get_next_indices(6)
    ii.__next__()

    assert len(ii.current_index_list) == len(index_list)
    assert numpy.array_equal(ii.current_index_list, numpy.asarray(index_list))
    assert ii.current_pos == (3 + 6) % len(index_list) + 2

    target = dict()
    ii.serialize(DummySerializer(target))

    ii = IndexIterator(index_list, shuffle=False, num=2)
    ii.serialize(DummyDeserializer(target))
    assert len(ii.current_index_list) == len(index_list)
    assert numpy.array_equal(ii.current_index_list, numpy.asarray(index_list))
    assert ii.current_pos == (3 + 6) % len(index_list) + 2


def _test_index_iterator_serialization_with_shuffle():
    index_list = [1, 3, 5, 10]
    ii = IndexIterator(index_list, shuffle=True, num=2)

    ii.get_next_indices(3)
    ii.get_next_indices(6)
    ii.__next__()

    assert len(ii.current_index_list) == len(index_list)
    for index in ii.current_index_list:
        assert index in index_list
    assert ii.current_pos == (3 + 6) % len(index_list) + 2

    target = dict()
    ii.serialize(DummySerializer(target))
    current_index_list_orig = ii.current_index_list

    ii = IndexIterator(index_list, shuffle=True, num=2)
    ii.serialize(DummyDeserializer(target))
    assert numpy.array_equal(ii.current_index_list, current_index_list_orig)
    assert ii.current_pos == (3 + 6) % len(index_list) + 2


testing.run_module(__name__, __file__)
