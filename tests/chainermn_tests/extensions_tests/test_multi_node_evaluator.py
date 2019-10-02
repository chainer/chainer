import chainer
import chainer.testing
from chainer.datasets import TupleDataset
from chainer.iterators import SerialIterator

from chainermn import create_communicator
from chainermn.extensions import GenericMultiNodeEvaluator


class ExampleModel(chainer.Chain):
    def forward(self, a, b, c):
        return a + b + c


def check_generic(comm, length, bs):
    assert bs > 0
    assert length > 0
    a = list(range(comm.rank, length, comm.size))
    b = list(range(comm.rank, length, comm.size))
    c = list(range(comm.rank, length, comm.size))

    model = ExampleModel()
    dataset = TupleDataset(a, b, c)
    iterator = SerialIterator(dataset, bs, shuffle=False, repeat=False)
    evaluator = GenericMultiNodeEvaluator(comm, iterator, model)

    results = evaluator(None)

    # Make expected answer
    iterator.reset()
    s = [[aa + bb + cc  # Same calculation as model
          for aa, bb, cc in batch] for batch in iterator]
    s = comm.gather_obj(s)

    if comm.rank == 0:
        # flatten list of lists gathered
        expected = []
        for e in zip(*s):
            expected.extend(e)

        for e, r in zip(expected, results):
            chainer.testing.assert_allclose(e, r)

    else:
        assert results is None


def test_generic():
    comm = create_communicator('naive')
    try:
        check_generic(comm, 97, 7)
        check_generic(comm, 9, 77)
    finally:
        comm.finalize()


class CustomMultiNodeEvaluator(GenericMultiNodeEvaluator):
    def __init__(self, *args, **kwargs):
        super(CustomMultiNodeEvaluator, self).__init__(*args, **kwargs)

    def calc_local(self, *args, **kwargs):
        assert len(args) == 3
        return 2

    def aggregate(self, results):
        for result in results:
            assert 2 == result
        return sum(results)


def check_custom(comm, length, bs):
    assert bs > 0
    assert length > 0
    a = list(range(comm.rank, length, comm.size))
    b = list(range(comm.rank, length, comm.size))
    c = list(range(comm.rank, length, comm.size))

    model = ExampleModel()
    dataset = TupleDataset(a, b, c)
    iterator = SerialIterator(dataset, bs, shuffle=False, repeat=False)
    evaluator = CustomMultiNodeEvaluator(comm, iterator, model)

    result = evaluator(None)

    iterator.reset()
    expected = comm.allreduce_obj(sum(2 for batch in iterator))

    if comm.rank == 0:
        assert expected == result

    else:
        assert result is None


def test_custom():
    comm = create_communicator('naive')
    try:
        check_custom(comm, 97, 7)
        check_custom(comm, 9, 77)
    finally:
        comm.finalize()
