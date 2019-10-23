import mock
import tempfile

import pytest

import chainer
import chainer.links as L
import chainer.functions as F
from chainer.training import extensions
from chainer.training import StandardUpdater
from chainer.training import Trainer
import chainermn
from chainermn import create_communicator
from chainermn.extensions import multi_node_snapshot
from chainermn.extensions import _multi_node_snapshot


@pytest.mark.parametrize('rs,size,expected', [
    ([0], 4, [{0}, {1, 2, 3}]),
    ([0, 1], 4, [{0}, {1}, {2, 3}]),
    ([[0, 1], [2, 3]], 4, [{0, 1}, {2, 3}]),
    ([], 4, [{0, 1, 2, 3}]),
    ([range(0, 16, 2), range(1, 16, 2)], 16,
     [set(range(0, 16, 2)), set(range(1, 16, 2))]),
    ([range(0, 16, 2)], 16, [set(range(0, 16, 2)), set(range(1, 16, 2))]),
    ([], 8, [set(range(8))]),
])
def test_parser(rs, size, expected):
    sets = _multi_node_snapshot._parse_replica_sets(rs, size)
    assert expected == sets


def test_smoke_wrapper():
    rs = [[0, 1], ]
    comm = create_communicator('naive')
    if comm.size < 2:
        pytest.skip()

    snapshot = extensions.snapshot()
    filename = '{}.{}'.format(snapshot.filename, comm.rank)

    replica_sets = rs
    mn_snapshot = multi_node_snapshot(comm, snapshot, replica_sets)
    if comm.rank == 0:
        assert mn_snapshot.is_master
        assert filename == mn_snapshot.snapshot.filename
    elif comm.rank == 1:
        assert not mn_snapshot.is_master
    elif comm.rank == 2:
        assert mn_snapshot.is_master
        assert filename == mn_snapshot.snapshot.filename
    else:
        assert not mn_snapshot.is_master

    comm.finalize()


def test_callable_filename():
    rs = [[0, 1], ]
    comm = create_communicator('naive')
    if comm.size < 2:
        pytest.skip()

    def filename_fun(t):
        return 'deadbeef-{.updater.iteration}'.format(t)

    snapshot = extensions.snapshot(filename=filename_fun)

    trainer = mock.MagicMock()
    filename = '{}.{}'.format(filename_fun(trainer), comm.rank)

    replica_sets = rs
    mn_snapshot = multi_node_snapshot(comm, snapshot, replica_sets)
    if comm.rank == 0:
        assert mn_snapshot.is_master
        assert filename == mn_snapshot.snapshot.filename(trainer)
    elif comm.rank == 1:
        assert not mn_snapshot.is_master
    elif comm.rank == 2:
        assert mn_snapshot.is_master
        assert filename == mn_snapshot.snapshot.filename(trainer)
    else:
        assert not mn_snapshot.is_master

    comm.finalize()


def test_smoke_multinode_snapshot():
    t = mock.MagicMock()
    c = mock.MagicMock(side_effect=[True, False])
    w = mock.MagicMock()
    snapshot = extensions.snapshot(target=t, condition=c, writer=w)
    trainer = mock.MagicMock()

    comm = create_communicator('naive')
    replica_sets = []
    mn_snapshot = multi_node_snapshot(comm, snapshot, replica_sets)

    mn_snapshot.initialize(trainer)
    mn_snapshot(trainer)
    mn_snapshot(trainer)
    mn_snapshot.finalize()

    if comm.rank == 0:
        assert mn_snapshot.is_master
        assert c.call_count == 2
        assert w.call_count == 1
    else:
        assert not mn_snapshot.is_master
        assert c.call_count == 0
        assert w.call_count == 0

    comm.finalize()


class MLP(chainer.Chain):
    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(784, n_units)
            self.l2 = L.Linear(n_units, n_units)
            self.l3 = L.Linear(n_units, n_out)

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


def _prepare_multinode_snapshot(n, result):
    n_units = 100
    batchsize = 10
    comm = create_communicator('naive')
    model = L.Classifier(MLP(n_units, 10))
    optimizer = chainermn.create_multi_node_optimizer(
        chainer.optimizers.Adam(), comm)
    optimizer.setup(model)

    if comm.rank == 0:
        train, _ = chainer.datasets.get_mnist()
    else:
        train, _ = None, None

    train = chainermn.scatter_dataset(train, comm, shuffle=True)
    train_iter = chainer.iterators.SerialIterator(train, batchsize)

    updater = StandardUpdater(train_iter, optimizer)
    trainer = Trainer(updater, out=result)

    snapshot = extensions.snapshot(target=updater, autoload=True)
    replica_sets = []
    mn_snapshot = multi_node_snapshot(comm, snapshot, replica_sets)
    mn_snapshot.initialize(trainer)
    for _ in range(n):
        updater.update()

    return updater, mn_snapshot, trainer


def test_multinode_autoload():
    n = 3
    with tempfile.TemporaryDirectory() as tempd:
        result = tempd
        updater0, snapshot, trainer0 = _prepare_multinode_snapshot(n, result)

        assert n == updater0.iteration
        snapshot(trainer0)

        updater, _, _ = _prepare_multinode_snapshot(0, result)

        assert n == updater.iteration
