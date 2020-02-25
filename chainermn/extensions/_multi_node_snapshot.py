import io

from chainer.serializers import load_npz
from chainer.serializers import save_npz
from chainer.training.extension import Extension
from chainer.training.extensions._snapshot import _find_latest_snapshot


def multi_node_snapshot(comm, snapshot, replica_sets):
    '''Create trainer extension for multi-node snapshots

    Provides generis multi-node snapshot saving and auto-load feature
    at multi-node environment, leveraging power of single-node
    snapshot.

    In many cases snapshot target may differ, e.g. only trainer of
    rank 0 process often has extensions such as ``LogReport`` and so
    on, to not confuse terminal output. Just loading at one process
    and broadcasting it to other processes does not work in that case.

    This wrapper addresses that issue by defining sets of replicas
    where within the set the target object is replicated and supposed
    to be same among processes. For example, a trainer example, only
    the trainer at rank ``0`` has special extensions and others
    doesn't::

        trainer = Trainer(updater)
        if comm.rank == 0:
            trainer.extend(extensions.DumpGraph('main/loss'))
            trainer.extend(extensions.LogReport())
            trainer.extend(extensions.PrintReport(
                ['epoch', 'main/loss', 'validation/main/loss',
                 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
            trainer.extend(extensions.ProgressBar())

    This case can be described with two replica sets, where each set
    can be represented as single integer that indicates rank number,
    or iterable set/list/generator of integers like this::

        replica_sets = [[0], range(1, comm.size)]

    Here the first replica set is described as ``[0]``, or simply in
    short just ``0``, and the second replica set is ``range(1,
    comm.size)``, representing rest of processes other than ``0``. The
    remaining list can be omitted. Thus in that case, it can be
    simplified more::

        replica_sets = [0,]

    In this case, the snapshot will be saved at rank ``0`` process and
    at rank ``1`` process. The latter represents the replica set of
    ``range(1, comm.size)`` . In this case autoloading at
    initialization of snapshot extension works after the restart
    cleanly, even though the size of the communicator differs.

    Once the replica sets are defined, it can be easily extended::

        replica_sets = [0,]
        snapshot = multi_node_snapshot(comm, extensions.snapshot(),
                                       replica_sets)
        trainer.extend(snapshot, trigger=(1, 'epoch'))


    More example tuples of replica set representation follows:

    ===================== ===== ==============================================
    code                  nproc actual sets
    ===================== ===== ==============================================
    ``[0]``               ``4`` ``[{0}, {1, 2, 3}]``
    ``[0, 1]``            ``4`` ``[{0}, {1}, {2, 3}]``
    ``[0, 1], [2, 3]]``   ``4`` ``[{0, 1}, {2, 3}]``
    ``[]``                ``4`` ``[{0, 1, 2, 3}]``
    ``[range(0, 8, 2)]``  ``8`` ``[set(range(0, 8, 2)), set(range(1, 8, 2))]``
    ===================== ===== ==============================================

    Args:
        comm (ChainerMN communicator): communicater object
        snapshot: Snapshot extension object obtained via
              :meth:`~chainer.training.extensions.snapshot` .
        replica_sets: list of replica set definition, where
              a replica set can be defined by single integer
              as rank number, or iterable integers.

    Returns:
        Trainer extension that wraps ``snapshot`` and properly
        controles number of snapshots.

    '''
    return _MultiNodeSnapshot(comm, snapshot, replica_sets)


def _parse_replica_sets(replica_sets, size):
    sets = []

    for replica_set in replica_sets:
        if isinstance(replica_set, int):
            assert replica_set >= 0
            assert replica_set < size
            sets.append({replica_set})
        else:
            # Must be iterable
            for i in replica_set:
                assert i >= 0
                assert i < size
            sets.append(set(replica_set))

    if size > sum(len(s) for s in sets):
        all_ranks = set(range(size))
        all_exp = set()
        for s in sets:
            all_exp |= s
        rest = all_ranks - all_exp
        if rest:
            sets.append(rest)

    # Must guarantee: no lack allowed
    assert size == sum(len(s) for s in sets)

    # Must guarantee: no two sets must have intersection.
    all_sum = set()
    for s in sets:
        all_sum |= s
    assert size == len(all_sum)
    return sets


class _MultiNodeSnapshot(Extension):
    def __init__(self, comm, snapshot, replica_sets):
        assert comm is not None
        assert snapshot is not None
        self.comm = comm
        self.snapshot = snapshot

        # Append rank number to snapshot filename format/function
        if callable(snapshot.filename):
            filename_fun = snapshot.filename

            def append_rank(trainer):
                filename = filename_fun(trainer)
                return '{}.{}'.format(filename, comm.rank)
            snapshot.filename = append_rank

        else:
            filename = '{}.{}'.format(snapshot.filename, comm.rank)
            snapshot.filename = filename

        sets = _parse_replica_sets(replica_sets, comm.size)

        self.master = None
        self.replica_set = []
        for s in sets:
            if self.comm.rank in s:
                self.master = min(s)
                self.replica_set = s
                break
        assert self.master is not None
        assert self.comm.rank in self.replica_set

    @property
    def is_master(self):
        return self.master == self.comm.rank

    def initialize(self, trainer):
        if self.is_master:
            self.snapshot.initialize(trainer)

        # If autoload is off, no need to re-init this extension.
        if not self.snapshot.autoload:
            return

        if self.snapshot._target is None:
            target = trainer
        else:
            target = self.snapshot._target

        # "Broadcast" the target here
        if self.is_master:
            # Find snapshot again
            # TODO(kuenishi): replace with cleaner way to know whether
            # a snapshot is autoloaded or not
            filename = _find_latest_snapshot(self.snapshot.filename,
                                             trainer.out)
            if filename is None:
                data = None
            else:
                buf = io.BytesIO()
                save_npz(buf, target)
                data = buf.getvalue()

            for rank in self.replica_set:
                if rank == self.comm.rank:
                    continue
                self.comm.send_obj(data, rank)

        # Get the loaded target from master
        else:
            data = self.comm.recv_obj(self.master)
            if data is None:
                return
            load_npz(io.BytesIO(data), target)

    def on_error(self, trainer, e, t):
        if self.is_master:
            self.snapshot.on_error(trainer, e, t)

    def __call__(self, trainer):
        if self.is_master:
            self.snapshot(trainer)

    def finalize(self):
        if self.is_master:
            self.snapshot.finalize()
