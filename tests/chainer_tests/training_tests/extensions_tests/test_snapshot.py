import glob
import itertools
import os
import random
import unittest

import mock
import pytest

from chainer import testing
from chainer import training
from chainer.training import extensions
from chainer.training.extensions._snapshot import _find_snapshot_files
from chainer.training.extensions import find_latest_snapshot
from chainer.training.extensions import find_stale_snapshots


class TestSnapshot(unittest.TestCase):

    def test_call(self):
        t = mock.MagicMock()
        c = mock.MagicMock(side_effect=[True, False])
        w = mock.MagicMock()
        snapshot = extensions.snapshot(target=t, condition=c, writer=w)
        trainer = mock.MagicMock()
        snapshot(trainer)
        snapshot(trainer)

        assert c.call_count == 2
        assert w.call_count == 1

    def test_savefun_and_writer_exclusive(self):
        # savefun and writer arguments cannot be specified together.
        def savefun(*args, **kwargs):
            assert False
        writer = extensions.snapshot_writers.SimpleWriter()
        with pytest.raises(TypeError):
            extensions.snapshot(savefun=savefun, writer=writer)

        trainer = mock.MagicMock()
        with pytest.raises(TypeError):
            extensions.snapshot_object(trainer, savefun=savefun, writer=writer)


class TestSnapshotSaveFile(unittest.TestCase):

    def setUp(self):
        self.trainer = testing.get_trainer_with_mock_updater()
        self.trainer.out = '.'
        self.trainer._done = True

    def tearDown(self):
        if os.path.exists('myfile.dat'):
            os.remove('myfile.dat')

    def test_save_file(self):
        w = extensions.snapshot_writers.SimpleWriter()
        snapshot = extensions.snapshot_object(self.trainer, 'myfile.dat',
                                              writer=w)
        snapshot(self.trainer)

        self.assertTrue(os.path.exists('myfile.dat'))

    def test_clean_up_tempdir(self):
        snapshot = extensions.snapshot_object(self.trainer, 'myfile.dat')
        snapshot(self.trainer)

        left_tmps = [fn for fn in os.listdir('.')
                     if fn.startswith('tmpmyfile.dat')]
        self.assertEqual(len(left_tmps), 0)


class TestSnapshotOnError(unittest.TestCase):

    def setUp(self):
        self.trainer = testing.get_trainer_with_mock_updater()
        self.trainer.out = '.'
        self.filename = 'myfile-deadbeef.dat'

    def tearDown(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def test_on_error(self):

        class TheOnlyError(Exception):
            pass

        @training.make_extension(trigger=(1, 'iteration'), priority=100)
        def exception_raiser(trainer):
            raise TheOnlyError()
        self.trainer.extend(exception_raiser)

        snapshot = extensions.snapshot_object(self.trainer, self.filename,
                                              snapshot_on_error=True)
        self.trainer.extend(snapshot)

        self.assertFalse(os.path.exists(self.filename))

        with self.assertRaises(TheOnlyError):
            self.trainer.run()

        self.assertTrue(os.path.exists(self.filename))


@pytest.mark.parametrize('fmt', ['snapshot_iter_{}',
                                 'snapshot_iter_{}.npz',
                                 '{}_snapshot_man_suffix.npz'])
def test_find_snapshot_files(fmt):
    files = dict((fmt.format(i), i) for i in range(1, 100))
    noise = dict(('dummy-foobar-iter{}'.format(i), i) for i in range(10, 304))
    noise2 = dict(('tmpsnapshot_iter_{}'.format(i), i) for i in range(10, 304))
    path = 'dummy'

    def lister(path):
        length = len(noise2) + len(noise) + len(files)
        chained = itertools.chain(noise.keys(), files.keys(), noise2.keys())
        return random.sample(list(chained), length)

    def get_ts(path, f):
        assert f not in noise
        assert f not in noise2
        return files[f]

    snapshot_files = _find_snapshot_files(fmt, path, lister=lister,
                                          get_ts=get_ts)

    ans = [(i, fmt.format(i)) for i in range(1, 100)]
    assert len(snapshot_files) == 99
    for lhs, rhs in zip(ans, snapshot_files):
        assert lhs == rhs


@pytest.mark.parametrize('fmt', ['snapshot_iter_{}_{}',
                                 'snapshot_iter_{}_{}.npz',
                                 '{}_snapshot_man_{}-suffix.npz',
                                 'snapshot_iter_{}.{}'])
def test_find_snapshot_files2(fmt):
    files = dict((fmt.format(i*10, j*10), j) for i, j
                 in itertools.product(range(0, 10), range(0, 10)))
    noise = dict(('tmpsnapshot_iter_{}.{}'.format(i, j), j)
                 for i, j in zip(range(10, 304), range(10, 200)))
    path = 'dummy'

    def lister(path):
        length = len(noise) + len(files)
        chained = itertools.chain(noise.keys(), files.keys())
        return random.sample(list(chained), length)

    def get_ts(path, f):
        assert f in files
        return files[f]

    snapshot_files = _find_snapshot_files(fmt, path, lister=lister,
                                          get_ts=get_ts)

    ans = ((j, fmt.format(i*10, j*10))
           for i, j in itertools.product(range(0, 10), range(0, 10)))

    for lhs, rhs in zip(sorted(ans), sorted(snapshot_files)):
        assert lhs == rhs


def test_find_latest_snapshot():
    fmt = 'snapshot_iter_{}'
    files = dict((fmt.format(i), i) for i in range(1, 100))
    path = 'dummy'

    def lister(path):
        return files.keys()

    def get_ts(path, f):
        assert f in files
        return files[f]

    assert 'snapshot_iter_99' == find_latest_snapshot(fmt, path,
                                                      lister=lister,
                                                      get_ts=get_ts)


@pytest.mark.parametrize('length,retain', [(100, 30), (10, 30), (1, 1000),
                                           (1000, 1), (1, 1), (1, 3), (2, 3)])
def test_find_stale_snapshot(length, retain):
    fmt = 'snapshot_iter_{}'
    files = dict(random.sample([(fmt.format(i), i) for i in range(0, length)],
                               length))
    path = 'dummy'

    def lister(path):
        return files.keys()

    def get_ts(path, f):
        return files[f]

    stale = list(find_stale_snapshots(fmt, path, retain,
                                      lister=lister, get_ts=get_ts))
    assert max(length-retain, 0) == len(list(stale))

    stales = [fmt.format(i) for i in range(0, max(length-retain, 0))]
    for lhs, rhs in zip(stales, stale):
        lhs == rhs


def test_remove_stale_snapshots():
    fmt = 'snapshot_iter_{.updater.iteration}'
    retain = 3
    snapshot = extensions.snapshot(filename=fmt, num_retain=retain,
                                   autoload=False)

    trainer = testing.get_trainer_with_mock_updater()
    trainer.out = '.'
    trainer.extend(snapshot, trigger=(1, 'iteration'))
    trainer.run()
    assert 10 == trainer.updater.iteration
    assert trainer._done

    pattern = os.path.join(trainer.out, "snapshot_iter_*")
    found = [path for path in glob.glob(pattern)]
    assert retain == len(found)
    found.sort()

    for lhs, rhs in zip(['snapshot_iter_{}'.format(i) for i in range(8, 10)],
                        found):
        lhs == rhs

    trainer2 = testing.get_trainer_with_mock_updater()
    trainer2.out = '.'
    assert not trainer2._done
    snapshot2 = extensions.snapshot(filename=fmt, autoload=True)
    # Just making sure no error occurs
    snapshot2.initialize(trainer2)

    # Cleanup
    for file in found:
        os.remove(file)


testing.run_module(__name__, __file__)
