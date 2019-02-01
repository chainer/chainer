import multiprocessing
import os
import shutil
from six.moves import queue
import threading

from chainer.serializers import npz
from chainer import utils


class Writer(object):

    """Base class of snapshot writers.

    :class:`~chainer.training.extensions.Snapshot` invokes ``__call__`` of this
    class everytime when taking a snapshot.
    This class determines how the actual saving function will be invoked.

    .. seealso::

        - :meth:`chainer.training.extensions.snapshot`
    """

    def __call__(self, filename, outdir, target):
        """Invokes the actual snapshot function.

        This method is invoked by a
        :class:`~chainer.training.extensions.Snapshot` object every time it
        takes a snapshot.

        Args:
            filename (str): Name of the file into which the serialized target
                is saved. It is a concrete file name, i.e. not a pre-formatted
                template string.
            outdir (str): Output directory. Corresponds to
                :py:attr:`Trainer.out <chainer.training.Trainer.out>`.
            target (dict): Serialized object which will be saved.
        """
        raise NotImplementedError

    def __del__(self):
        self.finalize()

    def finalize(self):
        """Finalizes the wirter.

        Like extensions in :class:`~chainer.training.Trainer`, this method
        is invoked at the end of the training.

        """
        pass

    def save(self, filename, outdir, target, savefun, **kwds):
        prefix = 'tmp' + filename
        with utils.tempdir(prefix=prefix, dir=outdir) as tmpdir:
            tmppath = os.path.join(tmpdir, filename)
            savefun(tmppath, target)
            shutil.move(tmppath, os.path.join(outdir, filename))


class SimpleWriter(Writer):
    """The most simple snapshot writer.

    This class just passes the arguments to the actual saving function.

    Args:
        savefun: Callable object. It takes three arguments: the output file
            path, the serialized dictionary object, and the optional keyword
            arguments.
        kwds: Keyword arguments for the ``savefun``.

    .. seealso::

        - :meth:`chainer.training.extensions.snapshot`
    """

    def __init__(self, savefun=npz.save_npz, **kwds):
        self._savefun = savefun
        self._kwds = kwds

    def __call__(self, filename, outdir, target):
        self.save(filename, outdir, target, self._savefun, **self._kwds)


class StandardWriter(Writer):
    """Base class of snapshot writers which use thread or process.

    This class creates a new thread or a process every time when ``__call__``
    is invoked.

    Args:
        savefun: Callable object. It takes three arguments: the output file
            path, the serialized dictionary object, and the optional keyword
            arguments.
        kwds: Keyword arguments for the ``savefun``.

    .. seealso::

        - :meth:`chainer.training.extensions.snapshot`
    """

    _started = False
    _finalized = False
    _worker = None

    def __init__(self, savefun=npz.save_npz, **kwds):
        self._savefun = savefun
        self._kwds = kwds
        self._started = False
        self._finalized = False

    def __call__(self, filename, outdir, target):
        if self._started:
            self._worker.join()
            self._started = False
        self._filename = filename
        self._worker = self.create_worker(filename, outdir, target,
                                          **self._kwds)
        self._worker.start()
        self._started = True

    def create_worker(self, filename, outdir, target, **kwds):
        """Creates a worker for the snapshot.

        This method creates a thread or a process to take a snapshot. The
        created worker must have :meth:`start` and :meth:`join` methods.

        Args:
            filename (str): Name of the file into which the serialized target
                is saved. It is already formated string.
            outdir (str): Output directory. Passed by `trainer.out`.
            target (dict): Serialized object which will be saved.
            kwds: Keyword arguments for the ``savefun``.

        """
        raise NotImplementedError

    def finalize(self):
        if self._started:
            if not self._finalized:
                self._worker.join()
            self._started = False
        self._finalized = True


class ThreadWriter(StandardWriter):
    """Snapshot writer that uses a separate thread.

    This class creates a new thread that invokes the actual saving function.

    .. seealso::

        - :meth:`chainer.training.extensions.snapshot`
    """

    def create_worker(self, filename, outdir, target, **kwds):
        return threading.Thread(
            target=self.save,
            args=(filename, outdir, target, self._savefun),
            kwargs=self._kwds)


class ProcessWriter(StandardWriter):
    """Snapshot writer that uses a separate process.

    This class creates a new process that invokes the actual saving function.

    .. note::
        Forking a new process from a MPI process might be danger. Consider
        using :class:`ThreadWriter` instead of ``ProcessWriter`` if you are
        using MPI.

    .. seealso::

        - :meth:`chainer.training.extensions.snapshot`
    """

    def create_worker(self, filename, outdir, target, **kwds):
        return multiprocessing.Process(
            target=self.save,
            args=(filename, outdir, target, self._savefun),
            kwargs=self._kwds)


class QueueWriter(Writer):
    """Base class of queue snapshot writers.

    This class is a base class of snapshot writers that use a queue.
    A Queue is created when this class is constructed, and every time when
    ``__call__`` is invoked, a snapshot task is put into the queue.

    Args:
        savefun: Callable object which is passed to the :meth:`create_task`
            if the task is ``None``. It takes three arguments: the output file
            path, the serialized dictionary object, and the optional keyword
            arguments.
        task: Callable object. Its ``__call__`` must have a same interface to
            ``Writer.__call__``. This object is directly put into the queue.

    .. seealso::

        - :meth:`chainer.training.extensions.snapshot`
    """

    _started = False
    _finalized = False
    _queue = None
    _consumer = None

    def __init__(self, savefun=npz.save_npz, task=None):
        if task is None:
            self._task = self.create_task(savefun)
        else:
            self._task = task
        self._queue = self.create_queue()
        self._consumer = self.create_consumer(self._queue)
        self._consumer.start()
        self._started = True
        self._finalized = False

    def __call__(self, filename, outdir, target):
        self._queue.put([self._task, filename, outdir, target])

    def create_task(self, savefun):
        return SimpleWriter(savefun=savefun)

    def create_queue(self):
        raise NotImplementedError

    def create_consumer(self, q):
        raise NotImplementedError

    def consume(self, q):
        while True:
            task = q.get()
            if task is None:
                q.task_done()
                return
            else:
                task[0](task[1], task[2], task[3])
                q.task_done()

    def finalize(self):
        if self._started:
            if not self._finalized:
                self._queue.put(None)
                self._queue.join()
                self._consumer.join()
            self._started = False
        self._finalized = True


class ThreadQueueWriter(QueueWriter):
    """Snapshot writer that uses a thread queue.

    This class creates a thread and a queue by :mod:`threading` and
    :mod:`queue` modules
    respectively. The thread will be a consumer of the queue, and the main
    thread will be a producer of the queue.

    .. seealso::

        - :meth:`chainer.training.extensions.snapshot`
    """

    def create_queue(self):
        return queue.Queue()

    def create_consumer(self, q):
        return threading.Thread(target=self.consume, args=(q,))


class ProcessQueueWriter(QueueWriter):
    """Snapshot writer that uses process queue.

    This class creates a process and a queue by :mod:`multiprocessing` module.
    The process will be a consumer of this queue, and the main process will be
    a producer of this queue.

    .. note::
        Forking a new process from MPI process might be danger. Consider using
        :class:`ThreadQueueWriter` instead of ``ProcessQueueWriter`` if you are
        using MPI.

    .. seealso::

        - :meth:`chainer.training.extensions.snapshot`
    """

    def create_queue(self):
        return multiprocessing.JoinableQueue()

    def create_consumer(self, q):
        return multiprocessing.Process(target=self.consume, args=(q,))
