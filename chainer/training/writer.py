import multiprocessing
import os
import shutil
from six.moves import queue
import tempfile
import threading

from chainer.serializers import npz


class Writer(object):

    """Base class of snapshot writer.

    :class:`~chainer.training.extensions.Snapshot` invokes this class'es
    ``__call__`` everytime if this process is in charge to take a snapshot.
    This class determine how to invoke the actual saving function. The most
    simple way is just passing the arguments to the saving function.
    """
    def __call__(self, filename, outdir, target):
        """Invokes the actual snapshot function.

        This method is invoked every time when this process is in charge to
        take a snapshot. All arguments are passed from
        :class:`~chainer.training.extensions.Snapshot` object.

        Args:
            filename (str): Name of the file into which the serialized target
                is saved. It is already formated string.
            outdir (str): Output directory. Passed by `trainer.out`.
            target (dict): Serialized object which will be saved.
        """
        raise NotImplementedError

    def __del__(self):
        self.finalize()

    def finalize(self):
        """Finalize the writer.

        Like an extension in :class:`~chainer.training.Trainer`, this method
        is invoked at the end of the training.
        """
        pass

    def save(self, filename, outdir, target, savefun, **kwds):
        prefix = 'tmp' + filename
        fd, tmppath = tempfile.mkstemp(prefix=prefix, dir=outdir)
        try:
            savefun(tmppath, target, **kwds)
        except Exception:
            os.close(fd)
            os.remove(tmppath)
            raise
        os.close(fd)
        shutil.move(tmppath, os.path.join(outdir, filename))


class SimpleWriter(Writer):
    """The most simple writer.

    This class just passes the arguments to the actual saving function.

    Args:
        savefun: Callable object. It takes three arguments: the output file
            path, the serialized dictionary object, and the optional keyword
            arguments.
        kwds: Keyword arguments for the ``savefun``.

    """
    def __init__(self, savefun=npz.save_npz, **kwds):
        self._savefun = savefun
        self._kwds = kwds

    def __call__(self, filename, outdir, target):
        self.save(filename, outdir, target, self._savefun, **self._kwds)


class StandardWriter(Writer):
    """Base class of snapshot writer which uses thread or process.

    This class creates a new thread or a process every time when ``__call__``
    is invoked.

    Args:
        savefun: Callable object. It takes three arguments: the output file
            path, the serialized dictionary object, and the optional keyword
            arguments.
        kwds: Keyword arguments for the ``savefun``.

    """
    def __init__(self, savefun=npz.save_npz, **kwds):
        self._savefun = savefun
        self._kwds = kwds
        self._flag = False

    def __call__(self, filename, outdir, target):
        if self._flag:
            self._worker.join()
        self._filename = filename
        self._worker = self.create_worker(filename, outdir, target,
                                          **self._kwds)
        self._worker.start()
        self._flag = True

    def create_worker(self, filename, outdir, target, **kwds):
        """Create a worker for the snapshot.

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
        if self._flag:
            self._worker.join()
        self._flag = False


class ThreadWriter(StandardWriter):
    """Writer that uses a thread.

    This class creates a new thread that invokes the actual saving function.

    """
    def create_worker(self, filename, outdir, target, **kwds):
        return threading.Thread(
            target=self.save,
            args=(filename, outdir, target, self._savefun),
            kwargs=self._kwds)


class ProcessWriter(StandardWriter):
    """Writer that uses a process.

    This class creates a new process that invokes the actual saving function.

    .. note::
        Forking a new process from a MPI process might be danger. Consider
        using ``ThreadWriter`` instead of ``ProcessWriter`` if you are using
        MPI.

    """
    def create_worker(self, filename, outdir, target, **kwds):
        return multiprocessing.Process(
            target=self.save,
            args=(filename, outdir, target, self._savefun),
            kwargs=self._kwds)


class QueueWriter(Writer):
    """Base class of queue snapshot writer.

    This class is a base class of writer that uses a queue. A Queue is created
    when this class is constructed, and every time when ``__call__`` is
    invoked, a snapshot task is put into the queue.

    Args:
        savefun: Callable object which is passed to the :meth:`create_task`
            if the task is ``None``. It takes three arguments: the output file
            path, the serialized dictionary object, and the optional keyword
            arguments.
        task: Callable object. Its ``__call__`` must have a same interface to
            ``Writer.__call__``. This object is directly put into the queue.

    """
    def __init__(self, savefun=npz.save_npz, task=None):
        if task is None:
            self._task = self.create_task(savefun)
        else:
            self._task = task
        self._queue = self.create_queue()
        self._consumer = self.create_consumer(self._queue)
        self._consumer.start()
        self._flag = True

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
        if self._flag:
            self._queue.put(None)
            self._queue.join()
            self._consumer.join()
        self._flag = False


class ThreadQueueWriter(QueueWriter):
    """Writer that uses thread queue.

    This class create a thread and a queue by `threading` and `queue` module
    respectively. The thread will be a consumer of the queue, and the main
    thread will be a producer of the queue.
    """
    def create_queue(self):
        return queue.Queue()

    def create_consumer(self, q):
        return threading.Thread(target=self.consume, args=(q,))


class ProcessQueueWriter(QueueWriter):
    """Writer that uses process queue.

    This class create a process and a queue by `multiprocessing` and `Queue`
    module respectively. The process will be a consumer of this queue,
    and the main process will be a producer of this queue.

    .. note::
        Forking a new process from MPI process might be danger. Consider using
        ``ThreadQueueWriter`` instead of ``ProcessQueueWriter`` if you are
        using MPI.

    """
    def create_queue(self):
        return multiprocessing.JoinableQueue()

    def create_consumer(self, q):
        return multiprocessing.Process(target=self.consume, args=(q,))
