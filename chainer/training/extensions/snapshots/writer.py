import multiprocessing
import os
import queue
import shutil
import tempfile
import threading

from chainer.training.extensions.snapshots import util


class Writer(object):
    """Base class of snapshot writer.

    :class:`Snapshot` invokes this :class:`Writer` object's :func:`__call__`
    everytime if this process is in charge to take a snapshot.
    This class determine how to invoke the actual saving function. The most
    simple way is just passing the arguments to the saving function.
    """

    def __init__(self):
        pass

    def __call__(self, filename, outdir, target):
        """Invokes the actual snapshot function.

        This method is invoked every time when this process is in charge to
        take a snapshot. All arguments are passed from :class:`Snapshot`
        object.

        Args:
            filename (str): Name of the file into which the serialized target
                is saved. It is already formated string.
            outdir (str): Output directory. Passed by `trainer.out`.
            target (dict): Serialized object which will be saved.
        """
        pass

    def finalize(self):
        """Finalize the writer.

        Like extensions in :class:`Trainer`, this method is invoked at the
        end of training.
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
        savefun: Callable object. Must have same interface to `Writer.__call__`
            method.
    """

    def __init__(self, savefun=util.save_npz, **kwds):
        self._savefun = savefun
        self._kwds = kwds

    def __call__(self, filename, outdir, target):
        self.save(filename, outdir, target, self._savefun, **self._kwds)


class ThreadWriter(Writer):
    """Writer that uses a thread.

    This class creates a thread that invokes the actual saving function.
    A new thread is created every time `__call__` is invoked.

    Args:
        savefun: Callable object. Must have same interface to `Writer.__call__`
            method. Also must need to be able to passed to a thread.
    """

    def __init__(self, savefun=util.save_npz, **kwds):
        self._savefun = savefun
        self._kwds = kwds
        self._flag = False

    def __call__(self, filename, outdir, target):
        self._thread = threading.Thread(
            target=self.save,
            args=(filename, outdir, target, self._savefun),
            kwargs=self._kwds)
        self._thread.start()
        self._flag = True

    def finalize(self):
        if self._flag:
            self._thread.join()


class ProcessWriter(Writer):
    """Writer that uses a process.

    This class creates a process that invokes the actual saving function.
    A new process is created every time `__call__` is invoked.

    .. note::
        Forking or spawing a new process from MPI process might be danger.
        Consider using `ThreadWriter` instead of `ProcessWriter` if you are
        using MPI in ChainerMN.

    Args:
        savefun: Callable object. Must have same interface to `Writer.__call__`
            method. Also must need to be able to passed to a thread.
    """

    def __init__(self, savefun=util.save_npz, **kwds):
        self._savefun = savefun
        self._kwds = kwds
        self._flag = False

    def __call__(self, filename, outdir, target):
        self._process = multiprocessing.Process(
            target=self.save,
            args=(filename, outdir, target, self._savefun),
            kwargs=self._kwds)
        self._process.start()
        self._flag = True

    def finalize(self):
        if self._flag:
            self._process.join()


class QueueWriter(Writer):
    """Base class of queue snapshot writer.

    This class a base class of writer that uses a queue. A Queue is created
    when this class is constructed, and every time when `__call__` is
    invoked, a snapshot task is put into the queue.

    Args:
        task: Callable object. Its `__call__` must have a same interface to
            `Writer.__call__`. This object is put into the queue.
    """

    def __init__(self, task=SimpleWriter()):
        raise NotImplementedError

    def __call__(self, filename, outdir, target):
        self._queue.put([self._task, filename, outdir, target])

    def consume(self, queue):
        while True:
            task = queue.get()
            if task is None:
                queue.task_done()
                return
            else:
                task[0](task[1], task[2], task[3])
                queue.task_done()

    def finalize(self):
        self._queue.put(None)
        self._queue.join()
        self._consumer.join()


class ThreadQueueWriter(QueueWriter):
    """Writer that uses thread queue.

    This class create a thread and a queue by `threading` and `Queue` module
    respectively. The thread will be a consumer of this queue, and the main
    thread will be a producer of this queue.

    Args:
        task: Callable object. Its `__call__` must have a same interface to
            `Writer.__call__`. This object is put into the queue.
    """

    def __init__(self, task=SimpleWriter()):
        self._task = task
        self._queue = queue.Queue()
        self._consumer = threading.Thread(target=self.consume,
                                          args=(self._queue,))
        self._consumer.start()


class ProcessQueueWriter(QueueWriter):
    """Writer that uses process queue.

    This class create a process and a queue by `multiprocessing` and `Queue`
    module respectively. The process will be a consumer of this queue,
    and the main process will be a producer of this queue.

    .. note::
        Forking or spawing a new process from MPI process might be danger.
        Consider using `ThreadWriter` instead of `ProcessWriter` if you are
        using MPI in ChainerMN.

    Args:
        task: Callable object. Its `__call__` must have a same interface to
            `Writer.__call__`. This object is put into the queue.
    """

    def __init__(self, task=SimpleWriter()):
        self._task = task
        self._queue = multiprocessing.JoinableQueue()
        self._consumer = multiprocessing.Process(target=self.consume,
                                                 args=(self._queue,),
                                                 daemon=True)
        self._consumer.start()
