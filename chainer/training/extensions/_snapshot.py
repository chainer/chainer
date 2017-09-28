import os
import shutil
import tempfile

from chainer.serializers import npz
from chainer.training import extension
from chainer.training.extensions.snapshots import condition as condition_module
from chainer.training.extensions.snapshots import util
from chainer.training.extensions.snapshots import writer as writer_module


def snapshot_object(target, filename, savefun=npz.save_npz):
    """Returns a trainer extension to take snapshots of a given object.

    This extension serializes the given object and saves it to the output
    directory.

    This extension is called once per epoch by default. To take a
    snapshot at a different interval, a trigger object specifying the
    required interval can be passed along with this extension
    to the `extend()` method of the trainer.

    The default priority is -100, which is lower than that of most
    built-in extensions.

    Args:
        target: Object to serialize.
        filename (str): Name of the file into which the object is serialized.
            It can be a format string, where the trainer object is passed to
            the :meth:`str.format` method. For example,
            ``'snapshot_{.updater.iteration}'`` is converted to
            ``'snapshot_10000'`` at the 10,000th iteration.
        savefun: Function to save the object. It takes two arguments: the
            output file path and the object to serialize.

    Returns:
        An extension function.

    """
    @extension.make_extension(trigger=(1, 'epoch'), priority=-100)
    def snapshot_object(trainer):
        _snapshot_object(trainer, target, filename.format(trainer), savefun)

    return snapshot_object


def snapshot(savefun=npz.save_npz,
             filename='snapshot_iter_{.updater.iteration}'):
    """Returns a trainer extension to take snapshots of the trainer.

    This extension serializes the trainer object and saves it to the output
    directory. It is used to support resuming the training loop from the saved
    state.

    This extension is called once per epoch by default. To take a
    snapshot at a different interval, a trigger object specifying the
    required interval can be passed along with this extension
    to the `extend()` method of the trainer.

    The default priority is -100, which is lower than that of most
    built-in extensions.

    .. note::
       This extension first writes the serialized object to a temporary file
       and then rename it to the target file name. Thus, if the program stops
       right before the renaming, the temporary file might be left in the
       output directory.

    Args:
        savefun: Function to save the trainer. It takes two arguments: the
            output file path and the trainer object.
        filename (str): Name of the file into which the trainer is serialized.
            It can be a format string, where the trainer object is passed to
            the :meth:`str.format` method.

    """
    @extension.make_extension(trigger=(1, 'epoch'), priority=-100)
    def snapshot(trainer):
        _snapshot_object(trainer, trainer, filename.format(trainer), savefun)

    return snapshot


def _snapshot_object(trainer, target, filename, savefun):
    fn = filename.format(trainer)
    prefix = 'tmp' + fn
    fd, tmppath = tempfile.mkstemp(prefix=prefix, dir=trainer.out)
    try:
        savefun(tmppath, target)
    except Exception:
        os.close(fd)
        os.remove(tmppath)
        raise
    os.close(fd)
    shutil.move(tmppath, os.path.join(trainer.out, fn))


class Snapshot(extension.Extension):
    """Takes a snapshot.

    Args:
        target: Object to serialize. If not specified, it will
            be trainer object.
        condition: Condition object. It must be a callable object that
            returns boolean in its call. If it returns True the snapshot will
            be done. If not it will be skipped.
        writer: Writer object. It need to be a callable object.
        filename (str): Name of the file into which the object is serialized.
            It can be a format string, where the trainer object is passed to
            the :meth:`str.format` method. For example,
            ``'snapshot_{.updater.iteration}'`` is converted to
            ``'snapshot_10000'`` at the 10,000th iteration.
            Also it can be a callable object, where the trainer is passed to
            that method.

    """

    def __init__(self,
                 target=None,
                 condition=condition_module.Always(),
                 writer=writer_module.SimpleWriter(),
                 filename='snapshot_iter_{.updater.iteration}'):
        self._target = target
        self.filename = filename
        self.condition = condition
        self.writer = writer

    def __call__(self, trainer):
        if self.condition(trainer, self):
            target = trainer if self._target is None else self._target
            serialized_target = util.serialize(target)
            filename = self.filename
            if callable(filename):
                filename = filename(trainer)
            else:
                filename = filename.format(trainer)
            outdir = trainer.out
            self.writer(filename, outdir, serialized_target)

    def finalize(self):
        if hasattr(self.writer, 'finalize'):
            self.writer.finalize()
