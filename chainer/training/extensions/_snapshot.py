from chainer.serializers import npz
from chainer.training import extension
from chainer.training import writer


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
        Snapshot extension object.

    """
    return Snapshot(
        target=target,
        writer=writer.SimpleWriter(savefun=savefun),
        filename=filename)


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

    Returns:
        Snapshot extension object.

    """
    return Snapshot(
        writer=writer.SimpleWriter(savefun=savefun),
        filename=filename)


class Snapshot(extension.Extension):
    """Trainer extension to take snapshots.

    This extension serializes the given object and saves it to the output
    directory.

    This extension is called once per epoch by default. To take a
    snapshot at a different interval, a trigger object specifying the
    required interval can be passed along with this extension
    to the `extend()` method of the trainer.

    The default priority is -100, which is lower than that of most
    built-in extensions.

    Args:
        target: Object to serialize. If it is not specified, it will
            be the trainer object.
        condition: Condition object. It must be a callable object that returns
            boolean without any arguments. If it returns True the snapshot will
            be done. If not it will be skipped. The default is a method that
            always returns True.
        writer: Writer object. It must be a callable object.
        filename (str): Name of the file into which the object is serialized.
            It can be a format string, where the trainer object is passed to
            the :meth:`str.format` method. For example,
            ``'snapshot_{.updater.iteration}'`` is converted to
            ``'snapshot_10000'`` at the 10,000th iteration.
            Also it can be a callable object, where the trainer is passed as
            an argument.

    """
    trigger = 1, 'epoch'
    priority = -100

    def __init__(self,
                 target=None,
                 condition=lambda: True,
                 writer=writer.SimpleWriter(),
                 filename='snapshot_iter_{.updater.iteration}'):
        self._target = target
        self.filename = filename
        self.condition = condition
        self.writer = writer

    def __call__(self, trainer):
        if self.condition():
            target = trainer if self._target is None else self._target
            serialized_target = npz.serialize(target)
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
