from chainer.serializers import npz
from chainer.training import extension
from chainer.training.extensions import snapshot_writers
from chainer.utils import argument


def snapshot_object(target, filename, savefun=None, **kwargs):
    """snapshot_object(target, filename, savefun=None, \
*, condition=None, writer=None, snapshot_on_error=False)

    Returns a trainer extension to take snapshots of a given object.

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
        condition: Condition object. It must be a callable object that returns
            boolean without any arguments. If it returns ``True``, the snapshot
            will be done.
            If not, it will be skipped. The default is a function that always
            returns ``True``.
        writer: Writer object.
            It must be a callable object.
            See below for the list of built-in writers.
            If ``savefun`` is other than ``None``, this argument must be
            ``None``. In that case, a
            :class:`~chainer.training.extensions.snapshot_writers.SimpleWriter`
            object instantiated with specified ``savefun`` argument will be
            used.
        snapshot_on_error (bool): Whether to take a snapshot in case trainer
            loop has been failed.

    Returns:
        Snapshot extension object.

    .. seealso::

        - :meth:`chainer.training.extensions.snapshot`
    """

    return snapshot(target=target, filename=filename, savefun=savefun,
                    **kwargs)


def snapshot(savefun=None,
             filename='snapshot_iter_{.updater.iteration}', **kwargs):
    """snapshot(savefun=None, filename='snapshot_iter_{.updater.iteration}', \
*, target=None, condition=None, writer=None, snapshot_on_error=False)

    Returns a trainer extension to take snapshots of the trainer.

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
            It is :meth:`chainer.serializers.save_npz` by default.
            If ``writer`` is specified, this argument must be ``None``.
        filename (str): Name of the file into which the trainer is serialized.
            It can be a format string, where the trainer object is passed to
            the :meth:`str.format` method.
        target: Object to serialize. If it is not specified, it will
            be the trainer object.
        condition: Condition object. It must be a callable object that returns
            boolean without any arguments. If it returns ``True``, the snapshot
            will be done.
            If not, it will be skipped. The default is a function that always
            returns ``True``.
        writer: Writer object.
            It must be a callable object.
            See below for the list of built-in writers.
            If ``savefun`` is other than ``None``, this argument must be
            ``None``. In that case, a
            :class:`~chainer.training.extensions.snapshot_writers.SimpleWriter`
            object instantiated with specified ``savefun`` argument will be
            used.
        snapshot_on_error (bool): Whether to take a snapshot in case trainer
            loop has been failed.

    Returns:
        Snapshot extension object.

    .. testcode::
       :hide:

       from chainer import training
       class Model(chainer.Link):
           def __call__(self, x):
               return x
       train_iter = chainer.iterators.SerialIterator([], 1)
       optimizer = optimizers.SGD().setup(Model())
       updater = training.updaters.StandardUpdater(
           train_iter, optimizer, device=0)
       trainer = training.Trainer(updater)

    .. admonition:: Using asynchronous writers

        By specifying ``writer`` argument, writing operations can be made
        asynchronous, hiding I/O overhead of snapshots.

        >>> from chainer.training import extensions
        >>> writer = extensions.snapshot_writers.ProcessWriter()
        >>> trainer.extend(extensions.snapshot(writer=writer), \
trigger=(1, 'epoch'))

        To change the format, such as npz or hdf5, you can pass a saving
        function as ``savefun`` argument of the writer.

        >>> from chainer.training import extensions
        >>> from chainer import serializers
        >>> writer = extensions.snapshot_writers.ProcessWriter(
        ...     savefun=serializers.save_npz)
        >>> trainer.extend(extensions.snapshot(writer=writer), \
trigger=(1, 'epoch'))

    This is the list of built-in snapshot writers.

        - :class:`chainer.training.extensions.snapshot_writers.SimpleWriter`
        - :class:`chainer.training.extensions.snapshot_writers.ThreadWriter`
        - :class:`chainer.training.extensions.snapshot_writers.ProcessWriter`
        - :class:`chainer.training.extensions.snapshot_writers.\
ThreadQueueWriter`
        - :class:`chainer.training.extensions.snapshot_writers.\
ProcessQueueWriter`

    .. seealso::

        - :meth:`chainer.training.extensions.snapshot_object`
    """
    target, condition, writer, snapshot_on_error = argument.parse_kwargs(
        kwargs,
        ('target', None), ('condition', None), ('writer', None),
        ('snapshot_on_error', False))
    argument.assert_kwargs_empty(kwargs)

    if savefun is not None and writer is not None:
        raise TypeError(
            'savefun and writer arguments cannot be specified together.')

    if writer is None:
        if savefun is None:
            savefun = npz.save_npz
        writer = snapshot_writers.SimpleWriter(savefun=savefun)

    return _Snapshot(
        target=target, condition=condition, writer=writer, filename=filename,
        snapshot_on_error=snapshot_on_error)


def _always_true():
    return True


class _Snapshot(extension.Extension):
    """Trainer extension to take snapshots.

    This extension serializes the given object and saves it to the output
    directory.

    This extension is called once per epoch by default. To take a
    snapshot at a different interval, a trigger object specifying the
    required interval can be passed along with this extension
    to the `extend()` method of the trainer.

    The default priority is -100, which is lower than that of most
    built-in extensions.
    """
    trigger = 1, 'epoch'
    priority = -100

    def __init__(
            self, target=None, condition=None, writer=None,
            filename='snapshot_iter_{.updater.iteration}',
            snapshot_on_error=False):
        if condition is None:
            condition = _always_true
        if writer is None:
            writer = snapshot_writers.SimpleWriter()
        self._target = target
        self.filename = filename
        self.condition = condition
        self.writer = writer
        self._snapshot_on_error = snapshot_on_error

    def on_error(self, trainer, exc, tb):
        super(_Snapshot, self).on_error(trainer, exc, tb)
        if self._snapshot_on_error:
            self._make_snapshot(trainer)

    def __call__(self, trainer):
        if self.condition():
            self._make_snapshot(trainer)

    def _make_snapshot(self, trainer):
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
