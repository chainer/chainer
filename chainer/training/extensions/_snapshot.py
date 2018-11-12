import os
import shutil

from chainer.serializers import npz
from chainer.training import extension
from chainer import utils


def snapshot_object(target, filename, savefun=npz.save_npz, **kwargs):
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
        snapshot_on_error (bool): Whether to take a snapshot in case trainer
            loop has been failed.

    Returns:
        An extension function.

    """

    snapshot_on_error = utils.argument.parse_kwargs(
        kwargs, ('snapshot_on_error', False))
    error_handler = None
    if snapshot_on_error:
        def h(trainer, exception, exc_info):
            _snapshot_object(trainer, trainer, filename.format(trainer),
                             savefun)
        error_handler = h

    @extension.make_extension(trigger=(1, 'epoch'), priority=-100,
                              on_error=error_handler)
    def snapshot_object(trainer):
        _snapshot_object(trainer, target, filename.format(trainer),
                         savefun)

    return snapshot_object


def snapshot(savefun=npz.save_npz,
             filename='snapshot_iter_{.updater.iteration}', **kwargs):
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
        snapshot_on_error (bool): Whether to take a snapshot in case trainer
            loop has been failed.

    """
    snapshot_on_error = utils.argument.parse_kwargs(
        kwargs, ('snapshot_on_error', False))

    error_handler = None
    if snapshot_on_error:
        def h(trainer, exception, exc_info):
            _snapshot_object(trainer, trainer, filename.format(trainer),
                             savefun)
        error_handler = h

    @extension.make_extension(trigger=(1, 'epoch'), priority=-100,
                              on_error=error_handler)
    def snapshot(trainer):
        _snapshot_object(trainer, trainer, filename.format(trainer),
                         savefun)

    return snapshot


def _snapshot_object(trainer, target, filename, savefun):
    fn = filename.format(trainer)
    prefix = 'tmp' + fn

    with utils.tempdir(prefix=prefix, dir=trainer.out) as tmpdir:
        tmppath = os.path.join(tmpdir, fn)
        savefun(tmppath, target)
        shutil.move(tmppath, os.path.join(trainer.out, fn))
