import json
import os
import shutil

from chainer.serializers import npz
from chainer.training import extension
from chainer import utils



class _SnapshotExtension(extension.Extension):
    def __init__(self, trigger, priority, target, filename, savefun=npz.save_npz, loadfun=npz.load_npz, state_filename='snapshot_state.json'):
        self.trigger = trigger
        self.priority = priority
        self.target = target
        self.savefun = savefun
        self.loadfun = loadfun
        self.filename = filename
        self.state_filename = state_filename

    def initialize(self, trainer):
        # Automatically resume from latest snapshot.
        state_fn = os.path.join(trainer.out, self.state_filename)
        target = trainer if self.target is None else self.target
        if os.path.exists(state_fn):
            state = json.load(open(state_fn))
            fn = os.path.join(trainer.out, state['last_snapshot'])
            self.loadfun(fn, target)

    def __call__(self, trainer):
        # Save snapshot with meta data (snapshot_state.json)
        state_fn = os.path.join(trainer.out, self.state_filename)
        target = trainer if self.target is None else self.target
        fn = self.filename.format(trainer)
        _snapshot_object(trainer, target, fn, self.savefun)
        json.dump({'last_snapshot': fn}, open(state_fn, 'w'))


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
    return _SnapshotExtension((1, 'epoch'), -100, target, filename, savefun)


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
    return _SnapshotExtension((1, 'epoch'), -100, None, filename, savefun)


def _snapshot_object(trainer, target, filename, savefun):
    fn = filename.format(trainer)
    prefix = 'tmp' + fn

    with utils.tempdir(prefix=prefix, dir=trainer.out) as tmpdir:
        tmppath = os.path.join(tmpdir, fn)
        savefun(tmppath, target)
        shutil.move(tmppath, os.path.join(trainer.out, fn))
