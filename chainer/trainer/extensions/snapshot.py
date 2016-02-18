from __future__ import print_function
import os
import tempfile

from chainer.serializers import npz
from chainer.trainer import extension


class Snapshot(extension.Extension):

    """Trainer extension to take snapshots of the trainer.

    This extension serializes the trainer object and saves it to the output
    directory. It is used to support resuming the training loop.

    This extension is called once for each epoch by default.

    .. note::
       This extension first writes the serialized object to a temporary file
       and then rename it to the target file name. Thus, if the program stops
       right before the renaming, then the temporary file might be left in the
       output directory.

    Args:
        savefun: Function to save the trainer. It accepts two arguments: the
            output file path and the trainer object.
        filename (str): Name of the file into which the trainer is serialized.

    """
    trigger = 1, 'epoch'

    def __init__(self, savefun=npz.save_npz, filename='snapshot'):
        self.savefun = savefun
        self.filename = filename

    def __call__(self, trainer):
        fd, tmppath = tempfile.mkstemp(prefix=self.filename, dir=trainer.out)
        try:
            self.savefun(tmppath, trainer)
        finally:
            os.close(fd)
        os.rename(tmppath, os.path.join(trainer.out, self.filename))
