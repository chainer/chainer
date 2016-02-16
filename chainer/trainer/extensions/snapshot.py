from __future__ import print_function
import os
import tempfile

from chainer.serializers import npz
from chainer.trainer import extension


class Snapshot(extension.Extension):

    """Extension to take snapshots of the trainer.

    TODO(beam2d): document it

    """
    trigger = 1, 'epoch'

    def __init__(self, savefun=npz.save_npz, filename='snapshot'):
        self.savefun = savefun
        self.filename = filename

    def __call__(self, out, trainer, **kwargs):
        fd, tmppath = tempfile.mkstemp(prefix=self.filename, dir=out)
        try:
            self.savefun(tmppath, trainer)
        finally:
            os.close(fd)
        os.rename(tmppath, os.path.join(out, self.filename))
