from __future__ import print_function
import os

from chainer.serializers import npz
from chainer.trainer import extension


class Snapshot(extension.Extension):

    """Extension to take snapshots of the trainer.

    TODO(beam2d): document it

    """
    default_trigger = 1, 'epoch'

    def __init__(self, savefun=npz.save_npz):
        self.savefun = savefun

    def __call__(self, out, trainer, t, **kwargs):
        path = os.path.join(out, 'snapshot')
        self.savefun(path, trainer)
