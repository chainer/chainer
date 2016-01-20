from __future__ import print_function
import os

from chainer.serializers import hdf5  # TODO(beam2d): use npz
from chainer.trainer import extension


class Snapshot(extension.Extension):

    """Extension to take snapshots of the trainer.

    TODO(beam2d): document it

    """
    default_trigger = 1, 'epoch'

    def __init__(self, savefun=hdf5.save_hdf5):
        self._save = savefun

    def __call__(self, out, trainer, t, **kwargs):
        root = os.path.join(out, 'snapshot')
        path = os.path.join(root, 'snapshot_t=%d' % t)
        try:
            os.makedirs(root)
        except:
            pass
        self._save(path, trainer)
        print('saved snapshot to', os.path.abspath(path))
