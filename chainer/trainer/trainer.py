from __future__ import print_function
import collections

import six

from chainer import cuda
from chainer.trainer import interval_trigger
import chainer.trainer.extension as extension_module
from chainer.trainer.extensions import print_result
from chainer.trainer.extensions import snapshot
import chainer.trainer.updater as updater_module


class Trainer(object):

    """Implementation of the standard training loop.

    TODO(beam2d): document it.

    """
    def __init__(self, dataset, target, optimizer,
                 updater=None, batchsize=1, epoch=None, iteration=None,
                 shuffle=True):
        if updater is None:
            updater = updater_module.StandardUpdater()

        self._dataset = dataset
        self._target = target
        self._optimizer = optimizer
        self._updater = updater
        self._max_epoch = epoch
        self._max_iteration = iteration
        self._device = -1  # cpu

        self._write_extensions = {}
        self._edit_extensions = {}
        self._read_extensions = {}

        self._iter = dataset.get_batch_iter(batchsize, shuffle)

    def to_cpu(self):
        self._target.to_cpu()
        self._device = -1

    def to_gpu(self, device=None):
        self._target.to_gpu(device)
        self._device = cuda.get_device(device).id

    def extend(self, extension, trigger=None, name=None):
        if isinstance(extension, extension_module.Extension):
            if name is None:
                name = extension.default_name
            if trigger is None:
                trigger = extension.default_trigger
            if isinstance(trigger, tuple):
                trigger = interval_trigger.IntervalTrigger(*trigger)

        if name is None:
            raise TypeError('name is not given for the extension')
        if name == 'training':
            raise ValueError('the name "training" is reserved')
        if not callable(trigger):
            raise TypeError('trigger must be callable')

        if extension.result_action == 'write':
            self._write_extensions[name] = trigger, extension
        elif extension.result_action == 'edit':
            self._edit_extensions[name] = trigger, extension
        elif extension.result_action == 'read':
            self._read_extensions[name] = trigger, extension
        else:
            raise ValueError(
                'result_action must be either of write, edit, or read')

    def run(self, out=None):
        if out is None:
            out = self._get_default_out_name()

        update = self._updater.update
        for inputs in self._iter:
            # TODO(beam2d): better device handling
            if self._device >= 0:
                with cuda.get_device(self._device):
                    inputs = tuple(cuda.to_gpu(x) for x in inputs)
            epoch = self._iter.epoch
            train_result = update(inputs, self._target, self._optimizer)
            is_new_epoch = epoch != self._iter.epoch
            if is_new_epoch:
                self._model.new_epoch()
            t = self._model.t

            result = collections.OrderedDict(training=updater_result)

            args = {'epoch': epoch, 'new_epoch': is_new_epoch,
                    'out': out, 't': t, 'trainer': self}
            for name, (trigger, extension) in self._write_extensions:
                if trigger(**args):
                    r = extension(**args)
                    if r is not None:
                        result[name] = r

            ext_args = args.copy()
            ext_args['result'] = result
            for extensions in self._edit_extensions, self._read_extensions:
                for name, (trigger, extension) in extensions:
                    if trigger(**args):
                        extension(**ext_args)

            # use < in order to support None (== endless loop)
            if not (epoch < self._max_epoch and t < self._max_iteration):
                break

    def serialize(self, serializer):
        self._target.serialize(serializer['_target'])
        self._optimizer.serialize(serializer['_optimizer'])
        self._iter.serialize(serializer['_iter'])

        wextensions = serializer['_write_extensions']
        eextensions = serializer['_edit_extensions']
        rextensions = serializer['_read_extensions']
        for extensions, s in (
                (self._write_extensions, serializer['_write_extensions']),
                (self._edit_extensions, serializer['_edit_extensions']),
                (self._read_extensions, serializer['_read_extensions'])):
            t = s['_trigger']
            for name, trigger, extension in extensions:
                extension.serialize(s[name])
                if hasattr(trigger, 'serialize'):
                    trigger.serialize(t[name])

    def _get_default_out_name(self):
        dataset = type(self._dataset).__name__
        target = type(self._target).__name__
        optimizer = type(self._optimizer).__name__
        return 'result-{}-{}-{}'.format(dataset, target, optimizer)


def create_standard_trainer(
        dataset, target, optimizer, updater=None, batchsize=1,
        epoch=None, iteration=None, suffle=True):
    tr = Trainer(dataset, target, optimizer, updater, batchsize, epoch,
                 iteration, shuffle)
    tr.extend(print_result.PrintResult())
    tr.extend(snapshot.Snapshot())
    return tr
