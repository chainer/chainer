from __future__ import print_function
import collections
import os

import six

from chainer import cuda
from chainer.trainer import interval_trigger
from chainer.trainer import extension as extension_module
from chainer.trainer.extensions import print_result
from chainer.trainer.extensions import snapshot
from chainer.trainer import updater as updater_module


class Trainer(object):

    """Implementation of the standard training loop.

    TODO(beam2d): document it.

    """
    def __init__(self, dataset, target, optimizer,
                 updater=None, batchsize=1, epoch=None, iteration=None,
                 device=None):
        if updater is None:
            updater = updater_module.StandardUpdater()

        self.target = target
        self.optimizer = optimizer
        self.updater = updater
        self._max_epoch = epoch
        self._max_iter = iteration

        self._write_extensions = {}
        self._edit_extensions = {}
        self._read_extensions = {}

        self._iter = dataset.get_batch_iterator(batchsize, device=device)

        optimizer.setup(target)

    def extend(self, extension, trigger=None, name=None):
        if isinstance(extension, extension_module.Extension):
            if name is None:
                name = extension.default_name
            if trigger is None:
                trigger = extension.default_trigger

        if name is None:
            raise TypeError('name is not given for the extension')
        if name == 'training':
            raise ValueError('the name "training" is reserved')

        if isinstance(trigger, tuple):
            trigger = interval_trigger.IntervalTrigger(*trigger)
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

    def get_extension(self, name):
        for extensions in (self._write_extensions, self._edit_extensions,
                           self._read_extensions):
            if name in extensions:
                return extensions[name][1]
        raise ValueError('extension {} not found'.format(name))

    def run(self, out='result'):
        try:
            os.makedirs(out)
        except:
            pass

        epoch = self._iter.epoch
        t = self.optimizer.t
        for inputs in self._iter:
            train_result = self.updater(inputs, self.target, self.optimizer)

            if t == self.optimizer.t:  # no update happens
                continue
            t = self.optimizer.t

            new_epoch = epoch != self._iter.epoch
            epoch = self._iter.epoch
            if new_epoch:
                self.optimizer.new_epoch()

            result = collections.OrderedDict(training=train_result)

            args = {'epoch': epoch, 'new_epoch': new_epoch, 'out': out, 't': t,
                    'trainer': self}
            for name, (trigger, extension) in (
                    six.iteritems(self._write_extensions)):
                if trigger(**args):
                    r = extension(**args)
                    if r is not None:
                        result[name] = r

            ext_args = args.copy()
            ext_args['result'] = result
            for extensions in self._edit_extensions, self._read_extensions:
                for name, (trigger, extension) in six.iteritems(extensions):
                    if trigger(**args):
                        extension(**ext_args)

            # use < in order to support None (== endless loop)
            if ((self._max_epoch is not None and epoch >= self._max_epoch) or
                (self._max_iter is not None and t >= self._max_iter)):
                break

    def serialize(self, serializer):
        self.target.serialize(serializer['target'])
        self.optimizer.serialize(serializer['optimizer'])
        self._iter.serialize(serializer['iter'])

        if hasattr(self.updater, 'serialize'):
            self.updater.serialize(serializer['updater'])

        for extensions, s in (
                (self._write_extensions, serializer['write_extensions']),
                (self._edit_extensions, serializer['edit_extensions']),
                (self._read_extensions, serializer['read_extensions'])):
            t = s['_trigger']
            for name, (trigger, extension) in six.iteritems(extensions):
                extension.serialize(s[name])
                if hasattr(trigger, 'serialize'):
                    trigger.serialize(t[name])


def create_standard_trainer(
        dataiter, target, optimizer, updater=None, batchsize=1,
        epoch=None, iteration=None, device=None):
    tr = Trainer(dataiter, target, optimizer, updater, batchsize,
                 epoch, iteration, device)
    tr.extend(print_result.PrintResult())
    tr.extend(snapshot.Snapshot())
    return tr
