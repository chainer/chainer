from __future__ import print_function
import collections
import os

import six

from chainer import cuda
from chainer.trainer import interval_trigger
from chainer.trainer import extension as extension_module
from chainer.trainer.extensions import log_result
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

        self._extensions = collections.OrderedDict()
        self._extension_priorities = {}

        self._iter = dataset.get_batch_iterator(batchsize, device=device)

        optimizer.setup(target)

    def extend(self, extension, trigger=None, name=None,
               invoke_before_training=None, priority=None):
        if name is None:
            name = getattr(extension, 'default_name', None)
        if trigger is None:
            trigger = getattr(extension, 'default_trigger', None)

        if name is None:
            raise TypeError('name is not given for the extension')
        if name == 'training':
            raise ValueError('the name "training" is reserved')

        if isinstance(trigger, tuple):
            trigger = interval_trigger.IntervalTrigger(*trigger)
        if not callable(trigger):
            raise TypeError('trigger must be a tuple or a callable')

        if invoke_before_training is None:
            invoke_before_training = getattr(
                extension, 'invoke_before_training', False)

        if priority is None:
            priority = extension.priority
        self._extensions[name] = trigger, invoke_before_training, extension
        self._extension_priorities[name] = priority

    def get_extension(self, name):
        if name in self._extensions:
            return self._extensions[name][2]
        raise ValueError('extension {} not found'.format(name))

    def run(self, out='result'):
        try:
            os.makedirs(out)
        except:
            pass

        epoch = self._iter.epoch
        t = self.optimizer.t

        extension_order = sorted(
            self._extensions.keys(),
            key=lambda name: self._extension_priorities[name],
            reverse=True)

        args = {'epoch': epoch, 'new_epoch': False, 'out': out, 'result': {},
                't': t, 'trainer': self}
        for name in extension_order:
            _, invoke_before_training, extension = self._extensions[name]
            if invoke_before_training:
                extension(**args)

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
            args = {'epoch': epoch, 'new_epoch': new_epoch, 'out': out,
                    'result': result, 't': t, 'trainer': self}
            for name in extension_order:
                trigger, _, extension = self._extensions[name]
                if trigger(**args):
                    r = extension(**args)
                    if r is not None:
                        result[name] = r

            # use < in order to support None (== endless loop)
            if ((self._max_epoch is not None and epoch >= self._max_epoch) or
                (self._max_iter is not None and t >= self._max_iter)):
                break

        self._iter.finalize()

    def serialize(self, serializer):
        self.target.serialize(serializer['target'])
        self.optimizer.serialize(serializer['optimizer'])
        self._iter.serialize(serializer['iter'])

        if hasattr(self.updater, 'serialize'):
            self.updater.serialize(serializer['updater'])

        s = serializer['extensions']
        t = serializer['_extension_triggers']
        for name, (trigger, _, extension) in six.iteritems(self._extensions):
            if hasattr(extension, 'serialize'):
                extension.serialize(s[name])
            if hasattr(trigger, 'serialize'):
                trigger.serialize(t[name])


def create_standard_trainer(
        dataset, target, optimizer, updater=None, batchsize=1,
        epoch=None, iteration=None, device=None):
    tr = Trainer(dataset, target, optimizer, updater, batchsize,
                 epoch, iteration, device)
    tr.extend(log_result.LogResult())
    tr.extend(print_result.PrintResult())
    tr.extend(snapshot.Snapshot())
    return tr
