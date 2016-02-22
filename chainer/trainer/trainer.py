from __future__ import print_function
import collections
import os

import six

from chainer.trainer.extensions import log_result
from chainer.trainer.extensions import print_result
from chainer.trainer.extensions import snapshot
from chainer.trainer import interval_trigger
from chainer.trainer import updater as updater_module


class Trainer(object):

    """The standard training loop.

    Trainer is an implementation of a training loop in Chainer. The training
    loop is implemented in the :meth:`run` method. Whole training process is
    done just by calling this method.

    Each iteration of the training loop proceeds as follows.

    - Fetch the next minibatch from the batch iterator. The batch iterator is
      created by the training dataset before entering the loop.
    - Update the model parameters using the minibatch. This is actually done by
      an `updater` function.
    - Invoke extensions in descending order of their priorities. Actually, each
      extension may be skipped depending on the decision made by the
      corresponding `trigger` object. Here trigger objects are callable objects
      that accept the trainer object as the argument and return a boolean
      value.

    Trainer manages objects involved in the training process: updater function,
    target chain, optimizer, and extensions. All these objects support
    serialization, and the Trainer class itself also supports it. It enables us
    to easily resume the training loop from a snapshot. Use the
    :class:`~chainer.trainer.extensions.Snapshot` extension to periodically
    save snapshots of the training loop.

    .. note::
       The serialization does not recover everything of the training loop. It
       only recovers the states which change over the training (e.g.
       parameters, optimizer states, the batch iterator state, extension
       states, etc.). You must initialize the objects correctly before
       deserializing the states.

    The Trainer class is `plain`, i.e. does not contain any extensions by
    default. The :func:`create_standard_trainer` function creates a trainer
    with convenient extensions already set up, which suits typical use cases.

    Args:
        dataset (Dataset): Training dataset.
        target (Link): Target link.
        optimizer (Optimizer): Optimizer.
        updater: Updater function. This is a :class:`~chainer.trainer.Updater`
            object or a callable object with the same ``__call__`` interface.
            If this object supports serialization, then the :meth:`serialize`
            method also (de)serializes the updater object.
        batchsize (int): Number of data points in each minibatch.
        epoch (int): Number of loops to iterate over the training dataset.
            If ``iteration`` and this argument is both None, then the
            :meth:`run` method enters an infinite loop.
        iteration (int): Number of updates to operate. If ``epoch`` and this
            argument is both None, then the :meth:`run` method enters an
            infinite loop.
        device: On which device to send each minibatch. This argument is just
            passed to the :meth:`Dataset.get_batch_iterator` method to create
            a batch iterator.

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
        """Register an extension to the trainer.

        :class:`Extension` is a callable object which is called after each
        update unless the corresponding trigger object decides to skip the
        iteration. The order of execution is determined by priorities:
        extensions with higher priorities are called earlier in each iteration.
        Extensions with the same priority are invoked in the order of
        registerations.

        See :class:`Extension` for the interface of extensions.

        Args:
            extension (~chainer.trainer.Extension): Extension to register.
            trigger (tuple or Trigger): Trigger object that determines when to
                invoke the extension. If it is None, then ``extension.trigger``
                is used instead. If the trigger is a tuple of an integer and
                a string ``'iteration'`` or ``'epoch'``, then
                :class:`~chainer.trainer.IntervalTrigger` is built from the
                tuple.
            name (str): Name of the extension. If it is None, then
                ``extension.name`` is used instead. This argument must be given
                if the extension does not have the ``name`` attribute.
            invoke_before_training (bool): If True, then the extension is also
                invoked right before entering the training loop. If this is
                None, then ``extension.invoke_before_trainng`` is used instead.
                This option is used for extensions that alter the training
                configuration: in such a case, resuming from snapshots require
                the call of extension to recover the configuration before any
                updates.
            priority (int): Invocation priority of the extension. Extensions
                are invoked in the descending order of priorities in each
                iteration. If this is None, then ``extension.priority`` is used
                instead.

        Returns:
            self.

        """
        if name is None:
            name = getattr(extension, 'name', None)
        if trigger is None:
            trigger = getattr(extension, 'trigger', None)

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

        return self

    def get_extension(self, name):
        """Returns the extension of a given name.

        Args:
            name (str): Name of the extension.

        Returns:
            Extension.

        """
        if name in self._extensions:
            return self._extensions[name][2]
        raise ValueError('extension {} not found'.format(name))

    def run(self, out='result'):
        """Executes the training loop.

        This method is the core of Trainer. It enters a long loop of training
        the target link.

        Args:
            out (str): Output directory. All the results are put under this
                directory.

        """
        try:
            os.makedirs(out)
        except OSError:
            pass

        self.epoch = self._iter.epoch
        self.new_epoch = False
        self.out = out
        self.result = collections.OrderedDict()
        self.t = self.optimizer.t

        extension_order = sorted(
            self._extensions.keys(),
            key=lambda name: self._extension_priorities[name],
            reverse=True)

        for name in extension_order:
            _, invoke_before_training, extension = self._extensions[name]
            if invoke_before_training:
                extension(self)

        for inputs in self._iter:
            train_result = self.updater(inputs, self.optimizer)

            if self.t == self.optimizer.t:  # no update happens
                continue
            self.t = self.optimizer.t

            self.new_epoch = self.epoch != self._iter.epoch
            self.epoch = self._iter.epoch
            if self.new_epoch:
                self.optimizer.new_epoch()

            self.result.clear()
            self.result['training'] = train_result
            for name in extension_order:
                trigger, _, extension = self._extensions[name]
                if trigger(self):
                    r = extension(self)
                    if r is not None:
                        self.result[name] = r

            # use < in order to support None (== endless loop)
            if ((self._max_epoch is not None and
                 self.epoch >= self._max_epoch) or
                (self._max_iter is not None and
                 self.t >= self._max_iter)):
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
    """Creates a trainer object with convenient extensions preinstalled.

    This is a convenient function to prepare a trainer object for typiecal use
    cases. The following extensions are preinstalled:

    - :class:`~chainer.trainer.extensions.LogResult`
    - :class:`~chainer.trainer.extensions.PrintResult`
    - :class:`~chainer.trainer.extensions.Snapshot`

    .. warning::
       The above list of preinstalled extensions are OUT OF COMPATIBILITY
       SUPPORT. It may be changed at any future releases of Chainer.

    See :class:`Trainer` for further details.

    Args:
        dataset (Dataset): Training dataset.
        target (Link): Target link.
        optimizer (Optimizer): Optimizer.
        updater: Updater function.
        batchsize (int): Number of data points in each minibatch.
        epoch (int): Number of loops to iterate over the datasets.
        iteration (int): Number of updates to operate.
        device: On which device to send each minibatch.

    Returns:
        Trainer: a trainer object with useful extensions installed.

    """
    tr = Trainer(dataset, target, optimizer, updater, batchsize,
                 epoch, iteration, device)
    tr.extend(log_result.LogResult())
    tr.extend(print_result.PrintResult())
    tr.extend(snapshot.Snapshot())
    return tr
