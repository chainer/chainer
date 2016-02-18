import collections
import json
import os
import tempfile

import six

import chainer.serializer as serializer_module
from chainer.trainer import extension
from chainer.trainer import interval_trigger
import chainer.utils.summary as summary_module


class LogResult(extension.Extension):

    """Trainer extension to output the accumulated result to a log file.

    This extension accumulates the result dictionaries given by an updater and
    other extensions, and periodically dumps the mean values to the log file in
    the JSON format. It also adds ``'epoch'`` and ``'iteration'`` entry to the
    result.

    .. note::
       This extension should be called every iteration in order to collect all
       the result dictionaries emitted by the updater.

    Args:
        keys (iterable of strs): Which values to accumulate. If this is None,
            then all the values are accumulated and output to the log file.
        trigger: Trigger that decides when to aggregate the result and output
            the mean values. This is distinct from the trigger of this
            extension itself. If it is a tuple of an integer and a string, then
            it is used to create an :class:`IntervalTrigger` object.
        postprocess: Callback to postprocess the result dictionaries. If it is
            set, then each result dictionary with mean values is passed to this
            callback. This callback may modify the result dictionaries.
        log_name (str): Name of the log file under the outptu directory.

    """
    def __init__(self, keys=None, trigger=(1, 'epoch'), postprocess=None,
                 log_name='log'):
        self._keys = keys
        self.set_trigger(trigger)
        self.postprocess = postprocess
        self.log_name = log_name
        self._log = []

        self._init_summary()

    def set_trigger(self, trigger):
        """Sets the trigger to create and output the mean values.

        This is equivalent to the ``trigger`` argument of the initializer.

        Args:
            trigger: Trigger that decides when to aggregate the result and
                output the mean values. See :class:`LogResult` for details.

        """
        if isinstance(trigger, tuple):
            trigger = interval_trigger.IntervalTrigger(*trigger)
        self._trigger = trigger

    def __call__(self, trainer):
        # update
        for key, value in six.iteritems(trainer.result):
            if self._keys is None or key in self._keys:
                self._summary[key].add(value)

        if self._trigger(trainer):
            # append an entry to the log
            means = {key: s.mean for key, s in six.iteritems(self._summary)}
            if self.postprocess is not None:
                self.postprocess(means)
            entry = collections.OrderedDict()
            entry['epoch'] = trainer.epoch
            entry['iteration'] = trainer.t
            for name, mean in six.iteritems(means):
                d = {}
                for key, value in six.iteritems(mean):
                    d[key] = float(value)
                entry[name] = d
            self._log.append(entry)

            # write to the file
            fd, path = tempfile.mkstemp(prefix=self.log_name, dir=trainer.out)
            with os.fdopen(fd, 'w') as f:
                json.dump(self._log, f, indent=4)
            os.rename(path, os.path.join(trainer.out, self.log_name))

            # reset the summary for next iterations
            self._init_summary()

    def serialize(self, serializer):
        if isinstance(serializer, serializer_module.Serializer):
            log = json.dumps(self._log)
            serializer('_log', log)
        else:
            log = serializer('_log', '')
            self._log = json.loads(log)

    def _init_summary(self):
        self._summary = collections.defaultdict(summary_module.DictSummary)
