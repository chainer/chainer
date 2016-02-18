from __future__ import print_function
import collections

import six

from chainer.trainer import extension
from chainer.trainer import interval_trigger
from chainer.utils import summary as summary_module


class PrintResult(extension.Extension):

    """Trainer extension to print the accumulated result.

    This extension is similar to the :class:`LogResult` extension, though this
    extension prints the result to the standard output instead of writing the
    result to the log file.

    See :class:`LogResult` for details.

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

    """
    def __init__(self, keys=None, trigger=(1, 'epoch'), postprocess=None):
        self._keys = keys
        self._summary = collections.defaultdict(summary_module.DictSummary)
        self.set_trigger(trigger)
        self.postprocess = postprocess

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
            means = {key: s.mean for key, s in six.iteritems(self._summary)}
            if self.postprocess is not None:
                self.postprocess(means)
            # print
            print('result @ %s iteration (%s epoch)' %
                  (trainer.t, trainer.epoch))
            for name, mean in six.iteritems(means):
                msg = ['  %s:' % name]
                msg += ['%s=%s' % pair for pair in six.iteritems(mean)]
                print('\t'.join(msg))
            # reset the summary
            self._summary = collections.defaultdict(summary_module.DictSummary)
