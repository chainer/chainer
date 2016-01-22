from __future__ import print_function
import collections

import six

from chainer.trainer import extension
from chainer.trainer import interval_trigger
from chainer.utils import summary as summary_module


class PrintResult(extension.Extension):

    """Trainer extension to print the accumulated result.

    TODO(beam2d): document it.

    """
    def __init__(self, keys=None, trigger=(1, 'epoch'), postprocess=None):
        self._keys = keys
        self._summary = collections.defaultdict(summary_module.DictSummary)
        if isinstance(trigger, tuple):
            trigger = interval_trigger.IntervalTrigger(*trigger)
        self._trigger = trigger
        self.postprocess = postprocess

    def __call__(self, epoch, new_epoch, result, t, **kwargs):
        # update
        for key, value in six.iteritems(result):
            if self._keys is None or key in self._keys:
                self._summary[key].add(value)

        if self._trigger(epoch=epoch, new_epoch=new_epoch, reuslt=result,
                         t=t, **kwargs):
            means = {key: s.mean for key, s in six.iteritems(self._summary)}
            if self.postprocess is not None:
                self.postprocess(means)
            # print
            print('result @ %s iteration (%s epoch)' % (t, epoch))
            for name, mean in six.iteritems(means):
                msg = ['  %s:' % name]
                msg += ['%s=%s' % pair for pair in six.iteritems(mean)]
                print('\t'.join(msg))
            # reset the summary
            self._summary = collections.defaultdict(summary_module.DictSummary)
