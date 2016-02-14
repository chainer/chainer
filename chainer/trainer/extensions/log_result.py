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

    TODO(beam2d): document it.

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
        if isinstance(trigger, tuple):
            trigger = interval_trigger.IntervalTrigger(*trigger)
        self._trigger = trigger

    def __call__(self, epoch, new_epoch, out, result, t, **kwargs):
        # update
        for key, value in six.iteritems(result):
            if self._keys is None or key in self._keys:
                self._summary[key].add(value)

        if self._trigger(epoch=epoch, new_epoch=new_epoch, result=result,
                         t=t, **kwargs):
            # append an entry to the log
            means = {key: s.mean for key, s in six.iteritems(self._summary)}
            if self.postprocess is not None:
                self.postprocess(means)
            entry = collections.OrderedDict()
            entry['epoch'] = epoch
            entry['iteration'] = t
            for name, mean in six.iteritems(means):
                d = {}
                for key, value in six.iteritems(mean):
                    d[key] = float(value)
                entry[name] = d
            self._log.append(entry)

            # write to the file
            fd, path = tempfile.mkstemp(prefix=self.log_name, dir=out)
            with os.fdopen(fd, 'w') as f:
                json.dump(self._log, f, indent=4)
            os.rename(path, os.path.join(out, self.log_name))

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
