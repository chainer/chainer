import json
import os
import shutil
import tempfile

import six

from chainer import reporter
from chainer import serializer as serializer_module
from chainer.training import extension
from chainer.training import trigger as trigger_module


class LogReport(extension.Extension):

    """Trainer extension to output the accumulated results to a log file.

    This extension accumulates the observations of the trainer to
    :class:`~chainer.DictSummary` at a regular interval specified by a supplied
    trigger, and writes them into a log file in JSON format.

    There are three triggers to handle this extension. One is the trigger to
    invoke this extension, which is used to handle the timing of accumulating
    the results. It is set to ``1, 'iteration'`` by default. The second is the
    trigger to determine when to emit the result. When this trigger returns
    True, this extension appends the summary of accumulated values to the list
    of past summaries. Then, this extension makes a new fresh summary object
    which is used until the next time that the trigger fires. Last is the
    trigger to determine when to write the list to the log file.

    It also adds some entries to each result dictionary.

    - ``'epoch'`` and ``'iteration'`` are the epoch and iteration counts at the
      output, respectively.
    - ``'elapsed_time'`` is the elapsed time in seconds since the training
      begins. The value is taken from :attr:`Trainer.elapsed_time`.

    Args:
        keys (iterable of strs): Keys of values to accumulate. If this is None,
            all the values are accumulated and output to the log file.
        trigger: Trigger that decides when to aggregate the result values.
            This is distinct from the trigger of this extension
            itself. If it is a tuple in the form ``<int>, 'epoch'`` or
            ``<int>, 'iteration'``, it is passed to :class:`IntervalTrigger`.
        write_trigger: Trigger that decides when to write the results the list
            to the log file. if `None` is given, default for this argument,
            use ``triger`` arguments, invoked after each aggregations.
        postprocess: Callback to postprocess the result dictionaries. Each
            result dictionary is passed to this callback on the output. This
            callback can modify the result dictionaries, which are used to
            output to the log file.
        log_name (str): Name of the log file under the output directory. It can
            be a format string: the last result dictionary is passed for the
            formatting. For example, users can use '{iteration}' to separate
            the log files for different iterations. If the log name is None, it
            does not output the log to any file.

    """

    def __init__(self, keys=None, trigger=(1, 'epoch'), write_trigger=None,
                 postprocess=None, log_name='log'):
        self._keys = keys
        self._trigger = trigger_module.get_trigger(trigger)
        if write_trigger:
            self._write_trigger = trigger_module.get_trigger(write_trigger)
        else:
            self._write_trigger = self._trigger
        self._postprocess = postprocess
        self._log_name = log_name
        self._log = []

        self._init_summary()

    def __call__(self, trainer):
        # accumulate the observations
        keys = self._keys
        observation = trainer.observation
        summary = self._summary

        if keys is None:
            summary.add(observation)
        else:
            summary.add({k: observation[k] for k in keys if k in observation})

        if self._trigger(trainer) or self._write_trigger(trainer):
            # output the result
            stats = self._summary.compute_mean()
            stats_cpu = {}
            for name, value in six.iteritems(stats):
                stats_cpu[name] = float(value)  # copy to CPU

            updater = trainer.updater
            stats_cpu['epoch'] = updater.epoch
            stats_cpu['iteration'] = updater.iteration
            stats_cpu['elapsed_time'] = trainer.elapsed_time

            if self._postprocess is not None:
                self._postprocess(stats_cpu)

            if self._trigger(trainer):
                self._log.append(stats_cpu)

            if self._write_trigger(trainer):
                self._write(stats_cpu, trainer.out)

            # reset the summary for the next output
            self._init_summary()

    def _write(self, stats_cpu, out_dir):
        # write to the log file
        if self._log_name is not None:
            log_name = self._log_name.format(**stats_cpu)
            fd, path = tempfile.mkstemp(prefix=log_name, dir=out_dir)
            with os.fdopen(fd, 'w') as f:
                json.dump(self._log, f, indent=4)
            new_path = os.path.join(out_dir, log_name)
            shutil.move(path, new_path)

    @property
    def log(self):
        """The current list of observation dictionaries."""
        return self._log

    def serialize(self, serializer):
        if hasattr(self._trigger, 'serialize'):
            self._trigger.serialize(serializer['_trigger'])

        # Note that this serialization may lose some information of small
        # numerical differences.
        if isinstance(serializer, serializer_module.Serializer):
            log = json.dumps(self._log)
            serializer('_log', log)
        else:
            log = serializer('_log', '')
            self._log = json.loads(log)

    def _init_summary(self):
        self._summary = reporter.DictSummary()
