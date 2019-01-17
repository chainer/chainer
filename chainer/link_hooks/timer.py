import collections
import os
import sys
import time

import numpy

from chainer.backends import cuda
from chainer import link_hook


# Select the best-resolution timer function
try:
    _get_time = time.perf_counter
except AttributeError:
    if os.name == 'nt':
        _get_time = time.clock
    else:
        _get_time = time.time


class TimerHook(link_hook.LinkHook):
    """Link hook for measuring elapsed time of :meth:`forward`.

    Example:
        Code example::

            from chainer.link_hooks import TimerHook
            hook = TimerHook()
            with hook:
                trainer.run()
            hook.print_report()

        Output example::

              LinkName  ElapsedTime  Occurrence
                Linear     41.42sec        2100
                   MLP     42.09sec         700
            Classifier     42.39sec         700

        where *LinkName* is the name of link that calls the hook,
        and *ElapsedTime* is the elapsed time the link consumed,
        and *Occurrence* is the number of calls.
    Warning:
        Call graph of links are hierarchical. That means reported elapsed times
        may be overlapping with each other and the sum may exceed the total
        time.
    Attributes:
        call_history: List of measurement results. It consists of pairs of
            the name of the link that calls this hook and the elapsed time
            the :meth:`forward` method of link consumes.
    """

    name = 'TimerHook'

    def __init__(self):
        self.call_history = []
        self._running_stack = []
        self._depth = 0
        self._total_time = 0

    def _preprocess(self):
        if self.xp is numpy:
            start = _get_time()
            self._running_stack.append(start)
        else:
            assert self.xp is cuda.cupy
            start = cuda.Event()
            stop = cuda.Event()
            start.record()
            self._running_stack.append((start, stop))
        self._depth += 1

    def forward_preprocess(self, args):
        self.xp = args.link.xp
        self._preprocess()

    def _postprocess(self, link):
        if self.xp is numpy:
            start = self._running_stack.pop()
            stop = _get_time()
            elapsed_time = stop - start
        else:
            assert self.xp is cuda.cupy
            start, stop = self._running_stack.pop()
            stop.record()
            stop.synchronize()
            # Note that `get_elapsed_time` returns result in milliseconds
            elapsed_time = cuda.cupy.cuda.get_elapsed_time(
                start, stop) / 1000
        self.call_history.append((link.__class__.__name__, elapsed_time))

        assert self._depth > 0
        self._depth -= 1
        if self._depth == 0:
            self._total_time += elapsed_time

    def forward_postprocess(self, args):
        link = args.link
        assert link.xp == self.xp
        self._postprocess(link)

    def total_time(self):
        """Returns total elapsed time in seconds."""
        return self._total_time

    def summary(self):
        """Returns a summary of time profiling in links.

        Returns:
            A summarized dictionary whose keys are link names and
            values are dictionaries of `elapsed_time` and `occurrence`.

        """
        summary = collections.OrderedDict()
        for link_name, elapsed_time in self.call_history:
            if link_name not in summary:
                summary[link_name] = {'elapsed_time': 0, 'occurrence': 0}
            record = summary[link_name]
            record['elapsed_time'] += elapsed_time
            record['occurrence'] += 1
        return summary

    def _humanized_time(self, second):
        """Returns a human readable time."""
        for unit in ['sec', 'ms', 'us']:
            if second >= 1:
                return '%3.2f%s' % (second, unit)
            second *= 1000.0
        return '%.2f%s' % (second, 'ns')

    def print_report(self, file=sys.stdout):
        """Prints a summary report of time profiling in links."""
        entries = [['LinkName', 'ElapsedTime', 'Occurrence']]
        for link_name, record in self.summary().items():
            elapsed_time = self._humanized_time(record['elapsed_time'])
            occurrence = str(record['occurrence'])
            entries.append([link_name, elapsed_time, occurrence])
        entry_widths = []
        entry_widths.append(max(len(f) for f, _, _ in entries))
        entry_widths.append(max(len(e) for _, e, _ in entries))
        entry_widths.append(max(len(o) for _, _, o in entries))
        template = '  '.join('{:>%d}' % w for w in entry_widths)
        for link_name, elapsed_time, occurrence in entries:
            line = template.format(link_name, elapsed_time, occurrence)
            file.write(line)
            file.write('\n')
        file.flush()

    # TODO(crcrpar): Support backward pre/post process.
    # See https://github.com/chainer/chainer/issues/5197
