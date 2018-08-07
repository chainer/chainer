import sys
import time

import numpy

from chainer.backends import cuda
from chainer import link_hook
from chainer import utils


class _CallHistoryRecord(object):
    """Call history record for a single call"""

    def __init__(self, link_name, elapsed_time):
        self.link_name = link_name
        self.elapsed_time = elapsed_time

    def __repr__(self):
        return utils._repr_with_named_data(
            self, link_name=self.link_name, elapsed_time=self.elapsed_time)


class TimerHook(link_hook.LinkHook):
    """Link hook for measuring elapsed time of :meth:`Link.forward`.

    Example:
        Code example::

            from chainer.link_hooks import TimerHook
            hook = TimerHook()
            with hook:
                trainer.run()
            hook.print_report()

        Output example::

              LinkName  ElapsedTime  Occurrence
            Classifier     42.39sec         700
                   MLP     42.09sec         700
                Linear     41.42sec        2100

        where *LinkName* is the name of link that calls the hook,
        and *ElapsedTime* is the elapsed time the link consumed,
        and *Occurrence* is the number of calls.

    Warning:
        Link calls are hierarchical and may overlap with each other.
        The sum of reported elapsed time of each link may exceed the one
        returned by :meth:`~TimerHook.total_time`.
    """

    name = 'TimerHook'

    def __init__(self):
        self._call_history = []
        self._running_stack = []
        self._depth = 0
        self._total_time = 0

    def _preprocess(self, link):
        xp = link.xp
        if xp is numpy:
            start = time.time()
            self._running_stack.append((xp, start))
        else:
            assert xp is cuda.cupy
            start = cuda.Event()
            stop = cuda.Event()
            start.record()
            self._running_stack.append((xp, start, stop))
        self._depth += 1

    def _postprocess(self, link):
        last = self._running_stack.pop()
        xp = link.xp
        assert last[0] is xp
        if xp is numpy:
            _, start = last
            stop = time.time()
            elapsed_time = stop - start
        else:
            assert xp is cuda.cupy
            _, start, stop = self._running_stack.pop()
            stop.record()
            stop.synchronize()
            # Note that `get_elapsed_time` returns result in milliseconds
            elapsed_time = cuda.cupy.cuda.get_elapsed_time(
                start, stop) / 1000
        self._call_history.append((link.__class__.__name__, elapsed_time))

        assert self._depth > 0
        self._depth -= 1
        if self._depth == 0:
            self._total_time += elapsed_time

    def forward_preprocess(self, args):
        self._preprocess(args.link)

    def forward_postprocess(self, args):
        self._postprocess(args.link)

    @property
    def call_history(self):
        """Iterable of measurement results.

        Returns:
            An iterable of record data, each of which represents a single
            invocation of link's forward method. A record data has the
            following attributes:

                * link_name (:class:`str`)
                    Name of the link
                * elapsed_time (:class:`float`)
                    Elapsed time of the forward method in seconds.
        """
        for link_name, elapsed_time in self._call_history:
            yield _CallHistoryRecord(link_name, elapsed_time)

    def total_time(self):
        """Returns the total elapsed time in seconds."""
        return self._total_time

    def summary(self):
        """Returns a summary of time profiling in links.

        Returns:
            A summarized dictionary whose keys are link names and
            values are dictionaries of `elapsed_time` and `occurrrence`.
        """
        summary = {}
        for link_name, elapsed_time in self._call_history:
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
        summary = self.summary()
        # Sort links in the descending order of elapsed time
        sorted_link_names = sorted(
            summary.keys(),
            key=lambda link_name: -summary[link_name]['elapsed_time'])
        for link_name in sorted_link_names:
            record = summary[link_name]
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
