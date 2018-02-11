import sys
import time

import numpy

from chainer.backends import cuda
from chainer import function_hook


class TimerHook(function_hook.FunctionHook):
    """Function hook for measuring elapsed time of functions.

    Example:
        Code example::

            from chainer.function_hooks import TimerHook
            hook = TimerHook()
            with hook:
                trainer.run()
            hook.print_report()

        Output example::

                   FunctionName  ElapsedTime  Occurrence
                 LinearFunction      1.24sec        3900
                           ReLU     593.05ms        2600
            SoftmaxCrossEntropy     824.11ms        1300
                       Accuracy     176.54ms         700

        where *FunctionName* is the name of function that calls the hook,
        and *ElapsedTime* is the elapsed time the function consumed,
        and *Occurrence* is the number of calls.
    Attributes:
        call_history: List of measurement results. It consists of pairs of
            the function that calls this hook and the elapsed time
            the function consumes.
    """

    name = 'TimerHook'

    def __init__(self):
        self.call_history = []
        self._running_stack = []
        self._depth = 0
        self._total_time = 0

    def _preprocess(self):
        if self.xp == numpy:
            start = time.time()
            self._running_stack.append(start)
        else:
            start = cuda.Event()
            stop = cuda.Event()
            start.record()
            self._running_stack.append((start, stop))
        self._depth += 1

    def forward_preprocess(self, function, in_data):
        self.xp = cuda.get_array_module(*in_data)
        self._preprocess()

    def backward_preprocess(self, function, in_data, out_grad):
        self.xp = cuda.get_array_module(*(in_data + out_grad))
        self._preprocess()

    def _postprocess(self, function):
        if self.xp == numpy:
            start = self._running_stack.pop()
            stop = time.time()
            elapsed_time = stop - start
        else:
            start, stop = self._running_stack.pop()
            stop.record()
            stop.synchronize()
            # Note that `get_elapsed_time` returns result in milliseconds
            elapsed_time = cuda.cupy.cuda.get_elapsed_time(
                start, stop) / 1000
        self.call_history.append((function, elapsed_time))

        assert self._depth > 0
        self._depth -= 1
        if self._depth == 0:
            self._total_time += elapsed_time

    def forward_postprocess(self, function, in_data):
        xp = cuda.get_array_module(*in_data)
        assert xp == self.xp
        self._postprocess(function)

    def backward_postprocess(self, function, in_data, out_grad):
        xp = cuda.get_array_module(*(in_data + out_grad))
        assert xp == self.xp
        self._postprocess(function)

    def total_time(self):
        """Returns total elapsed time in seconds."""
        return self._total_time

    def summary(self):
        """Returns a summary of time profiling in functions.

        Returns:
            A summarized dictionary whose keys are function names and
            values are dictionaries of `elapsed_time` and `occurrrence`.
        """
        summary = {}
        for func, elapsed_time in self.call_history:
            function_name = func._impl_name
            if function_name not in summary:
                summary[function_name] = {'elapsed_time': 0, 'occurrence': 0}
            record = summary[function_name]
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
        """Prints a summary report of time profiling in functions."""
        entries = [['FunctionName', 'ElapsedTime', 'Occurrence']]
        for function_name, record in self.summary().items():
            elapsed_time = self._humanized_time(record['elapsed_time'])
            occurrence = str(record['occurrence'])
            entries.append([function_name, elapsed_time, occurrence])
        entry_widths = []
        entry_widths.append(max(len(f) for f, _, _ in entries))
        entry_widths.append(max(len(e) for _, e, _ in entries))
        entry_widths.append(max(len(o) for _, _, o in entries))
        template = '  '.join('{:>%d}' % w for w in entry_widths)
        for function_name, elapsed_time, occurrence in entries:
            line = template.format(function_name, elapsed_time, occurrence)
            file.write(line)
            file.write('\n')
        file.flush()
