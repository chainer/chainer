import os
import sys
import time

import numpy

from chainer import backend
from chainer.backends import cuda
from chainer import function_hook


# Select the best-resolution timer function
try:
    _get_time = time.perf_counter
except AttributeError:
    if os.name == 'nt':
        _get_time = time.clock
    else:
        _get_time = time.time


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
                           ReLU      0.59sec        2600
            SoftmaxCrossEntropy      0.82sec        1300
                       Accuracy      0.18sec         700

        where *FunctionName* is the name of function that calls the hook,
        and *ElapsedTime* is the elapsed time the function consumed,
        and *Occurrence* is the number of calls.
    Attributes:
        call_history: List of measurement results. It consists of pairs of
            the name of the function that calls this hook and the elapsed time
            the function consumes.
    """

    name = 'TimerHook'
    table = {'sec': 1, 'ms': 10 ** 3, 'us': 10 ** 6, 'ns': 10 ** 9}

    def __init__(self):
        self.call_history = []
        self._running_stack = []
        self._depth = 0
        self._total_time = 0

    def _preprocess(self):
        if self.xp == numpy:
            start = _get_time()
            self._running_stack.append(start)
        else:
            start = cuda.Event()
            stop = cuda.Event()
            start.record()
            self._running_stack.append((start, stop))
        self._depth += 1

    def forward_preprocess(self, function, in_data):
        self.xp = backend.get_array_module(*in_data)
        self._preprocess()

    def backward_preprocess(self, function, in_data, out_grad):
        self.xp = backend.get_array_module(*(in_data + out_grad))
        self._preprocess()

    def _postprocess(self, function):
        if self.xp == numpy:
            start = self._running_stack.pop()
            stop = _get_time()
            elapsed_time = stop - start
        else:
            start, stop = self._running_stack.pop()
            stop.record()
            stop.synchronize()
            # Note that `get_elapsed_time` returns result in milliseconds
            elapsed_time = cuda.cupy.cuda.get_elapsed_time(
                start, stop) / 1000
        self.call_history.append((function._impl_name, elapsed_time))

        assert self._depth > 0
        self._depth -= 1
        if self._depth == 0:
            self._total_time += elapsed_time

    def forward_postprocess(self, function, in_data):
        xp = backend.get_array_module(*in_data)
        assert xp == self.xp
        self._postprocess(function)

    def backward_postprocess(self, function, in_data, out_grad):
        xp = backend.get_array_module(*(in_data + out_grad))
        assert xp == self.xp
        self._postprocess(function)

    def total_time(self):
        """Returns total elapsed time in seconds."""
        return self._total_time

    def summary(self):
        """Returns a summary of time profiling in functions.

        Returns:
            A summarized dictionary whose keys are function names and
            values are dictionaries of `elapsed_time` and `occurrence`.
        """
        summary = {}
        for function_name, elapsed_time in self.call_history:
            if function_name not in summary:
                summary[function_name] = {'elapsed_time': 0, 'occurrence': 0}
            record = summary[function_name]
            record['elapsed_time'] += elapsed_time
            record['occurrence'] += 1
        return summary

    def _choose_unit(self, second):
        """Choose optimal unit."""
        factor = 1
        for unit in ['sec', 'ms', 'us']:
            if second * factor >= 1:
                return factor, unit
            factor *= 1000.0
        return factor, 'ns'

    def print_report(self, unit='auto', file=sys.stdout):
        """Prints a summary report of time profiling in functions.

        Args:
            unit (str): Supplementary units used for computational times.
                `sec`, `ms`, `us`, `ns`, `auto`(default) and `auto_foreach`
                are supported. If `auto`, units of times are aligned to the
                largest, and if `auto_foreach`, units of times are adjusted for
                each element.
        """
        entries = [['FunctionName', 'ElapsedTime', 'Occurrence']]
        auto_foreach = (unit == 'auto_foreach')
        if unit == 'auto':
            max_time = max(
                record['elapsed_time'] for record in self.summary().values())
            factor, unit = self._choose_unit(max_time)
        elif unit != 'auto_foreach':
            factor = self.table[unit]
        for function_name, record in self.summary().items():
            second = record['elapsed_time']
            if auto_foreach:
                factor, unit = self._choose_unit(second)
            elapsed_time = '%3.2f%s' % (second * factor, unit)
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
        if hasattr(file, 'flush'):
            file.flush()
