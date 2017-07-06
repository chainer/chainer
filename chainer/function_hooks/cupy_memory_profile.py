import sys

import six

from chainer.cuda import memory_pool
from chainer import function


class CupyMemoryProfileHook(function.FunctionHook):
    """Function hook for measuring memory usage of functions in cupy memory pool.

    Example:
        Code example::

            from chainer.function_hooks import CupyMemoryProfileHook
            hook = CupyMemoryProfileHook()
            with hook:
                trainer.run()
            hook.print_report()

        Output example::

            FunctionName        UsedBytes AcquiredBytes Occurrence
            LinearFunction      5.16GB    179.98MB      3900
            ReLU                992.77MB  458.97MB      2600
            SoftmaxCrossEntropy 5.76MB    5.08MB        1300
            Accuracy            350.00KB  351.00KB      700

        where *FunctionName* is the name of function that calls the hook, and
        *UsedBytes* is the memory bytes the function used from cupy memory
        pool, and *AcquiredBytes* is the actual memory bytes the cupy memory
        pool acquired from GPU device on the function call, and *Occurrence*
        is the number of calls.
    Attributes:
        call_history: List of measurement results. It consists of the function
            that calls this hook,  the memory bytes the function used from cupy
            memory pool, and the memory bytes the cupy memory pool acquired
            from GPU device on the function call.
    """

    name = 'CupyMemoryProfileHook'

    def __init__(self):
        self.call_history = []

    def _preprocess(self):
        self.start_used_bytes = memory_pool.used_bytes()
        self.start_total_bytes = memory_pool.total_bytes()

    def forward_preprocess(self, function, in_data):
        self._preprocess()

    def backward_preprocess(self, function, in_data, out_grad):
        self._preprocess()

    def _postprocess(self, function):
        used_bytes = memory_pool.used_bytes() - self.start_used_bytes
        acquired_bytes = memory_pool.total_bytes() - self.start_total_bytes
        self.call_history.append((function, used_bytes, acquired_bytes))

    def forward_postprocess(self, function, in_data):
        self._postprocess(function)

    def backward_postprocess(self, function, in_data, out_grad):
        self._postprocess(function)

    def summary(self):
        """Returns a summary of memory profiling in functions.

        Returns:
            A summarized dictionary whose keys are function names and
            values are dictionaries of ``used_bytes``, ``acquired_bytes``,
            and ``occurrrence``.
        """
        summary = {}
        for func, used_bytes, acquired_bytes in self.call_history:
            function_name = func.__class__.__name__
            if function_name not in summary:
                summary[function_name] = {'used_bytes': 0,
                                          'acquired_bytes': 0, 'occurrence': 0}
            record = summary[function_name]
            record['used_bytes'] += used_bytes
            record['acquired_bytes'] += acquired_bytes
            record['occurrence'] += 1
        return summary

    def _humanized_size(self, size):
        """Returns a human redable bytes string."""
        for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E']:
            if size < 1024.0:
                return '%3.2f%sB' % (size, unit)
            size /= 1024.0
        return '%.2f%sB' % (size, 'Z')

    def print_report(self, end='\n', file=sys.stdout):
        """Prints a summary report of memory profiling in functions."""
        entries = [['FunctionName', 'UsedBytes', 'AcquiredBytes', 'Occurrence']]
        for function_name, record in self.summary().items():
            used_bytes = self._humanized_size(record['used_bytes'])
            acquired_bytes = self._humanized_size(record['acquired_bytes'])
            occurrence = str(record['occurrence'])
            entries.append([function_name, used_bytes, acquired_bytes, occurrence])
        entry_widths = []
        entry_widths.append(max(len(f) for f, _, _, _ in entries))
        entry_widths.append(max(len(u) for _, u, _, _ in entries))
        entry_widths.append(max(len(a) for _, _, a, _ in entries))
        entry_widths.append(max(len(o) for _, _, _, o in entries))
        template = '  '.join('{:>%d}' % w for w in entry_widths)
        for function_name, used_bytes, acquired_bytes, occurrence in entries:
            line = template.format(function_name, used_bytes, acquired_bytes, occurrence)
            six.print_(line, end=end, file=file)
