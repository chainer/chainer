import collections
import sys
import typing as tp  # NOQA

from chainer.backends import cuda
from chainer import function_hook


try:
    MemoryHook = cuda.cupy.cuda.memory_hook.MemoryHook  # type: tp.Any # to handle https://github.com/python/mypy/issues/2477 # NOQA
    memory_hook_available = True
except Exception as e:
    _resolution_error = e
    MemoryHook = object
    memory_hook_available = False


class CupyMemoryProfileHook(function_hook.FunctionHook):
    """Function hook for measuring memory usage of functions in cupy memory pool.

    Example:
        Code example::

            from chainer.function_hooks import CupyMemoryProfileHook
            hook = CupyMemoryProfileHook()
            with hook:
                trainer.run()
            hook.print_report()

        Output example::

                   FunctionName  UsedBytes  AcquiredBytes  Occurrence
                 LinearFunction     5.16GB       179.98MB        3900
                           ReLU     0.99GB       458.97MB        2600
            SoftmaxCrossEntropy     0.01GB         5.08MB        1300
                       Accuracy     0.00GB         0.35MB         700

        where *FunctionName* is the name of function that calls the hook, and
        *UsedBytes* is the memory bytes the function used from cupy memory
        pool, and *AcquiredBytes* is the actual memory bytes the cupy memory
        pool acquired from GPU device on the function call, and *Occurrence*
        is the number of calls.
    Attributes:
        call_history: List of measurement results. It consists of the name of
            the function that calls this hook, the memory bytes the function
            used from cupy memory pool, and the memory bytes the cupy memory
            pool acquired from GPU device on the function call.
    """

    name = 'CupyMemoryProfileHook'
    _units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB']
    _table = {u: 1024.0 ** i for i, u in enumerate(_units)}

    def __init__(self):
        cuda.check_cuda_available()
        if not memory_hook_available:
            msg = 'CuPy >= 2.0 is required. %s' % str(_resolution_error)
            raise RuntimeError(msg)
        self.call_history = []
        self._memory_hook = CupyMemoryCumulativeHook()
        self._running_stack = []
        self._total_used_bytes = 0
        self._total_acquired_bytes = 0

    def added(self, function=None):
        self._memory_hook.__enter__()

    def deleted(self, function=None):
        self._memory_hook.__exit__()

    def _preprocess(self):
        start_used_bytes = self._memory_hook.used_bytes
        start_acquired_bytes = self._memory_hook.acquired_bytes
        self._running_stack.append((start_used_bytes, start_acquired_bytes))

    def forward_preprocess(self, function, in_data):
        self._preprocess()

    def backward_preprocess(self, function, in_data, out_grad):
        self._preprocess()

    def _postprocess(self, function):
        start_used_bytes, start_acquired_bytes = self._running_stack.pop()
        end_used_bytes = self._memory_hook.used_bytes
        end_acquired_bytes = self._memory_hook.acquired_bytes
        used_bytes = end_used_bytes - start_used_bytes
        acquired_bytes = end_acquired_bytes - start_acquired_bytes
        depth = len(self._running_stack)
        self.call_history.append(
            (function._impl_name, used_bytes, acquired_bytes, depth))
        if depth == 0:
            self._total_used_bytes += used_bytes
            self._total_acquired_bytes += acquired_bytes

    def forward_postprocess(self, function, in_data):
        self._postprocess(function)

    def backward_postprocess(self, function, in_data, out_grad):
        self._postprocess(function)

    def total_used_bytes(self):
        """Returns total bytes that functions used from cupy memory pool."""
        return self._total_used_bytes

    def total_acquired_bytes(self):
        """Returns total bytes that cupy memory pool acquired from GPU."""
        return self._total_acquired_bytes

    def summary(self):
        """Returns a summary of memory profiling in functions.

        Returns:
            A summarized dictionary whose keys are function names and
            values are dictionaries of
            ``used_bytes``, ``acquired_bytes``, and ``occurrrence``.
        """
        # TODO(sonots): PROBLEM: takes count of nested functions duplicately
        summary = collections.OrderedDict()
        for func_name, used_bytes, acquired_bytes, depth in self.call_history:
            if func_name not in summary:
                summary[func_name] = {'used_bytes': 0,
                                      'acquired_bytes': 0, 'occurrence': 0}
            record = summary[func_name]
            record['used_bytes'] += used_bytes
            record['acquired_bytes'] += acquired_bytes
            record['occurrence'] += 1
        return summary

    def _choose_unit(self, size):
        """Choose optimal unit.

        Returns:
            Tuple of denomi (float) and human-readable unit (str).
        """
        denomi = 1.0
        if size <= 0:
            return denomi, self._units[0]
        for unit in self._units[:-1]:
            if size / (denomi * 1024) < 1:
                return denomi, unit
            denomi *= 1024
        return denomi, self._units[-1]

    def print_report(self, unit='auto', file=sys.stdout):
        """Prints a summary report of memory profiling in functions.

        Args:
            unit (str): Supplementary units used for used memories.
                `B`, `KB`, `MB`, `GB`, `TB`, `PB`, `EB`, `ZB`, `auto`(default)
                and `auto_foreach` are supported. If `auto`, units of memories
                are aligned to the largest values of 'used_bytes' and
                'acquired_bytes'. If `auto_foreach`, units of memories are
                adjusted for each element.
        """
        entries = [[
            'FunctionName', 'UsedBytes', 'AcquiredBytes', 'Occurrence']]
        if unit == 'auto':
            max_used = max(
                record['used_bytes'] for record in self.summary().values())
            max_acquired = max(
                record['acquired_bytes'] for record in self.summary().values())
            denomi_used, unit_used = self._choose_unit(max_used)
            denomi_acquired, unit_acquired = self._choose_unit(max_acquired)
        elif unit != 'auto_foreach':
            denomi_used = denomi_acquired = self._table[unit]
            unit_used = unit_acquired = unit
        for function_name, record in self.summary().items():
            used_bytes = record['used_bytes']
            acquired_bytes = record['acquired_bytes']
            if unit == 'auto_foreach':
                denomi_used, unit_used = self._choose_unit(used_bytes)
                denomi_acquired, unit_acquired = self._choose_unit(
                    acquired_bytes)
            used_bytes = '%3.2f%s' % (used_bytes / denomi_used, unit_used)
            acquired_bytes = '%3.2f%s' % (
                acquired_bytes / denomi_acquired, unit_acquired)
            occurrence = str(record['occurrence'])
            entries.append(
                [function_name, used_bytes, acquired_bytes, occurrence])
        entry_widths = []
        entry_widths.append(max(len(f) for f, _, _, _ in entries))
        entry_widths.append(max(len(u) for _, u, _, _ in entries))
        entry_widths.append(max(len(a) for _, _, a, _ in entries))
        entry_widths.append(max(len(o) for _, _, _, o in entries))
        template = '  '.join('{:>%d}' % w for w in entry_widths)
        for function_name, used_bytes, acquired_bytes, occurrence in entries:
            line = template.format(
                function_name, used_bytes, acquired_bytes, occurrence)
            file.write(line)
            file.write('\n')
        if hasattr(file, 'flush'):
            file.flush()


class CupyMemoryCumulativeHook(MemoryHook):
    """A simple memory hook for cupy measuring memory usage cumulatively.

    Attributes:
        used_bytes (int): cumulative bytes that application used from cupy
            memory pool.
        acquired_bytes (int): cumulative bytes that cupy memory pool acquired
            from GPU device.
    """

    name = 'CupyMemoryCumulativeHook'

    def __init__(self):
        self.used_bytes = 0
        self.acquired_bytes = 0

    def alloc_preprocess(self, **kwargs):
        self.acquired_bytes += kwargs['mem_size']

    def malloc_preprocess(self, **kwargs):
        self.used_bytes += kwargs['mem_size']
