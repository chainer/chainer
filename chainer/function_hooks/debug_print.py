import sys
import warnings

from chainer import backend
from chainer import function_hook
from chainer import variable


class PrintHook(function_hook.FunctionHook):
    """Function hook that prints debug information.

    This function hook outputs the debug information of input arguments of
    ``forward`` and ``backward`` methods involved in the hooked functions
    at preprocessing time (that is, just before each method is called).

    Unlike simple "debug print" technique, where users insert print functions
    at every function to be inspected, we can show the information
    of all functions involved with single ``with`` statement.

    Further, this hook enables us to show the information of
    ``backward`` methods without inserting print functions into
    Chainer's library code.

    Args:
        sep: *(deprecated since v4.0.0)* Ignored.
        end: Character to be added at the end of print function.
        file: Output file_like object that that redirect to.
        flush: If ``True``, this hook forcibly flushes the text stream
            at the end of preprocessing.

    .. admonition:: Example

        The basic usage is to use it with ``with`` statement.

        >>> from chainer import function_hooks
        >>> l = L.Linear(10, 10)
        >>> x = chainer.Variable(np.zeros((1, 10), np.float32))
        >>> with chainer.function_hooks.PrintHook():
        ...     y = l(x)
        ...     z = F.sum(y)
        ...     z.backward() # doctest:+SKIP

        In this example, ``PrintHook`` shows the debug information of
        forward propagation of ``LinearFunction`` (which is implicitly
        called by ``l``) and ``Sum`` (called by ``F.sum``)
        and backward propagation of ``z`` and ``y``.

    """

    name = 'PrintHook'

    def __init__(self, sep=None, end='\n', file=sys.stdout, flush=True):
        if sep is not None:
            warnings.warn('sep argument in chainer.function_hooks.PrintHook '
                          'is deprecated.', DeprecationWarning)
        self.sep = sep  # Keep sep because it was originally documented
        self.end = end
        self.file = file
        self.flush = flush

    def _print(self, msg):
        self.file.write(msg + self.end)

    def _process(self, function, in_data, out_grad=None):
        self._print('function\t{}'.format(function.label))
        self._print('input data')
        for d in in_data:
            if d is None:
                # Some inputs can be removed with `retain_grad`.
                self._print('(removed)')
                continue
            self._print(variable.Variable(d).debug_print())
        if out_grad is not None:
            self._print('output gradient')
            for d in out_grad:
                if d is None:
                    v = variable.Variable()
                else:
                    xp = backend.get_array_module(d)
                    v = variable.Variable(xp.zeros_like(d, dtype=d.dtype))
                    v.grad = d
                self._print(v.debug_print())
        if self.flush and hasattr(self.file, 'flush'):
            self.file.flush()

    def forward_preprocess(self, function, in_data):
        self._process(function, in_data)

    def backward_preprocess(self, function, in_data, out_grad):
        self._process(function, in_data, out_grad)
