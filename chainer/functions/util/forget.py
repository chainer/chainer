from chainer.backends import cuda
from chainer import function
from chainer import variable


class _DummyFunction(function.Function):

    def __init__(self, grads):
        self.grads = grads

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        return xp.array(0),

    def backward(self, inputs, outputs):
        return self.grads


class Forget(function.Function):

    def __init__(self, func):
        if not callable(func):
            raise TypeError('func must be callable')

        self.func = func

    def _call_func(self, xs):
        outs = self.func(*xs)

        if isinstance(outs, tuple):
            for i, out in enumerate(outs):
                if isinstance(out, variable.Variable):
                    continue
                n = i + 1
                suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(
                    n if n < 20 else n % 10, 'th')
                msg = ('{}{} element of a returned tuple is not Variable, '
                       'but is {}').format(n, suffix, type(out))
                raise RuntimeError(msg)
        elif isinstance(outs, variable.Variable):
            outs = (outs,)
        else:
            msg = ('A tuple of Variables or a Variable are expected, but {} '
                   'is returned.'.format(type(outs)))
            raise RuntimeError(msg)

        return outs

    def forward(self, inputs):
        with function.no_backprop_mode():
            xs = [variable.Variable(x) for x in inputs]
            outs = self._call_func(xs)
        return tuple(out.data for out in outs)

    def backward(self, inputs, grads):
        with function.force_backprop_mode():
            xs = [variable.Variable(x) for x in inputs]
            outs = self._call_func(xs)
            _DummyFunction(grads)(*outs).backward()
        return tuple(x.grad for x in xs)


def forget(func, *xs):
    """Calls a function without storing intermediate results.

    On a forward propagation, Chainer normally stores all intermediate results
    of :class:`~chainer.variable.VariableNode`\\ s on a computational graph as
    they are required on backward propagation.
    Sometimes these results consume too much memory.
    ``F.forget`` *forgets* such intermediate results on forward propagation,
    and still supports backpropagation with recalculation.

    On a forward propagation, ``F.forget`` calls a given function with given
    variables without creating a computational graph. That means, no
    intermediate results are stored.
    On a backward propagation, ``F.forget`` calls the given function again to
    create a computational graph for backpropagation.

    ``F.forget`` reduces internal memory usage, whereas it requires more
    calculation time as it calls the function twice.

    .. admonition:: Example

       Let ``f`` be a function defined as:

       >>> def f(a, b):
       ...   return a + b + a

       and, ``x`` and ``y`` be :class:`~chainer.Variable`\\ s:

       >>> x = chainer.Variable(np.random.uniform(-1, 1, 5).astype('f'))
       >>> y = chainer.Variable(np.random.uniform(-1, 1, 5).astype('f'))

       When ``z`` is calculated as ``z = f(x, y)``, its intermediate result
       ``x + y`` is stored in memory. Instead, if you call ``f`` with
       ``F.forget``:

       >>> z = F.forget(f, x, y)

       intermediate ``x + y`` is forgotten.

    .. note::

        ``F.forget`` does not support functions which behave differently in
        multiple calls with the same inputs, such as
        :meth:`F.dropout() <chainer.functions.dropout>` and
        :meth:`F.negative_sampling() <chainer.functions.negative_sampling>`.

    .. note::

        In case input argument variables are of class :class:`numpy.ndarray` or
        :class:`cupy.ndarray` objects, arguments will automatically be
        converted to :class:`~chainer.Variable`\\ s.
        This conversion takes place to ensure that this function is included
        in the computational graph to enable backward computations.

    Args:
        func (callable): A function to call. It needs to be called with
            :class:`~chainer.Variable` object(s) and to return a
            :class:`~chainer.Variable` object or a tuple of
            :class:`~chainer.Variable` objects.
        xs (~chainer.Variable): Argument variables of the function.

    Returns:
        ~chainer.Variable: A variable ``func`` returns. If it returns a tuple,
        the method returns a tuple too.

    """
    xs = tuple(x if isinstance(x, variable.Variable) else
               variable.Variable(x, requires_grad=True) for x in xs)
    return Forget(func)(*xs)
