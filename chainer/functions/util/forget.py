import chainer
from chainer import function
from chainer import function_node
from chainer import variable


def _call_func(func, xs):
    outs = func(*xs)

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


class Forget(function_node.FunctionNode):

    def __init__(self, func):
        if not callable(func):
            raise TypeError('func must be callable')
        self.func = func

    def forward(self, inputs):
        self.retain_inputs(tuple(range(len(inputs))))
        with function.no_backprop_mode():
            xs = [variable.Variable(x) for x in inputs]
            outs = _call_func(self.func, xs)
        return tuple(out.data for out in outs)

    def backward(self, indexes, grad_outputs):
        # Double backprop is not allowed
        if chainer.config.enable_backprop:
            raise RuntimeError('double backpropagation in functions.forget is '
                               'not allowed.')

        inputs = self.get_retained_inputs()
        # Create new variables that have no creators
        dummy_inputs = tuple([variable.Variable(inp.array) for inp in inputs])

        with function.force_backprop_mode(),\
                chainer.using_config('in_recomputing', True):
            outs = _call_func(self.func, dummy_inputs)
            assert len(outs) == len(grad_outputs)

        output_tuples = []
        for out, grad_output in zip(outs, grad_outputs):
            if grad_output is not None:
                output_tuples.append((out.node, grad_output))
        # TODO(kataoka): use outer backward's `retain_grad` and `loss_scale`
        chainer.variable._backprop_to_all(output_tuples, False, None)

        return tuple([inp.grad_var for inp in dummy_inputs])


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
       ...   return (a + b) * a

       and, ``x`` and ``y`` be :class:`~chainer.Variable`\\ s:

       >>> x = chainer.Variable(np.random.uniform(-1, 1, 5).astype(np.float32))
       >>> y = chainer.Variable(np.random.uniform(-1, 1, 5).astype(np.float32))

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

        In case input argument variables are of :ref:`ndarray` objects,
        arguments will automatically be
        converted to :class:`~chainer.Variable`\\ s.
        This conversion takes place to ensure that this function is included
        in the computational graph to enable backward computations.

    .. note::

        ``F.forget`` does not support double backpropagation.

    .. note::

        If you want to use ``F.forget`` to a link which updates the link's
        internal information every time the forward computation is called,
        please ensure that the information is updated just once in a single
        iteration. You may use the ``chainer.config.in_recomputing`` flag to
        check if the forward computation is the first call in an iteration.
        Please see the implementation of
        :class:`~chainer.links.BatchNormalization` for detail.

    Args:
        func (callable): A function to call. It needs to be called with
            :class:`~chainer.Variable` object(s) and to return a
            :class:`~chainer.Variable` object or a tuple of
            :class:`~chainer.Variable` objects.
        xs (:class:`tuple` of :class:`~chainer.Variable` or :ref:`ndarray`):
            Argument variables of the function.

    Returns:
        ~chainer.Variable: A variable ``func`` returns. If it returns a tuple,
        the method returns a tuple too.

    """
    xs = tuple(x if isinstance(x, variable.Variable) else
               variable.Variable(x, requires_grad=True) for x in xs)
    y = Forget(func).apply(xs)
    if len(y) == 1:
        y, = y
    return y
