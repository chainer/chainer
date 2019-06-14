import collections

from chainer.functions.theano import theano_function
from chainer import link
from chainer.utils import collections_abc


def _to_var_tuple(vs):
    import theano
    msg = ('inputs and outputs must be a TensorVariable, a list '
           'of TensorVariable or a tuple of TensorVariable')

    if isinstance(vs, theano.tensor.TensorVariable):
        return vs,
    elif isinstance(vs, collections_abc.Iterable):
        vs = tuple(vs)
        if not all(isinstance(v, theano.tensor.TensorVariable) for v in vs):
            raise TypeError(msg)
        return vs
    else:
        raise TypeError(msg)


class TheanoFunction(link.Link):

    """Theano function wrapper.

    .. warning::

        This feature is experimental. The interface can change in the future.

    This function wraps Theano function as a :class:`chainer.Link`.
    A user needs to make input Theano variables and output Theano variables.
    This function automatically creates Theano function for forward calculation
    and backward calculation from inputs and ouptuts. And then, it sends data
    in :class:`chainer.Variable` to the function and gets results from Theano.

    .. rubric:: Example

    .. doctest::
       # See chainer/chainer#5997
       :skipif: doctest_helper.skipif_requires_satisfied( \
               'Theano<=1.0.3', 'numpy>=1.16.0')

       >>> import theano
       >>> x = theano.tensor.fvector()
       >>> y = theano.tensor.fvector()
       >>> z = x + y
       >>> w = x - y
       >>> f = L.TheanoFunction(inputs=[x, y], outputs=[z, w])
       >>> a = chainer.Variable(np.array([1, 2], dtype=np.float32))
       >>> b = chainer.Variable(np.array([2, 3], dtype=np.float32))
       >>> c, d = f(a, b)
       >>> c.array
       array([3., 5.], dtype=float32)
       >>> d.array
       array([-1., -1.], dtype=float32)

    .. note::

       The current implementation always copys :class:`cupy.ndarray` to CPU.

    Args:
        inputs (tuple of ``theano.tensor.TensorVariable``): Input variables of
            Theano. This function accepts the same number of
            :class:`~chainer.Variable`\\ s in forward computation.
        outputs (tuple of ``theano.tensor.TensorVariable``):
            Output variables of Theano.
            The function returns the same number of
            :class:`~chainer.Variable`\\ s as ``outputs``.

    """

    def __init__(self, inputs, outputs):
        try:
            # When Theano library is imported, it executes a lot of
            # initialization process. To minimize its side effect,
            # we need import theano here.
            import theano
        except ImportError:
            msg = '''theano is not installed on your environment.
Please install theano to activate theano function.

  $ pip install theano'''
            raise RuntimeError(msg)

        super(TheanoFunction, self).__init__()

        inputs = _to_var_tuple(inputs)
        outputs = _to_var_tuple(outputs)

        # TODO(unno): We can remove redundant gpu-cpu copy using
        # theano.sandbox.cuda.basic_ops.gpu_from_host
        self.forward_func = theano.function(inputs=inputs, outputs=outputs)

        gs = tuple(
            o.type('g_{}'.format(i)) for i, o in enumerate(outputs))
        known_grads = collections.OrderedDict(zip(outputs, gs))
        grad = theano.tensor.grad(
            cost=None, wrt=inputs, known_grads=known_grads,
            disconnected_inputs='ignore')

        self.backward_func = theano.function(
            inputs=inputs + gs,
            outputs=grad,
            on_unused_input='ignore')

    def forward(self, *args):
        return theano_function.theano_function(
            self.forward_func, self.backward_func, *args)
