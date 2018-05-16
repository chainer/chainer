import warnings
import weakref

import six

from chainer.backends import cuda
from chainer import configuration
# for backward compatibility
from chainer.function_hook import FunctionHook  # NOQA
from chainer import function_node
from chainer import variable


def no_backprop_mode():
    """Make a context manager which disables back-propagation.

    In this context, Chainer does not make a computational graph. It has the
    benefit of reducing memory consumption. However, a
    :class:`~chainer.Variable` created in this context does not hold a
    reference to the :class:`~chainer.FunctionNode` that created itself so no
    gradients are accumulated by :func:`~chainer.Variable.backward`.

    In the following example, ``y`` is created in this context, which means
    that calling :func:`~chainer.Variable.backward` on ``y`` has no effect on
    the gradients of ``x``.

    >>> x = chainer.Variable(np.array([1,], np.float32))
    >>> with chainer.no_backprop_mode():
    ...     y = x + 1
    >>> y.backward()
    >>> x.grad is None
    True

    .. seealso::

       See :func:`force_backprop_mode` for details on how to override this
       context.

    """
    return configuration.using_config('enable_backprop', False)


def force_backprop_mode():
    """Make a context manager which enables back-propagation.

    When you want to enable back-propagation in :func:`no_backprop_mode`, call
    this method. A :class:`~chainer.Variable` created in this context always
    has a computational graph unless overridden by deeper contexts. If you call
    this method outside of :func:`no_backprop_mode` context, it changes
    nothing.

    In the following example, ``y`` has a computational graph and calling
    :func:`~chainer.Variable.backward` on ``y`` will compute and accumulate the
    gradients of the variables in the graph, in this case only ``x``.

    >>> x = chainer.Variable(np.array([1,], np.float32))
    >>> with chainer.no_backprop_mode():
    ...     with chainer.force_backprop_mode():
    ...         y = x + 1
    >>> y.backward()
    >>> x.grad
    array([1.], dtype=float32)

    .. seealso::

       See :func:`no_backprop_mode` for details on disabled back-propagation
       mode.

    """
    return configuration.using_config('enable_backprop', True)


class FunctionAdapter(function_node.FunctionNode):

    """Adapter class to wrap Function with FunctionNode.

    While :class:`~chainer.FunctionNode` provides the interface
    of new-style differentiable functions, the old-style
    :class:`~chainer.Function` can still be used for the backward
    compatibility.
    This class provides an adapter of there interface; it adds
    :class:`~chainer.FunctionNode` interface to any
    :class:`~chainer.Function` object by delegation.

    .. note::

       The ownership of :class:`FunctionAdapter` and :class:`~chainer.Function`
       is a bit tricky.
       At the initialization, :class:`FunctionAdapter` is owned by the
       :class:`~chainer.Function` object.
       Once the function is applied to variables, the ownership is reversed;
       the adapter becomes the owner of the
       :class:`~chainer.Function` object and the :class:`~chainer.Function`
       object changes the reference to a weak one.

    Args:
        function (Function): The function object to wrap.

    .. versionadded:: 3.0.0

    """

    _function = None
    _weak_function = None

    def __init__(self, function):
        super(FunctionAdapter, self).__init__()
        self._weak_function = weakref.ref(function)
        function._owned_node = self

    @property
    def function(self):
        """The :class:`Function` object that this adapter is wrapping."""
        func = self._function
        if func is not None:
            return func

        weak_func = self._weak_function
        return weak_func and weak_func()

    @property
    def label(self):
        return self._function.label

    @property
    def _impl_name(self):
        return self._function.__class__.__name__

    def check_type_forward(self, in_types):
        self._function.check_type_forward(in_types)

    def forward(self, inputs):
        # Retain all inputs by default in old-style functions.
        self.retain_inputs(six.moves.range(len(inputs)))
        return self._function.forward(inputs)

    def backward(self, target_input_indexes, grad_outputs):
        in_data = tuple([input.data for input in self.inputs])
        grad_out_data = tuple([None if grad is None else grad.data
                               for grad in grad_outputs])

        with cuda.get_device_from_array(*(in_data + grad_out_data)):
            gxs = self._function.backward(in_data, grad_out_data)
        for x, gx in six.moves.zip(self.inputs, gxs):
            variable._check_grad_type(self, x, gx)

        ret = []
        for i in target_input_indexes:
            if gxs[i] is None:
                g = None
            else:
                # Intentionally not passing requires_grad=False so that
                # backprop routines can raise an error when a further backprop
                # is attempted against this gradient variable.
                g = variable.Variable(gxs[i])
                g.node._old_style_grad_generator = self._function.label
            ret.append(g)

        return tuple(ret)


class Function(object):

    """Old-style interface of a differentiable function.

    This class provides an interface to implement an old-style differentiable
    function (i.e., the function application is recorded to the computational
    graph). The subclass of :class:`Function` that implement :meth:`forward`
    and :meth:`backward` can be used to run the forward computation and
    automatically induce the backpropagation procedure.

    There is another way to implement such a function: subclassing
    :class:`~chainer.FunctionNode`. There are mainly two
    differences between them.

    1. The *differentiable backprop* is available for
       :class:`~chainer.FunctionNode`,
       while it is not for :class:`Function` because the :meth:`backward`
       of the latter directly operates on the arrays instead of
       :class:`Variable` objects so that it cannot record the history of
       the computation.
    2. The information passed to :meth:`backward` is different. In
       :class:`~chainer.FunctionNode`,
       which inputs the function node has to compute
       the gradients w.r.t. is passed so that it can omit unnecessary
       computations, while :class:`Function` always has to compute gradients
       w.r.t. all the input nodes.
       The :class:`~chainer.FunctionNode` also accepts the
       current gradient values of the input nodes so that the accumulation
       work can be merged with the gradient computation if an efficient kernel
       is available.

    This class uses :class:`~chainer.FunctionAdapter` to convert
    the interface to that of :class:`~chainer.FunctionNode` and
    adds the :class:`~chainer.FunctionNode` object to the
    computational graph.

    See :class:`~chainer.FunctionNode` for the details of
    building the computational graph in Chainer.

    """

    _node = None
    _owned_node = None

    def __call__(self, *inputs):
        """Applies forward propagation with chaining backward references.

        This method creates a new :class:`~chainer.FunctionAdapter`
        object and runs the forward propagation using it.

        See :class:`~chainer.FunctionNode` for the detailed
        behavior of building the computational graph.

        Args:
            inputs: Tuple of input :class:`Variable`, :class:`numpy.ndarray` or
                :class:`cupy.ndarray` objects.
                If the input is an :class:`numpy.ndarray` or a
                :class:`cupy.ndarray`, it is automatically wrapped with
                :class:`Variable`.

        Returns:
            One :class:`Variable` object or a tuple of multiple
            :class:`Variable` objects.

        """
        node = self.node

        # Swap the ownership
        node._function = self
        node._weak_function = None
        self._node = weakref.ref(node)
        self._owned_node = None

        ret = node.apply(inputs)

        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)

    @property
    def inputs(self):
        """The input nodes of the function."""
        return self.node.inputs

    @property
    def outputs(self):
        """Weak references to the output nodes of the function."""
        return self.node.outputs

    @property
    def node(self):
        """The :class:`FunctionAdapter` object that wraps this Function.

        If the Function does not have a node object, this property
        automatically creates a new one.

        """
        noderef = self._node
        nd = (noderef and noderef()) or self._owned_node
        if nd is not None:
            return nd

        nd = FunctionAdapter(self)
        self._owned_node = nd
        return nd

    @property
    def local_function_hooks(self):
        """Ordered Dictionary of registered function hooks.

        See :attr:`FunctionNode.local_function_hooks` for the detail.

        """
        return self.node.local_function_hooks

    @property
    def label(self):
        """Short text that represents the function.

        The default implementation returns its type name.
        Each function should override it to give more information.

        """
        return self.__class__.__name__

    @property
    def output_data(self):
        """A tuple of the retained output arrays.

        It has the same length as the :attr:`outputs`. Elements that are not
        retained are set to ``None``.

        """
        return self.node.output_data

    @property
    def rank(self):
        """The topological ordinal of the corresponding function node."""
        return self.node.rank

    @property
    def stack(self):
        return self.node.stack

    def check_type_forward(self, in_types):
        """Checks types of input data before forward propagation.

        Before :meth:`forward` is called, this function is called.
        You need to validate types of input data in this function
        using :ref:`the type checking utilities <type-check-utils>`.

        Args:
            in_types (~chainer.utils.type_check.TypeInfoTuple): The type
                information of input data for :meth:`forward`.
        """
        pass

    def forward(self, inputs):
        """Applies forward propagation to input arrays.

        It delegates the procedure to :meth:`forward_cpu` or
        :meth:`forward_gpu` by default. Which it selects is determined by the
        type of input arrays.
        Implementations of :class:`Function` must implement either CPU/GPU
        methods or this method.

        Args:
            inputs: Tuple of input array(s).

        Returns:
            Tuple of output array(s).

        .. warning::

            Implementations of :class:`Function` must take care that the
            return value must be a tuple even if it returns only one array.

        """
        if any(isinstance(x, cuda.ndarray) for x in inputs):
            return self.forward_gpu(inputs)
        else:
            return self.forward_cpu(inputs)

    def forward_cpu(self, inputs):
        """Applies forward propagation to input arrays on CPU.

        Args:
            inputs: Tuple of :class:`numpy.ndarray` object(s).

        Returns:
            tuple: Tuple of :class:`numpy.ndarray` object(s).

        .. warning::

            Implementations of :class:`Function` must take care that the
            return value must be a tuple even if it returns only one array.

        """
        raise NotImplementedError()

    def forward_gpu(self, inputs):
        """Applies forward propagation to input arrays on GPU.

        Args:
            inputs: Tuple of :class:`cupy.ndarray` object(s).

        Returns:
            tuple: Tuple of :class:`cupy.ndarray` object(s).

        .. warning::

            Implementations of :class:`Function` must take care that the
            return value must be a tuple even if it returns only one array.

        """
        raise NotImplementedError()

    def backward(self, inputs, grad_outputs):
        """Applies backprop to output gradient arrays.

        It delegates the procedure to :meth:`backward_cpu` or
        :meth:`backward_gpu` by default. Which it selects is determined by the
        type of input arrays and output gradient arrays. Implementations of
        :class:`Function` must implement either CPU/GPU methods or this method,
        if the function is intended to be backprop-ed.

        Args:
            inputs: Tuple of input arrays.
            grad_outputs: Tuple of output gradient arrays.

        Returns:
            tuple: Tuple of input gradient arrays. Some or all of them can be
            ``None``, if the function is not differentiable on
            inputs.

        .. warning::

            Implementations of :class:`Function` must take care that the
            return value must be a tuple even if it returns only one array.

        """
        if any(isinstance(x, cuda.ndarray) for x in inputs + grad_outputs):
            return self.backward_gpu(inputs, grad_outputs)
        else:
            return self.backward_cpu(inputs, grad_outputs)

    def backward_cpu(self, inputs, grad_outputs):
        """Applies backprop to output gradient arrays on CPU.

        Args:
            inputs: Tuple of input :class:`numpy.ndarray` object(s).
            grad_outputs: Tuple of output gradient :class:`numpy.ndarray`
                object(s).

        Returns:
            tuple: Tuple of input gradient :class:`numpy.ndarray` object(s).
            Some or all of them can be ``None``, if the function is not
            differentiable on corresponding inputs.

        .. warning::

            Implementations of :class:`Function` must take care that the
            return value must be a tuple even if it returns only one array.

        """
        return tuple(None for _ in inputs)

    def backward_gpu(self, inputs, grad_outputs):
        """Applies backprop to output gradient arrays on GPU.

        Args:
            inputs: Tuple of input :class:`cupy.ndarray`
                object(s).
            grad_outputs: Tuple of output gradient
                :class:`cupy.ndarray` object(s).

        Returns:
            tuple: Tuple of input gradient :class:`cupy.ndarray`
            object(s). Some or all of them can be ``None``, if the function is
            not differentiable on corresponding inputs.

        .. warning::

            Implementations of :class:`Function` must take care that the
            return value must be a tuple even if it returns only one array.

        """
        return tuple(None for _ in inputs)

    def unchain(self):
        """Purges in/out nodes and this function itself from the graph.

        See :meth:`FunctionNode.unchain() <chainer.FunctionNode.unchain>`
        for the detail.

        """
        self.node.unchain()

    def add_hook(self, hook, name=None):
        """Registers a function hook.

        See :meth:`FunctionNode.add_hook` for the detail.

        Args:
            hook(~chainer.FunctionHook):
                Function hook to be registered.
            name(str): Name of the function hook.
                name must be unique among function hooks
                registered to the function. If ``None``,
                default name of the function hook is used.

        """
        self.node.add_hook(hook, name)

    def delete_hook(self, name):
        """Unregisters the specified function hook.

        Args:
            name(str): the name of the function hook
                to be unregistered.

        """
        self.node.delete_hook(name)

    def retain_inputs(self, indexes):
        """Lets specified input variable nodes keep data arrays.

        By calling this method from :meth:`forward`, the function can specify
        which inputs are required for backprop.

        If this method is not called, the function keeps all input arrays. If
        you want to release all input arrays, call this method by passing an
        empty sequence. *Note that this behavior is different from that of*
        :meth:`FunctionNode.retain_inputs() \
        <chainer.FunctionNode.retain_inputs>`.

        Note that **this method must not be called from the outside of**
        :meth:`forward`.

        Args:
            indexes (iterable of int): Indexes of input variables that the
                function will require for backprop.

        """
        self.node.retain_inputs(indexes)

    def retain_outputs(self, indexes, retain_after_backward=False):
        """Lets specified output variable nodes keep data arrays.

        By calling this method from :meth:`forward`, the function can specify
        which outputs are required for backprop. If this method is not called,
        any output variables are not marked to keep the data array at the point
        of returning from :meth:`__call__`. The retained arrays are stored to
        :attr:`output_data`.

        .. note::
           It is STRONGLY RECOMMENDED to use this method if the function
           requires some or all output arrays in backprop. The function can
           also use output arrays just by keeping references to them directly,
           whereas it might influence on the performance of later function
           applications to the output variables.

        Note that **this method must not be called from the outside of**
        :meth:`forward`.

        Args:
            indexes (iterable of int): Indexes of input variables that the
                function will require for backprop.

            retain_after_backward (bool): This option has no effect. It is
                left only for the backward compatibility.

        """
        if retain_after_backward:
            warnings.warn('retain_after_backward option has no effect',
                          DeprecationWarning)
        self.node.retain_outputs(indexes)
