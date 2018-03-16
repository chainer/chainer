import collections
import heapq
import traceback
import weakref

import numpy
import six

import chainer
from chainer import _backprop_utils
from chainer.backends import cuda
from chainer import configuration
from chainer import function_hook
from chainer.utils import experimental
from chainer.utils import type_check
from chainer import variable


class FunctionNode(object):

    """Function node of the computational graph.

    FunctionNode is a class representing a node in a computational graph. The
    node corresponds to an application of a differentiable function to input
    variables.

    When a differentiable function is applied to :class:`~chainer.Variable`
    objects,
    it creates an instance of FunctionNode implementation and calls its
    :meth:`apply` method. The :meth:`apply` method basically does the following
    three things.

    1. Adding an edge from the function node to the variable node corresponding
       to each input. The node of each input is extracted by
       :attr:`Variable.node <chainer.Variable.node>`.
    2. Computing the output arrays of the function.
    3. Creating a :class:`~chainer.Variable` object for each output array and
       adding an edge from the node of the variable to the function node.

    The output variables are then returned.

    .. admonition:: Example

       Let ``x`` be an instance of :class:`~chainer.Variable` and ``f`` be an
       instance of :class:`FunctionNode` taking only one argument.
       Then the following code

       >>> import numpy, chainer, chainer.functions as F
       >>> x = chainer.Variable(numpy.zeros(10))
       >>> f = F.Identity()
       >>> y = f.apply((x,))[0]

       computes a new variable ``y`` and creates backward references. The
       backward references are actually set as per the following diagram::

           x.node <--- f <--- y.node

       If an application of another function ``g`` occurs as

       >>> g = F.Identity()
       >>> z = g.apply((x,))[0]

       then the graph grows with a branch::

                    |--- f <--- y.node
           x.node <-+
                    |--- g <--- z.node

       Note that the branching is correctly managed on backward computation,
       i.e. the gradients from ``f`` and ``g`` are accumulated to the gradient
       of ``x``.

    Every function-node implementation should provide :meth:`forward` and
    :meth:`backward`. Instead of overriding :meth:`forward`, one can also
    implement :meth:`forward_cpu` and :meth:`forward_gpu` when the
    implementations for CPU and GPU arrays are totally different.

    Note that the input and output variables are inaccessible from
    :meth:`backward` by default. If it needs accesses to these variables, the
    :meth:`forward` method (or its CPU/GPU variants) has to call
    :meth:`retain_inputs` and :meth:`retain_outputs` appropriately. The
    retained input/output variables can be accessed from :meth:`backward` by
    calling :meth:`get_retained_inputs` and :meth:`get_retained_outputs`.

    .. note::

       There are two types of differentiable functions in Chainer (since v3).
       The first type is of a function using a subclass of
       :class:`~chainer.Function`,
       which is called *old-style differentiable function*. The second type is
       of a function using a subclass of :class:`FunctionNode`, which is called
       **new-style differentiable function**. There are several advantages on
       using the new-style differentiable function.

       - The new-style differentiable function supports *differentiable
         backpropagation*. The backpropagated gradients computed through the
         new-style differentiable functions themselves support further
         backpropagations so that the automatic higher-order differentiation is
         available.
       - The backpropagation of the new-style differentiable function can be
         more computationally efficient because the interface allows an
         implementation to omit the computation of unneeded input gradients.

       Note that the new-style differentiable function is the standard way of
       defining a function node of the computational graph in Chainer; old-
       style differentiable functions are implemented as wrappers of the new-
       style differentiable functions.

    Attributes:
        ~FunctionNode.inputs: A tuple of the input
            :class:`~chainer.variable.VariableNode` objects.
        ~FunctionNode.outputs: A tuple of weak references to the output
            :class:`~chainer.variable.VariableNode` objects.
        ~FunctionNode.rank (int): An ordinal following the topological order
            of the computational graph.
        ~FunctionNode.stack: Stack trace retrieved at the forward computation.
            The stack trace is available only in the debug mode.

    .. versionadded:: 3.0.0

    """

    inputs = None
    outputs = None
    rank = 0
    stack = None
    _input_indexes_to_retain = None
    _output_indexes_to_retain = None
    _retained_output_data = None
    _local_function_hooks = None
    lazy_grad_sum = False

    @property
    def local_function_hooks(self):
        """Ordered dictionary of registered function hooks.

        Contrary to ``chainer.thread_local.function_hooks``,
        which registers its elements to all functions,
        Function hooks in this property is specific to this function.

        """
        if self._local_function_hooks is None:
            self._local_function_hooks = collections.OrderedDict()
        return self._local_function_hooks

    @property
    def _n_local_function_hooks(self):
        return (0 if self._local_function_hooks is None
                else len(self._local_function_hooks))

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

        This property is mainly used by :class:`Function`. Users basically do
        not have to use this property; use :meth:`get_retained_outputs`
        instead.

        """
        if self._retained_output_data is None:
            raise RuntimeError('retained output data is gone')
        out_data = [None] * len(self.outputs)
        for index, data in six.moves.zip(self._output_indexes_to_retain,
                                         self._retained_output_data):
            out_data[index] = data
        return tuple(out_data)

    @property
    def _impl_name(self):
        return self.__class__.__name__

    def __call__(self, *args, **kwargs):
        if self.__class__.__module__.startswith('chainer.'):
            msg = '''\
Chainer's built-in function class object ({}) which is derived from \
chainer.FunctionNode has been called as if it were a callable. \
Use FunctionNode.apply() method instead.
Furthermore, it's not recommended to use built-in function classes directly; \
use corresponding function aliases (those with snake_case name, such as \
F.convolution_nd) instead.\
'''.format(self.__class__.__name__)
        else:
            msg = '''\
A function class object ({}) which is derived from \
chainer.FunctionNode has been called as if it were a callable. \
Use apply() method instead.\
'''.format(self.__class__.__name__)

        raise RuntimeError(msg)

    def apply(self, inputs):
        """Computes output variables and grows the computational graph.

        Basic behavior is expressed in the documentation of
        :class:`FunctionNode`.

        .. note::

           If the :data:`~Variable.data` attribute of input variables exist on
           a GPU device, that device is made current before calling
           :meth:`forward`, so implementors do not need to take care of device
           selection in most cases.

        Args:
            inputs: Tuple of input variables. Each element can be either
                :class:`~chainer.Variable`, :class:`numpy.ndarray`,
                or :class:`cupy.ndarray`. If the element is an ndarray, it is
                automatically wrapped with :class:`~chainer.Variable`.

        Returns:
            A tuple of output :class:`~chainer.Variable` objects.

        """
        input_vars = [chainer.as_variable(x) for x in inputs]
        in_data = tuple([x.data for x in input_vars])
        requires_grad = any([x.requires_grad for x in input_vars])

        # Check for input array types
        if not chainer.is_arrays_compatible(in_data):
            raise TypeError(
                'incompatible array types are mixed in the forward input '
                '({}).\n'
                'Actual: {}'.format(
                    self.label,
                    ', '.join(str(type(x)) for x in in_data)))

        is_debug = chainer.is_debug()
        if is_debug:
            # Keep stack trace for debug
            self.stack = traceback.extract_stack()

        if configuration.config.type_check:
            self._check_data_type_forward(in_data)

        hooks = chainer.get_function_hooks()
        if self._n_local_function_hooks > 0:
            hooks = collections.OrderedDict(hooks)
            hooks.update(self.local_function_hooks)
        hooks = hooks.values()  # avoid six for performance

        for hook in hooks:
            hook.forward_preprocess(self, in_data)

        # Forward propagation
        with cuda.get_device_from_array(*in_data):
            self._input_indexes_to_retain = None
            self._output_indexes_to_retain = None
            outputs = self.forward(in_data)

        # Check for output array types
        if not isinstance(outputs, tuple):
            raise TypeError(
                'forward output must be a tuple ({})\n'
                'Actual: {}'.format(self.label, type(outputs)))

        if not chainer.is_arrays_compatible(outputs):
            raise TypeError(
                'incompatible array types are mixed in the forward output '
                '({}).\n'
                'Actual: {}'.format(
                    self.label,
                    ', '.join(str(type(x)) for x in outputs)))

        for hook in hooks:
            hook.forward_postprocess(self, in_data)

        # NaN check of output values
        if is_debug:
            if any(out.dtype.kind == 'f' and
                   cuda.get_array_module(out).isnan(out).any()
                   for out in outputs):
                msg = ('NaN is detected on forward computation of '
                       '{}'.format(self.label))
                raise RuntimeError(msg)

        ret = tuple([variable.Variable(y, requires_grad=requires_grad)
                     for y in outputs])

        if configuration.config.enable_backprop:
            # Topological ordering
            self.rank = max([x.rank for x in input_vars]) if input_vars else 0
            # Add backward edges
            for y in ret:
                y.creator_node = self
            self.inputs = tuple([x.node for x in input_vars])
            # Add forward edges (must be weak references)
            self.outputs = tuple([weakref.ref(y.node) for y in ret])

            if self._input_indexes_to_retain is not None:
                for index in self._input_indexes_to_retain:
                    input_vars[index].retain_data()

            if self._output_indexes_to_retain is not None:
                retained_data = []
                for index in self._output_indexes_to_retain:
                    ret[index].retain_data()
                    retained_data.append(outputs[index])
                self._retained_output_data = tuple(retained_data)

            self.lazy_grad_sum = configuration.config.lazy_grad_sum
            if self.lazy_grad_sum:
                experimental('config.lazy_grad_sum')

        return ret

    def _check_data_type_forward(self, in_data):
        in_type = type_check.get_light_types(in_data)
        try:
            with type_check.light_mode:
                self.check_type_forward(in_type)
            return
        except type_check.InvalidType:
            # Ignore errors on first run
            pass

        in_type = type_check.get_types(in_data, 'in_types', False)
        with type_check.get_function_check_context(self):
            self.check_type_forward(in_type)

    def check_type_forward(self, in_types):
        """Checks types of input data before forward propagation.

        This method is called before :meth:`forward` and validates the types of
        input variables using
        :ref:`the type checking utilities <type-check-utils>`.

        Args:
            in_types (~chainer.utils.type_check.TypeInfoTuple): The type
                information of input variables for :meth:`forward`.

        """
        pass

    def forward(self, inputs):
        """Computes the output arrays from the input arrays.

        It delegates the procedure to :meth:`forward_cpu` or
        :meth:`forward_gpu` by default. Which of them this method selects is
        determined by the type of input arrays. Implementations of
        :class:`FunctionNode` must implement either CPU/GPU methods or this
        method.

        Args:
            inputs: Tuple of input array(s).

        Returns:
            Tuple of output array(s).

        .. warning::

            Implementations of :class:`FunctionNode` must take care that the
            return value must be a tuple even if it returns only one array.

        """
        assert len(inputs) > 0
        if isinstance(inputs[0], cuda.ndarray):
            return self.forward_gpu(inputs)
        return self.forward_cpu(inputs)

    def forward_cpu(self, inputs):
        """Computes the output arrays from the input NumPy arrays.

        Args:
            inputs: Tuple of input :class:`numpy.ndarray` objects.

        Returns:
            Tuple of output arrays. Each element can be NumPy or CuPy arrays.

        .. warning::

            Implementation of :class:`FunctionNode` must take care that the
            return value must be a tuple even if it returns only one array.

        """
        raise NotImplementedError

    def forward_gpu(self, inputs):
        """Computes the output arrays from the input CuPy arrays.

        Args:
            inputs: Tuple of input :class:`cupy.ndarray` objects.

        Returns:
            Tuple of output arrays. Each element can be NumPy or CuPy arrays.

        .. warning::

            Implementation of :class:`FunctionNode` must take care that the
            return value must be a tuple even if it returns only one array.

        """
        raise NotImplementedError

    def retain_inputs(self, indexes):
        """Lets specified input variable nodes keep data arrays.

        By calling this method from :meth:`forward`, the function node can
        specify which inputs are required for backprop. The input variables
        with retained arrays can then be obtained by calling
        :meth:`get_retained_inputs` from inside :meth:`backward`.

        Unlike :class:`~chainer.Function`, the function node **DOES NOT** keep
        input
        arrays by default. If you want to keep some or all input arrays, do not
        forget to call this method.

        Note that **this method must not be called from the outside of**
        :meth:`forward`.

        Args:
            indexes (iterable of int): Indexes of input variables that the
                function will require for backprop.

        """
        self._input_indexes_to_retain = indexes

    def retain_outputs(self, indexes):
        """Lets specified output variable nodes keep data arrays.

        By calling this method from :meth:`forward`, the function node can
        specify which outputs are required for backprop. If this method is not
        called, no output variables will be marked to keep their data array at
        the point of returning from :meth:`apply`. The output variables with
        retained arrays can then be obtained by calling
        :meth:`get_retained_outputs` from inside :meth:`backward`.

        .. note::

           It is recommended to use this method if the function requires some
           or all output arrays in backprop. The function can also use output
           arrays just by keeping references to them directly, although it
           might affect the performance of later function applications on the
           output variables.

        Note that **this method must not be called from the outside of**
        :meth:`forward`.

        Args:
            indexes (iterable of int): Indexes of output variables that the
                function will require for backprop.

        """
        self._output_indexes_to_retain = indexes

    def backward(self, target_input_indexes, grad_outputs):
        """Computes gradients w.r.t.\\  specified inputs given output gradients.

        This method is used to compute one step of the backpropagation
        corresponding to the forward computation of this function node.
        Given the gradients w.r.t. output variables, this method computes the
        gradients w.r.t. specified input variables. Note that this method does
        not need to compute any input gradients not specified by
        ``target_input_indices``.

        Unlike :meth:`Function.backward() <chainer.Function.backward>`,
        gradients are given as  :class:`~chainer.Variable` objects and this
        method itself has to return input gradients as
        :class:`~chainer.Variable` objects. It enables the function node to
        return the input gradients with the full computational history, in
        which case it supports *differentiable backpropagation* or
        *higher-order differentiation*.

        The default implementation returns ``None`` s, which means the
        function is not differentiable.

        Args:
            target_input_indexes (tuple of int): Indices of the input variables
                w.r.t. which the gradients are required. It is guaranteed that
                this tuple contains at least one element.
            grad_outputs (tuple of :class:`~chainer.Variable`\\ s): Gradients
                w.r.t. the output variables.
                If the gradient w.r.t. an output variable is not
                given, the corresponding element is ``None``.

        Returns:
            Tuple of variables that represent the gradients w.r.t. specified
            input variables. The length of the tuple can be same as either
            ``len(target_input_indexes)`` or the number of inputs. In the
            latter case, the elements not specified by ``target_input_indexes``
            will be discarded.

        .. seealso::

           :meth:`backward_accumulate` provides an alternative interface that
           allows you to implement the backward computation fused with the
           gradient accumulation.

        """
        return (None,) * len(target_input_indexes)

    def backward_accumulate(self, target_input_indexes, grad_outputs,
                            grad_inputs):
        """Computes gradients w.r.t.\\  specified inputs and accumulates them.

        This method provides a way to fuse the backward computation and the
        gradient accumulations in the case that the multiple functions are
        applied to the same variable.

        Users have to override either of this method or :meth:`backward`.
        It is often simpler to implement :meth:`backward` and is recommended
        if you do not need to provide efficient gradient accumulation.

        Args:
            target_input_indexes (tuple of int): Indices of the input variables
                w.r.t. which the gradients are required. It is guaranteed that
                this tuple contains at least one element.
            grad_outputs (tuple of Variable): Gradients w.r.t. the output
                variables. If the gradient w.r.t. an output variable is not
                given, the corresponding element is ``None``.
            grad_inputs (tuple of Variable): Gradients w.r.t. the input
                variables specified by ``target_input_indexes``. These values
                are computed by other computation paths. If there is no
                gradient value existing for the variable, the corresponding
                element is ``None``. See also the note below.

        Returns:
            Tuple of variables that represent the gradients w.r.t. specified
            input variables. Unlike :meth:`backward`, the length of the tuple
            **must** be same as that of ``target_input_indices``.

        .. note::

           When the same variable is passed to the multiple input arguments of
           a function, only the first position of ``grad_inputs`` corresponding
           to these input arguments may contain the gradient variable
           corresponding to that input variable, and other entries are set to
           ``None``. This is an implementation-detail convention to avoid the
           complication of correctly accumulating gradients in such a case.
           This behavior might be changed in a future version.

        """
        assert isinstance(target_input_indexes, tuple)
        assert isinstance(grad_outputs, tuple)
        assert isinstance(grad_inputs, tuple)

        # The default implementation uses backward(). You can override this
        # method without using backward().
        gxs = self.backward(target_input_indexes, grad_outputs)

        len_gxs = len(gxs)
        if len_gxs == len(self.inputs):
            gxs = tuple([gxs[i] for i in target_input_indexes])
        elif len_gxs != len(target_input_indexes):
            raise ValueError(
                'number of gradients returned by %s (%s) is incorrect.'
                % (self._impl_name, self.label))

        if self.lazy_grad_sum:
            gxs_output = ()
            for i, (gx, g_input) in enumerate(six.moves.zip(gxs, grad_inputs)):
                sum_gx = _backprop_utils.concat_variable(gx, g_input)
                j = target_input_indexes[i]
                if self.inputs[j].creator is None and \
                        isinstance(sum_gx, tuple):
                    sum_gx = chainer.functions.add(*sum_gx)
                gxs_output += sum_gx,
            return gxs_output
        else:
            return tuple([gx if g_input is None else
                          g_input if gx is None else
                          gx + g_input
                          for gx, g_input in six.moves.zip(gxs, grad_inputs)])

    def get_retained_inputs(self):
        """Returns a tuple of retained input variables.

        This method is used to retrieve the input variables retained in
        :meth:`forward`.

        Returns:
            A tuple of retained input variables.

        """
        inputs = self.inputs
        return tuple([inputs[index].get_variable()
                      for index in self._input_indexes_to_retain])

    def get_retained_outputs(self):
        """Returns a tuple of retained output variables.

        This method is used to retrieve the output variables retained in
        :meth:`forward`.

        Returns:
            A tuple of retained output variables.

        .. note::

           This method does a tricky thing to support the case of an output
           node garbage-collected before this method is called; in this case,
           this method creates a fresh variable node that acts as an output
           node of the function node.

        """
        ret = []
        outputs = self.outputs

        new_outputs = list(outputs)
        outputs_modified = False
        for index, data in six.moves.zip(self._output_indexes_to_retain,
                                         self._retained_output_data):
            output = outputs[index]()
            if output is None:
                # The output node is garbage collected, so create a fresh
                # Variable object.
                output_var = variable.Variable(data)
                output_var.creator_node = self
                new_outputs[index] = weakref.ref(output_var)
                outputs_modified = True
            else:
                output_var = output.get_variable()
            ret.append(output_var)

        if outputs_modified:
            self.outputs = tuple(new_outputs)

        return tuple(ret)

    def unchain(self):
        """Purges in/out nodes and this function node itself from the graph."""
        for y in self.outputs:
            y_ref = y()
            if y_ref is not None:
                y_ref.unchain()
        self.inputs = None

    def add_hook(self, hook, name=None):
        """Registers a function hook.

        Args:
            hook (~chainer.FunctionHook): Function hook to be
                registered.
            name (str): Name of the function hook. The name must be unique
                among function hooks registered to this function. If ``None``,
                the default name of the function hook is used.

        """
        if not isinstance(hook, function_hook.FunctionHook):
            raise TypeError('Hook must be of type FunctionHook')
        if name is None:
            name = hook.name
        hooks = self.local_function_hooks
        if name in hooks:
            raise KeyError('Hook %s already exists' % name)
        hooks[name] = hook
        hook.added(function=self)

    def delete_hook(self, name):
        """Unregisters the function hook.

        Args:
            name (str): The name of the function hook to be unregistered.

        """
        if name in self.local_function_hooks:
            self.local_function_hooks[name].deleted(function=self)
            del self.local_function_hooks[name]
        else:
            raise KeyError('Hook %s does not exist' % name)


def grad(outputs, inputs, grad_outputs=None, grad_inputs=None, set_grad=False,
         retain_grad=False, enable_double_backprop=False, loss_scale=None):
    """Computes the gradient of output variables w.r.t.\\  the input variables.

    This function implements the backpropagation algorithm. While
    :meth:`Variable.backward` also implements backprop, this function selects
    the smallest paths in the computational graph needed to compute the
    gradients w.r.t. inputs. The error is backpropagated only through these
    selected paths, which may reduce the overall computational cost.

    This function also differs from :meth:`Variable.backward` in the way to
    return the gradients; it directly returns the gradient variables as a list
    instead of setting gradients to the :attr:`Variable.grad_var` attribute of
    the original variable. It means users do not need to clear the gradient
    w.r.t. each variable before computing the gradient using this function.
    If ``set_grad`` option is set to ``True``, the computed gradient is also
    stored in the :attr:`Variable.grad_var` attribute of each variable, in
    which case any original value of :attr:`Variable.grad_var` will be updated
    even if it had already been set.

    Args:
        outputs (tuple or list of :class:`~chainer.Variable`):
            A sequence of output variables from which backprop starts.
        inputs (tuple or list of :class:`~chainer.Variable`):
            A sequence of input variables each of which this function computes
            the gradient w.r.t.
        grad_outputs (tuple or list of :class:`~chainer.Variable` or None):
            A sequence of variables that gives the initial value of each output
            gradient.
            If an element is set to ``None``, an array filled with 1 is used.
            If this argument itself is ``None``, it is treated as a sequence of
            ``None``\\ s.
        grad_inputs (tuple or list of :class:`~chainer.Variable` or None):
            A sequence of variables that gives the initial value of each input
            gradient. The gradients computed by the backprop
            algorithm are accumulated to them (not in-place). If an element
            is set to ``None``, the gradient is not accumulated to this value.
            If this argument itself is ``None``, it is treated as a sequence of
            ``None``\\ s.
        set_grad (bool): If it is ``True``, the :attr:`Variable.grad_var`
            attribute of each input variable is set to the corresponding
            computed gradient variable.
        retain_grad (bool): If it is ``True``, the gradients w.r.t. all the
            intermediate variables are stored in the :attr:`Variable.grad_var`
            attribute. In this case, the ``set_grad`` option is ignored.
        enable_double_backprop (bool): If it is ``True``, the computed
            gradients can be further backpropagated. Enabling it may increase
            the memory consumption (and possibly the computational time) to
            remember the intermediate gradient values for the second
            backpropagation.
        loss_scale (float): Loss scaling factor. Loss scaling is a usefull
            technique to mitigate vanishing gradient issue that tends to happen
            when low precision data type like float16 is used during training.
            If you set loss scaling factor, gradients of loss values are to be
            multiplied by the factor before backprop starts. The factor is
            propagated to whole gradients in a computational graph along the
            backprop. The gradients of parameters are divided by the factor
            just before the parameters are to be updated.

    Returns:
        A list of gradient variables w.r.t. the inputs.

    """
    if not isinstance(outputs, (tuple, list)):
        raise TypeError(
            'outputs must be a tuple or a list, not {}.'.format(type(outputs)))
    if not isinstance(inputs, (tuple, list)):
        raise TypeError(
            'inputs must be a tuple or a list, not {}.'.format(type(inputs)))
    if not (grad_outputs is None or isinstance(grad_outputs, (tuple, list))):
        raise TypeError(
            'grad_outputs must be a tuple or a list or None, not {}.'.format(
                type(grad_outputs)))
    if not (grad_inputs is None or isinstance(grad_inputs, (tuple, list))):
        raise TypeError(
            'grad_inputs must be a tuple or a list or None, not {}.'.format(
                type(grad_inputs)))

    for v in outputs:
        # Raise error here if v is created by Function.backward.
        # In such case, we don't know exact inputs of the creator.
        v.node._check_old_style_gradient()

    # The implementation consists of three steps.

    # 1. Backward enumeration: all the nodes reachable backward from the output
    #    nodes are enumerated. The forward direction links are collected in
    #    this step. Note that the variable nodes whose requires_grad is false
    #    are ignored and their creators are not searched.
    candidate_funcs = [v.creator_node for v in outputs
                       if v.creator_node is not None]
    visited_funcs = set()
    forward_graph = collections.defaultdict(list)
    while candidate_funcs:
        func = candidate_funcs.pop()
        if func in visited_funcs:
            continue
        visited_funcs.add(func)
        for x in func.inputs:
            # Raise error here if x is created by Function.backward.
            # In such case, we don't know exact inputs of the creator.
            x._check_old_style_gradient()

            if not x.requires_grad:
                continue
            forward_graph[x].append(func)
            creator = x.creator_node
            if creator is not None and creator not in visited_funcs:
                candidate_funcs.append(creator)

    # 2. Forward enumeration: all the nodes in the subgraph reachable from the
    #    input nodes are enumerated. The extracted (sub-)subgraph is the union
    #    of all paths that backpropagation will visit.
    candidate_vars = [x.node for x in inputs]
    visited_funcs = set()
    grad_required = set()
    while candidate_vars:
        x = candidate_vars.pop()
        grad_required.add(x)
        for func in forward_graph[x]:
            if func in visited_funcs:
                continue
            visited_funcs.add(func)
            for y_ref in func.outputs:
                y = y_ref()
                if y is not None and y in forward_graph:
                    candidate_vars.append(y)

    # 3. Backpropagation: the backpropagation is executed along the
    #    (sub-)subgraph. It uses the topological order of the subgraph which is
    #    induced by the reversed order of function applications ("rank").
    grads = {}  # mapping from variable nodes to their gradients

    # Initialize the gradient mapping.
    if grad_outputs is None:
        grad_outputs = (None,) * len(outputs)
    for y, gy in zip(outputs, grad_outputs):
        if gy is None:
            with cuda.get_device_from_array(y.data) as device:
                if device is cuda.DummyDevice:
                    gy_data = numpy.ones_like(y.data)
                else:
                    gy_data = cuda.cupy.ones_like(y.data)
                gy = variable.Variable(gy_data, requires_grad=False)
            if loss_scale is not None:
                gy.data *= loss_scale
        grads[y.node] = gy

    if grad_inputs is not None:
        for x, gx in zip(inputs, grad_inputs):
            if gx is not None:
                grads[x.node] = gx

    # Backprop implementation. It edits grads which will only contain the
    # gradients w.r.t. the inputs.
    with chainer.using_config('enable_backprop', enable_double_backprop):
        _backprop(outputs, inputs, grad_required, retain_grad, grads,
                  loss_scale)

    # Extract the gradients w.r.t. the inputs and return them.
    ret = [grads.get(x.node, None) for x in inputs]
    if set_grad:
        for x, gx in zip(inputs, ret):
            x.grad_var = gx

    return ret


def _backprop(outputs, inputs, grad_required, retain_grad, grads, loss_scale):
    candidate_funcs, push_candidate, pop_candidate = _get_ordered_func_heap()

    for y in outputs:
        creator = y.creator_node
        if creator is not None:
            push_candidate(creator)

    input_nodes = set(x.node for x in inputs)

    while candidate_funcs:
        func = pop_candidate()

        # Collect the gradients w.r.t. the outputs
        gys = []
        for y_ref in func.outputs:
            y = y_ref()
            if y is None:
                # output is not a part of the selected subgraph and has already
                # been released.
                gys.append(None)
                continue
            gys.append(grads.get(y, None))
        gys = tuple(gys)

        # Collect the gradients w.r.t. the inputs
        #
        # Note (Tokui): when the same variable is passed multiple times as
        # inputs in the same function (e.g. an expression like f(x, x)), the
        # current implementation passes None as the current gradient w.r.t.
        # such an input except for the first one (i.e., it builds gxs like
        # (gx, None) where gx is the current gradient w.r.t. x).
        gxs = []
        input_indexes = []
        selected_inputs = set()
        for i, x in enumerate(func.inputs):
            if x not in grad_required:
                continue
            input_indexes.append(i)
            if x in selected_inputs:
                gxs.append(None)
            else:
                gxs.append(grads.get(x, None))
                selected_inputs.add(x)
        gxs = tuple(gxs)
        input_indexes = tuple(input_indexes)

        if not input_indexes:
            continue

        # Do backward
        gys = tuple([gy if not isinstance(gy, tuple) else
                     chainer.functions.add(*gy)
                     for gy in gys])
        new_gxs = func.backward_accumulate(input_indexes, gys, gxs)

        # Delete output gradients that are not required to return
        for y_ref in func.outputs:
            y = y_ref()
            if y is not None and y in grads and y not in input_nodes:
                del grads[y]

        # Update grads
        selected_inputs = set()
        for i, g in zip(input_indexes, new_gxs):
            if g is None:
                continue

            node = func.inputs[i]
            if node in selected_inputs:
                # Accumulate the duplicated gradients here
                cur_gx = grads.get(node, None)
                if cur_gx is not None:
                    if func.lazy_grad_sum:
                        if x.creator is None:
                            g = _backprop_utils.add(g, cur_gx)
                        else:
                            g = _backprop_utils.concat_variable(g, cur_gx)
                    # cur_gx can't be tuple, the lazy_grad_sum can't
                    # be enabled in its sibling node.
                    else:
                        g = g + cur_gx
            else:
                selected_inputs.add(node)

            grads[node] = g

            if retain_grad:
                v = node.get_variable_or_none()
                if v is not None:
                    v.grad_var = g
                    v._loss_scale = loss_scale

            creator = node.creator_node
            if creator is not None:
                push_candidate(creator)


def _get_ordered_func_heap():
    heap = []
    visited_funcs = set()

    def push_heap(func):
        if func not in visited_funcs:
            # Negate since heapq is min-heap
            # The second element is used to make each item unique
            ordered_func = -func.rank, len(visited_funcs), func
            visited_funcs.add(func)
            heapq.heappush(heap, ordered_func)

    def pop_heap():
        _, _, func = heapq.heappop(heap)
        return func

    return heap, push_heap, pop_heap
