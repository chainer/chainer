import collections
import traceback
import weakref

import six

import chainer
from chainer import configuration
from chainer import cuda
from chainer import function_hook
from chainer.utils import type_check
from chainer import variable


class FunctionNode(object):

    """Function node of the computational graph.

    FunctionNode is a class representing a node in a computational graph. The
    node corresponds to an application of a differentiable function to input
    variables.

    When a differentiable function is applied to :class:`Variable` objects,
    it creates an instance of FunctionNode implementation and calls its
    :meth:`apply` method. The :meth:`apply` method basically does the following
    three things.

    1. Adding an edge from the function node to the variable node corresponding
       to each input. The node of each input is extracted by
       :attr:`Variable.node`.
    2. Computing the output arrays of the function.
    3. Creating a :class:`Variable` object for each output array and adding an
       edge from the node of the variable to the function node.

    The output variables are then returned.

    .. admonition:: Example

       Let ``x`` be an instance of :class:`Variable` and ``f`` be an instance
       of :class:`FunctionNode` taking only one argument. Then a line

       >>> import numpy, chainer, chainer.functions as F
       >>> x = chainer.Variable(numpy.zeros(10))
       >>> f = F.Identity()
       >>> y = f.apply((x,))[0]

       computes a new variable ``y`` and creates backward references. Actually,
       backward references are set as per the following diagram::

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
       The first type is of a function using a subclass of :class:`Function`,
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
         more computationally saving because the interface allows an
         implementation to omit the computation of unneeded input gradients.

       Note that the new-style differentiable function is the standard way of
       defining a function node of the computational graph in Chainer; old-
       style differentiable functions are implemented as wrappers of the new-
       style differentiable functions.

    Attributes:
        inputs: A tuple of the input :class:`VariableNode` objects.
        outputs: A tuple of weak references to the output
            :class:`VariableNode` objects.
        rank (int): An ordinal following the topological order of the
            computational graph.
        stack: Stack trace retrieved at the forward computation. The stack
            trace is available only in the debug mode.

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
                :class:`Variable`, :class:`numpy.ndarray`,
                or :class:`cupy.ndarray`. If the element is an ndarray, it is
                automatically wrapped with :class:`Variable`.

        Returns:
            A tuple of output :class:`Variable` objects.

        """
        input_vars = [x if isinstance(x, variable.Variable)
                      else variable.Variable(x, requires_grad=False)
                      for x in inputs]
        in_data = tuple([x.data for x in input_vars])
        requires_grad = any([x.requires_grad for x in input_vars])

        if chainer.is_debug():
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
            assert type(outputs) is tuple

        for hook in hooks:
            hook.forward_postprocess(self, in_data)

        # NaN check of output values
        if chainer.is_debug():
            if any(out.dtype.kind == 'f' and
                   cuda.get_array_module(out).isnan(out).any()
                   for out in outputs):
                msg = 'NaN is detected on forward computation'
                raise RuntimeError(msg)

        ret = tuple([variable.Variable(y, requires_grad=requires_grad)
                     for y in outputs])

        if configuration.config.enable_backprop:
            # Topological ordering
            self.rank = max([x.rank for x in input_vars]) if input_vars else 0
            # Add backward edges
            for i, y in enumerate(ret):
                y.creator_node = self
            self.inputs = twrdple([x.node for x in input_vars])
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
        with retained arrays can be obtained by :meth:`get_retained_inputs`
        from :meth:`backward`.

        Unlike :class:`Function`, the function node **DOES NOT** keep input
        arrays by default. If you want to keep some or all input arrays, do not
        forget to call this method.

        Note that **this method must not be called from the outside of
        forward method.**

        Args:
            indexes (iterable of int): Indexes of input variables that the
                function does not require for backprop.

        """
        self._input_indexes_to_retain = indexes

    def retain_outputs(self, indexes):
        """Lets specified output variable nodes keep data arrays.

        By calling this method from :meth:`forward`, the function node can
        specify which outputs are required for backprop. If this method is not
        called, any output variables are not marked to keep the data array at
        the point of returning from :meth:`apply`. The output variables with
        retained arrays can be obtained by :meth:`get_retained_outputs` from
        :meth:`backward`.

        .. note::

           It is recommended to use this method if the function requires some
           or all output arrays in backprop. The function can also use output
           arrays just by keeping references to them directly, whereas it might
           influence on the performance of later function applications to the
           output variables.

        Note that **this method must not be called from the outside of
        forward method.**

        Args:
            indexes (iterable of int): Indexes of input variables that the
                function does not require for backprop.

        """
        self._output_indexes_to_retain = indexes

    def backward(self, target_input_indexes, grad_outputs):
        """Computes gradients w.r.t. specified inputs given output gradients.

        This method is used to compute one step of the backpropagation
        corresponding to the forward computation of this function node.
        Given the gradients w.r.t. output variables, this method computes the
        gradients w.r.t. specified input variables. Note that this method does
        not need to compute any input gradients not specified by
        ``target_input_indices``.

        Unlike :meth:`Function.backward`, gradients are given as
        :class:`Variable` objects and this method itself has to return input
        gradients as :class:`Variable` objects. It enables the function node to
        return the input gradients with the full computational history, in
        which case it supports *differentiable backpropagation* or
        *higher-order differentiation*.

        The default implementation returns ``None`` s, which means the
        function is not differentiable.

        Args:
            target_input_indexes (tuple of int): Indices of the input variables
                w.r.t. which the gradients are required. It is guaranteed that
                this tuple contains at least one element.
            grad_outputs (tuple of Variable): Gradients w.r.t. the output
                variables. If the gradient w.r.t. an output variable is not
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
        """Computes gradients w.r.t. specified inputs and accumulates them.

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

        return ret

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
            hook (~chainer.function.FunctionHook): Function hook to be
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

    def delete_hook(self, name):
        """Unregisters the function hook.

        Args:
            name (str): The name of the function hook to be unregistered.

        """
        del self.local_function_hooks[name]
