import contextlib

import chainer
import chainer.function_node

# fixme: add an attribute/property to all function nodes that support static graph so
# that we can raise an exception if a function that does not support it is called
# from a static chain.

# fixme: check that all array copies use x[:] = y.

# fixme: clean up documentation and API.

# todo: add test that use the same random seed with two models: a static chain
# and a (non-static) chain. Enable `chainer.config.cudnn_deterministic` and
# run both models and check that the outputs are identical.

# todo: modify optimizers so that they never attempt to change the grad/data
# reference of a Parmaeter, such as setting to None.
# They should only copy zeros into the parameter
# when necessary, and copy updated parameter values into the .data attributes.

class StaticScheduleFunction(chainer.function_node.FunctionNode):
    """A function that executes the static schedule of a Chain.

    An instance of this class executes the static schedule of computations
    that are equivalent to executing the define-by-run code of a Chain.

    This class is used by the `static_graph` decorator to wrap the define-by-run
    computations of a chain into two static schedules:
    - The forward schedule corresponds to the computations that are executed by
    the define-by-run code of the `__call__()` method of a chain. The static
    schedule corresponding to these computations can be executed by calling the
    `forward()` method of this class.
    - The backward schedule corresponds to the computations that are executed
    by the sequence of calls to `Function.backward()` that occur during when
    backpropagating the gradients through the same chain. That is, for each
    `Function.forward()` that was called during the forward propagation,
    there will be a corresponding call to `Function.backward()` (of the
    same Function object) in the backward schedule. This backward schedule
    can be executed by calling the `backward()` method of this class.

    Note the intended usage of this class:
    During the first forward pass of a static chain (that is, a chain that is
    decorated by `static_graph`) the define-by-run code is executed. However,
    for subsequent iterations, that define-by-run code is replaced by an instance
    of this function and this function will be called instead. Since the static
    schedules contained by this function perform the same computations, it is
    safe (and potentially much more efficient) to simply execute the static schedule instead
    of the define-by-run code. See `static_graph` for details.

    Args:
        in_vars (tuple of Variable): The tuple of input variables that is supplied to
            `__call__()` method of the chain that this schedule corresponds to.

        is_forward (bool): The type of static schedule (i.e., forward or backward).
    """

    def __init__(self, in_vars, is_forward=True):
        assert isinstance(in_vars, tuple)
        self._static_schedule_forward = []
        #self._static_schedule_backward = []
        self._backward_schedule_func = None
        self._chain_in_vars = in_vars

        # fixme: remove underscore.
        self._in_arrays = tuple([x.data.copy() if isinstance(x, chainer.Variable) else x.copy() for x in in_vars])

        # This maps the id of an array (from the data attribute of an a variable
        # that is an input of the sub-graph corresponding to this static schedule)
        # to the corresponding static array index.
        self._input_var_array_to_static_array_index = dict()
        if is_forward:
            # In the case that this is a forward schedule, in_vars contains
            # the values that will be used during the forward pass. We will
            # copy these values into the corresponding static arrays

            # For each input variable, save the id of its data array as the key
            # in a dictionary that maps to the corresponding static array.
            # This is needed so that the copy from input variable to static
            # array can take place before commputing the functions in the
            # static schedule, which will cause these functions to refer
            # to the static arrays rather than the (dynamically) allocated
            # data attributes of the input variables.
            for n in range(len(in_vars)):
                self._input_var_array_to_static_array_index[id(in_vars[n].data)] = n
        else:
            # This is a backward schedule, so just zero all of the static arrays
            # for now.
            [x.fill(0) for x in self._in_arrays]
        self._backward_out_arrays = None
        self._backward_grad_outputs = None
        self._chain_return_vars = None
        self._is_forward = is_forward
        self._out_arrays = None # fixme: remove undescore.
        self._out_var_to_tuple_index = None

    def is_forward_schedule(self):
        """Return True if this is a forward schedule.

        """
        return self._is_forward

    def copy_input_arrays_dynamic_to_static(self, x):
        """Copy an input to FunctionNode.forward() to a static array.

        If ``x`` corresponds to the data attribute of an input variable to the
        static subgraph corresponding to this schedule, copy it into the
        corresponding static array of this schedule. Otherwise, do nothing.

        Args:
            x (ndarray): An input array argument to FunctionNode.forward().

        Return:
            (index, ndarray): If a copy was performed, return the index within the
            input tuple of variables to this schedule's static subgraph. Also
            Return the static array.
            Otherwise, return (``None``, ``None``).
        """
        if id(x) in self._input_var_array_to_static_array_index:
            index = self._input_var_array_to_static_array_index[id(x)]
            if not self.is_forward_schedule():
                # We only need to copy if this is a backward schedule because
                # for a forward schedule, the copy was already performed in
                # the initializer.
                self._in_arrays[index][:] = x
            return index, self._in_arrays[index]
        return None, None

    def contains_backward_schedule_funct(self):
        """Check whether a backward schedule function exists.

        Returns:
            bool: True if a backward schedule function has already been
            created. Otherwise, returns False.
        """
        return self._backward_schedule_func is not None

    def create_backward_schedule_func(self, out_vars):
        """Create the backward schedule function.

        Creates a new `StaticScheduleFunction` instance that will be used for
        the backward static schedule. Note that ``out_vars`` corresponds to the
        forward-pass output variables that are returned by the `@static_graph`-decorated Chain.


        During the backward pass, the ``grad`` atributes of different variables
        having the same shapes as ``out_vars`` will be the inputs to the functions
        of the backward static schedule.

        Note that it is not safe
        to assume that the pointers to the backing arrays of the ``grad``
        arrays will remain the same over subsequent iterations. Therefore, all
        functions in the backward static schedule that read from these inputs
        should be sure to copy the ``grad`` arrays into statically-allocated arrays
        in the backward static schedule before executing the schedule.

        To ensure that these copy operations take place when reading from each
        input variable, this implementation follows the backward references from
        each variable in ``out_vars`` to its creator function and adds an attribute
        to the function that will then be checked by the `@static_backward` decorator during the
        backward pass, which will perform the needed copy operation and insert it into the static
        schedule.

        Agrs:
            out_vars(tuple of Variable): The tuple of output variables that is
            returned from the Chain that contains this forward schedule. Note
            that the tuple of input (gradient) variables to the created backward schedule
            will have the same shapes as ``out_vars``.

        Returns:
            StaticScheduleFunction: The backward schedule function node.
        """
        assert isinstance(out_vars, tuple)
        if self._backward_schedule_func is None:
            self._backward_schedule_func = StaticScheduleFunction(out_vars, is_forward=False)
        else:
            raise RuntimeError("Cannot create. A backward schedule already exists.")
        # Now follow the backward references for each variable in out_vars to find out which
        # function created it and also find the corresponding index of the variable in the
        # output tuple of the function. This information will then be used during the first
        # backward pass. Specifically, each gradients variable that corresponds to an input
        # argument of static_chain.backward() will need to have its `data` attribute first
        # copied into a corresponding static array before it is used by the backward-pass
        # functions. To accomplish this, we will mark each creator function of a variable in
        # `out_vars` with an attribute that contains the needed information.
        self._creator_funcs_to_backward_static_arrays = dict()
        for chain_arg_index in range(len(out_vars)):
            var = out_vars[chain_arg_index]
            creator_func = var.creator
            print('Creator function of current variable: ', creator_func)
            # Create a dictionary that will map the creator_function to another dictionary
            # that will map each positional index of the variable in the function's output tuple
            # to its corresponding static array.

            # In @static_backward, these inputs can be checked and added to
            # the dict(): ._input_var_array_to_static_array_index of the backward schedule.

            # Now find the positional index of var in the creator function's output tuple.
            var_id = id(var.node)
            for func_arg_index in range(len(creator_func.outputs)):
                creator_output_id = id(creator_func.outputs[func_arg_index]())
                if var_id == creator_output_id:
                    # Found the variable.
                    # During the backward pass, we will need to copy the
                    # `data` array from the func_arg_index position in the
                    # tuple of input gradients variables to the
                    # chain_arg_index'th static input array of the
                    # backward schedule.
                    # (func_arg_index, chain_arg_index)
                    backward_static_arrays_info = getattr(creator_func, '_backward_static_arrays_info', None)
                    if backward_static_arrays_info is None:
                        backward_static_arrays_info = list()
                        creator_func._backward_static_arrays_info = backward_static_arrays_info
                    backward_static_arrays_info.append((func_arg_index, chain_arg_index))

        return self._backward_schedule_func

    def get_backward_schedule_func(self):
        """Get the backward schedule function.

        Returns:
            static_graph.StaticScheduleFunction:
        """
        if self._backward_schedule_func is not None:
            return self._backward_schedule_func
        else:
            raise RuntimeError("Cannot return. A backward schedule does not exist.")

    def create_out_arrays(self, out_vars):
        """Create the static output arrays for the forward schedule.

        Create a tuple of statically-allocated arrays that will be used to
        temporarily hold the outputs before they are returned in dynamically-
        allocated arrays.

        This method must be called before the forward schedule can be executed.

        Args:
            out_vars (tuple of Varible): The output variables that
            are returned by the `__call__()` method of the chain that this schedule
            corresponds to.
        """
        if self.is_forward_schedule():
            # Since this is a forward schedule, the forward pass has already completed
            # and we can therefore use the `data` attributes from `out_vars` as
            # the static output arrays.
            self._out_arrays = tuple([y.data for y in out_vars])
        else:
            # Note that this method is called at the end of the forward pass of
            # the static schedule graph, but before the backward pass.
            # Thus, the output arrays are not yet avaiilable since
            # they will be allocated during the backward pass. So far now,
            # the best we can do is just set some None placeholders which
            # will be replaced by actual arrays in the backward pass.
            self._out_arrays = [None,] * len(out_vars)

    def append_forward_function(self, forward):
        """Append a function to the forward static schedule.

        Args:
            forward: The function to append to the schedule. The function
            should not take any arguments and should not return any results.

        """
        self._static_schedule_forward.append(forward)

    def append_backward_function(self, backward):
        """Append a function to the backward static schedule.

        Args:
            backward: The function to append to the schedule. The function
            should not take any arguments and should not return any results.

        """
        self._static_schedule_backward.append(backward)

    def forward(self, inputs):
        # Note: This method will be invoked every iteration starting from the second
        # iteration. That is because the corresponding define-by-run code runs instead
        # during the first iteration.
        #if self._backward_grad_outputs is None:
        #    self._backward_grad_outputs = tuple([x.grad for x in self._chain_return_vars])
        # Copy any external input arrays into the statically-allocated arrays:
        assert len(self._in_arrays) == len(inputs)
        for n in range(len(inputs)):
            # Debug:
            assert self._in_arrays[n].shape == inputs[n].shape
            assert self._in_arrays[n][0].dtype == inputs[n][0].dtype
            self._in_arrays[n][:] = inputs[n]

        # The following line should be the new performance bottleneck after the first iteration
        # has completed. Note that we have several options for the implementation:
        # - Simply iterate through the schedule (in Python), calling each function.
        # - Export the schedule to C/C++ code. The forward computation can then
        # potentially be performed without any dependency on Python.
        # - Optimize the schedule code in Cython, calling optimized C/C++ code.
        [x() for x in self._static_schedule_forward]
        # Retain all inputs? (fixme)
        #self.retain_inputs(range(len(inputs)))
        #return self._out_arrays

        # Return a copy of the static arrays here because it is possible that the
        # static chain containing this schedule is repeated several times in the
        # computation graph (such as when the static chain correspond to a single
        # time slice in an RNN).
        return tuple([y.copy() for y in self._out_arrays])

    def backward(self, target_input_indexes, grad_outputs):
        # Note: This method will be invoked every iteration starting from the second
        # iteration. That is because the corresponding define-by-run code runs instead
        # during the first iteration.
        #print('StaticScheduleFunction: backward()...')
        return self.get_backward_schedule_func().apply(grad_outputs)


def static_schedule_func(func):
    """Decorator to mark a function for inclusion in the forward schedule.

    This decorator is used to wrap a function `func` that is a forward-pass
    method of a sub-class of Function. This will cause it to be added to
    the forward static schedule when the `static_graph` feature is
    enabled on a Chain that deeply calls it from the chain's
    `__call__()` method.

    The function to be wrapped should only return `None` because any return value
    will be ignored. Instead of returning its results, any result arrays must
    be supplied as input arguments and must have already have been initialized
    to the appropriate dimensions and data types.

    Usage:

    Typical usage is to allocate any required arrays (Numpy or CuPy) in Python
    code in an instance of FunctionNode (See `LinearFunction` function for an example).
    Generally, this will involve first allocating storage for the results arrays
    in the `forward()` method of a sub-class of Function. Then, the
    FunctionNode.foward()
     method should call another
    (private) method that is wrapped using this decorator. The
    decorated function will take all required input and output arrays as
    arguments and will not return anything (that is, `None` will be implicitly 
    returned). 
    
    Note that by following this usage convention, all input and output activations,
    along with any parameter arrays will have been statically allocated by the
    end of the first forward pass. Since the the forward-pass functions that
    are used inside the forward static schedule (that is, the functions that
    use this decorator) do not allocate any results arrays, this results in code that
    looks like 'define-by-run' to the user, and which can be debugged during
    the first iteration, but then becomes static in terms of memory allocations and
    scheduling starting from the second iteration. Thus, we get the benefit of
    both ease of use and optimized performance.

    It is important that all of the required computations that occur during the
    second  and later forward passes must be contained inside the functions 
    that use this decorator. That is, any other code (that is not wrapped inside this
    decorator) in the various FunctionNode and Link instances can be viewed as
    setup code that only checks types, allocates result arrays, initializes
    parameters etc., but does not perform any computations that must
    be repeated after the first forward pass.

    The reason for this is that after the first iteration (that is, starting
    from the second forward pass), when the chain's `__call__()` is called,
    the forward static schedule will be invoked and it will only call the
    functions that were wrapped with this decorator. Note that this can potentially
    lead to difficult to find bugs if one forgets to decorate a required function,
    since the corresponding computations would no longer execute after the
    first iteration. As a general rule, any code that is intended to exectue on
    each iteration should be called by a function that uses this decorator.

    Restrictions:

    This feature currently assumes that the inputs to the wrapped function
    Will continue to have the same shapes and types each time it is called.
    There are currently no checks to ensure that this constraint is satisfied.
    Such a type check may be added in the future. For example, the current code
    will break if the mini-batch size changes at any point. 
    todo: add checks for this and run the define-by-run code again whenever any
    input sizes change. If such changes are frequent, we can consider caching multiple
    static schedules and using the appropriate one for the current input sizes. 

    Args:
        func: A forward-pass method of a sub-class of FunctionNode that will be inserted
            into the static forward schedule when `static_graph` is enabled. The function
            must not return anything because any return values will be ignored.

    Returns: The wrapped function.

    """
    def wrapped_func(*args, **kwargs):
        # Save arguments, function, and results pointers/references to the schedule list:
        def no_arg_func():
            #print('In no_arg_func: Calling: ', func)
            ret = func(*args, **kwargs)
            if ret is not None:
                raise RuntimeError("This function is not supposed to return anything: ", func)
            #print("Arguments were: %s, %s" % (args, kwargs))

        # no_arg_func() requires no arguments to call since the arguments of the decorated function
        # are captured by the closure.
        no_arg_func()

        #schedule_function = getattr(_thread_local, 'schedule_func', None)
        schedule_function = chainer.config.schedule_func
        # If trace mode is on, add to schedule.
        if schedule_function is not None:
            schedule_function.append_forward_function(no_arg_func)
            # Add the schedule function as an attribute of the FunctionNode instance
            # that contains the wrapped function as a method
            # This attribute will be needed by the corresponding @static_backward
            # function.
            instance = args[0]
            assert isinstance(instance, chainer.function_node.FunctionNode)
            instance._supports_static_optimizations = True
            print('Adding function to the forward static schedule.')
            #print('static_forward: instance: ', instance)
            instance.schedule_func = schedule_function

    return wrapped_func


def static_graph(func):
    """Decorator to mark a Chain's ``__call__()`` as a static sub-graph.

    This decorator marks the define-by-run code inside the `__call__()`
    method of a Chain instance as corresponding to a static computation
    graph or sub-graph. Only the top-most (that is, largest) static
    sub-graph should be decorated.

    This decorator will cause the wrapped `__call__()` method to
    execute its define-by-run code once (the first time it is called). 
    Subsequent calls
    will then invoke optimized code that performs the same computations
    but without the Python overhead of define-by-run. Such optimized
    code can potentially be executed as optimized C or C++ code, and
    potentially deployed to platforms that do not support Python (todo).

    Starting from the second iteration, the code inside ``__call__()``
    is basically replaced by a single call to a FunctionNode such that
    FunctionNode.forward() implements the forward static schedule and
    FunctionNode.backward() implements the backward static schedule.

    Usage:

    Apply this decorator only to the top-most Chain in the hierarchy that
    contains a static sub-graph. It is not necessary (and not allowed) to
    mark a chain as static if it is contained within
    another chain that is also marked as being static (todo: this is not checked yet). 
    That is, suppose a
    static graph `A` contains a static sub-graph `B`. Then, only the chain
    corresponding to `A` should be marked as static and the chain corresponding
    to `B` should not be marked as static.

    Notes:
        It is required to set retain_grad=True when calling loss.backward()
        on a model that uses the static graph feature. This is because
        the gradient arrays that were allocated during the first backward
        pass will be reused in the backward static schedule. If retain_grad
        were set to False, then these arrays would be set to None in
        `Variable.backward()` which would break the functionality.

    Args:
        forward: The forward `__call__()` method of an instance of Chain
        that is wrapped by this decorator.

    Returns:

    """
    def wrapped_func(*args, **kwargs):
        chain = args[0]
        in_vars = args[1:]
        if hasattr(chain, 'static_schedule'):
            # Call the optimized static schedule code.
            #print('This is the 2nd or greater iteration. Calling the optimized schedule...')
            # Note: out_vars are dynamically allocated because FunctionNode.apply()
            # will dynamically allocate variables on each call, which is the desired
            # behavior.
            out_vars = chain.static_schedule.apply(in_vars)
            if len(out_vars) == 1:
                out_vars, = out_vars
            return out_vars
        else:
            # This is the first iteration. Calling the define-by-run code.
            assert isinstance(chain, chainer.Chain)
            print('This is the first iteration. Calling the define-by-run code.: ', func)
            # First check that this chain is not called from inside another
            # static chain because it is not allowed.
            if chainer.config.schedule_func is not None:
                raise RuntimeError("Not allowed to nest static chains: ", chain)

            chain.static_schedule = StaticScheduleFunction(in_vars)
            with chainer.using_config('schedule_func', chain.static_schedule):
                out_vars = func(*args, **kwargs)

            print('Creating a new backward schedule function.')
            # Force out_vars to be a tuple of variables.
            if isinstance(out_vars, chainer.Variable):
                tuple_out_vars = out_vars,
            else:
                tuple_out_vars = out_vars
            chain.static_schedule.create_out_arrays(tuple_out_vars)
            backward_sched = chain.static_schedule.create_backward_schedule_func(tuple_out_vars)
            backward_sched.create_out_arrays(in_vars)
            #print("Arguments were: %s, %s" % (args, kwargs))
            return out_vars

    return wrapped_func
