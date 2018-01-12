import contextlib

import chainer

# These function are intended to by called from chainer.FunctionNode and
# chainer.Variable. They should not be directly called from user code.


def is_static_func(func):
    """Check if the function node is included in a static schedule.

    Returns:
        bool: True if the supplied function is included in a static
            schedule. Otherwise, return False.
    """
    return hasattr(func, 'schedule_func')


def get_static_schedule(func):
    """Get the forward static schedule that contains the supplied function node.

    If the supplied function node is contained in a static schedule, return
    the static schedule. Otherwise, return ``None``. Note in order for
    ``func`` to be contained in a static schedule, ``func`` must have already
    been called in the forward pass from a ``@static_graph``-decorated
    chain.

    Args:
        func (FunctionNode): The supplied function node.

    Returns:
        StaticScheduleFunction or None: Depending on whether or not the
        supplied function is contained in a static schedule.
    """
    return getattr(func, 'schedule_func', None)


def is_trace_mode():
    """Check if trace mode is on.

    If this function is called by the define-by-run code of a @static_graph
    decorated ``__call__()`` of a chain, return True.

    Returns:
        bool: True if trace mode is on. Otherwise, return False.
    """
    return chainer.config.schedule_func is not None


def mark_static_vars(input_vars):
    """Mark variables as static if inside a static chain.

    If trace mode is currently on, set the ``is_static`` attribute of
    each variable in ``input_vars`` to True.

    Args:
        input_vars (list of variable): The supplied list of variables
            (including parameters).

    """
    if is_trace_mode():
        for var in input_vars:
            # todo: consider only marking a variable if it is a parameter.
            var.is_static = True


def static_schedule_func(*dec_args, **dec_kwargs):
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
        delay_call (bool): Optional keyword argument. If True, don't call the wrapped
        function during the define by run pass, but only add it to the static schedule.
        Default value is False.

        func: A forward-pass method of a sub-class of FunctionNode that will be inserted
            into the static forward schedule when `static_graph` is enabled. The function
            must not return anything because any return values will be ignored.

    Returns: The wrapped function.

    """
    delay_call = False
    zero_args = False
    if len(dec_args) == 1 and not dec_kwargs and callable(dec_args[0]):
        func = dec_args[0]
        zero_args = True
    elif dec_kwargs:
        if 'delay_call' in dec_kwargs:
            delay_call = dec_kwargs['delay_call']

    def wrap(func):
        def wrapped_func(*args, **kwargs):
            # Save arguments, function, and results pointers/references to the schedule list:
            def no_arg_func():
                # print('In no_arg_func: Calling: ', func)
                ret = func(*args, **kwargs)
                if ret is not None:
                    raise RuntimeError("This function is not supposed to return anything: ", func)
                # print("Arguments were: %s, %s" % (args, kwargs))

            # no_arg_func() requires no arguments to call since the arguments of the decorated function
            # are captured by the closure.
            if not delay_call:
                no_arg_func()

            schedule_function = chainer.config.schedule_func
            # If trace mode is on, add to schedule.
            if schedule_function is not None:
                schedule_function.append_function(no_arg_func)
                # Add the schedule function as an attribute of the FunctionNode instance
                # that contains the wrapped function as a method
                # This attribute will be needed by the corresponding @static_backward
                # function.
                instance = args[0]
                # assert isinstance(instance, chainer.function_node.FunctionNode)
                instance._supports_static_optimizations = True
                print('Adding function to static schedule: ', func)
                # print('static_forward: instance: ', instance)
                instance.schedule_func = schedule_function

        return wrapped_func

    if zero_args:
        return wrap(func)
    else:
        return wrap


def static_forward_optimizations(func, in_vars, in_data, outputs):
    """Perform checks needed for creation of a static schedule.

    For each variable ``x`` in ``in_vars``, check if ``x`` is an
    input variable to a static chain. If so, then save the
    information to the function so that it can be used during the
    backward pass schedule creation.

    Also check if `func` supports static graph optimizations. If not, try
    to automatically wrap it to be compatible.

    This function should be called from the ``FunctionNode`` apply() method
    just after func.forward() is called.

    Args:
        func (instance of FunctionNode):
        in_vars (tuple of variable): input variables to func.apply()
        in_data (tuple of ndarray): input arrays to func
        outputs (tuple of ndarray): outputs of func.

    """

    schedule_function = chainer.config.schedule_func
    if schedule_function is not None:
        for func_arg_index, var in enumerate(in_vars):
            if id(var.data) in schedule_function._input_var_array_to_static_array_index:
                chain_arg_index = schedule_function._input_var_array_to_static_array_index[id(var.data)]
                # Add this index information to the func_node so that it can be used in
                # backward() to copy corresponding gradient outputs into static arrays.
                forward_static_arrays_info = getattr(func, '_forward_static_arrays_info', None)
                if forward_static_arrays_info is None:
                    forward_static_arrays_info = list()
                    func._forward_static_arrays_info = forward_static_arrays_info
                forward_static_arrays_info.append((func_arg_index, chain_arg_index))

        if not func._supports_static_optimizations:
            print("Adding automatic static graph support to function: ",
                func)
            # func does not support static optimizations, so let's try to wrap it
            # automatically.
            @static_schedule_func(delay_call=True)
            def generic_static_forward(func, in_data, out_data):
                """

                in_arrs: tuple of input arrays
                out_arrs: tuple of output arrays
                func: compatible with out_arrs = func(in_arrs)
                """
                temp_out_data = func.forward(in_data)
                assert len(temp_out_data) == len(out_data)
                for ind, static_ar in enumerate(out_data):
                    static_ar[:] = temp_out_data[ind]

            generic_static_forward(func, in_data, outputs)


def static_forward_optimizations_old(func, in_vars):
    """Perform checks needed for creation of a static schedule.

    For each variable ``x`` in ``in_vars``, check if ``x`` is an
    input variable to a static chain. If so, then save the
    information to the function so that it can be used during the
    backward pass schedule creation.

    This function should be called from the ``FunctionNode`` apply() method
    just after func.forward() is called.

    Args:
        in_vars (iterable of chainer.Variable):
        func (FunctionNode):
    """

    schedule_function = chainer.config.schedule_func
    if schedule_function is not None:
        for func_arg_index, var in enumerate(in_vars):
            if id(var.data) in schedule_function._input_var_array_to_static_array_index:
                chain_arg_index = schedule_function._input_var_array_to_static_array_index[id(var.data)]
                # Add this index information to the func_node so that it can be used in
                # backward() to copy corresponding gradient outputs into static arrays.
                forward_static_arrays_info = getattr(func, '_forward_static_arrays_info', None)
                if forward_static_arrays_info is None:
                    forward_static_arrays_info = list()
                    func._forward_static_arrays_info = forward_static_arrays_info
                forward_static_arrays_info.append((func_arg_index, chain_arg_index))

        if not func._supports_static_optimizations:
            raise RuntimeError(
                "The following function was called inside a static chain but it does not support static optimizations: ",
                func)


def check_func_backward_outputs(func, grad_outputs):
    """Update schedule information if conditions are satisfied.

    If the supplied function node created output variables of a static chain
    during the forward pass, just before performing the backward pass,
    add information to the backward schedule. Specifically, the
    backward schedule is updated to contain information so that its
    input variables (i.e., ``grad_outputs``) can be first copied into
    statically-allocated arrays. This copy operation will need to be
    performed on each iteration of the backward schedule.
    this function does not actually perform the copy operation.

    Args:
        func (FunctionNode): The supplied function node.
        grad_outputs (tuple of Variable): The input gradients for the
        backward method of ``func``. These correspond to the "outputs"
        of ``func``.
    """
    backward_static_arrays_info = getattr(func, '_backward_static_arrays_info', None)
    if backward_static_arrays_info is not None:
        forward_schedule = get_static_schedule(func)
        backward_schedule = forward_schedule.get_backward_schedule_func()
        #print('Found _backward_static_arrays_info during static_bakcward().')
        for func_arg_index, chain_arg_index in backward_static_arrays_info:
            input_var = grad_outputs[func_arg_index]
            # Modify the data attribute of input_var to refer to a statically allocated array.
            backward_schedule._in_arrays[chain_arg_index][:] = input_var.data
            input_var.data = backward_schedule._in_arrays[chain_arg_index]


def check_func_backward_inputs(func, grad_inputs):
    """Update schedule information if conditions are satisfied.

    If any of the input variables to ``func`` (in forward pass) are also input variables to the
    static sub-graph (i.e., static chain), update the backward schedule for
    the sub-graph. Specifically, the ``data`` array references from such
    variables will be copied into the static schedule for easy access
    when the schedule is run.

    Args:
        func (FunctionNode): The supplied function node.
        grad_inputs (tuple of Variable or None): The output gradients from the
        backward method of ``func``. These correspond to the "inputs"
        of ``func`` in the forward pass.
        One or more of the items in the tuple are allowed to be None.

    """
    # Check if func_node returns any variables that should have their
    # data attributes copied into the static outputs array of the
    # backward schedule.
    forward_static_arrays_info = getattr(func, '_forward_static_arrays_info', None)
    if forward_static_arrays_info is not None:
        forward_schedule = get_static_schedule(func)
        backward_schedule = forward_schedule.get_backward_schedule_func()
        #print('Found static_arrays_list in backward(): ', forward_static_arrays_info)
        for func_arg_index, chain_arg_index in forward_static_arrays_info:
            # Need to make the chain_arg_index'th output array of the schedule refer
            # to the array of the variable in chain_arg_index'th position of the
            # grad_outputs tuple.
            # Note: if the input variable of the static chain was input to
            # multiple function in the forward pass, then the following
            # static array reference will be set multiple time for the same
            # variable. This is fine, though, since only the final reference
            # is needed for the output gradients from the static schedule.

            # assert backward_schedule._out_arrays[chain_arg_index] is None

            # Since the data array was allocated statically, we must return a copy.
            # Note: In Chainer, it is allowed for one or more of the grad inputs
            # to be None instead of a Variable.
            if grad_inputs[func_arg_index] is not None:
                backward_schedule._out_arrays[chain_arg_index] = grad_inputs[func_arg_index].data
                grad_inputs[func_arg_index].data = grad_inputs[func_arg_index].data.copy()