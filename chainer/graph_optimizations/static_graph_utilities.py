import inspect

import chainer


def static_code(*dec_args, **dec_kwargs):
    """Decorator to mark a function for inclusion in the static schedule.

    This decorator is used to mark a function or method to be included
    in the static schedule. This will only occur if the function is
    called (directly or deeply) from inside a static chain's `__call__()`
    method. That is, if a chain's `__call__()` uses the `@static_graph`
    decorator, then any code that is executed while inside `__call__()`
    that uses `@static_code` will be included in the corresponding
    static schedule. Such code will be added to the static schedule in the
    order that it was called.

    This decorator should be applied to any code that needs to run each
    iteration. This should ideally only include the code that performs
    the actual forward and/or backward computations, and not include code
    for initializaing parameters, checking types, etc.. If the user would
    like to include any other code in a static chain's `__call__()` method
    that needs to run every iteration, then it should also use this
    decorator.

    Usage:

    This decorator can be applied to either a function or a method (typically
    of a `FunctionNode`). There are no required arguments, and so a user can
    apply it to "side effect" code to cause an operation to be executed each
    iteration. The more usual use case is where the core framework code
    will apply it to the all of (and only) the functions
    that actually perform the computations needed to compute the forward
    and backward passes.

    The simplest usage is when we would like to force a particular
    user-defined function to run each iteration. For example, such a function
    might increment a counter, check conditions, and possibly print
    information to the console. In this use, it is only required to add
    this decorator to the function definition and then call it during
    the first iteration from the context of the static chain's
    `__call__()` method.

    There are no constraints on the function's
    arguments, unless the function needs to access, modify, and/or
    create ndarrays that are used by other functions in the static
    schedule. In these cases, the following contraints specify how
    such arrays should be passed into and returned from a function
    that uses this decorator.

    Passing and returing arrays:

    If the function needs an array as an input argument that was
    used elsewhere in the static schedule, it must appear as an
    item in list of arrays that is supplied in the `inputs` keyword
    argument.

    If the function will return results in one or more arrays, there are
    two options:
        1. Write the results in-place into preallocated arrays that are
            supplied in a list in the `outputs` keyword argument.
        OR
        2. Dynamically allocate the result array(s) inside the function
            and return them inside a tuple.

    Note: Care must be taken for the case where two schedule functions
    "func_A" and "func_B" operate on the same array `x`. In such cases,
    `x` must explicitly appear in the `inputs` list, `outputs` list,
    or returned tuple of both "func_A" and "func_B". For
    example, it would be an error to have schedule function "func_A"
    return a dynamically allocated array `x` and then have schedule
    function "func_B" later
    read from `x` without it appearing in "func_B"'s `inputs` list.
    Note that this would work during the first iteration, but during
    the next iteration when "func_A" is called, it would allocate and
    return a new array for `x` leading to "func_B" reading from the
    stale reference to `x` from the previous iteration. Actually, such
    usage is allowed in some special cases by the framework code, but
    is not allowed for user-defined functions.

    Performance notes:

    It is suggested to have the function return any output arrays in-place
    into pre-allocated arrays (1. above) when possible since this provides
    the most flexability to the scheduler to make various computation speed
    vs memory usage tradeoffs. For example, this allows the use of a
    completely static array allocations (no allocations after the first
    iteration), if desired. However, if memory reduction is needed, the
    scheduler may delete arrays in `inputs` once they are no longer
    needed in an iteration and then reallocate them again in the next
    iteration just before the function is called. Note, however, that
    completely static array allocations if of course not possible if
    any of the schedule functions return a tuple of dynamically allocated
    arrays.

    The following optional arguments apply to the wrapped function or method.

    Args:
        inputs (list of ndarray): An optional keyword argument that
            supplies all arrays needed as input by the function. If the
            function needs an array that is used by another function
            in the static schedule, it must appear in this list.
        outputs (list of ndarray): An optional keyword argument that
            supplies all output arrays of this function. These arrays
            must already have been initialized to the correct shapes
            and dtypes before the function is called. The function
            must write its results in-place into these arrays. Any
            output arrays that may be used inside another schedule
            function must appear in this list.
    Returns:
        None or a tuple of ndarray: If the function dynamically
            allocates its output arrays, they must be returned in a tuple
            of arrays.

    """
    func_name = None
    zero_args = False
    if len(dec_args) == 1 and not dec_kwargs and callable(dec_args[0]):
        callable_arg = dec_args[0]
        zero_args = True
    elif dec_kwargs:
        if 'func_name' in dec_kwargs:
            func_name = dec_kwargs['func_name']

    def wrap(func):
        def wrapped_func(*args, **kwargs):
            # Save arguments, function, and results pointers/references
            # to the schedule list:
            # If trace mode is on, add to schedule.
            schedule_function = chainer.config.schedule_func
            if schedule_function is not None:
                # Note: 'ret = func(*args, **kwargs)' is called inside
                # the following method.
                ret = schedule_function.append_function(func, args, kwargs,
                                                        func_name=func_name)
                # Add the schedule function as an attribute of the
                # FunctionNode instance (or more generally, to any class)
                # that contains the wrapped function as a method
                if len(args) > 0:
                    instance = args[0]
                    if inspect.isclass(instance):
                        # note: this is not currently needed.
                        instance.schedule_func = schedule_function
            else:
                ret = func(*args, **kwargs)
            return ret
        return wrapped_func
    if zero_args:
        return wrap(callable_arg)
    else:
        return wrap


def static_forward_optimizations(func, inputs):
    """Perform checks needed for creation of a static schedule.

    Check if `func` supports static graph optimizations. If not, try
    to automatically wrap it to be compatible.

    This function should be called from the ``FunctionNode`` apply() method
    in place of the original `func.forward(inputs)` call.

    Args:
        func (instance of FunctionNode):
        inputs (tuple of ndarray): input arrays to `func`

    Returns:
        (tuple of ndarray): The outputs of the function.
    """

    schedule_function = chainer.config.schedule_func
    if schedule_function is not None:
        if not func._supports_static_optimizations:
            if schedule_function.verbosity_level >= 2:
                print("Adding automatic static graph support to "
                      "function: ", func)

            @static_code(func_name=str(func))
            def generic_static_forward(func, inputs):
                """Auto-wrap the supplied function.

                func (instance of FunctionNode): The function to include in
                    the static schedule.
                inputs (list of input arrays): The input arguments to `func`.

                Returns: a tuple of output arrays.

                """
                # Convert back to tuple because func.forward() requires it.
                in_data = tuple(inputs)
                ret = func.forward(in_data)
                return ret

            # Note: we convert inputs to a list because the API for
            # static_code requires it.
            return generic_static_forward(func, inputs=list(inputs))
    return func.forward(inputs)
