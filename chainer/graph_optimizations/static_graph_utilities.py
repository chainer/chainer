import inspect

import chainer


def static_code(*dec_args, **dec_kwargs):
    """Decorator to mark a function for inclusion in the static schedule.

    This decorator is used to mark a function or method to be included
    in a static schedule. There are multiple types of static schedules, such
    as "forward pass schedule", "backward pass schedule", "double backward
    pass schedule" etc.. The type of schedule that the decorated function's
    code is added to will depend on the context in which this decorator
    is used. For example, the decorated code will be added to the
    "forward pass schedule" if it is called while executing the define-by-
    run code of a static subgraph. To inform the framework that a particular
    portion of define-by-run code corresponds to a static subgraph, the
    code should be placed inside the `__call__()` method of a chain and
    then apply the `@static_graph` decorator to the `__call__()` method.
    We will refer to such a chain as a "static chain."
    This will cause any functions
    decorated with `static_code` that are called while inside of `__call__()`
    to be included in the forward pass static
    schedule in the same order in which they were executed in the
    define-by-run code.

    Likewise, for any `FunctionNode` instances that are called inside
    a static chain, any code that is run while inside the `backward()`
    method that calls a function using this decorator will be added to
    the corresponding "backward pass schedule."

    Usage:

    This decorator should be applied to any code called from a static chain
    that needs to run each
    iteration. This should only include the code that performs
    the actual forward and/or backward computations and not include code
    for initializing parameters, checking types, etc..

    As long as a chain is marked as static, the framework
    will automatically wrap any `FunctionNode` instances so that the
    code inside their `forward()` and `backward()` methods is added to
    the corresponding forward and backward static schedules, respectively.
    As a result, any built-in Chainer function and
    link calls will be automatically included in the static schedule.

    However, there are two cases where the user will need to use this
    decorator:

    1. Code with side effects that is called from a static chain's define-by-
    run code must be placed in a function decorated with `@static_code`.

    2. Any user-defined links that contain code other chain Chainer
    function calls that must run every iteration must place such code
    in a function decorated with `@static_graph`.


    This decorator can be applied to either a function or a method (usually
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

    Passing and returing arrays:

    If the function needs an array as an input argument that was
    used elsewhere in the static schedule, it must appear as an
    item in list of arrays that is supplied in the `inputs` keyword
    argument. An example would be the typical case where one layer
    in the network produces output activations `y` which are then
    used as the input of the next layer. If the corresponding
    `FunctionNode` instances wrap their computaions using this decorator,
    this will result in multiple functions that operate on `y`.
    The following constraints specify how
    such arrays should be passed into and returned from a function
    that uses this decorator.

    If the function will return results in one or more arrays, there are
    two options:

    1. Write the results in-place into preallocated arrays that are
    supplied in a list in the `outputs` keyword argument.

    2. Dynamically allocate the result array(s) inside the function
    and return them inside a tuple.

    When two schedule functions
    "func_A" and "func_B" operate on the same array `x`,
    `x` must explicitly appear as an input argument and/or output
    of both functions. For
    example, it would be an error to have schedule function "func_A"
    return a dynamically allocated array `x` and then have schedule
    function "func_B" later
    read from `x` without it appearing in "func_B"'s `inputs` list.
    Note that this would work during the first iteration, but during
    the next iteration when "func_A" is called, it would allocate and
    return a new array for `x` leading to "func_B" reading from the
    stale reference to `x` from the previous iteration. This
    usage is allowed in some special cases by the framework code, but
    is not allowed for user-defined functions.

    Performance notes:

    The function should return any output arrays in-place
    into pre-allocated arrays (1. above) when possible since this this allows
    the scheduler to make tradeoffs
    between computation efficiency and memory usage.
    For example, this allows the use of a
    completely static array allocations (no allocations after the first
    iteration), if desired. If memory reduction is needed, the
    scheduler may choose to delete the arrays in `inputs` once they are no
    longer
    needed in an iteration and then reallocate them again in the next
    iteration just before the function is called. Note that
    completely static array allocations are not possible if
    any of the schedule functions return a tuple of dynamically allocated
    arrays, as the existing chainer functions do.

    The following optional arguments apply to the wrapped function or method.

    Args (of this decorater):
        func_name (str): An optional descriptive name that will be associated
            with this function in the static schedule. It is intended
            for debugging purposes.

    Args (of the wrapped fuction):
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

    Check if `func` supports static graph optimizations. If not,
    automatically wrap it to be compatible.

    This function is called from the `FunctionNode` apply() method
    in place of the original `func.forward(inputs)` call if
    `chainer.config.schedule_func` is not None.

    Args:
        func (instance of FunctionNode):
        inputs (tuple of ndarray): input arrays to `func`

    Returns:
        (tuple of ndarray): The outputs of the function.
    """

    schedule_function = chainer.config.schedule_func
    if not func._supports_static_optimizations:
        if schedule_function.verbosity_level >= 2:
            print('Adding automatic static graph support to '
                  'function: ', func)

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
