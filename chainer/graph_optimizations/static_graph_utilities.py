import contextlib

import chainer


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

def static_forward_optimizations(func, in_data):
    # Check if any of the input arrays correspond to input
    # variables to a static chain. If so, replace these arrays
    # with statically-allocated arrays of the static schedule.
    schedule_function = chainer.config.schedule_func
    if schedule_function is not None:

        in_arrays = list(in_data)
        # Check if any of the input arrays correspond to a (dynamically
        # allocated) data attribute of an in variable to this subgraph.
        for func_arg_index in range(len(in_arrays)):
            in_array = in_arrays[func_arg_index]
            chain_arg_index, static_array = schedule_function.copy_input_arrays_dynamic_to_static(in_array)
            if chain_arg_index is not None:
                # Replace with the static array in arguments list.
                assert in_arrays[func_arg_index].shape == static_array.shape
                in_arrays[func_arg_index] = static_array
                # Add this index information to the func_node so that it can be used in
                # backward() to copy corresponding gradient outputs into static arrays.
                forward_static_arrays_info = getattr(func, '_forward_static_arrays_info', None)
                if forward_static_arrays_info is None:
                    forward_static_arrays_info = list()
                    func._forward_static_arrays_info = forward_static_arrays_info
                forward_static_arrays_info.append((func_arg_index, chain_arg_index))

        mod_in_data = tuple(in_arrays)
        outputs = func.forward(mod_in_data)
        # trace mode is on, so check that func is compatible:
        if not func._supports_static_optimizations:
            raise RuntimeError(
                "The following function was called inside a static chain but it does not support static optimizations: ",
                func)
        return outputs

    else:
        return func.forward(in_data)

def check_func_backward_outputs(func, grad_outputs):
    """Update schedule information is conditions are satisfied.

    If the supplied function produces output variables of a static sub-graph,
    update the backward schedule for the sub-graph. Specifically, the
    backward schedule is updated to contain information so that its
    input variables (i.e., ``grad_outputs``) can be first copied into
    statically-allocated arrays. This copy operation will need to be
    performed on each execution of the backward schedule. Note that
    this function does not actually perform the copy operation.

    Args:
        func (FunctionNode): The supplied function node.
        grad_outputs (tuple of Variable): The input gradients for the
        backward method of ``func``. These correspond to the "outputs"
        of ``func``.
    """
    if hasattr(func, '_backward_static_arrays_info'):
        forward_schedule = get_static_schedule(func)
        backward_schedule = forward_schedule.get_backward_schedule_func()
        backward_static_arrays_info = getattr(func, '_backward_static_arrays_info', None)
        print('Found _backward_static_arrays_info during static_bakcward().')
        for func_arg_index, chain_arg_index in backward_static_arrays_info:
            # Get data array from the input variable (this is
            # allocated dynamically outside the static chain)
            data = grad_outputs[func_arg_index].data
            backward_schedule._input_var_array_to_static_array_index[id(data)] = chain_arg_index


def check_func_backward_inputs(func, grad_inputs):
    """Update schedule information is conditions are satisfied.

    If any of the input variables to ``func`` are also input variables to the
    static sub-graph (i.e., static chain), update the backward schedule for
    the sub-graph. Specifically, the ``data`` array references from such
    variables will be copied into the static schedule for easy access
    when the schedule is run.

    Args:
        func (FunctionNode): The supplied function node.
        grad_inputs (tuple of Variable): The output gradients from the
        backward method of ``func``. These correspond to the "inputs"
        of ``func``.

    """
    # Check if func_node returns any variables that should have their
    # data attributes copied into the static outputs array of the
    # backward schedule.
    forward_static_arrays_info = getattr(func, '_forward_static_arrays_info', None)
    if forward_static_arrays_info is not None:
        forward_schedule = get_static_schedule(func)
        backward_schedule = forward_schedule.get_backward_schedule_func()
        print('Found static_arrays_list in backward(): ', forward_static_arrays_info)
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
            backward_schedule._out_arrays[chain_arg_index] = grad_inputs[func_arg_index].data