import contextlib

import chainer
import chainer.function_node


# todo: add test that use the same random seed with two models: a static chain
# and a (non-static) chain. Enable `chainer.config.cudnn_deterministic` and
# run both models and check that the outputs are identical.


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
        schedule_manager (): fixme
        in_vars (tuple of Variable): The tuple of input variables that is supplied to
            `__call__()` method of the chain that this schedule corresponds to.

        is_forward (bool): The type of static schedule (i.e., forward or backward).
    """

    def __init__(self, schedule_manager, in_vars, is_forward=True):
        # todo: maybe need weak reference here for schedule_manager?
        self._schedule_manager = schedule_manager
        assert isinstance(in_vars, tuple)
        self._static_schedule_forward = []
        self._backward_schedule_func = None
        self._chain_in_vars = in_vars
        self._in_arrays = tuple([x.data.copy() for x in in_vars])

        # Maps the id of an array (from the data attribute of an a variable
        # that is an input of the sub-graph corresponding to this static schedule)
        # to the corresponding static array index.
        self._input_var_array_to_static_array_index = dict()
        if is_forward:
            # In the case that this is a forward schedule, in_vars contains
            # the values that will be used during the forward pass. We will
            # copy these values into the corresponding static arrays

            # For each input variable, save the id of its data array as the key
            # in a dictionary that maps to the corresponding static array.
            for n, var in enumerate(in_vars):
                var.data = self._in_arrays[n]
                self._input_var_array_to_static_array_index[id(var.data)] = n
        else:
            # This is a backward schedule, so just zero all of the static arrays
            # for now.
            for x in self._in_arrays:
                x.fill(0)
        self._backward_out_arrays = None
        self._backward_grad_outputs = None
        self._chain_return_vars = None
        self._is_forward = is_forward
        self._out_arrays = None
        self._out_var_to_tuple_index = None

    def is_forward_schedule(self):
        """Return True if this is a forward schedule.

        """
        return self._is_forward

    def copy_input_var_dynamic_to_static(self, x):
        """Copy an input to FunctionNode.forward() to a static array.

        If ``x.data`` corresponds to the data attribute of an input variable to the
        static subgraph corresponding to this schedule, copy it into the
        corresponding static array of this schedule and also make ``x.data``
        refer to its corresponding static array.

        Args:
            x (chainer.Variable): An input variable argument to FunctionNode.apply()
            or input gradients variable to the backward static FunctionNode.


        """
        if id(x.data) in self._input_var_array_to_static_array_index:
            index = self._input_var_array_to_static_array_index[id(x.data)]
            if not self.is_forward_schedule():
                # We only need to copy if this is a backward schedule because
                # for a forward schedule, the copy was already performed in
                # the initializer.
                self._in_arrays[index][:] = x.data
            x.data = self._in_arrays[index]

    def create_backward_schedule_func(self, out_vars):
        """Create the backward schedule function.

        Creates a new `StaticScheduleFunction` instance that will be used for
        the backward static schedule.

        During the backward pass, the ``grad`` atributes of different variables
        having the same shapes as ``out_vars`` will be the inputs to the functions
        of the backward static schedule.

        Note that it is not safe
        to assume that the references to the backing arrays of the ``grad``
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
            returned from the static chain of this forward schedule. Note
            that the tuple of input (gradient) variables to the created backward schedule
            will have the same shapes as ``out_vars``.

        Returns:
            StaticScheduleFunction: The backward schedule function node.
        """
        assert isinstance(out_vars, tuple)
        if self._backward_schedule_func is None:
            self._backward_schedule_func = StaticScheduleFunction(self._schedule_manager, out_vars, is_forward=False)
        else:
            raise RuntimeError("Cannot create. A backward schedule already exists.")
        # Now follow the backward references for each variable in out_vars to find out which
        # function created it and also find the corresponding index of the variable in the
        # output tuple of that function. This information will then be used during the first
        # backward pass. Specifically, each gradients variable that corresponds to an input
        # argument of static_chain.backward() will need to have its `data` attribute first
        # replaced by a corresponding static array before it is used by the backward-pass
        # functions. To accomplish this, we will mark each creator function of a variable in
        # `out_vars` with an attribute that contains the needed information.
        self._creator_funcs_to_backward_static_arrays = dict()
        for chain_arg_index in range(len(out_vars)):
            var = out_vars[chain_arg_index]
            creator_func = var.creator
            #print('Creator function of current variable: ', creator_func)
            if creator_func is None:
                print("Warning: Trying to create backward schedule but creator function of output variable of static chain is None!")
            else:
                # map each positional index of the variable in the function's output tuple
                # to its corresponding static array.

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

    def is_empty(self):
        """Return True if this schedule is empty.

        """
        return len(self._static_schedule_forward) == 0

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
            self._out_arrays = [y.data for y in out_vars]
        else:
            # Note that this method is called at the end of the forward pass of
            # the static schedule graph, but before the backward pass.
            # Thus, the output arrays are not yet avaiilable since
            # they will be allocated during the backward pass. So far now,
            # the best we can do is just set some None placeholders which
            # will be replaced by actual arrays in the backward pass.
            self._out_arrays = [None,] * len(out_vars)

    def append_function(self, forward):
        """Append a function to the (forward) static schedule.

        Args:
            forward: The function to append to the schedule. The function
            should not take any arguments and should not return any results.

        """
        if not self._is_forward:
            self._schedule_manager.end_forward()

        self._static_schedule_forward.append(forward)

    def forward(self, inputs):
        if not self._is_forward:
            self._schedule_manager.end_forward()

        # Note: This method will be invoked every iteration starting from the second
        # iteration. That is because the corresponding define-by-run code runs instead
        # during the first iteration.
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

        # Return a copy of the static arrays here because it is possible that the
        # static chain containing this schedule is repeated several times in the
        # computation graph (such as when the static chain correspond to a single
        # time slice in an RNN).

        #return tuple([y.copy() for y in self._out_arrays])
        return tuple([None if y is None else y.copy() for y in self._out_arrays])

    def backward(self, target_input_indexes, grad_outputs):
        #fixme: inform schedule manager that forward pass has finished.

        # Note: This method will be invoked every iteration starting from the second
        # iteration. That is because the corresponding define-by-run code runs instead
        # during the first iteration.
        #print('StaticScheduleFunction: backward()...')
        return self.get_backward_schedule_func().apply(grad_outputs)


def static_schedule_func_fixme_remove(func):
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

        schedule_function = chainer.config.schedule_func
        # If trace mode is on, add to schedule.
        if schedule_function is not None:
            schedule_function.append_function(no_arg_func)
            # Add the schedule function as an attribute of the FunctionNode instance
            # that contains the wrapped function as a method
            # This attribute will be needed by the corresponding @static_backward
            # function.
            instance = args[0]
            # If the instance has a 'node' attribute, assume it is an old-style Function.
            #if hasattr(instance, '_node'):
            #    pass
            #assert isinstance(instance, chainer.function_node.FunctionNode)
            if not isinstance(instance, chainer.function_node.FunctionNode):
                print("Warning: static_schedule_func was used to wrap an object that is not a FunctionNode: ",
                      instance)
                if isinstance(instance, chainer.function.Function):
                    print("It's an old-style Function.")
                    instance.node._supports_static_optimizations = True
            else:
                instance._supports_static_optimizations = True
            print('Adding function to static schedule: ', func)
            #print('static_forward: instance: ', instance)
            instance.schedule_func = schedule_function

    return wrapped_func


class ScheduleManager(object):

    """A manager of static schedules for a static chain.

    Args:


    """

    def __init__(self):
        # Maps a key string to a list of schedule functions.
        self.schedules = dict()

        self.in_use_count = dict()
        self._end_forward = False

    def get_schedule(self, in_vars):
        """Get a static schedule.

        Try to return an existing static schedule (instance of
        ``StaticScheduleFunction``) that is compatible with
        the current configuration and input variables to the supplied chain.
        If there is no existing schedule available, return an empty schedule
        object.

        If `chainer.config.train` is `True`, then this function will return a
        distinct schedule object on each call during the forward pass. That is,
        that returned schedule will either be a distinct cached schedule or
        a distinct empty schedule. These returned schedules cannot be reused
        (that is, returned again) until the next iteration. Then end of the
        forward pass is signified by calling ``loss.backward()`` on a variable
        that contains the supplied chain in its computation graph.

        If `chainer.config.train` is `False`, and this function is called multiple
        times during the forward pass, then this function is allowed to return
        the same schedule object multiple times provided that it is compatable
        with the supplied input variables.

        Args:
            in_vars (tuple of :class:`~chainer.Variable`): The input variables to the chain.

        Returns:
            An instance of ``StaticScheduleFunction``.
        """
        if self._end_forward:
            self._end_forward = False
        if chainer.config.train is False:
            # Test mode is active, so always reuse any existing schedule that
            # is compatable with in_vars.
            key_str = 'test:' + ''.join(str(x.shape) for x in in_vars)
            if key_str in self.schedules:
                sched_list = self.schedules[key_str]
                sched = sched_list[0]
            else:
                sched = StaticScheduleFunction(self, in_vars)
                self.schedules[key_str] = [sched]
            return sched
        else:
            # Training mode is active, so always return a distinct schedule
            # instance.
            key_str = 'train:' + ''.join(str(x.shape) for x in in_vars)
            if key_str in self.schedules:
                sched_list = self.schedules[key_str]
                available_index = self.in_use_count[key_str]
                if available_index >= len(sched_list):
                    sched = StaticScheduleFunction(self, in_vars)
                    sched_list.append(sched)

                sched = sched_list[available_index]
                self.in_use_count[key_str] = available_index + 1
            else:
                sched = StaticScheduleFunction(self, in_vars)
                self.schedules[key_str] = [sched]
                self.in_use_count[key_str] = 1

            return sched

    def end_forward(self):
        """Make in-use schedules available for use in next iteration.

        If training mode is active, free all in-use schedules so that they
        can be reused in the next iteration.

        """
        if not self._end_forward:
            for key in self.in_use_count:
                self.in_use_count[key] = 0
            self._end_forward = True


def static_graph(func):
    """Decorator to mark a Chain's ``__call__()`` as a static sub-graph.

    This decorator marks the define-by-run code inside the `__call__()`
    method of a Chain instance as corresponding to a static computation
    graph or sub-graph. Such a chain will be referred to as a 'static chain'
    in the documentation.

    When a chain is marked as static, it causes it to execute as
    define-by-run during the first iteration, in which a trace is performed
    to create an optimized static schedule.
    Starting from the second iteration, the code inside ``__call__()``
    is basically replaced by a single call to a FunctionNode such that
    FunctionNode.forward() implements the forward static schedule and
    FunctionNode.backward() implements the backward static schedule.
    The static schedule code performs the same computations but without the
    Python overhead of define-by-run. For some chains, this can result
    in significant runtime performance improvements.

    This feature is intended to provide the following benefits:
    - Define-by-run is still used. There is no change to the way that users define the model except that
    this decorator is used to explicitly mark the chains corresponding
    to the largest static sub-graphs in the network.
    - Since the define-by-run code is executed during the first iteration, it
    still supports easy debugging.
    - Since an optimized static schedule is executed starting from the second
    iteration, it can potentially provide the speed of a static graph framework.

    A static schedule
    representation can potentially be further optimized to reduce memory and/or perform
    operations such as kernel fusion and export of computations to non-Python
    platforms. Such advanced operations are not currently implemented, however.

    A model (that is, the complete computation graph)
    is allowed to contain an arbitrary number of static chains, each of
    which may be called an arbitrary number of times in an iteration.
    However, in any hierarchical nesting of chains corresponding to a
    static graph, only the top-level chain should be explicitly
    marked by this decorator as being static. This is because any other chains
    that are called within a static chain will be implicitly assumed to be
    static as well.

    Usage:

    Apply this decorator only to the top-most Chain in the hierarchy that
    contains a static sub-graph. It is not necessary (and not allowed) to
    mark a chain as static if it is contained within
    another chain that is also marked as being static.
    For example, suppose a
    static graph `A` contains a static sub-graph `B`. Then, only the chain
    corresponding to `A` should be marked as static and the chain corresponding
    to `B` should not be marked as static.

    In some models, such as RNNs used in NLP applications, the RNN may be
    unrolled a different number of times each iteration. If a single
    time slice of the RNN is represented by a static chain, this chain
    may potentially be called an arbitrary number of times during each
    forward pass. Note that in order for gradient propagation to work
    correctly during the backward pass, it is required that each call
    of the chain during the forward pass invoke a distinct `FunctionNode`
    object that implements the static schedule. Thus, we need to ensure
    that the same schedule instance is not reused during the same forward
    pass. If it is not necessary to compute gradients, such as during
    test mode, then it is fine to reuse the same schedule instance during
    the same forward pass.

    In order to ensure that a static chain works correctly with models
    such as the RNN just described and without other modifications to existing code,
    we chose to make
    the behavior of a static chain depend on the training mode flag,
    `chainer.config.train`. If the it is `True`, then a static chain that is
    called multiple times will try to use a distinct static schedule object
    (that is, call a distinct instance of a FunctionNode that implements
    that static schedule) on each call. The same schedule instance cannot
    be reused until the forward pass has completed, which is signaled by
    performing a backward pass through the model. It is therefore important
    that the backward pass be performed after each forward pass during
    training. Since this is usually the case, most usages of static chain
    will not required any modifications to existing code other than applying
    this decorator. However, if you would like to perform multiple forward
    passes during training before performing a backward pass, then you
    must explicitly set an attribute ``iteration_finished`` of the
    static chain to `True` after each forward pass.
    If test mode is active (`chainer.config.train` is `False`) then it
    is not necessary to inform the chain at the end of each forward pass
    because in test mode, a static chain always attempts to reuse
    existing static schedule objects whenever possible. It is acceptable
    to reuse the same static schedule during a single forward pass because
    we assume that there is no need to compute gradients and hence no
    need to ever perform a backward pass during test mode.

    Important:
        Regarding parameters of a static chain: In the current
        implementation, it is not allowed to change a parameter's ``data`` or
        ``grad`` array references once they have been allocated. Any optimizer
        that operates on a model containing a static chain must therefore
        take care to only update a parameter's array in-place. This
        restriction could be lifted in the future.
        The current implementation automatically sets ``is_static`` to `True`
        for each parameter of a static chain.

    Notes:
        There are additional optimizations (to reduce memory usage and increase
        performance) that are not yet implemented. When using statc graph
        optimizations, all intermediate activations are currently allocated
        statically even if they are not needed for backpropagation,
        which can result in higher memory usage than the corresponding define-by-
        run code. Also, there is the potential to perform kernel/function
        fusion to further increase performance. Exporting of static graph
        to C/C++ and/or an intermediate level graph representations is also
        possible and may be considered in the future.

    Args:
        func: The `__call__()` method of an instance of Chain
        that is wrapped by this decorator.

    Returns:
        Wrapped ``__call__()`` method with static chain support.
    """
    def wrapped_func(*args, **kwargs):
        chain = args[0]
        in_vars = args[1:]
        # Since it is allowed for in_vars to be either variables or arrays,
        # we force to variables.
        new_in_vars = []
        for x in in_vars:
            if not isinstance(x, chainer.Variable):
                new_in_vars.append(chainer.Variable(x))
            else:
                new_in_vars.append(x)
        in_vars = tuple(new_in_vars)

        if not hasattr(chain, 'schedule_manager'):
            chain.schedule_manager = ScheduleManager()

        schedule_manager = chain.schedule_manager
        chain.static_schedule = schedule_manager.get_schedule(in_vars)

        if not chain.static_schedule.is_empty():
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

            new_args = []
            new_args.append(chain)
            for var in new_in_vars:
                new_args.append(var)
            args = tuple(new_args)

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
            return out_vars

    return wrapped_func
