import contextlib

import chainer
import chainer.function_node

import numpy as np

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
        in_vars (tuple of Variable): The flattened tuple of input variables that is supplied to
            `__call__()` method of the chain that this schedule corresponds to.

    """

    def __init__(self, schedule_manager, verbosity_level=0,
                 enable_double_backprop=False):
        print('Creating new static schedule!')
        self._schedule_manager = schedule_manager
        self._static_schedule_forward = []
        self._backward_schedule_func = None
        self.verbosity_level = verbosity_level
        self.enable_double_backprop = enable_double_backprop
        self._chain_return_vars = None
        self._local_in_vars = None
        self.chain =None


    def is_empty(self):
        """Return True if this schedule is empty.

        """
        return len(self._static_schedule_forward) == 0


    # fixme: remove?
    def add_input_variables(self, in_vars):
        """Add input variables for this schedule.


        :param in_vars:
        :return:
        """
        self._in_arrays = [var.data for var in in_vars] # fixme: remove?
        self._local_in_vars = in_vars

    def append_function(self, func):
        """Append a function to the (forward) static schedule.

        Args:
            func: The function to append to the schedule. The function
            should not take any arguments and should not return any results.

        """

        if self.verbosity_level >= 2:
            print('Adding function to static schedule: ', func)
        self._static_schedule_forward.append(func)

    def forward(self, inputs):
        #print('StaticScheduleFunction.forward(): enable_backprop: ', chainer.config.enable_backprop)

        # Note: This method will be invoked every iteration starting from the second
        # iteration. That is because the corresponding define-by-run code runs instead
        # during the first iteration.
        # Copy any external input arrays into the statically-allocated arrays:
        assert len(self._in_arrays) == len(inputs)
        for n in range(len(inputs)):
            # Debug:
            if inputs[n] is not None:
                in_array = self._in_arrays[n]
                assert in_array.shape == inputs[n].shape
                assert in_array[0].dtype == inputs[n][0].dtype
                in_array[:] = inputs[n]
                #print('static function node forward: self._in_arrays[n]: ', stats.describe(in_array, axis=None))
                #print('and id of array: ', id(in_array))

        # The following line should be the new performance bottleneck after the first iteration
        # has completed. Note that we have several options for the implementation:
        # - Simply iterate through the schedule (in Python), calling each function.
        # - Export the schedule to C/C++ code. The forward computation can then
        # potentially be performed without any dependency on Python.
        # - Optimize the schedule code in Cython, calling optimized C/C++ code.
        for x in self._static_schedule_forward:
            x()


        #print('shape of inputs: ', [x.shape for x in inputs])
        #sched_size = len(self._static_schedule_forward)
        #print('size of forward schedule: ', sched_size)

        # Return a copy of the static arrays here because it is possible that the
        # static chain containing this schedule is repeated several times in the
        # computation graph.

        #print('static function node forward: self._out_arrays', stats.describe(self._out_arrays, axis=None))

        #return tuple([None if y.data is None else y.data.copy() for y in self._out_vars])
        return tuple([None if y is None else None if y.data is None else y.data.copy() for y in self._out_vars])

    def backward(self, target_input_indexes, grad_outputs):
        #print('StaticScheduleFunction: backward()')
        # The first time this method is called, the define-by-run code is
        # executed in order to create a static schedule.
        self._schedule_manager.end_forward()
        #
        #debug_grad = grad_outputs[0].data
        #print('static function node backward: grad_outputs', stats.describe(debug_grad, axis=None))
        #print('with id = ', id(debug_grad))

        #if True:
        if self._backward_schedule_func is None:
            print('Creating new backward schedule...')
            # Create backward schedule and run define-by-run backward code.
            self._backward_schedule_func = StaticScheduleFunction(self._schedule_manager)

            # Make local copies of the variables in grad_outputs.
            new_grad_outputs = []
            for var in grad_outputs:
                # Replace each input variable with a new variable having
                # the same data.
                new_grad_outputs.append(chainer.Variable(var.data.copy()))

            self._backward_schedule_func.add_input_variables(new_grad_outputs)



            #out_vars = 0
            with chainer.using_config('schedule_func', self._backward_schedule_func):
                with chainer.using_config('enable_backprop', True):
                    print('StaticScheduleFunction.backward(): enable_backprop: ', chainer.config.enable_backprop)
                    for ind, var in enumerate(new_grad_outputs):
                        dbr_out_var = self._out_vars[ind]
                        dbr_out_var.grad = new_grad_outputs[ind].data
                        self._out_vars[ind].grad = new_grad_outputs[ind].data
                        #if ind == 0:
                        #    out_vars = dbr_out_var
                        #else:
                        #    out_vars = out_vars + dbr_out_var
                        #if ind == len(new_grad_outputs) - 1:
                        #    #    dbr_out_var.backward(retain_grad=True,
                        #    #                         enable_double_backprop=self.enable_double_backprop)
                        #    out_vars.grad = np.zeros(out_vars.data.shape, dtype=np.float32)
                        #    out_vars.backward(retain_grad=True,
                        #                 enable_double_backprop=self.enable_double_backprop)
                    inputs = [param for param in self.chain.params()]
                    for var in self._local_in_vars:
                        inputs.append(var)
                    temp_out = chainer.grad(self._out_vars, inputs,
                             grad_outputs=new_grad_outputs,
                             set_grad=True,
                                        enable_double_backprop=self.enable_double_backprop)
                    #print(temp_out)

            # Note: var.grad_var is allowed to be None below:
            self._backward_schedule_func._out_vars = [var.grad_var for var in self._local_in_vars]

            #del self._local_in_vars
            print('Static backward schedule created!')

        #ret = self._backward_schedule_func.apply(grad_outputs)
        #assert isinstance(ret, tuple)
        return self._backward_schedule_func.apply(grad_outputs)


class ScheduleManager(object):

    """A manager of static schedules for a static chain.

    Args:
        minimize_cache_size (bool): If `True`, attempt to reduce memory
        usage by clearing the cached schedules whenever the training
        mode changes (that is, whenever `chainer.config.train` changes
        value) or whenever the mini-batch size changes.


    """

    def __init__(self, minimize_cache_size=True, verbosity_level=0):
        # Maps a key string to a list of schedule functions.
        self.schedules = dict()
        self._minimize_cache_size = minimize_cache_size
        self.in_use_count = dict()
        self._end_forward = False
        self._prev_train_config = None
        self._max_in_use_train = 0
        self._train_count = 0
        self._verbosity_level = verbosity_level

    def get_schedule(self, in_vars, enable_double_backprop=False):
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
        that contains the supplied chain in its computation graph. It is
        therefore necessary to call ``loss.backward()`` each iteration after
        the forward pass has completed. Otherwise, this method would always
        return a distinct schedule object which would eventually cause an
        out-of-memory error to occur.

        If `chainer.config.train` is `False` and this function is called multiple
        times during the forward pass, then the same schedule object can be
        returned multiple times provided that it is compatible with the
        types and shapes of the input variables.

        Args:
            in_vars (tuple of :class:`~chainer.Variable`): The input variables to the chain.

        Returns:
            An instance of ``StaticScheduleFunction``.
        """
        if self._end_forward:
            self._end_forward = False
        if self._minimize_cache_size:
            if chainer.config.train != self._prev_train_config:
                # Training config changed, so clear caches.
                self._prev_train_config = chainer.config.train
                if self._verbosity_level >= 2:
                    print("Clearing schedule cache...")
                self.schedules.clear()
                self.in_use_count.clear()
            # todo (vogel): Also check if minibatch size has changed and clear schedules.

        if chainer.config.train is False:
            key_str = 'test:' + ''.join(str(x.shape) for x in in_vars)
            # If the maximum number of in-use schedules in any iteration
            # during training mode was exactly 1, assume it should also
            # be 1 for test mode.

            if key_str in self.schedules:
                sched_list = self.schedules[key_str]
                sched = sched_list[0]
            else:
                sched = StaticScheduleFunction(self,
                                               verbosity_level=self._verbosity_level,
                                               enable_double_backprop=enable_double_backprop)
                self.schedules[key_str] = [sched]
            return sched

        else:
            key_str = 'train:' + ''.join(str(x.shape) for x in in_vars)
            #print("key: \n", key_str)
            self._train_count += 1

            if key_str in self.schedules:
                sched_list = self.schedules[key_str]
                available_index = self.in_use_count[key_str]
                if available_index >= len(sched_list):
                    sched = StaticScheduleFunction(self,
                                                   verbosity_level=self._verbosity_level,
                                                   enable_double_backprop=enable_double_backprop)
                    sched_list.append(sched)

                sched = sched_list[available_index]
                self.in_use_count[key_str] = available_index + 1
            else:
                sched = StaticScheduleFunction(self,
                                               enable_double_backprop=enable_double_backprop)
                self.schedules[key_str] = [sched]
                self.in_use_count[key_str] = 1

        return sched

    def end_forward(self):
        """Make in-use schedules available for use in next iteration.

        Set the in-use status of all schedules to "not in use" so that
        they can be reused in the next iteration.

        In the case that test mode is active
        (`chainer.config.train` is `False`) and the static chain corresponding
        to this manager was not called more than once in any iteration during
        trainign mode, then this method will be called automatically.

        """
        if not self._end_forward:
            for key in self.in_use_count:
                self.in_use_count[key] = 0
            self._end_forward = True

            if self._train_count > self._max_in_use_train:
                self._max_in_use_train = self._train_count
                if self._verbosity_level >= 2:
                    print("Maximum in-use schedules per training iteration: ",
                        self._max_in_use_train)
            self._train_count = 0
            #self._test_count = 0


def static_graph(*args, **kwargs):
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
    It is also possible to disable static optimzations while in test mode
    such as to maintain compatibility with some existing models that require
    dynamic behavior in test mode.

    Double-backprop:
        Double-backpropagation is not enabled by default. It can be enabled by
        supplying the keyword argument ``enable_double_backprop=True``
        to this decorator.

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

    Restrictions on input arguments and return values of a static chain:
        Recall that unlike a function, there is no restrictions on the
        arguments to a chain. However, there currently are some restrictions
        when a static chain is used. Specifically, the arguments to a static
        chain must consist of a variable, list or tuple. In the case of a list
        or tuple, the elements are required to be an instance of variable,
        list, or tuple. There can be an arbitrary number of nested lists/
        tuples. No other object types are allowed.
        In addition to this, it is not allowed to use keyword arguments.
        The return value of a static chain must also consist of either a
        variable, list, or tuple in which each element of the list or
        tuple is also a variable, list, or tuple.

    This decorator can be supplied with the following optional keyword
    arguments:

    Args:
        force_test_define_by_run (bool): If `True`, disable static graph
            optimizations during test mode (that is, when
            `chainer.config.train` is False). This may be needed in order
            for some existing RNN links such as LSTM to work correctly,
            since some existing links do not correspond to a static graph
            in some cases.
            The default is `False`.

        minimize_cache_size (bool): If `True`, minimize the number of cached
            static schedules in order to reduce memory usage. The default
            value is `False`. fixme: Don't enable yet due to memory bug or
            slow garbage collection?

        verbosity_level (int): Depending on the value, print additional
            information:
            0: Warnings only. (the default value)
            1: Print when a function is added to a static schedule.
            2: Detailed debugging information.

        enable_double_backprop (bool): If `True`, enable double-backprop.
            The default value is `False` (not enabled).


    Returns:
        Wrapped ``__call__()`` method with static chain support.
    """
    force_test_define_by_run = False
    minimize_cache_size = False # todo: enable after debug
    verbosity_level = 0
    enable_double_backprop = False
    zero_args = False
    if len(args) == 1 and not kwargs and callable(args[0]):
        callable_arg = args[0]
        zero_args = True
    elif kwargs:
        if 'force_test_define_by_run' in kwargs:
            force_test_define_by_run = kwargs['force_test_define_by_run']
        if 'minimize_cache_size' in kwargs:
            minimize_cache_size = kwargs['minimize_cache_size']
        if 'verbosity_level' in kwargs:
            verbosity_level = kwargs['verbosity_level']
        if 'enable_double_backprop' in kwargs:
            enable_double_backprop = kwargs['enable_double_backprop']

    def wrap(func):
        def wrapped_func(*inner_args, **inner_kwargs):
            chain = inner_args[0]
            # The arguments to `__call__()` of the static chain.
            # These could consist of any combination of nested lists and/or
            # tuples of variables or arrays.
            chain_args = inner_args[1:]
            if chainer.config.train is False and force_test_define_by_run:
                return func(*inner_args, **inner_kwargs)

            chain_args_flat, in_unflatten_inds, __ = _flatten_args(chain_args)

            # Since it is allowed for in_vars to be either variables or arrays,
            # we force to variables.
            flat_vars = []
            for x in chain_args_flat:
                if not isinstance(x, chainer.Variable):
                    flat_vars.append(chainer.Variable(x))
                else:
                    flat_vars.append(x)

            flat_vars = tuple(flat_vars)

            if not hasattr(chain, 'schedule_manager'):
                chain.schedule_manager = ScheduleManager(
                    minimize_cache_size=minimize_cache_size,
                    verbosity_level=verbosity_level)

            schedule_manager = chain.schedule_manager
            chain.static_schedule = schedule_manager.get_schedule(flat_vars,
                                                                  enable_double_backprop=enable_double_backprop)
            chain.static_schedule.chain = chain # fixme: clean up
            if not chain.static_schedule.is_empty():
                # Call the optimized static schedule code.
                #print('This is the 2nd or greater iteration. Calling the optimized schedule...')
                # Note: out_vars are dynamically allocated because FunctionNode.apply()
                # will dynamically allocate variables on each call, which is the desired
                # behavior.
                out_vars_flat = chain.static_schedule.apply(flat_vars)

                out_vars = _unflatten_args(out_vars_flat, chain._out_vars_unflatten_inds)

            else:
                # This is the first iteration. Calling the define-by-run code.
                print('Creating new forward schedule...')
                assert isinstance(chain, chainer.Chain)
                if verbosity_level >= 2:
                    print('This is the first iteration. Calling the define-by-run code.: ', func)
                # First check that this chain is not called from inside another
                # static chain because it is not allowed.
                if chainer.config.schedule_func is not None:
                    raise RuntimeError("Not allowed to nest static chains: ", chain)

                new_args = []
                new_args.append(chain)
                new_flat_vars = []
                for var in flat_vars:
                    # Replace each input variable with a new variable having
                    # the same data.
                    new_flat_vars.append(chainer.Variable(var.data.copy()))

                unflat_in_args = _unflatten_args_as_list(new_flat_vars, in_unflatten_inds)

                for item in unflat_in_args:
                    new_args.append(item)

                inner_args = tuple(new_args)

                with chainer.using_config('schedule_func', chain.static_schedule):

                    out_vars = func(*inner_args, **inner_kwargs)

                chain.static_schedule.add_input_variables(new_flat_vars)
                # fixme: maybe make a method of static schedule function to do this?
                #chain.static_schedule._in_arrays = [var.data for var in new_flat_vars] # fixme: clean up

                # note: we have to save these variables because we will need to read the .grad members
                # from them after the backward pass.
                #chain.static_schedule._local_in_vars = new_flat_vars # fixme: clean up
                out_vars_flat_dbr, chain._out_vars_unflatten_inds, __ = _flatten_args(out_vars)

                chain.static_schedule._out_vars = [var for var in out_vars_flat_dbr]

                # Now that the static schedule is available, call it using the
                # flattened input variables. This will cause the
                # static schedule function node to be included in the
                # computational graph.
                out_vars_flat = chain.static_schedule.apply(flat_vars)

                out_vars = _unflatten_args(out_vars_flat, chain._out_vars_unflatten_inds)

                if verbosity_level >= 2:
                    print('Creating a new backward schedule function.')

            return out_vars

        return wrapped_func

    if zero_args:
        return wrap(callable_arg)
    else:
        return wrap

# fixme: remove unused function
def _copy_vars_no_creators(var_tuple):
    """Return a tuple containing new variables using existing data arrays.

    Return a tuple of variables having the same shape, type, and data arrays
    as `var_tuple`. These new variables are initialized using the data array
    of the variables in `var_tuple`. Therefore, the new variables will not
    have a `creator`.

    Args:
        var_tuple (tuple of Variable): Input tuple of variables.

    Returns:
        a tuple of variables having the same data as those of the input
        tuple.
    """
    out_vars = []
    for var in var_tuple:
        out_vars.append(chainer.Variable(var.data))
    return tuple(out_vars)


def _flatten_args(xs):
    """Flatten the input into a tuple of variables.

    In the typical case, `xs` is a tuple or list of objects where each
    object is either a variable, list, or tuple. In the case where it is
    a list of tuple, the objects in the list or tuple could also be either
    a variable, list or tuple. Although the non-list and non-tuple items
    are typically an instance of variable, any object other than list or
    tuple is allowed.

    This function simply flattens the hierarchical lists/tuples so that all
    objects that are deeply contained in `xs` that are non-list and non-tuple
    will be returned in a single tuple.

    Args:
        xs:

    Returns:
        The flattened tuple, allong with the indecies and count so that the
        items can be unflattened later (i.e., by calling `_unflatten_args()`.

    fixme: does not work if xs is a variable only.
    """
    inds = []
    ys = []
    i = 0
    if not isinstance(xs, (list, tuple)):
        inds.append(('s', ))
        #return [xs], inds, 0
        return (xs,), inds, 0
    for x in xs:
        if isinstance(x, (list, tuple)):
            x, sub_inds, total = _flatten_args(x, )
            inds.append(('i', i, i+total, sub_inds))
            i += total
        else:
            x = [x]
            inds.append(('f', i))
            i += 1
        ys.extend([y for y in x])
    return tuple(ys), inds, i


# fixme: this only outputs tuples of tuples. Any list in the original input
# will be converted to a tuple, changing the types of the input arguments
# to the static chain.
def _unflatten_args(xs, inds):
    ys = []
    for ind in inds:
        code = ind[0]
        if code == 's':
            return xs[0]
        elif code == 'i':
            i_start, i_end, sub_inds = ind[1:]
            y = _unflatten_args(xs[i_start:i_end], sub_inds)
        else:
            i = ind[1]
            y = xs[i]
        ys.append(y)
    return tuple(ys)

def _unflatten_args_as_list(xs, inds):
    ys = []
    for ind in inds:
        code = ind[0]
        if code == 's':
            return xs[0]
        elif code == 'i':
            i_start, i_end, sub_inds = ind[1:]
            y = _unflatten_args(xs[i_start:i_end], sub_inds)
        else:
            i = ind[1]
            y = xs[i]
        ys.append(y)
    return ys


#todo: move code below into Chainer tests after initial debug.


def test1():
    print('testing var copy')
    import numpy as np
    N = 3
    xs = []
    for n in range(N):
        data = np.random.rand(2,3)
        xs.append(chainer.Variable(data))
    xs = tuple(xs)
    print('xs: ', xs)
    copied_vars = _copy_vars_no_creators(xs)
    print('copied: ', copied_vars)

if __name__ == '__main__':
    test1()
