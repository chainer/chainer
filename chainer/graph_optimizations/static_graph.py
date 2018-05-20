import contextlib

#import sys

import chainer
from chainer.backends import cuda
import chainer.function_node

import numpy as np

# todo: add test that use the same random seed with two models: a static chain
# and a (non-static) chain. Enable `chainer.config.cudnn_deterministic` and
# run both models and check that the outputs are identical.

#fixme: need to check the dtype of inputs when checking the type signature for static schedules.

# todo: Add a debug mode (perhaps enabled by defaut) that checks that the parameter references
# do not change from iteration to iteration. This is needed because some existing parts of Chainer,
# such as the gradient_check.py change the references of the parameter arrays, which is not allowed!
# Without any checks, this will lead to mysterious and dificult to detect errors. We should therefore
# check for it and raise an exception if outside code attempts to change the refernce.
# Implementation: before returning from the __call__(), get a list of all parmaeters. Then save
# the references of each parameter to a dict or something. (or we can just save the id() of the
# .data arrays? Then, in each subsequent call of the static chain, verify that the references
# /ids have not changed. We can do this for the backward pass, too? how? That is, we can check both
# the .data and .grad attributes of all parameters.

# todo: make a more general version of this function that finds the id of each array.
# It should also be modified to support cupy arrays.
def _debug_print_stats(args):
    for arg in args:
        if isinstance(arg, np.ndarray) or isinstance(item, cuda.ndarray):
            print('no-arg-func stats: ', stats.describe(arg, axis=None))
            print('id of array: ', id(arg))
        elif isinstance(arg, (list, tuple)):
            _debug_print_stats(arg)
        else:
            print('no-arg-func name: ', str(arg))

class ScheduleInfo(object):

    def __init__(self, func, args, kwargs, hooks, unique_arrays, func_name=None):
        """A callable wrapper for a function in the static schedule.

        Args:
            func (FunctionNode): A function in the static schedule.
            args: Arguments to 'func'.
            kwargs: Keyword arguments to 'func'.
            hooks (list of tuples): A list of hooks that instruct how to
                update the ndarray references in 'args' so that they
                refer to the correct master array in 'unique_arrays'.
            unique_arrays (list of ndarray): The master list of all unique ndarrays
                that appear in the static schedule.
            func_name (str): An optional name of the static function.
        """
        self.func = func
        # If the type is a tuple, change to a list.
        self.args = list(args)

        self.kwargs = kwargs
        self.hooks = hooks
        self.unique_arrays = unique_arrays
        self.func_name = func_name

    def _run_hooks(self):
        """Run hooks to set correct references.

        Process the list of hooks which will modify the array references in
        the arguments list of the static function. This method must be
        called before executing the static function.

        The hooks specify that
        each array argument points to a "master" array reference in the
        unique_arrays list. If the reference in unique_arrays changes, then
        we must also change the corresponding array reference in the arguments
        list. The hooks specify the mapping and this method updates the
        references in args to the corresponding values from unique_arrays.

        There are currently two possible formats for an item in the hooks
        list:

        Format 1:

        In this format, each item in args is either an ndarray or another
        type that we don't care about.

        (arg_index, unique_list_index)
        which is an instruction to perform the operation
        args[arg_index] = self.unique_arrays[unique_list_index]

        Format 2 (only used by the automatic wrapping function):

        In this format, each item in args is either a list of ndarrays or
        another type that we don't care about.

        (arg_index, sub_index, unique_list_index)
        which is an instruction to perform the operations

        temp_list = args[arg_index]
        temp_list[sub_index] = self.unique_arrays[unique_list_index]

        So, this method simply checks the format of each instruction tuple in
        'hooks' and executes it.
        """
        for hook in self.hooks:
            if len(hook) == 2:
                # Format 1.
                (arg_index, unique_list_index) = hook
                self.args[arg_index] = self.unique_arrays[unique_list_index]
            elif len(hook) == 3:
                # Format 2.
                (arg_index, sub_index, unique_list_index) = hook
                temp_list = self.args[arg_index]
                temp_list[sub_index] = self.unique_arrays[unique_list_index]
            else:
                raise RuntimeError('Unkown hook format.')

    def __call__(self):
        self._run_hooks()
        #print('New---In no_arg_func: Calling: ', self.func)
        # _debug_print_stats(args)
        ret = self.func(*self.args, **self.kwargs)
        if ret is not None:
            # todo: We can allow it to return tuple of arrays in the future.
            raise RuntimeError("This function is not supposed to return anything: ", func)

    def __repr__(self):
        return "function: {0}\n name: {1}\n args: {2}\n kwargs: {3}\n hooks: {4}".format(self.func,
                                                                            self.func_name,
                                                                self.args,
                                                                self.kwargs, self.hooks)


# todo: rename to StaticSchedule. (the Function part makes it sound like
# it refers to only 1 function iside the static schedule)
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

    Recall that a "static chain" referes to a chain that is decorated by the
    `static_graph` decorator.
    During the first forward pass of a static chain, the define-by-run code is executed. However,
    for subsequent iterations, that define-by-run code is replaced by an instance
    of this function and this function will be called instead. Since the static
    schedules contained by this function perform the same computations, it is
    safe (and potentially much more efficient) to simply execute the static schedule instead
    of the define-by-run code. See `static_graph` for details.

    Args:
        schedule_manager (ScheduleManager): The schedule manager of this schedule instance.
        in_vars (tuple of Variable): The flattened tuple of input variables that is supplied to
            `__call__()` method of the chain that this schedule corresponds to.

    """

    def __init__(self, schedule_manager, verbosity_level=0,
                 enable_double_backprop=False):
        self._schedule_manager = schedule_manager
        # todo: Consider creating _schedule_funcs as a post-processing step.
        #self._schedule_funcs_v1 = [] # todo: remove?
        #self._schedule_funcs = []
        self.schedule_info_list = []
        # A list of all unique ndarrays used in the static schedule, including
        # input variables and parameter arrays.
        self.unique_arrays = []
        # Maps id(ndarray) to its position in self.unique_arrays
        self.array_id_to_unique_index = dict()
        self._backward_schedule_func = None
        self.verbosity_level = verbosity_level
        self.enable_double_backprop = enable_double_backprop
        self._chain_return_vars = None
        self._local_in_vars = None
        self.chain =None
        self.schedule_built = False
        # A list of all parameters in the model (i.e., that exist when
        # build_schedule() is called.
        self.params_list = []
        # This list contains the grad_var corresponding to each variable
        # in params_list. This is needed so that we can restore any grad_var
        # that is set to None by outside code.
        self.grad_var_list = []
        self.debug_grad_var_list = []
        # Maps an array id (of a parameter) to its location.
        # id(array) -> (index_in_self.params_list, attribute_location)
        self.array_id_to_param_map = dict()
        # A list of tuples that specify the mappings from static schedule
        # arrays to parameter attributes.
        self.param_hooks = []



    def is_empty(self):
        """Return True if this schedule is empty.

        """
        return len(self.schedule_info_list) == 0


    # fixme: remove?
    def add_input_variables(self, in_vars):
        """Add input variables for this schedule.


        :param in_vars:
        :return:
        """
        self._in_arrays = [var.data for var in in_vars] # fixme: remove?
        self._local_in_vars = in_vars

    def append_function(self, func, args, kwargs, func_name=None):
        """Append a function to the (forward) static schedule.

        Args:
            func: The function to append to the schedule. The function
            should not take any arguments and should not return any results.

        """
        # If this kwarg is True, we need to use a special format for args.
        is_generic_static_forward = False
        if 'is_generic_static_forward' in kwargs:
            is_generic_static_forward = kwargs['is_generic_static_forward']

        # A list of hooks (each is a tuple) that will be used to set
        # correct array references in 'args' before executing the static schedule
        # function 'func'
        # See ScheduleInfo._run_hooks() for details.
        hooks = []

        if is_generic_static_forward:
            # Use special arg format consiting of lists of arrays.
            # Each argument that is a list, contains arrays (not nested).
            for arg_index, item in enumerate(args):
                if isinstance(item, list):
                    for sub_index, sub_item in enumerate(item):
                        if isinstance(sub_item, np.ndarray) or isinstance(sub_item, cuda.ndarray):
                            if id(sub_item) not in self.array_id_to_unique_index:
                                self.unique_arrays.append(sub_item)
                                unique_list_index = len(self.unique_arrays) - 1
                                self.array_id_to_unique_index[id(sub_item)] = unique_list_index
                            else:
                                unique_list_index = self.array_id_to_unique_index[id(sub_item)]
                                #print('found same array: \n')
                                #print('unique_array item: \n', self.unique_arrays[unique_list_index])
                                #print('is equal to the sub_item:\n', sub_item)
                            hooks.append((arg_index, sub_index, unique_list_index))

        # Iterate over args, appending each unique ndarray onto unique_arrays.
        else:
            # Use normal arg format:
            # arg[i] is either an array or another type that we don't care about.
            for arg_index, item in enumerate(args):
                if isinstance(item, np.ndarray) or isinstance(item, cuda.ndarray):
                    if id(item) not in self.array_id_to_unique_index:
                        self.unique_arrays.append(item)
                        unique_list_index = len(self.unique_arrays) - 1
                        self.array_id_to_unique_index[id(item)] = unique_list_index
                    else:
                        unique_list_index = self.array_id_to_unique_index[id(item)]
                    hooks.append((arg_index, unique_list_index))

        if self.verbosity_level >= 2:
            print('Adding function to static schedule: ', func)


        self.schedule_info_list.append(ScheduleInfo(func, args, kwargs, hooks, self.unique_arrays, func_name=func_name))

    def __repr__(self):
        out = "StaticSchedule:\n"
        out += "Length of unique_arrays: " + str(len(self.unique_arrays)) + "\n"
        for x in self.schedule_info_list:
            out += str(x)
        return out

    def _run_param_hooks(self):
        """Run parameter reference updater hooks.

        This updates the 'data' and/or 'grad' attributes of the parameters,
        if necessary. It also handles the case where the 'grad' attribute
        was set to 'None' by outside Chainer code.

        """
        for hook in self.param_hooks:
            #print('running parameter hook: ', hook)
            (unique_array_index, param_attribute_location) = hook
            (params_list_index, attribute_location) = param_attribute_location
            param = self.params_list[params_list_index]
            schedule_grad_var = self.grad_var_list[params_list_index]
            was_grad_var = self.debug_grad_var_list[params_list_index]
            #if was_grad_var:
            if schedule_grad_var is not None:
                if param.grad_var is None:
                    if self.verbosity_level >= 2:
                        print('----------- USER CODE REMOVED GRAD VAR!------')
                    if schedule_grad_var.data is not None:
                        schedule_grad_var.data.fill(0)
                        param.grad_var = schedule_grad_var
            #print('curent reference count is: ', sys.getrefcount(self.params_list[params_list_index]))
            # This is the original array that was used during the first
            # iteration and at that time corresponded to the data
            # attribute of the parameter.
            #schedule_array = self.unique_arrays[unique_array_index]
            if attribute_location == 'data':
                # This is the corresponding parameter array, which might
                # have had its reference changed to a different array or set
                # to None.
                possibly_modified_parameter_array = \
                    self.params_list[params_list_index].data
                if self.unique_arrays[unique_array_index] is not possibly_modified_parameter_array:
                    if self.verbosity_level >= 2:
                        print('The parameter data attribute has changed: ', self.params_list[params_list_index])
                    if self.unique_arrays[unique_array_index] is not None and possibly_modified_parameter_array is not None:


                        # Set the schedule's array to refer to the 'data'
                        # attribute of the parameter.
                        if self.verbosity_level >= 2:
                            print('Setting schedule array equal to data attribute reference.')
                        self.unique_arrays[unique_array_index] = possibly_modified_parameter_array
                    elif self.unique_arrays[unique_array_index] is not None and possibly_modified_parameter_array is None:
                        # The data attribute was set to None by outside code.
                        # Assume that it is safe to replace a None-valued
                        # array with 0.
                        self.unique_arrays[unique_array_index].fill(0)
                        if self.verbosity_level >= 2:
                            print('Zero-ing and updating parameter data attribute reference.')
                        self.params_list[params_list_index].data = self.unique_arrays[unique_array_index]
            elif attribute_location == 'grad':
                # This is the corresponding parameter array, which might
                # have had its reference changed to a different array or set
                # to None.
                possibly_modified_parameter_array = \
                    self.params_list[params_list_index].grad
                if self.unique_arrays[unique_array_index] is not possibly_modified_parameter_array:
                    if self.verbosity_level >= 2:
                        print('The parameter grad attribute has changed: ', self.params_list[params_list_index])
                    if self.unique_arrays[unique_array_index] is not None and possibly_modified_parameter_array is not None:
                        # Set the schedule's array to refer to the 'grad'
                        # attribute of the parameter.
                        if self.verbosity_level >= 2:
                            print('Setting schedule array equal to grad attribute reference.')
                        self.unique_arrays[unique_array_index] = possibly_modified_parameter_array
                    elif self.unique_arrays[unique_array_index] is not None and possibly_modified_parameter_array is None:
                        # The grad attribute was set to None by outside code.
                        # Assume that it is safe to replace a None-valued
                        # array with 0.
                        self.unique_arrays[unique_array_index].fill(0)
                        if self.verbosity_level >= 2:
                            print('Zero-ing and updating parameter grad attribute reference.')
                        self.params_list[params_list_index].grad = self.unique_arrays[unique_array_index]


    def build_schedule(self, chain):
        """Build the static chedule

        Perform processing on the functions and arguments that were
        previously supplied in 'append_function()' to create the static
        schedule.

        This method must be called after the final call of 'append_function()'
        and before calling 'forward()' for the first time.

        Details:

        (todo)


        Args:
            chain: The static chain that uses this scheudle.
        """
        self.chain = chain
        assert len(self.params_list) == 0
        assert len(self.array_id_to_param_map) == 0

        # Check that all array references are actually unique.
        unique_ids = set()
        for ar in self.unique_arrays:
            assert id(ar) not in unique_ids
            unique_ids.add(id(ar))
            #print('unique array id:\n', id(ar))

        for param in chain.params():
            #print('build_schedule: param: \n', param)
            self.params_list.append(param)
            grad_var = param.grad_var
            # Note: grad_var might be None, but it is fine to append None.
            if self.verbosity_level >= 2:
                print('For param: ', param)
                print('adding its grad_var: ', grad_var)
            self.grad_var_list.append(grad_var)
            if grad_var is None:
                self.debug_grad_var_list.append(False)
            else:
                self.debug_grad_var_list.append(True)
            params_list_index = len(self.params_list) - 1
            if param.data is not None:
                key = id(param.data)
                assert key not in self.array_id_to_param_map
                self.array_id_to_param_map[key] = (params_list_index, 'data')
            if param.grad is not None:
                key = id(param.grad)
                assert key not in self.array_id_to_param_map
                self.array_id_to_param_map[key] = (params_list_index, 'grad')


        #print('self.params_list:\n', self.params_list)
        #print('self.array_id_to_param_map:\n', self.array_id_to_param_map)

        # Iterate over all arrays used in the schedule and check which ones
        # correspond to parameter arrays.
        assert len(self.unique_arrays) > 0
        for unique_array_index, ar in enumerate(self.unique_arrays):
            if id(ar) in self.array_id_to_param_map:
                param_attribute_location = self.array_id_to_param_map[id(ar)]
                hook = (unique_array_index, param_attribute_location)
                self.param_hooks.append(hook)

        if self.verbosity_level >= 2:
            print('self.param_hooks: ', self.param_hooks)


        # todo: We can potentially reduce memory usage by freeing memory
        # of intermediate arrays
        # once they are no longer needed in the schedule or by
        # parameters. However, we cannot know without checking
        # if retain_inputs/outputs has been called and/or checking
        # the backward schedule.
        #for func_info in self.schedule_info_list:
        #    func = func_info.func
        #    args = func_info.args
        #    kwargs = func_info.kwargs
        #    func_name = func_info.func_name






        self.schedule_built = True

    def forward(self, inputs):
        if self.verbosity_level >= 2:
            print('Calling StaticScheduleFunction.forward()...')

        # Note: This method will be invoked every iteration starting from the second
        # iteration. That is because the corresponding define-by-run code runs instead
        # during the first iteration.
        # Copy any external input arrays into the statically-allocated arrays:
        if not self.schedule_built:
            raise RuntimeError('forward() was called before build_schedule()!')
        self._run_param_hooks()
        assert len(self._in_arrays) == len(inputs)
        for n in range(len(inputs)):
            # Debug:
            if inputs[n] is not None:
                in_array = self._in_arrays[n]
                assert in_array.shape == inputs[n].shape
                if in_array[0].dtype != inputs[n][0].dtype:
                    print('in_array[0].dtype: ', in_array[0].dtype)
                    print('inputs[n][0].dtype: ', inputs[n][0].dtype)
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
        if self.verbosity_level >= 2:
            print('Running static schedule...')
        for x in self.schedule_info_list:
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
        if self.verbosity_level >= 2:
            print('Calling StaticScheduleFunction.backward()...')
        # The first time this method is called, the define-by-run code is
        # executed in order to create a static schedule.
        self._schedule_manager.end_forward()
        #
        #debug_grad = grad_outputs[0].data
        #print('static function node backward: grad_outputs', stats.describe(debug_grad, axis=None))
        #print('with id = ', id(debug_grad))

        if self._backward_schedule_func is None:
            print('Creating new backward schedule...')
            # Create backward schedule and run define-by-run backward code.
            self._backward_schedule_func = StaticScheduleFunction(self._schedule_manager,
                                                                  verbosity_level=self.verbosity_level)

            # Make local copies of the variables in grad_outputs.
            new_grad_outputs = []
            for var in grad_outputs:
                # Replace each input variable with a new variable having
                # the same data.
                new_grad_outputs.append(chainer.Variable(var.data.copy()))

            self._backward_schedule_func.add_input_variables(new_grad_outputs)

            with chainer.using_config('schedule_func', self._backward_schedule_func):
                with chainer.using_config('enable_backprop', True):
                    for ind, var in enumerate(new_grad_outputs):
                        dbr_out_var = self._out_vars[ind]
                        dbr_out_var.grad = new_grad_outputs[ind].data
                        self._out_vars[ind].grad = new_grad_outputs[ind].data

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
            self._backward_schedule_func.build_schedule(self.chain)

        #return self._backward_schedule_func.apply(grad_outputs)
        ret = self._backward_schedule_func.apply(grad_outputs)
        return ret


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

        Return a static schedule object (that is, an instance of
        ``StaticScheduleFunction``) that is compatible with
        the current configuration and input variables to the supplied chain.
        If there is no existing schedule available, return an empty schedule
        object.

        During the usual "training mode" (that is, when both
        `chainer.config.enable_backprop` and `chainer.config.train`
        are `True`), this method will always return a distince static
        schedule within the same iteration. It will also try to reuse
        existing schedules across iterations. Therefore, any schedule that
        is returned in a given iteration cannot be returned again until
        the following iteration. However, if either of these flags is
        'False', then this method may return the same schedule instance
        multiple times within the same iteration, as long as it is
        compatible with `in_vars`.

        Note that in order to implement the above behavior, the schedule
        manager must be informed when the current iteration has finished.
        This is accomplished by calling `end_forward()` after the
        iteration has finished.

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

        if chainer.config.train is False or chainer.config.enable_backprop is False:
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
        training mode, then this method will be called automatically.

        fixme: what about: enable_backprop? If it is off, then we still want
        a static schedule to be generated, but loss.backward() cannot be called!
        Therefore, if backprop is disabled, there should be a way to manually
        inform Chainer that the forward pass has completed.

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

    def __repr__(self):
        out = "ScheduleManager:\n"
        for key_str in self.schedules:
            out += "key string: " + key_str
            sched_list = self.schedules[key_str]
            out += " -> schedule list of length: " + str(len(sched_list)) + '\n'
            for sched in sched_list:
                out += str(sched)
        return out

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

    Note: If either 'chainer.config.enable_backprop' or 'chainer.config.train'
    is set to 'False', then cached static schedules will be reused when
    possible to reduce memory usage. We assume that the user will not need
    to perform back propagation if either of these flags is 'False', and so
    it should be safe to reuse the same schedule. That is, if the same static
    chain is  called 'N' times in the same iteration with the same input types,
    a static schedule will be created on the first call and then reused for
    each of the 'N-1' remaining calls.

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
            if verbosity_level >= 2:
                print('Calling static chain...')

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
            if not chain.static_schedule.is_empty():
                # Call the static schedule code.
                if verbosity_level >= 2:
                    print('This is the 2nd or greater iteration. Calling the existing static schedule...')
                # Note: out_vars are dynamically allocated because FunctionNode.apply()
                # will dynamically allocate variables on each call, which is the desired
                # behavior.
                out_vars_flat = chain.static_schedule.apply(flat_vars)

                out_vars = _unflatten_args(out_vars_flat, chain._out_vars_unflatten_inds)

            else:
                # This is the first iteration. Calling the define-by-run code.
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

                # Mark the static schedule as complete.
                chain.static_schedule.build_schedule(chain)

                # Now that the static schedule is available, call it using the
                # flattened input variables. This will cause the
                # static schedule function node to be included in the
                # computational graph.
                out_vars_flat = chain.static_schedule.apply(flat_vars)

                out_vars = _unflatten_args(out_vars_flat, chain._out_vars_unflatten_inds)

                if verbosity_level >= 2:
                    print('Returing from 1st call of the static chain.')

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
