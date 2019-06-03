import sys
import weakref

import numpy as np

import chainer
from chainer.backends import cuda
import chainer.function_node


def _is_xp(x):
    return isinstance(x, np.ndarray) or isinstance(x, cuda.ndarray)


class ScheduleInfo(object):

    """A callable wrapper for a function in the static schedule.

    Args:
        func (FunctionNode): A function in the static schedule.
        args: Arguments to 'func'.
        kwargs: Keyword arguments to 'func'.
        inputs_hooks (list of tuples): A list of hooks that instruct how to
            update the ndarray references in 'args' so that they
            refer to the correct master array in 'unique_arrays'.
        return_hooks (list of tuples): A list of hooks that instruct how
            to update the ndarray references in 'unique_arrays' so that
            they refer to the correct arrays that were dynamically
            allocated and returned by 'func'. These run after
            'func' is called.
        unique_arrays (list of ndarray): The master list of all unique
            ndarrays that appear in the static schedule.
        func_name (str): An optional name of the static function. This is
            the name (if any) that was used as a decorater argument to
            `@static_code(func_name=name)`.
    """

    def __init__(self, func, args, kwargs, inputs_hooks, outputs_hooks,
                 return_hooks, delete_hooks, unique_arrays, array_infos,
                 func_name=None):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.inputs_hooks = inputs_hooks
        self.outputs_hooks = outputs_hooks
        self.return_hooks = return_hooks
        self.unique_arrays = unique_arrays
        self.array_infos = array_infos
        assert len(self.array_infos) == len(self.unique_arrays)
        self.func_name = func_name
        self.in_list = None
        if self.inputs_hooks:
            self.in_list = self.kwargs['inputs']
        if self.outputs_hooks:
            self.out_list = self.kwargs['outputs']
        # Check if 'func' wraps code of a 'FunctionNode':
        self.function_node = None
        if self.args:
            maybe_func = self.args[0]
            if isinstance(maybe_func, chainer.FunctionNode):
                self.function_node = maybe_func
        # List of indices in unique_arrays to delete.
        self.delete_hooks = delete_hooks

    def run_pre_hooks(self):
        """Run hooks to set correct references.

        This method is called from '__call__()'.
        Process the list of hooks which will modify the array references in
        the arguments list of the static function. This method must be
        called before executing the static function.

        The hooks specify that
        each array argument points to a "master" array reference in the
        unique_arrays list. If the reference in unique_arrays changes, then
        we must also change the corresponding array reference in the arguments
        list. The hooks specify the mapping and this method updates the
        references in args to the corresponding values from unique_arrays.
        """
        for hook in self.inputs_hooks:
            (ind, unique_ind) = hook
            self.in_list[ind] = self.unique_arrays[unique_ind]

        for hook in self.outputs_hooks:
            (ind, unique_ind) = hook
            self.out_list[ind] = self.unique_arrays[unique_ind]

        for ind in self.delete_hooks:
            self.unique_arrays[ind] = None

    def run_post_hooks(self, return_arrays):
        """Run post-hooks.

        This method should be called after calling the static function
        `self.func(*self.args)`. This method sets any array references that
        appear in `self.args` to None. This is safe because the master
        array reference is still kept in `self.unique_arrays`.

        Also, process the list of post-hooks which will modify the array
        references in
        the unique_arrays list to refer to the new dynamically-allocated arrays
        that were returned by 'func'.

        Args:
            return_arrays (list of ndarray or None): The list of arrays that
                were returned by the schedule function, if not None.
        """
        for hook in self.inputs_hooks:
            (ind, unique_ind) = hook
            self.in_list[ind] = None

        for hook in self.outputs_hooks:
            (ind, unique_ind) = hook
            self.out_list[ind] = None

        for hook in self.return_hooks:
            # Update the array reference in unique_arrays to refer to the
            # array in the results array.
            (ret_index, unique_list_index) = hook

            # Note: input/output variables to a FunctionNode that are
            # retained using retain_inputs() or retain_outputs() are
            # not currently explicitly used as input arguments to the
            # auto-wrapped functions, and so their corresponding array
            # reference could be used inside a function wrapped with
            # @static_code without the array explicitly appearing in the
            # 'inputs' argument. It is therefore not safe to change the
            # reference of such arrays, and so for them, we must be
            # sure to copy the dynamically-allocated array into the
            # same array that was used in the define-by-run code and
            # set 'need_copy' to True in such cases.
            need_copy = self.array_infos[unique_list_index].retain
            # todo: possible memory leak when need_copy False is allowed?
            if need_copy:
                # This must be used if the model used retain_inputs() or
                # retain_outputs().
                self.unique_arrays[unique_list_index][...] = \
                    return_arrays[ret_index]
            else:
                # This is preferred, when possible, since it should
                # be faster than a copy to simply update the array
                # reference.
                self.unique_arrays[unique_list_index] = \
                    return_arrays[ret_index]

    def __call__(self):
        self.run_pre_hooks()
        ret = self.func(*self.args, **self.kwargs)
        self.run_post_hooks(ret)

    def __repr__(self):
        out = 'function: ' + str(self.func) + '\n'
        out += 'name: ' + str(self.func_name) + '\n'
        out += 'args: ' + str(self.args) + '\n'
        out += 'kwargs: ' + str(self.args) + '\n'
        return out


class ArrayInfo(object):

    """Array information needed by the scheduler.

    This contains information about one array used in the naive static
    schedule corresponding to the define-by-run code.

    """

    def __init__(self, array):
        # Weak reference to the array in the define-by-run code.
        self.weak_ref = weakref.ref(array)
        self.id = id(array)
        # The array (normal reference). Do not create in initializer.
        self.array = None
        self.shape = array.shape
        self.dtype = array.dtype
        # either numpy or cupy
        self.ndarray_module = cuda.get_array_module(array)
        if self.ndarray_module is cuda.cupy:
            # device id, if available.
            self.device = cuda.get_device_from_array(array)
        else:
            # numpy (cpu)
            self.device = -1
        # todo: save array order ('C', 'F' as well?
        # It specifies the input variable corresponding
        # to this array as the tuple (pass_depth, in_var_index).
        self.in_var_index = None
        # It specifies the output variable corresponding
        # to this array as the tuple (pass_depth, out_var_index).
        self.out_var_index = None
        # todo: set in initializer as keyword arg?
        self.dynamically_allocated = False
        # If the array was returned as a dynamically allocated array
        # in the define-by-run code, this specifies the location
        # in the schedule as the tuple (pass_depth, sched_func_index)
        # where sched_func_index is the index of the corresponding
        # ScheduleInfo object in the StaticScheduleFunction's
        # self.schedule_info_list
        self.dynamic_allocation_index = None
        self.dynamic_allocation_pass_depth = None
        self.dynamic_deletion_index = None
        self.dynamic_deletion_pass_depth = None
        # This is the same as self.dynamic_allocation_index, but for the
        # case where the array was statically allocated in the
        # define-by-run code.
        self.static_allocation_index = None
        # If the array needs to be retained (was included in
        # retain_inputs/retain_outputs),
        # this will be set to True later.
        self.retain = False

    def was_deleted(self):
        return self.weak_ref() is None

    def get_new_empty_array(self):
        """Make and return a new empty ndarray.

        Make and return a new empty ndarray that has the same shape,
        dtype, and device as the array that was supplied to the
        initializer.

        """
        # todo: set device id
        return self.ndarray_module.empty(self.shape, dtype=self.dtype)

    def __repr__(self):
        out = 'shape: {}\n'.format(self.shape)
        if self.was_deleted():
            out += 'Weak reference: dead\n'
        else:
            out += 'Weak reference: alive\n'
        if self.retain:
            out += 'Retained with retain_inputs()/retain_outputs().\n'
        if self.dynamically_allocated:
            out += 'Dynamically allocated at\n'
            out += \
                '  pass_depth: {}\n'.format(self.dynamic_allocation_pass_depth)
            out += '  sched_index: {}\n'.format(self.dynamic_allocation_index)
        out += 'array id: {}'.format(self.id)
        return out


class StaticScheduleFunction(chainer.function_node.FunctionNode):

    """A function that executes the static schedule of a Chain.

    An instance of this class executes the static schedule of computations
    that are equivalent to executing the define-by-run code of a Chain.

    This class is used by the `static_graph` decorator to wrap the
    define-by-run
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
    During the first forward pass of a static chain, the define-by-run code
    is executed. However,
    for subsequent iterations, that define-by-run code is replaced by an
    instance
    of this function and this function will be called instead. Since the
    static
    schedules contained by this function perform the same computations, it is
    safe (and potentially much more efficient) to simply execute the static
    schedule instead
    of the define-by-run code. See `static_graph` for details.

    Args:
        schedule_manager (ScheduleManager): The schedule manager of this
            schedule instance.
        in_vars (tuple of Variable): The flattened tuple of input variables
            that is supplied to
            `__call__()` method of the chain that this schedule corresponds to.
        unique_arrays (list of ndarray): A list of all unique array references
            deeply used in an StaticScheduleFunction instance. It is 'None'
            for the StaticScheduleFunction that corresponds to the "forward"
            schedule, but the contained StaticScheduleFunction for the
            "backward" schedule should take the unique_arrays of the
            "forward" schedule.

    """

    def __init__(self, schedule_manager, verbosity_level=0,
                 enable_double_backprop=False):
        # A pass depth of 0 corresponds to the schedule for the forward pass.
        # A pass depth of 1 corresponds to the schedule for the backward pass.
        # A pass depth of 2 corresponds to the schedule for the
        # double-backward pass, and so on.
        self.pass_depth = 0
        self.schedule_manager = schedule_manager
        # A list of ScheduleInfo objects, each of which contains one function
        # in the static schedule. The order of functions in this list is
        # the order they should be called in the schedule.
        self.schedule_info_list = []
        # A list of all unique ndarrays used in this schedule and any deeply
        # contained schedules (backward, double-backward schedules).
        # That is, it is shared among all pass depths.
        # Note that this typically includes the ndarray attributes of the
        # parameters of the chain, the input variables to the chain,
        # and any intermediate arrays (activations, etc) created while
        # executing the define-by-run code of the chain.
        self.unique_arrays = []
        # A list of UniqueArray objects, where
        # each object contains information such as what the array corresponds
        # to (variable, parameter.data, etc), weak or regular reference,
        # whether
        # it was dynamically allocated or read-only in the schedule.
        # It is the same length as unique_arrays.
        self.unique_array_infos = []
        # Maps id(ndarray) to its position in self.unique_arrays
        # This is shared by this schedule and all deeply-contained schedules.
        self.array_id_to_unique_index = dict()
        self.backward_schedule_func = None
        self.verbosity_level = verbosity_level
        self.enable_double_backprop = enable_double_backprop
        self.in_vars = None
        self.chain = None
        self.schedule_built = False
        # A list of all parameters in the model (i.e., that exist when
        # build_schedule() is called.
        # This is shared among all deeply-contained schedules of this schedule.
        self.params_list = []
        # This list contains the grad_var corresponding to each variable
        # in params_list. This is needed so that we can restore any grad_var
        # that is set to None by outside code.
        # This is shared among all deeply-contained schedules of this schedule.
        self.grad_var_list = []
        # Maps an array id (of a parameter) to its location.
        # id(array) -> (index_in_self.params_list, attribute_location)
        self.array_id_to_param_map = dict()
        # Maps an array id (of an input variable for forward()) to its
        # positional index.
        # id(array) -> (index in inputs argument of forward())
        self.array_id_to_input_var_map = dict()
        # maps a Parameter id to the parameter's index in self.params_list
        self.param_id_to_index = dict()
        # A list of tuples that specify the mappings from static schedule
        # arrays to parameter attributes.
        # These are pre-hooks that are run before running the schedule.
        self.param_hooks = []
        # These are post hooks that are run after executing the schedule.
        # They are used to update parameter attributes from dynamically-
        # allocated arrays in the schedule.
        self.param_post_hooks = []
        # A list of tuples that specify the mappings from static schedule
        # arrays to 'data' array attributes of the output variables.
        self.out_var_hooks = []
        # A list of tuples that specify the mapping from static schedule
        # arrays to input variable index in the "inputs" argument of forward()
        # This is used to update the array references in the static schedule
        # that refer to the data attribute of input variables.
        self.in_var_hooks = []
        self.dynamically_allocated_unique_index = set()
        # Maps an index in unique_arrays to the index in the returned
        # output variables, if the index corresponds to an output
        # variable.
        self.unique_ind_to_out_var_ind = dict()

    def get_unique_index_from_array(self, array):
        """Return the array index if it exists.

        Return the index of the array in self.unique_array_infos if the
        array already exists in self.unique_array_info with a valid
        reference. Otherwise, return None.
        """
        ar_id = id(array)
        if ar_id in self.array_id_to_unique_index:
            # It is possible that this id is stale if a previous
            # array that had the same id has already been deleted.
            # So, verify that the existing array with this id is
            # still alive.
            unique_ind = self.array_id_to_unique_index[ar_id]
            info = self.unique_array_infos[unique_ind]
            assert ar_id == info.id
            if info.was_deleted():
                # id was stale, so remove from the dict.
                del self.array_id_to_unique_index[ar_id]
                return None
            else:
                return self.array_id_to_unique_index[ar_id]

    def get_contained_schedule(self):
        # Make and return the backward schedule (relative to
        # this schedule).
        sched = StaticScheduleFunction(self.schedule_manager,
                                       self.verbosity_level,
                                       self.enable_double_backprop)
        sched.pass_depth = self.pass_depth + 1
        sched.unique_arrays = self.unique_arrays
        sched.unique_array_infos = self.unique_array_infos
        sched.array_id_to_unique_index = self.array_id_to_unique_index
        sched.params_list = self.params_list
        sched.grad_var_list = self.grad_var_list
        sched.array_id_to_param_map = self.array_id_to_param_map
        sched.param_hooks = self.param_hooks
        sched.param_id_to_index = self.param_id_to_index
        return sched

    def is_empty(self):
        """Return True if this schedule is empty.

        """
        return len(self.schedule_info_list) == 0

    def append_function(self, func, args, kwargs, func_name=None):
        """Append a function to the static schedule.

        Append a function `func` to the static schedule. `func` can
        be any function that is decorated with `@static_code` and that
        was called while executing the static chain's `__call___()`
        method, which contains the define-by-run code. The code
        in the `@static_code` decorator will call this method to
        add the function to the schedule just after it executes in
        the define-by-run code as follows:

        `return_arrays = func(*args, **kwargs)`

        During the next iteration when the static chain switches from define-
        by-run to the static schedule, a corresponding `ScheduleInfo`
        object will call `func` as above, except that the scheduler might
        make modifications
        to some of the arrays in `kwargs` before and after the function is
        called to implement various memory optimizations.

        Args:
            func (function or method): The function to append to the schedule.
                This is a function that was decorated with `@static_code`.
            args: The arguments that were originally supplied to `func` in
                the define-by-run code of the static chain.
            kwargs: The keyword arguments that were originally supplied to
                `func` in the define-by-run code of the static chain.
            func_name (str): Optional name for `func`, for debugging
                purposes.
            return_arrays (tuple of ndarray) or None: The value that is
                returned by `func`, if any.

        """

        # Check previous function in the schedule, if available.
        # Check the arrays in the retained inputs/outputs and force them
        # to remain statically allocated in the schedule.
        # ids of any retained arrays.
        retained_ids = set()

        last_sched_info_ind = len(self.schedule_info_list) - 1
        if last_sched_info_ind >= 0:
            prev_sched_info = self.schedule_info_list[last_sched_info_ind]
            if prev_sched_info.function_node is not None:
                # get retained inputs/outputs.
                retained_in_vars = \
                    prev_sched_info.function_node.get_retained_inputs()
                retained_out_vars = \
                    prev_sched_info.function_node.get_retained_outputs()
                if (retained_in_vars is not None and
                        retained_out_vars is not None):
                    retained_vars = retained_in_vars + retained_out_vars
                elif retained_in_vars is not None:
                    retained_vars = retained_in_vars
                elif retained_out_vars is not None:
                    retained_vars = retained_out_vars
                else:
                    retained_vars = None
                if retained_vars is not None:
                    for var in retained_vars:
                        retained_ids.add(id(var.data))

        for keep_id in retained_ids:
            unique_ind = self.array_id_to_unique_index[keep_id]
            array_info = self.unique_array_infos[unique_ind]
            array_info.retain = True
            # Note: the following line is not actually needed.
            # array_info.array = array_info.weak_ref()

        delete_hooks = []
        for unique_ind, ar_info in enumerate(self.unique_array_infos):
            # todo: this is O(N^2) and maybe too slow for large graphs.
            # Optimize it later.
            if ar_info.was_deleted():
                if ar_info.dynamic_deletion_index is None:
                    if self.verbosity_level >= 2:
                        print('Adding delete hook:')
                    delete_hooks.append(unique_ind)
                    ar_info.dynamic_deletion_index = last_sched_info_ind + 1
                    ar_info.dynamic_deletion_pass_depth = self.pass_depth

        # Call the `@static_code`-decorated function.
        ret = func(*args, **kwargs)

        inputs_hooks = []
        if 'inputs' in kwargs:
            in_list = kwargs['inputs']
            assert isinstance(in_list, list)
            for ind, x in enumerate(in_list):
                if _is_xp(x):
                    unique_ind = self.get_unique_index_from_array(x)
                    if unique_ind is None:
                        # Note: we append None here because we cannot store any
                        # additional reference to the array.
                        # Otherwise, it would
                        # prevent garbage collection. Note that a
                        # weak reference
                        # will be stored in the ArrayInfo below.
                        self.unique_arrays.append(None)
                        self.unique_array_infos.append(ArrayInfo(x))
                        unique_ind = len(self.unique_arrays) - 1
                        self.array_id_to_unique_index[id(x)] = unique_ind
                    inputs_hooks.append((ind, unique_ind))
                    # Now that the hook has been added, we can delete
                    # array reference from 'args'.
                    in_list[ind] = None

        outputs_hooks = []
        if 'outputs' in kwargs:
            out_list = kwargs['outputs']
            assert isinstance(out_list, list)
            for ind, x in enumerate(out_list):
                if _is_xp(x):
                    unique_ind = self.get_unique_index_from_array(x)
                    if unique_ind is None:
                        self.unique_arrays.append(x)
                        # todo: enable the following line instead once the
                        # auto-intializing hooks are added. This will further
                        # reduce memory usage.
                        # self.unique_arrays.append(None)
                        self.unique_array_infos.append(ArrayInfo(x))
                        unique_ind = len(self.unique_arrays) - 1
                        self.array_id_to_unique_index[id(x)] = unique_ind
                    outputs_hooks.append((ind, unique_ind))
                    # Now that the hook has been added, we can delete
                    # array reference from 'args'.
                    out_list[ind] = None

        # A list of hooks (each is a tuple) that will be used to set
        # correct array references in 'unique_arrays' after executing
        # the static schedule function 'func'. These hooks update
        # the references in 'unique_arrays' to refer to the arrays
        # that were dynamically allocated in the return value of
        # 'func'.
        return_hooks = []
        if ret is not None:
            assert (isinstance(ret, list) or
                    isinstance(ret, tuple))
            for ret_index, item in enumerate(ret):
                if _is_xp(item):
                    # note: id might not be unique if objects have been
                    # garbage collected.
                    item_id = id(item)
                    unique_index = self.get_unique_index_from_array(item)
                    if unique_index is None:
                        # Note: Append None instead of 'item' to prevent an
                        # extra reference from being stored. Otherwise it
                        # would prevent garbage collection.
                        self.unique_arrays.append(None)
                        ar_info = ArrayInfo(item)
                        ar_info.dynamically_allocated = True
                        sched_info_ind = len(self.schedule_info_list)
                        ar_info.dynamic_allocation_index = sched_info_ind
                        ar_info.dynamic_allocation_pass_depth = self.pass_depth
                        self.unique_array_infos.append(ar_info)
                        unique_index = len(self.unique_arrays) - 1
                        self.array_id_to_unique_index[item_id] = \
                            unique_index
                    else:
                        # Since all of the return arrays are supposed to
                        # have been dynamically allocated inside 'func',
                        # they had better not already be in unique_arrays.
                        # If so, it is an error.
                        unique_index = self.array_id_to_unique_index[item_id]
                        print('the current id: ', item_id)
                        print('the unique_index: ', unique_index)
                        print('array info: ',
                              self.unique_array_infos[unique_ind])
                        raise RuntimeError('Found result array from schedule '
                                           'function already in '
                                           'unique_arrays!')
                    return_hooks.append((ret_index, unique_index))
                    self.dynamically_allocated_unique_index.add(unique_index)

        if self.verbosity_level >= 2:
            print('Adding function to static schedule: ', func)

        self.schedule_info_list.append(ScheduleInfo(func, args, kwargs,
                                                    inputs_hooks,
                                                    outputs_hooks,
                                                    return_hooks,
                                                    delete_hooks,
                                                    self.unique_arrays,
                                                    self.unique_array_infos,
                                                    func_name=func_name))

        return ret

    def __repr__(self):
        out = 'StaticSchedule:\n'
        if self.pass_depth == 0:
            depth = 'forward pass'
        elif self.pass_depth == 1:
            depth = 'backward pass'
        elif self.pass_depth == 2:
            depth = 'double backward pass'
        else:
            depth = str(self.pass_depth)
        out += 'Pass depth: ' + depth + '\n'
        out += 'Length of unique_arrays: ' + \
               str(len(self.unique_arrays)) + '\n'
        for x in self.schedule_info_list:
            out += str(x)
        return out

    def debug_print_ref_counts(self):
        print('reference counts in unique_arrays:')
        for ind in range(len(self.unique_arrays)):
            print('index: ', ind)
            print('reference count: ',
                  sys.getrefcount(self.unique_arrays[ind]))

    def run_param_pre_hooks(self):
        """Run parameter reference updater hooks.

        It also handles the case where the 'grad' attribute
        was set to 'None' by outside Chainer code.

        """
        for hook in self.param_hooks:
            (unique_array_index, param_attribute_location) = hook
            (params_list_index, attribute_location) = param_attribute_location
            if attribute_location == 'data':
                # This is the corresponding parameter array, which might
                # have had its reference changed to a different array or set
                # to None.
                self.unique_arrays[unique_array_index] = \
                    self.params_list[params_list_index].data
            elif attribute_location == 'grad':
                # This is the corresponding parameter array, which might
                # have had its reference changed to a different array or set
                # to None.
                self.params_list[params_list_index].grad = \
                    self.unique_arrays[unique_array_index]

    def run_param_post_hooks(self):
        """Update parameter attributes after schedule is executed.

        If any dynamically-allocated arrays in the schedule correspond to
        a parameter attribute, it must be updated after the schedule is
        run.
        """
        if self.verbosity_level >= 2:
            print('run_param_post_hooks()...')
        for hook in self.param_post_hooks:
            (unique_array_index, param_attribute_location) = hook
            (params_list_index, attribute_location) = param_attribute_location
            if attribute_location == 'data':
                self.params_list[params_list_index].data = \
                    self.unique_arrays[unique_array_index]
            elif attribute_location == 'grad':
                self.params_list[params_list_index].grad = \
                    self.unique_arrays[unique_array_index]

    def run_in_var_hooks(self, input_var_arrays):
        """Run hooks to update variable array references.

        Args:
            input_var_arrays (tuple of ndarray): The 'data' array attributes
                of the input variables to this function.
        """
        for hook in self.in_var_hooks:
            (unique_array_index, in_var_ind) = hook
            if self.verbosity_level >= 2:
                print('input var hook:')
                print('unique_array_index: ', unique_array_index)
                print('in_var_ind: ', in_var_ind)
                print('_run_in_var_hooks(): Using this input variable array '
                      'for forward pass: ', input_var_arrays[in_var_ind])
            self.unique_arrays[unique_array_index] = \
                input_var_arrays[in_var_ind]

    def debug_print_unique_arrays_info(self):
        for ind, item in enumerate(self.unique_arrays):
            print('--- unique_arrays ---')
            print('index: {0}; id: {1}'.format(ind, id(item)))

            if item is not None:
                print('shape: ', item.shape)
            if ind in self.unique_ind_to_out_var_ind:
                out_var_ind = self.unique_ind_to_out_var_ind[ind]
                print('output variable at return index: ', out_var_ind)
            if ind in self.dynamically_allocated_unique_index:
                print('Dynamically allocated inside schedule.')

    def run_out_var_hooks(self):
        """Run hooks to update output variable array references.


        """
        for hook in self.out_var_hooks:
            (out_var_ind, unique_list_index) = hook
            out_var = self.out_vars[out_var_ind]
            out_var.data = self.unique_arrays[unique_list_index]
            if self.verbosity_level >= 2:
                print('StaticScheduleFunction: running output variable hook: '
                      'out_var_ind, unique_list_index): ', hook)

    def set_out_variables(self, out_vars):
        """Set output variables.

        This should be called after the define-by-run code in the
        chain's `__call__()` has already run but before running the
        static schedule.

        Args:
            out_vars (list of Variable): The (flattened) list of output
                variables obtained by performing a define-by-run
                forward pass (or corresponding backward pass) on the
                local sub-graph corresponding to the static chain.
        """
        self.out_vars = out_vars
        # Create output-variable update hooks.
        for var_ind, var in enumerate(out_vars):
            if var is not None:
                key = id(var.data)
                if key in self.array_id_to_unique_index:
                    unique_list_index = self.array_id_to_unique_index[key]
                    self.out_var_hooks.append((var_ind, unique_list_index))
                    self.unique_ind_to_out_var_ind[unique_list_index] = var_ind
                else:
                    raise RuntimeError('Could not find output variable in '
                                       'unique_arrays.')

    def build_schedule(self, chain, in_vars):
        """Build the static schedule.

        Perform one-time post-processing on the functions and arguments
        that were
        previously supplied in 'append_function()' to create the static
        schedule.

        This method must be called after the final call of 'append_function()'
        and before calling 'forward()' for the first time.

        Args:
            chain: The static chain that uses this scheudle.
            in_vars (list of Variable): The input variables to this static
                schedule. This are the input variables (each having no
                creator) of the local sub-graph corresponding to the
                static chain.
        """
        self.chain = chain
        self.in_vars = in_vars

        # Iterate through all array info objects and for any arrays that
        # still have a valid reference, copy into unique_arrays.
        if self.verbosity_level >= 2:
            print('Building schedule for pass depth: ', self.pass_depth)
        for ind, info in enumerate(self.unique_array_infos):
            if self.verbosity_level >= 2:
                print('unique array index: ', ind)
                print('array info: ', info)
            if not info.was_deleted():
                self.unique_arrays[ind] = info.weak_ref()

        # Verify that all array references are actually unique.
        unique_ids = set()
        for ar in self.unique_arrays:
            if ar is not None:
                assert id(ar) not in unique_ids
                unique_ids.add(id(ar))

        for param in chain.params():
            param_key = id(param)
            if param_key not in self.param_id_to_index:
                self.params_list.append(param)
                grad_var = param.grad_var
                self.grad_var_list.append(grad_var)
                param_index = len(self.params_list) - 1
                self.param_id_to_index[param_key] = param_index
            else:
                # We have seen this parameter before.
                param_index = self.param_id_to_index[param_key]
            grad_var = param.grad_var
            self.grad_var_list[param_index] = grad_var
            if param.data is not None:
                key = id(param.data)
                if key not in self.array_id_to_param_map:
                    self.array_id_to_param_map[key] = (param_index, 'data')
            if param.grad is not None:
                key = id(param.grad)
                if key not in self.array_id_to_param_map:
                    self.array_id_to_param_map[key] = (param_index, 'grad')

        for var_ind, in_var in enumerate(self.in_vars):
            assert in_var.data is not None
            key = id(in_var.data)
            self.array_id_to_input_var_map[key] = var_ind

        # Iterate over all arrays used in the schedule and check which ones
        # correspond to parameter arrays or input variables. When a match
        # is found, create a corresponding hook function. This hook will
        # run just before executing the schedule and set the array
        # references used in the schedule to be consistent with the
        # input variables and parameters.
        assert len(self.unique_arrays) > 0
        for unique_array_index, ar in enumerate(self.unique_arrays):
            key = id(ar)
            # Create pre-run parameter hooks.
            if key in self.array_id_to_param_map:
                param_attribute_location = self.array_id_to_param_map[key]
                param_hook = (unique_array_index, param_attribute_location)
                self.param_hooks.append(param_hook)
            # Create pre-run input variable hooks.
            if key in self.array_id_to_input_var_map:
                in_var_ind = self.array_id_to_input_var_map[key]
                in_var_hook = (unique_array_index, in_var_ind)
                self.in_var_hooks.append(in_var_hook)
                if self.verbosity_level >= 2:
                    print('build_schedule(): Adding input variable hook: ',
                          in_var_hook)
                    print('For input variable: ', ar)
            # Create post-run hooks for any arrays that are dynamically
            # allocated inside the schedule.
            if unique_array_index in self.dynamically_allocated_unique_index:
                if key in self.array_id_to_param_map:
                    param_attribute_location = self.array_id_to_param_map[key]
                    param_hook = (unique_array_index, param_attribute_location)
                    self.param_post_hooks.append(param_hook)

        if self.verbosity_level >= 2:
            print('self.param_hooks: ', self.param_hooks)
            self.debug_print_unique_arrays_info()

        # todo: We can potentially reduce memory usage by freeing memory
        # of intermediate arrays in self.unique_arrays
        # once they are no longer needed in the schedule or by
        # parameters.

        print('end of build_schedule()')
        self.schedule_built = True

    def forward(self, inputs):
        if self.verbosity_level >= 2:
            print('Calling StaticScheduleFunction.forward()...')

        # Note: This method will be invoked every iteration starting
        # from the second
        # iteration. That is because the corresponding define-by-run
        # code runs instead
        # during the first iteration.
        if not self.schedule_built:
            raise RuntimeError('forward() was called before '
                               'build_schedule()!')
        self.run_param_pre_hooks()
        self.run_in_var_hooks(inputs)

        if self.verbosity_level >= 2:
            print('Running static schedule...')
        # Run each function in the static schedule.
        for x in self.schedule_info_list:
            x()
        if self.verbosity_level >= 2:
            self.debug_print_unique_arrays_info()

        self.run_out_var_hooks()
        self.run_param_post_hooks()
        ret = []
        for y in self.out_vars:
            if y is None or y.data is None:
                ret.append(None)
            else:
                # todo: add test case for an example where the following
                # copy is required (evaluation mode, repeated calls of
                # chain that reuse same schedule).
                ret.append(y.data.copy())
        return tuple(ret)

    def backward(self, target_input_indexes, grad_outputs):
        if self.verbosity_level >= 2:
            print('Calling StaticScheduleFunction.backward()...')
        # The first time this method is called, the define-by-run code is
        # executed in order to create a static schedule.
        self.schedule_manager.end_forward()
        if self.backward_schedule_func is None:
            print('Creating new backward schedule...')
            # Create backward schedule and run define-by-run backward code.
            self.backward_schedule_func = self.get_contained_schedule()
            # Make local copies of the variables in grad_outputs.
            new_grad_outputs = []
            for var in grad_outputs:
                # Replace each input variable with a new variable having
                # the same data.
                new_grad_outputs.append(chainer.Variable(var.data))
            with chainer.using_config('schedule_func',
                                      self.backward_schedule_func):
                with chainer.using_config('enable_backprop', True):
                    for ind, var in enumerate(new_grad_outputs):
                        # todo: possibly don't need the following:
                        self.out_vars[ind].grad = new_grad_outputs[ind].data

                    inputs = [param for param in self.chain.params()]
                    for var in self.in_vars:
                        inputs.append(var)
                    # Need shorter var to avoid "line too long error"
                    ugh = self.enable_double_backprop
                    chainer.grad(self.out_vars,
                                 inputs,
                                 grad_outputs=new_grad_outputs,
                                 set_grad=True,
                                 enable_double_backprop=ugh)

            # We no longer need the backward graph from self.out_vars, so
            # unchain them.
            # todo (vogel): enable this eventually. For now, it
            # causes some needed variables to be set to None
            # in some models such as CIFAR example.
            # for var in self.out_vars:
            #    var.unchain_backward()

            # Note: var.grad_var is allowed to be None below:
            backward_out_vars = [var.grad_var for var in self.in_vars]
            self.backward_schedule_func.set_out_variables(backward_out_vars)
            for n in range(len(self.in_vars)):
                self.in_vars[n] = None
            if self.verbosity_level >= 2:
                print('building backward schedule.')
            self.backward_schedule_func.build_schedule(self.chain,
                                                       new_grad_outputs)

        return self.backward_schedule_func.apply(grad_outputs)


class ScheduleManager(object):

    """A manager of static schedules for a static chain.

    This is a container of the static schedules that are used by a static
    chain.

    Args:
        minimize_cache_size (bool): If `True`, attempt to reduce memory
        usage by clearing the cached schedules whenever the training
        mode changes (that is, whenever `chainer.config.train` changes
        value) or whenever the mini-batch size changes.


    """

    def __init__(self, minimize_cache_size=True, verbosity_level=0):
        # Maps a key string to a list of schedule functions.
        self.schedules = dict()
        self.minimize_cache_size = minimize_cache_size
        self.in_use_count = dict()
        self.forward_over = False
        self.prev_train_config = None
        self.max_in_use_train = 0
        self.train_count = 0
        self.verbosity_level = verbosity_level

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
        schedule each time it is called within the same iteration.
        It will also try to reuse
        existing schedules across iterations. Therefore, any schedule that
        is returned in a given iteration cannot be returned again until
        the following iteration. However, if either of these flags is
        'False', then this method may return the same schedule instance
        multiple times within the same iteration, as long as it is
        compatible with `in_vars`.

        Note that in order to implement the above behavior, the schedule
        manager must be informed when the current iteration has finished.
        This is accomplished by calling `end_forward()` after the
        iteration has finished. If a backward pass is performed, then
        `end_forward()` will be automatically called. Otherwise, it
        will not be called and the user will be responsible for calling
        it.

        Args:
            in_vars (tuple of :class:`~chainer.Variable`): The input
                variables to the chain.

        Returns:
            An instance of ``StaticScheduleFunction``.
        """
        if self.forward_over:
            self.forward_over = False
        if self.minimize_cache_size:
            if chainer.config.train != self.prev_train_config:
                # Training config changed, so clear caches.
                self.prev_train_config = chainer.config.train
                if self.verbosity_level >= 2:
                    print('Clearing schedule cache...')
                self.schedules.clear()
                self.in_use_count.clear()

        if (chainer.config.train is False or
                chainer.config.enable_backprop is False):
            key_str = 'test:' + \
                      ''.join(str(x.shape) + str(x.dtype) for x in in_vars)
            # If the maximum number of in-use schedules in any iteration
            # during training mode was exactly 1, assume it should also
            # be 1 for test mode.
            if key_str in self.schedules:
                sched_list = self.schedules[key_str]
                sched = sched_list[0]
            else:
                # avoid "line too long":
                vb = self.verbosity_level
                edb = enable_double_backprop
                sched = StaticScheduleFunction(self,
                                               verbosity_level=vb,
                                               enable_double_backprop=edb)
                self.schedules[key_str] = [sched]
            return sched
        else:
            key_str = 'train:' + \
                      ''.join(str(x.shape) + str(x.dtype) for x in in_vars)
            self.train_count += 1

            if key_str in self.schedules:
                sched_list = self.schedules[key_str]
                available_index = self.in_use_count[key_str]
                if available_index >= len(sched_list):
                    # avoid "line too long":
                    vb = self.verbosity_level
                    edb = enable_double_backprop
                    sched = StaticScheduleFunction(self,
                                                   verbosity_level=vb,
                                                   enable_double_backprop=edb)
                    sched_list.append(sched)

                sched = sched_list[available_index]
                self.in_use_count[key_str] = available_index + 1
            else:
                # avoid "line too long":
                vb = self.verbosity_level
                edb = enable_double_backprop
                sched = StaticScheduleFunction(self,
                                               verbosity_level=vb,
                                               enable_double_backprop=edb)
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

        """
        if not self.forward_over:
            for key in self.in_use_count:
                self.in_use_count[key] = 0
            self.forward_over = True

            if self.train_count > self.max_in_use_train:
                self.max_in_use_train = self.train_count
                if self.verbosity_level >= 2:
                    print('Maximum in-use schedules per training iteration: ',
                          self.max_in_use_train)
            self.train_count = 0

    def __repr__(self):
        out = 'ScheduleManager:\n'
        for key_str in self.schedules:
            out += 'key string: ' + key_str
            sched_list = self.schedules[key_str]
            out += ' -> schedule list of length: ' + \
                   str(len(sched_list)) + '\n'
            for sched in sched_list:
                out += str(sched)
        return out


def static_graph(*args, **kwargs):
    """Decorator to mark a Chain's ``__call__()`` as a static sub-graph.

    This decorator marks the define-by-run code inside the `__call__()`
    method of a Chain instance as corresponding to a static computation
    graph or sub-graph. Such a chain will be referred to as a 'static chain'.
    This allows various "static graph" optimizations to be performed, which
    can result in significant speedups for some models.

    When this decorator is used, the chain's define-by-run code executes
    during the first iteration as usual. However, while the define-by-run
    code is executing, a trace is also performed to incrementally create a
    corresponding static schedule. This static schedule will only contain
    the subset of the computations inside the define-by-run code that actually
    needs to run every iteration. Specifically, this will contain the code
    inside any functions called that were annotated with the `@static_code`
    decorator, which will include all Chainer built-in functions, as well as
    any user-defined functions that use `@static_code`. Then, starting
    from the second iteration, when the static chain is called, its
    static schedule code will be executed instead of its define-by-run code.

    However, the user must also be careful of the following:
    - The user is responsible for applying this decorator correctly. The
    framework
    does not check that the define-by-run code corresponds to a static
    graph. The graph can be different between training and
    evaluation mode (such as when dropout and/or batch normalization are
    used), but should otherwise be static.
    - When `chainer.config.enable_backprop` is enabled, if a backward pass
    is not performed each iteration, then the user code must call a method
    `chain.schedule_manager.end_forward()`on the static chain each iteration.
    - Static graphs allow tradeoffs between computation and memory usage.
    For example, the `minimize_cache_size` argument will typically result in
    higher memory useage when set to `False` because all cached schedules
    are retained.
    - When this feature is enabled, only the Chainer function and/or link
    calls inside the chain's `__call__()` method will be included in the
    static schedule by default. An other code that the user puts in
    `__call__()`, such as a print statement or code to increment a counter
    for example, will not automatically get added. We will refer to such
    code other than Chainer function/link calls as "side-effect" code.
    Since side-effect code does not get included in the static schedule
    by default, this means that it will only every execute once, during
    the first iteration. There is a way to force side-effect code to be
    included in the static schedule, however: the user can wrapp such
    code inside a function that is decorated with
    `@static_code` to ensure that it gets added to the static schedule.
    For an example of this, refer to the documentation.
    - This feature is experimental and advanced optimizations such
    as kernel fusion and various memory optimizations are not implemented
    yet.

    Usage:

    This decorator should only be applied
    to define-by-run code that actually corresponds to a static subgraph.
    Refer to the documenation for additional details and examples of
    correct usage.
    This decorator should be applied to each of the largest static
    subgraphs in the model; it can also be applied to a static subgraph
    that is not the largest subgraph, but that could result in reduced
    performance.
    It is not currently allowed to
    mark a chain as static if it is contained within
    another chain that is also marked as being static.
    For example, suppose a
    static graph `A` contains a static sub-graph `B`. Then, only the chain
    corresponding to `A` should be marked as static and the chain
    corresponding
    to `B` should not be marked as static.

    The behavior of a static chain depends on the training mode flag,
    `chainer.config.train`. If it is `True`, then a static chain that is
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
    must call `chain.schedule_manager.end_forward()` after the end
    of each forward pass.

    If test mode is active (`chainer.config.train` is `False`) then it
    is not necessary to inform the chain at the end of each forward pass
    because in test mode, a static chain always attempts to reuse
    existing static schedule objects. The same static schedule can be reused
    during a single forward pass, because it is not necessary to compute
    gradients.
    It is also possible to disable static optimzations while in test mode by
    setting the decorator argument `force_test_define_by_run=True`.

    Note: If either 'chainer.config.enable_backprop' or 'chainer.config.train'
    is set to 'False', then cached static schedules will be reused when
    possible to reduce memory usage.

    Double-backprop:
        Double-backpropagation is not enabled by default. It can be enabled by
        supplying the keyword argument ``enable_double_backprop=True``
        to this decorator. Note: this feature has not been tested yet.

    Restrictions on input arguments and return values of a static chain:
        Recall that unlike a function, there is no restrictions on the
        arguments to a chain. However, there currently are some restrictions
        when a static chain is used. Specifically, the arguments to a static
        chain must consist of a variable, list or tuple. In the case of a list
        or tuple, the elements are required to be an instance of variable,
        list, or tuple. There can be an arbitrary number of nested lists/
        tuples. No other object types are allowed.
        In addition, keyword arguments are not allowed.
        The return value of a static chain must be a
        variable, list, or tuple in which each element of the list or
        tuple is also a variable, list, or tuple.

    This decorator can be supplied with the following optional keyword
    arguments. This is an experimental feature, and the API and arguments
    might change

    Args:
        force_test_define_by_run (bool): If `True`, disable static graph
            optimizations during test mode (that is, when
            `chainer.config.train` is False). This may be needed in order
            for some existing RNN links such as LSTM to work correctly,
            since some existing links do not correspond to a static graph
            in some cases.
            The default is `False`.

        minimize_cache_size (bool): If `True`, minimize the number of cached
            static schedules in order to reduce memory usage. For example,
            if the mini-batch size changes or the training mode changes,
            the schedules will need to be recomputed, but memory is also
            saved by not retaining all cached schedules.
            The default value is `True`.

        verbosity_level (int): Depending on the value, print additional
            information:
            0: Warnings only. (the default value)
            1: Show only information that is collected during the first
            iteration and when a new static schedule is created.
            2: Detailed debugging information, possibly showing new
            information every iteration.

        enable_double_backprop (bool): If `True`, enable double-backprop.
            The default value is `False` (not enabled).

    Returns:
        Wrapped ``__call__()`` method with static chain support.

    """

    # todo: consider to allow nested use of this decorator.
    force_test_define_by_run = False
    # todo: enable eventually
    minimize_cache_size = False
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
                # This assumes x is either a variable or ndarray.
                # todo: check this and handle case when it is not.
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
            # To prevent "line too long" error
            edb = enable_double_backprop
            chain.static_schedule = \
                schedule_manager.get_schedule(flat_vars,
                                              enable_double_backprop=edb)

            if verbosity_level >= 2:
                print('Current schedule manager info: ', schedule_manager)
            if not chain.static_schedule.is_empty():
                # Call the static schedule code.
                if verbosity_level >= 2:
                    print('This is the 2nd or greater iteration. Calling '
                          'the existing static schedule...')
                    chain.static_schedule.debug_print_ref_counts()
                out_vars_flat = chain.static_schedule.apply(flat_vars)
                out_vars = _unflatten_args(out_vars_flat,
                                           chain._out_vars_unflatten_inds)
            else:
                # This is the first iteration. Calling the define-by-run code.
                assert isinstance(chain, chainer.Chain)
                if verbosity_level >= 2:
                    print('This is the first iteration. Calling the '
                          'define-by-run code.: ', func)
                # First check that this chain is not called from inside another
                # static chain because it is not allowed.
                if chainer.config.schedule_func is not None:
                    raise RuntimeError('Not allowed to nest static chains: ',
                                       chain)

                new_args = []
                new_args.append(chain)
                new_flat_vars = []
                for var in flat_vars:
                    # Replace each input variable with a new variable having
                    # the same data. This is needed so that the chain-local
                    # computation graph will be rooted at the input variables.
                    new_flat_vars.append(chainer.Variable(var.data))

                unflat_in_args = _unflatten_args_as_list(new_flat_vars,
                                                         in_unflatten_inds)

                for item in unflat_in_args:
                    new_args.append(item)

                inner_args = tuple(new_args)

                with chainer.using_config('schedule_func',
                                          chain.static_schedule):
                    # Execute the chain's call() method. As the define-by-run
                    # code executes, the static schedule is constructed.
                    out_vars = func(*inner_args, **inner_kwargs)

                out_vars_flat_dbr, chain._out_vars_unflatten_inds, __ = \
                    _flatten_args(out_vars)
                sched_out_vars = list(out_vars_flat_dbr)
                chain.static_schedule.set_out_variables(sched_out_vars)

                # Mark the static schedule as complete.
                chain.static_schedule.build_schedule(chain, new_flat_vars)

                # Now that the static schedule is available, call it using the
                # flattened input variables. This will cause the
                # static schedule function node to be included in the
                # computational graph.
                out_vars_flat = chain.static_schedule.apply(flat_vars)

                out_vars = _unflatten_args(out_vars_flat,
                                           chain._out_vars_unflatten_inds)

                if verbosity_level >= 2:
                    print('Returing from 1st call of the static chain.')

            return out_vars

        return wrapped_func

    if zero_args:
        return wrap(callable_arg)
    else:
        return wrap


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


# todo: this only outputs tuples of tuples. Any list in the original input
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
