import collections
import copy
import heapq
import traceback
import warnings
import weakref

import numpy
import six

import chainer
from chainer import cuda
from chainer import initializers
from chainer.initializers import constant
from chainer import utils
from chainer.utils import argument


def _check_grad_type(func, x, gx):
    def make_message(message):
        if func:
            detail = 'Function `{0}` ({1}) has a bug.\n'.format(
                type(func).__name__, func.label)

            stack = func.stack
            if stack:
                detail += 'Stacktrace of the function is below:\n'
                for line in traceback.format_list(func._stack):
                    detail += line

            detail += '''
Please report this error to the issue tracker with the stack trace,
the information of your environment, and your script:
https://github.com/pfnet/chainer/issues/new.
'''.format(type(func).__name__, func.label)

        else:
            detail = ''

        detail += message
        return detail

    if x.data is None or gx is None:
        # ``x.data is None`` implies that the data array is not retained
        return
    if not isinstance(gx, type(x.data)):
        msg = ('Type of data and grad mismatch\n%s != %s' %
               (type(x.data), type(gx)))
        raise TypeError(make_message(msg))
    if gx.dtype != x.data.dtype:
        msg = ('Dtype of data and grad mismatch\n%s != %s' %
               (x.data.dtype, gx.dtype))
        raise TypeError(make_message(msg))
    if gx.shape != x.data.shape:
        msg = ('Shape of data and grad mismatch\n%s != %s' %
               (x.data.shape, gx.shape))
        raise ValueError(make_message(msg))


def variable_repr(var):
    """Return the string representation of a variable.

    Args:
        var (~chainer.Variable): Input Variable.
    .. seealso:: numpy.array_repr
    """
    xp = cuda.get_array_module(var)
    if xp is numpy:
        arr = var.data
    else:
        arr = var.data.get()

    if var.name:
        prefix = 'variable ' + var.name
    else:
        prefix = 'variable'

    if arr is None:
        lst = 'None'
    elif arr.size > 0 or arr.shape == (0,):
        lst = numpy.array2string(arr, None, None, None, ', ', prefix + '(')
    else:  # show zero-length shape unless it is (0,)
        lst = '[], shape=%s' % (repr(arr.shape),)

    return '%s(%s)' % (prefix, lst)


def variable_str(var):
    """Return the string representation of a variable.

    Args:
        var (~chainer.Variable): Input Variable.
    .. seealso:: numpy.array_str
    """
    xp = cuda.get_array_module(var)
    if xp is numpy:
        arr = var.data
    else:
        arr = var.data.get()

    if var.name:
        prefix = 'variable ' + var.name
    else:
        prefix = 'variable'

    if arr is None:
        lst = 'None'
    else:
        lst = numpy.array2string(arr, None, None, None, ' ', prefix + '(')

    return '%s(%s)' % (prefix, lst)


class VariableNode(object):

    """Node in the backward computational graph representing a variable.

    This object represents a variable node in a computational graph. The node
    is used in error backpropagation (a.k.a. backprop) to determine which
    gradient to be passed to each function.

    A variable node is held by the corresponding :class:`Variable` object,
    which is managed by users. :class:`Function` objects that take the variable
    as an input also hold references to the variable node.

    Note that the node does not hold a reference to the corresponding data
    array in general. The data array is actually accessible by the node in the
    following cases.

    1. If there exists a :class:`Variable` object that holds a reference to the
       variable node, the variable node holds a weak reference to the variable
       object, and thus the data array is accessible via the weak reference.
    2. If :meth:`retain_data` is called, the node holds a reference to the data
       array. It is mainly called by a function that needs the input or output
       data array in its backprop procedure. See :meth:`Function.retain_inputs`
       and :meth:`Function.retain_outputs` for more details.

    Users usually do not need to touch this variable node object. The
    computational graph is automatically managed by Chainer, and any interface
    that is beneficial for users is also provided by :class:`Variable`.

    Args:
        variable (Variable): The corresponding variable object.
        name (str): Name of the variable node.

    Attributes:
        dtype: Data type of the data array.
        shape: Shape of the data array.
        name (str): Name of the variable node.

    """

    def __init__(self, variable, name, grad=None):
        self._variable = weakref.ref(variable)
        self._creator = None
        self._data = None
        self._rank = 0
        self.name = name
        self._requires_grad = variable.requires_grad

        vdata = variable.data
        self._set_data_type(vdata)

        self.grad = grad

    @property
    def creator(self):
        """Function node that created this variable node."""
        return self._creator

    @creator.setter
    def creator(self, func):
        self._creator = func
        if func is not None:
            self._rank = func.rank + 1

    @property
    def data(self):
        """Data array of the corresponding variable.

        If the data is not available, it returns ``None``.

        """
        return self._data

    @data.setter
    def data(self, d):
        self._data = d
        self._set_data_type(d)

    @property
    def grad(self):
        """Gradient array of the corresponding variable."""
        return self._grad

    @grad.setter
    def grad(self, g):
        _check_grad_type(None, self, g)
        self._grad = g

    @property
    def label(self):
        """Short text that represents the variable node."""
        if self.shape == ():
            return str(self.dtype)
        return '(%s), %s' % (', '.join(map(str, self.shape)),
                             str(self.dtype))

    @property
    def rank(self):
        return self._rank

    @property
    def requires_grad(self):
        """It indicates that ``grad`` will be set in backward calculation."""
        return self._requires_grad

    def set_creator(self, creator):
        """Sets a :class:`Function` object that created this node.

        This method is equivalent to ``self.creator = creator``.

        Args:
            creator (Function): Function object that created this node.

        """
        self.creator = creator

    def unchain(self):
        """Deletes the reference to the creator of this variable node.

        This method is equivalent to ``self.creator = None``.

        """
        self.creator = None

    def retain_data(self):
        """Lets the node hold a reference to the underlying data array.

        This method gets the data array of the corresponding variable and keeps
        it. If the weak reference to the corresponding variable is dead, it
        raises an error.

        """
        variable = self._variable()
        if variable is not None:
            self.data = variable.data
        else:
            raise RuntimeError('cannot retain variable data: the variable has '
                               'been already released')

    def _set_data_type(self, d):
        if d is None:
            self.dtype = None
            self.shape = None
        else:
            self.dtype = d.dtype
            self.shape = d.shape

    def _set_grad_with_check(self, g, func, var):
        _check_grad_type(func, var, g)
        self._grad = g


def _create_variable(data, name, grad, requires_grad):
    return Variable(
        data, name=name, grad=grad, requires_grad=requires_grad)


class Variable(object):

    """__init__(data=None, *, name=None, grad=None, initializer=None, update_rule=None, requires_grad=True)

    Array with a structure to keep track of computation.

    Every variable holds a data array of type either :class:`numpy.ndarray` or
    :class:`cupy.ndarray`.

    A variable object holds a data array and a :class:`VariableNode` object of
    a computational graph. If the variable is constructed by the user, the node
    is _root_ and does not hold any parent. If the variable is constructed by a
    :class:`Function` object, the node holds a reference to its parent called
    `creator`. This reference is used in backpropagation to backtrack the
    graph.

    Users can disable (resp. enable) this chaining behavior by calling
    :func:`~chainer.no_backprop_mode` (resp.
    :func:`~chainer.force_backprop_mode`).
    In the former context, a variable never creates a computational graph,
    whereas in the latter context, it is forced to create.

    .. warning::

       ``volatile`` argument is not supported anymore since v2.
       Instead, use :func:`chainer.no_backprop_mode`.

    Args:
        data (numpy.ndarray or cupy.ndarray): Initial data array.
        name (str): Name of the variable.
        grad (numpy.ndarray or cupy.ndarray): Initial gradient array.
        requires_grad (bool): Boolean indicating whether ``grad`` will be set
            in backward calculation.

    Attributes:
        data: Data array of type either :class:`numpy.ndarray` or
            :class:`cupy.ndarray`. If it is None, the variable is left in an
            uninitialized state.
        grad: Gradient array.
        creator: The function who creates this variable. It is ``None`` if the
            variable is not created by any function.

    """  # NOQA

    def __init__(self, data=None, **kwargs):
        argument.check_unexpected_kwargs(
            kwargs, volatile='volatile argument is not supported anymore. '
            'Use chainer.using_config')
        name, grad, requires_grad \
            = argument.parse_kwargs(
                kwargs, ('name', None), ('grad', None),
                ('requires_grad', True))

        if (data is not None and
                not isinstance(data, (numpy.ndarray, cuda.ndarray))):
            msg = '''numpy.ndarray or cuda.ndarray are expected.
Actual: {0}'''.format(type(data))
            raise TypeError(msg)

        # Use a list as a data structure to hold the data array indirectly to
        # abstract its initialized/uninitialized state.
        self._data = [data]
        self._requires_grad = requires_grad
        self._node = VariableNode(self, name, grad)

    def __copy__(self):
        return self._copy_to(Variable())

    def _copy_to(self, target):
        target.__dict__ = copy.copy(self.__dict__)
        target._node = VariableNode(target, self.name)
        return target

    def __reduce__(self):
        return _create_variable, (self.data, self.name, self._node._grad,
                                  self._requires_grad)

    def __repr__(self):
        return variable_repr(self)

    def __str__(self):
        return variable_str(self)

    @property
    def name(self):
        return self._node.name

    @name.setter
    def name(self, n):
        self._node.name = n

    def summary(self):
        if self.name:
            return '<variable %s>' % self.name
        else:
            return '<variable at 0x%x>' % id(self)

    def debug_print(self):
        """Display a summary of the stored data and location of the Variable"""

        msg = """{summary}
- device: {device}
- backend: {background}
- shape: {shape}
- dtype: {dtype}
- statistics: {stats}
- grad: {grad}"""

        stats_msg = 'mean={0:.8f}, std={1:.8f}'

        try:
            device = self.data.device
        except AttributeError:
            device = 'CPU'

        with cuda.get_device_from_array(self.data) as dev:
            xp = numpy if int(dev) == -1 else cuda.cupy

            if self.grad is None:
                grad = None
            elif xp.all(self.grad == 0):
                grad = 0
            else:
                grad = stats_msg.format(float(xp.mean(self.grad)),
                                        float(xp.std(self.grad)))

            stats = stats_msg.format(float(xp.mean(self.data)),
                                     float(xp.std(self.data)))

        return msg.format(summary=self.summary(),
                          grad=grad, shape=self.data.shape,
                          background=type(self.data),
                          dtype=self.data.dtype, device=device,
                          stats=stats)

    def __pos__(self):
        return self

    def __len__(self):
        """Returns the first dimension of the data array.

        Returns:
            int: Number of the first dimension of the data array.

        """
        return len(self.data)

    @property
    def label(self):
        """Short text that represents the variable."""
        return self._node.label

    @property
    def creator(self):
        """:meth:`Function` object that created this variable.

        This property has a setter to which ``None`` can be set. Setting
        ``None`` to this property is equivalent to call :meth:`unchain`;
        it purges the variable from the function that created this variable.

        The setter also accepts the original :meth:`Function` object that
        created this variable. For example, you can once set ``None`` to this
        property and then set the original value again.

        .. note::
           Setting an irrelevant :meth:`Function` object does not emit any
           error immediately, whereas the behavior is undefined. Do not set
           a :meth:`Function` object that did not create this variable object.

        """
        return self._node._creator

    @creator.setter
    def creator(self, func):
        self._node.creator = func

    @property
    def data(self):
        return self._data[0]

    @data.setter
    def data(self, d):
        self._data[0] = d
        self._node._set_data_type(d)

    @property
    def grad(self):
        return self._node._grad

    @grad.setter
    def grad(self, g):
        self._node._set_grad_with_check(g, None, self)

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def rank(self):
        return self._node.rank

    @property
    def node(self):
        return self._node

    @property
    def requires_grad(self):
        """It indicates that ``grad`` will be set in backward calculation."""
        return self._requires_grad

    def to_cpu(self):
        """Copies the data and gradient arrays to CPU."""
        if self.data is None:
            return

        self._data = [cuda.to_cpu(self.data)]
        # ensure that the node tracks the device migration
        node = self._node
        if node._data is not None:
            node.retain_data()
        if node._grad is not None:
            node._grad = cuda.to_cpu(node._grad)

    def to_gpu(self, device=None):
        """Copies the data and gradient arrays to specified GPU.

        Args:
            device: Target device specifier. If omitted, the current device is
                used.

        """
        if self.data is None:
            current = cuda.Device().id
            self._initial_device = current if device is None else device
        else:
            self._data = [cuda.to_gpu(self.data, device)]
            # ensure that the node tracks the device migration
            node = self._node
            if node._data is not None:
                node.retain_data()
            if node._grad is not None:
                node._grad = cuda.to_gpu(node._grad, device)

    def cleargrad(self):
        """Clears the gradient array."""
        self._node._grad = None

    def zerograd(self):
        """Initializes the gradient array by zeros.

        .. deprecated:: v1.15
           Use :meth:`cleargrad` instead.

        """
        warnings.warn(
            'Variable.zerograd is deprecated. Use Variable.cleargrad instead.',
            DeprecationWarning)

        if self.data is None:
            return

        with cuda.get_device_from_array(self.data) as dev:
            node = self._node
            if node._grad is None:
                xp = numpy if int(dev) == -1 else cuda.cupy
                node._grad = xp.zeros_like(self.data)
            else:
                node._grad.fill(0)

    def copydata(self, var):
        """Copies the data array from given source variable.

        This method copies the data array from given variable to this variable.
        The copy is done even if the arrays reside on different devices,
        including across the host and a GPU device. If this variable has an
        uninitialized data array, this method initializes it by the data array
        of the given variable. Similarly, if the given variable has an
        uninitialized data array, this method initializes it by the data array
        of this variable (``self``). If both are uninitialized, this method
        does nothing.

        Args:
            var (Variable): Source variable.

        """
        src = var.data
        dst = self.data
        if src is None:
            if dst is None:
                return
            var.initialize(self.shape)
            src = var.data
        elif dst is None:
            self.initialize(src.shape)
            dst = self.data
        src_xp = cuda.get_array_module(src)
        dst_xp = cuda.get_array_module(dst)
        if dst_xp is src_xp:
            dst_xp.copyto(dst, src)
        elif dst_xp is numpy:
            dst_xp.copyto(dst, src.get())
        else:
            dst.set(src)

    def addgrad(self, var):
        """Accumulates the gradient array from given source variable.

        This method adds the gradient of a given variable to the gradient of
        this variable. The accumulation is even done across the host and
        different devices. If this variable has uninitialized data/grad arrays,
        this method initializes it with the shape of the given varaible and
        then accumulates the gradient.

        Args:
            var (Variable): Source variable.

        """
        src = var._node._grad
        if src is None:
            return

        if self.data is None:
            self.initialize(var.shape)
        dst = self._node._grad

        src_dev = cuda.get_device_from_array(src)
        dst_dev = cuda.get_device_from_array(self.data)

        if src_dev.id == dst_dev.id:
            with dst_dev:
                if dst is None:
                    xp = cuda.get_array_module(src)
                    self._node.grad = xp.copy(src)
                else:
                    dst += src
            return

        if dst_dev.id < 0:
            src_grad = cuda.to_cpu(src)
        else:
            src_grad = cuda.to_gpu(src, device=dst_dev)

        if dst is None:
            self._node.grad = src_grad
        else:
            with dst_dev:
                dst += src_grad

    def set_creator(self, gen_func):
        """Notifies the variable that the given function is its creator.

        Args:
            gen_func (Function): Function object that creates this variable as
                one of its outputs.

        """
        self._node.set_creator(gen_func)

    def backward(self, retain_grad=False):
        """Runs error backpropagation (a.k.a. backprop) from this variable.

        On backprop, :meth:`Function.backward` is called on each
        :class:`Function` object appearing in the backward graph starting from
        this variable. The backward graph is represented by backward references
        from variable nodes to their creators, and from functions to their
        input variable nodes. The backprop stops at all root nodes. Some
        functions set ``None`` as gradients of some inputs, where further
        backprop does not take place at such inputs.

        This method uses :data:`grad` as the initial error array. User can
        manually set a gradient array before calling this method. If
        :data:`data` contains only one element (i.e., it is scalar) and
        :data:`grad` is ``None``, then this method automatically complements
        1.0 as the initial error. This is useful on starting backprop from
        some scalar loss value.

        Args:
            retain_grad (bool): If ``True``, the gradient arrays of all
                intermediate variables are kept. Otherwise, :data:`grad` of the
                intermediate variables are set to ``None`` on appropriate
                timing, which may reduce the maximum memory consumption.

                In most cases of training some models, the purpose of backprop
                is to compute gradients of parameters, not of all variables,
                and therefore it is recommended to set this flag ``False``.

        """
        if self.creator is None:
            return
        initial_device = None
        if cuda.available and isinstance(self.data, cuda.cupy.ndarray):
            try:
                initial_device = cuda.Device()
            except cuda.cupy.cuda.runtime.CUDARuntimeError as e:
                if e.status != 38:  # cudaErrorNoDevice
                    raise

        is_debug = chainer.is_debug()

        cand_funcs = []
        seen_set = set()
        seen_vars = set()
        need_copy = set()

        # Initialize error by 1, if this is a loss variable
        if self.data.size == 1 and self.grad is None:
            with cuda.get_device_from_array(self.data) as device:
                if device is cuda.DummyDevice:
                    self.grad = numpy.ones_like(self.data)
                else:
                    self.grad = cuda.cupy.ones_like(self.data)

        def add_cand(cand):
            if cand not in seen_set:
                # Negate since heapq is min-heap
                heapq.heappush(cand_funcs, (-cand.rank, len(seen_set), cand))
                seen_set.add(cand)

        add_cand(self.creator)

        while cand_funcs:
            _, _, func = heapq.heappop(cand_funcs)
            outputs = [y() for y in func.outputs]  # access via weak ref

            in_data = tuple([x.data for x in func.inputs])
            out_grad = tuple([None if y is None else y.grad for y in outputs])
            hooks = chainer.get_function_hooks()
            if func._n_local_function_hooks != 0:
                hooks = collections.OrderedDict(hooks)
                hooks.update(func.local_function_hooks)

            cuda.get_device_from_array(*(in_data + out_grad)).use()
            for hook in six.itervalues(hooks):
                hook.backward_preprocess(func, in_data, out_grad)
            func.output_data = tuple(
                [None if y is None else y.data for y in outputs])
            gxs = func.backward(in_data, out_grad)
            assert len(gxs) == len(in_data)
            if not getattr(func, '_retain_after_backward', False):
                func.output_data = None
            for hook in six.itervalues(hooks):
                hook.backward_postprocess(func, in_data, out_grad)

            if is_debug:
                for gx in gxs:
                    if gx is None:
                        continue
                    cuda.get_device_from_array(gx).use()
                    if cuda.get_array_module(gx).isnan(gx).any():
                        msg = 'NaN is detected on backward computation'
                        raise RuntimeError(msg)

            if not retain_grad:
                for y in outputs:
                    if y is not None and y is not self.node:
                        y.grad = None
            for x, gx in zip(func.inputs, gxs):
                if gx is None:
                    continue
                if not x.requires_grad:
                    continue

                _check_grad_type(func, x, gx)

                # Accumulate the gradient to x. It is a bit tricky to handle
                # branches and parameter gradient accumulation correctly.
                id_x = id(x)
                if x.creator is None:  # leaf
                    if x._grad is None:
                        x.grad = gx
                        need_copy.add(id_x)
                    else:
                        cuda.get_device_from_array(gx).use()
                        if id_x in need_copy:
                            x.grad = utils.force_array(x._grad + gx)  # copy
                            need_copy.remove(id_x)
                        else:
                            x._grad += gx
                else:  # not a leaf
                    add_cand(x.creator)
                    if id_x not in seen_vars:  # 1st visit
                        x.grad = gx
                        seen_vars.add(id_x)
                        need_copy.add(id_x)
                    else:
                        cuda.get_device_from_array(gx).use()
                        if id_x in need_copy:  # 2nd visit
                            x.grad = utils.force_array(gx + x._grad)  # copied
                            need_copy.remove(id_x)
                        else:  # 3rd or later visit
                            x._grad += gx
            del gxs  # to reduce memory usage
            if initial_device is not None:
                initial_device.use()

    def reshape(self, *shape):
        """Returns a variable of a different shape and the same content.

        .. seealso::
           :func:`chainer.functions.reshape` for full documentation,

        """
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return chainer.functions.reshape(self, shape)

    def transpose(self, *axes):
        """Permute the dimensions of an input variable without copy.

        .. seealso::
           :func:`chainer.functions.transpose` for full documentation.

        """
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1 and (isinstance(axes[0], (tuple, list)) or
                                 axes[0] is None):
            axes = axes[0]
        return chainer.functions.transpose(self, axes)

    def unchain(self):
        """Deletes the reference to the creator of this variable.

        This method deletes the reference to the creator from the corresponding
        variable node. Unlike :meth:`unchain_backward`, it does not backtrack
        the graph.

        This method is equivalent to ``self.creator = None``.

        """
        self.creator = None

    def unchain_backward(self):
        """Deletes references between variable nodes and functions backward.

        After this method completes, intermediate variable nodes and functions
        that are not referenced from anywhere are deallocated by reference
        count GC. Also this variable itself deletes the reference to its
        creator function from the node, i.e. the node becomes root in the
        computation graph. It indicates that backprop after unchaining stops at
        this variable. This behavior is useful to implement truncated BPTT.

        """
        cand_funcs = []
        seen_set = set()

        def add_cand(cand):
            if cand is not None and cand not in seen_set:
                cand_funcs.append(cand)
                seen_set.add(cand)

        add_cand(self.creator)

        while cand_funcs:
            func = cand_funcs.pop()
            for var in func.inputs:
                add_cand(var.creator)
            func.unchain()

    def retain_data(self):
        """Lets the corresponding variable node keep the underlying array."""
        self._node.data = self._data[0]

    def __lt__(self, other):
        raise NotImplementedError()

    def __le__(self, other):
        raise NotImplementedError()

    def __eq__(self, other):
        raise NotImplementedError()

    def __ne__(self, other):
        raise NotImplementedError()

    def __gt__(self, other):
        raise NotImplementedError()

    def __ge__(self, other):
        raise NotImplementedError()

    def __nonzero__(self):
        raise NotImplementedError()

    def __bool__(self):
        raise NotImplementedError()

    def __hash__(self):
        return super(Variable, self).__hash__()

    __array_priority__ = 200


class Parameter(Variable):

    """Parameter variable that can be registered to a link.

    Parameter is a subclass of :class:`Variable`. It almost behaves as same
    as a usual variable except that a parameter can be registered to a
    :class:`~chainer.Link` object just by assigning it to an attribute of
    the link within an :meth:`~chainer.Link.init_scope` context.

    Parameter also supports an initialization by an initializer. It can have
    two initializers: one for the data array, and the other for the gradient
    array. The initializer only specifies the way of filling the elements of
    these arrays, and the shape information is specified at the initialization
    point.

    When a link that the parameter has been registered to is passed to an
    :class:`~chainer.GradientMethod`, an update rule is set to the parameter.
    This update rule specifies how to update the data array of the parameter
    using its gradient array.

    Args:
        initializer (~chainer.Initializer or numpy.ndarray or cupy.ndarray):
            Initializer of the data array. If ``shape`` is given, this
            initializer is immediately used to initialize the data array.
            Otherwise, if it is an array, it is immediately used as the data
            array, and otherwise the data array is left uninitialized and will
            be initialized by this initializer in :meth:`initialize`. It can
            also be a scalar, in which case the data array will be filled by
            this scalar. Note that float32 is used in this case.
        shape (int or tuple of int or None): Shape of the parameter. If it is
            ``None``, the initialization is deferred to the call of
            :meth:`initialize`.
        name (str): Name of the parameter.

    Attributes:
        initializer: Initializer of the data array. It is used for
            initializing the data array of an uninitialized variable.
        update_rule: :class:`~chainer.optimizer.UpdateRule` instance that
            updates this variable as a parameter. This argument is set to
            :attr:`update_rule`.

    """

    initializer = None
    _grad_initializer = None
    _initial_device = -1

    def __init__(self, initializer=None, shape=None, name=None):
        if initializer is None:
            initializer = constant.NaN()
        elif numpy.isscalar(initializer):
            initializer = constant.Constant(initializer)
        if shape is None:
            if isinstance(initializer, (numpy.ndarray, cuda.ndarray)):
                # parameter initialized by the initial array
                super(Parameter, self).__init__(initializer, name=name)
            else:
                # uninitialized parameter
                super(Parameter, self).__init__(name=name)
                self.initializer = initializer
                dtype = getattr(initializer, 'dtype', numpy.float32)
                self._grad_initializer = constant.NaN(dtype)
        else:
            # parameter initialized with a given shape
            if isinstance(initializer, (numpy.ndarray, cuda.ndarray)):
                xp = cuda.get_array_module(initializer)
                initializer = constant.Constant(initializer)
            else:
                xp = numpy
            data = initializers.generate_array(initializer, shape, xp)
            grad = xp.full_like(data, numpy.nan)
            super(Parameter, self).__init__(data, name=name, grad=grad)

        self.update_rule = None

    def __copy__(self):
        return self._copy_to(Parameter())

    def __reduce__(self):
        return _recover_parameter, (self.data, self.name, self.grad,
                                    self.initializer, self.update_rule)

    def to_cpu(self):
        super(Parameter, self).to_cpu()
        if self.data is None:
            self._initial_device = -1

    def to_gpu(self, device=None):
        super(Parameter, self).to_gpu(device)
        if self.data is None:
            if device is None:
                device = cuda.Device().id
            self._initial_device = device

    def cleargrad(self):
        super(Parameter, self).cleargrad()
        if self.data is None:
            self._grad_initializer = None

    def zerograd(self):
        super(Parameter, self).zerograd()
        if self.data is None:
            dtype = getattr(self.initializer, 'dtype', None)
            self._grad_initializer = initializers.Zero(dtype)

    def initialize(self, shape):
        """Initializes the uninitialized variable.

        Uninitialized variable is a variable created with the data array set to
        None. This method creates and initializes the data array. The shape of
        the variable can be left unknown until this method is called.

        Args:
            shape (tuple of int): Shape of the data array.

        """
        data = initializers.generate_array(self.initializer, shape, numpy)

        ginit = self._grad_initializer
        grad = None if ginit is None else initializers.generate_array(
            ginit, shape, numpy)

        if self._initial_device >= 0:
            data = cuda.to_gpu(data, device=self._initial_device)
            if grad is not None:
                grad = cuda.to_gpu(grad, device=self._initial_device)

        self._data[0] = data
        self._node._grad = grad

    def update(self):
        """Updates the data array using the gradient and the update rule.

        This method updates the parameter using the attached update rule.

        """
        if self.update_rule is not None:
            self.update_rule.update(self)


def _recover_parameter(data, name, grad, initializer, update_rule):
    p = Parameter(initializer=initializer, name=name)
    p.data = data
    p.grad = grad
    p.update_rule = update_rule
    return p
