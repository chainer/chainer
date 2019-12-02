from __future__ import absolute_import
import copy
import threading
import traceback
import typing as tp  # NOQA
import warnings
import weakref

import numpy

import chainer
from chainer import _backprop
from chainer import backend
from chainer.backends import _cpu
from chainer.backends import cuda
from chainer.backends import intel64
from chainer import initializers
from chainer.initializers import constant
from chainer import types  # NOQA
import chainer.utils._collections
from chainer.utils import argument
import chainerx


_thread_local = threading.local()


def _raise_grad_error(exc_type, func, msg):
    detail = ''
    if func:
        detail = 'Function `{0}` ({1}) has a bug.\n'.format(
            type(func)._impl_name, func.label)
        stack = func.stack
        if stack:
            detail += 'Stacktrace of the function is below:\n'
            for line in traceback.format_list(func.stack):
                detail += line
        detail += '''
Please report this error to the issue tracker with the stack trace,
the information of your environment, and your script:
https://github.com/chainer/chainer/issues/new.
'''

    raise exc_type(detail + msg)


def _check_grad_type(func, x, is_node_x, gx):
    # is_node_x: equivalent to isinstance(x, VariableNode)

    assert gx is not None

    # x_shape is the raw shape

    # TODO(kataoka): avoid `isinstance`
    if isinstance(x, _ChainerxVariableNodeProps):
        x_data = None
        x_layout = None
        x_shape = x.shape
    elif is_node_x:
        x_data = x._data
        x_layout = x._layout
        x_shape = x.shape
        if x_layout is not None:
            # to raw shape
            x_shape = chainer.memory_layouts._transpose_shape(
                x_shape, None, x_layout)
    else:
        # assert isinstance(x, Variable)
        x_data = x._data[0]
        x_layout = x._layout
        x_shape = None if x_data is None else x_data.shape

    # TODO(kataoka): Make _update_data_info store the array module.
    # ``is_node_x and x_data is None`` implies that the data array is not
    # retained.
    # ``not is_node_x and x_data is None`` implies that grad of uninitialized
    # variable is checked here.

    if x_data is None and not is_node_x:
        # TODO(kataoka): This should be an error.
        return
    if x_layout is None:
        if x.dtype is None or x.shape is None:
            # unretained Variable(None)
            # TODO(kataoka): This should be an error.
            return

    if not isinstance(gx, chainer.get_array_types()):
        _raise_grad_error(
            TypeError,
            func,
            ('Type of grad is invalid:\n'
             + 'Expected: Any of {}\n'.format(chainer.get_array_types())
             + 'Actual: {}'.format(type(gx))))
    elif x_data is not None and not chainer.is_arrays_compatible((gx, x_data)):
        _raise_grad_error(
            TypeError,
            func,
            ('Type of data and grad mismatch\ngrad: %s != data: %s' %
             (type(gx), type(x_data))))
    elif gx.dtype != x.dtype:
        _raise_grad_error(
            TypeError,
            func,
            ('Dtype of data and grad mismatch\ngrad: %s != data: %s' %
             (gx.dtype, x.dtype)))
    elif gx.shape != x_shape:  # comparing semantic shapes (not semantic)
        _raise_grad_error(
            ValueError,
            func,
            ('Shape of data and grad mismatch\ngrad: %s != data: %s' %
             (gx.shape, x_shape)))


def variable_repr(var):
    """Return the string representation of a variable.

    Args:
        var (~chainer.Variable): Input Variable.
    .. seealso:: numpy.array_repr
    """
    arr = _cpu._to_cpu(var.array)

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
    arr = _cpu._to_cpu(var.array)

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

    A variable node is held by the corresponding :class:`~chainer.Variable`
    object, which is managed by users. :class:`~chainer.FunctionNode` objects
    that take the variable as an input also hold references to the variable
    node.

    Note that the node does not hold a reference to the corresponding data
    array in general. The data array is actually accessible by the node in the
    following cases.

    1. If there exists a :class:`~chainer.Variable` object that holds a
       reference to the variable node, the variable node holds a weak reference
       to the variable object, and thus the data array is accessible via the
       weak reference.
    2. If :meth:`retain_data` is called, the node holds a reference to the data
       array. It is mainly called by a function that needs the input or output
       data array in its backprop procedure.
       See :meth:`FunctionNode.retain_inputs()
       <chainer.FunctionNode.retain_inputs>`
       and :meth:`FunctionNode.retain_outputs()
       <chainer.FunctionNode.retain_outputs>` for more details.

    Users usually do not need to touch this variable node object. The
    computational graph is automatically managed by Chainer, and any interface
    that is beneficial for users is also provided by
    :class:`~chainer.Variable`.

    Args:
        variable (~chainer.Variable): The corresponding variable object.
        name (str): Name of the variable node.

    Attributes:
        dtype: Data type of the data array.
        shape: Shape of the data array.
        name (str): Name of the variable node.

    """

    dtype = None
    shape = None  # semantic shape

    _creator_node = None
    _data = None  # type: types.NdArray
    _rank = 0  # type: int
    # Name of the Function is assigned if this variable is a gradient generated
    # by an old-style Function
    _old_style_grad_generator = None  # type: str
    _layout = None

    def __init__(
            self,
            variable: 'Variable',
            name: tp.Optional[str],
            **kwargs: tp.Any
    ) -> None:
        if kwargs:
            argument.check_unexpected_kwargs(
                kwargs,
                grad='unexpected keyword argument "grad": '
                     'pass the gradient to Variable instead'
            )
        self._variable = weakref.ref(variable)
        self.name = name
        self._requires_grad = variable.requires_grad
        self._layout = variable.layout

        vdata = variable.raw_array
        self._update_data_info(vdata)

    @property
    def creator(self):
        """Function object that created this variable node.

        When the function is implemented with the old-style API (i.e., it uses
        :class:`~chainer.Function` class),
        this property returns the :class:`~chainer.Function` object.
        The object is extracted from the :class:`~chainer.FunctionAdapter`
        object, so the returned object is not the function node, but instead
        the actual implementation of forward and backward procedures.

        When the function is implemented with the new-style API (i.e., it uses
        :class:`~chainer.FunctionNode` class),
        this property returns the function node
        object. In this case, the returned object is same as
        :attr:`creator_node`.

        .. warning::

           As of v3.0.0, when the creator is an old-style function, the
           following code is invalid:

           .. code-block:: python

              creator = v.creator
              v.creator = None
              ...
              v.creator = creator

           The point is that :class:`~chainer.FunctionNode` objects are used
           as nodes in the computational graph instead of
           :class:`~chainer.Function`, and each :class:`~chainer.Function`
           object only holds a *weak reference* to the corresponding
           :class:`~chainer.FunctionNode`.
           Since ``creator`` returns the :class:`~chainer.Function` object,
           the :class:`~chainer.FunctionNode` object is not kept by preserving
           ``creator``.

           The above code should be fixed as follows.

           .. code-block:: python

              creator_node = v.creator_node
              v.creator_node = None
              ...
              v.creator_node = creator_node

        """
        node = self._creator_node
        if node is None:
            return None

        if isinstance(node, chainer.function.FunctionAdapter):
            return node.function
        return node

    @creator.setter
    def creator(self, func):
        self.creator_node = func

    @property
    def creator_node(self):
        """Function node that has this variable as an output.

        See :class:`~chainer.FunctionNode` for the definition of a function
        node.

        """
        return self._creator_node

    @creator_node.setter
    def creator_node(self, func):
        if isinstance(func, chainer.Function):
            func = func.node
        self._creator_node = func
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
        self._update_data_info(d)

    @property
    def grad(self):
        """Gradient array of the corresponding variable.

        If the variable is not available, it returns ``None``.

        """
        var = self._variable()
        return None if var is None else var.grad

    @property
    def grad_var(self):
        """Gradient variable of the corresponding variable.

        If the corresponding variable is not available, it return ``None``.

        """
        var = self._variable()
        return None if var is None else var.grad_var

    def _set_grad_var_if_available(self, g):
        var = self._variable()
        if var is not None:
            var._set_grad_var_without_check(g)

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

    def get_variable(self):
        """Returns the corresponding :class:`~chainer.Variable` object.

        VariableNode object holds a weak reference of the variable object. If
        the reference is alive, it is returned by this property. Otherwise,
        this property creates a new :class:`~chainer.Variable` object from
        this node object and returns it.

        Returns:
            ~chainer.Variable: The variable object that refers this node.

        """
        var = self._variable()
        if var is not None:
            return var
        var = Variable._init_unchecked(
            self.data,
            name=self.name,
            requires_grad=self.requires_grad,
            node=self,
            layout=self._layout)
        return var

    def get_variable_or_none(self):
        """Returns the holding :class:`~chainer.Variable` object or ``None``.

        VariableNode object holds a weak reference of the variable object.If
        the reference is alive, it is returned by this property. Otherwise,
        returns ``None``.

        Returns:
            ~chainer.Variable: The variable object that refers this node.

        """
        return self._variable()

    def set_creator(self, creator):
        """Sets a :class:`~chainer.Function` object that created this node.

        This method is equivalent to ``self.creator = creator``. A
        :class:`~chainer.FunctionNode` object can also be passed.

        Args:
            creator (Function or FunctionNode): Function that has created this
                variable.

        """
        self.creator = creator

    def set_creator_node(self, creator_node):
        """Sets a :class:`~chainer.FunctionNode` object that created this node.

        This method is equivalent to ``self.creator_node = creator_node``. A
        :class:`~chainer.Function` object can also be passed, in which case the
        :attr:`Function.node <chainer.Function.node>` attribute is used.

        Args:
            creator_node (FunctionNode or Function): Function node that has
                this variable as an output.

        """
        self.creator_node = creator_node

    def unchain(self):
        """Deletes the reference to the creator of this variable node.

        This method is equivalent to ``self.creator_node = None``.

        """
        self.creator_node = None

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

    def _update_data_info(self, d):
        # d is a raw array (with raw shape)
        if d is None:
            self.dtype = None
            self.shape = None
        else:
            self.dtype = d.dtype

            if self._layout is None:
                self.shape = d.shape
            else:
                self.shape = chainer.memory_layouts._transpose_shape(
                    d.shape, self._layout, None)

        # If the node has a reference to data, update it as well.
        if self._data is not None:
            self._data = d

    def _check_old_style_gradient(self):
        if self._old_style_grad_generator is not None:
            raise RuntimeError(
                'cannot twice-differentiate an old style Function "%s"' %
                self._old_style_grad_generator)


def _create_variable(data, name, grad, requires_grad, device):
    var = Variable(
        data, name=name, grad=grad, requires_grad=requires_grad)
    var.to_device(device)
    return var


class Variable(object):

    """__init__(data=None, *, name=None, grad=None, requires_grad=True)

    Array with a structure to keep track of computation.

    Every variable holds a data array of type either :class:`numpy.ndarray` or
    :class:`cupy.ndarray`.

    A variable object holds a data array and a
    :class:`~chainer.variable.VariableNode` object of
    a computational graph. If the variable is constructed by the user, the node
    is *root* and does not hold any parent. If the variable is constructed by a
    :class:`~chainer.FunctionNode` object (i.e., by calling functions under
    ``chainer.functions`` or user-defined functions), or by using operators
    (see the list below), the node holds a reference to its parent called
    :attr:`creator_node`.
    This reference is used in backpropagation to backtrack the graph.

    Users can disable (resp. enable) this chaining behavior by calling
    :func:`~chainer.no_backprop_mode` (resp.
    :func:`~chainer.force_backprop_mode`).
    In the former context, a variable never creates a computational graph,
    whereas in the latter context, it is forced to create.

    .. note::

        The following operators are defined for variable(s).

        * Indexing: ``a[slices]`` (:meth:`__getitem__`)
        * Addition: ``a + b`` (:meth:`__add__`, :meth:`__radd__`)
        * Subtraction: ``a - b`` (:meth:`__sub__`, :meth:`__rsub__`)
        * Multiplication: ``a * b`` (:meth:`__mul__`, :meth:`__rmul__`)
        * Division: ``a / b`` (:meth:`__div__`, :meth:`__rdiv__`, \
                               :meth:`__truediv__`, :meth:`__rtruediv__`)
        * Floor Division: ``a // b`` (:meth:`__floordiv__`, \
                                      :meth:`__rfloordiv__`)
        * Exponentiation: ``a ** b`` (:meth:`__pow__`, :meth:`__rpow__`)
        * Matrix Multiplication: ``a @ b`` (:meth:`__matmul__`, \
                                            :meth:`__rmatmul__`)
        * Negation (Arithmetic): ``- a`` (:meth:`__neg__`)
        * Absolute value: ``abs(a)`` (:meth:`__abs__`)

    Args:
        data (:ref:`ndarray`): Initial data array.
        name (str): Name of the variable.
        grad (:ref:`ndarray`): Initial gradient array.
        requires_grad (bool): Boolean indicating whether ``grad`` will be set
            in backward calculation.

    """

    # Cached value of `self.xp is chainerx`. It prevents from initializing
    # self._device as much as possible because it is really costly.
    _has_chainerx_array = False

    # Cached grad-stopped view of chainerx array. This is the return value
    # of `array` and `data` properties.
    _chainerx_nobp_array_cache = None

    # Cached grad-stopped view of the array returned by `grad` property.
    # It's a 2-element tuple, where the first is the original grad array and
    # the second is a grad-stopped view of the first. `grad` property returns
    # the second element.
    _chainerx_grad_cache = None

    _chainerx_name = None  # type: tp.Optional[str]

    # A NumPy, CuPy array cache to avoid redundant conversions between
    # NumPy/CuPy and ChainerX.
    # TODO(hvy): Avoid modifying this variable from outside this class.
    _chainerx_fallback_array = None

    # Used in non-ChainerX variables. The gradient array is stored in
    # this attribute on Variable.grad setter to delay creation of grad_var
    # instance.
    _grad = None

    _layout = None

    def as_layout(self, layout):
        src_layout = self._layout
        if src_layout == layout:
            return self

        y, = chainer.memory_layouts.AsLayout(layout).apply((self,))
        return y

    def __init__(
            self,
            data: tp.Optional[types.NdArray] = None,
            **kwargs: tp.Any
    ) -> None:
        name, grad, requires_grad, grad_valid, layout = argument.parse_kwargs(
            kwargs, ('name', None), ('grad', None), ('requires_grad', True),
            ('_grad_valid', True), ('layout', None),
            volatile='volatile argument is not supported anymore. '
                     'Use chainer.using_config')
        # _grad_valid is for internal use, hence the prefix _.

        assert isinstance(requires_grad, bool)
        if data is not None:
            array_types = chainer.get_array_types()
            if not isinstance(data, array_types):
                msg = '{} or {} are expected. Actual: {}'.format(
                    ', '.join([str(at) for at in array_types[:-1]]),
                    array_types[-1], type(data))
                raise TypeError(msg)

        self._init_impl(
            data, None, name, grad, grad_valid, requires_grad, None, None,
            layout)

    @staticmethod
    def _init_unchecked(
            data=None, device=None, name=None, grad=None, grad_valid=True,
            requires_grad=True, is_chainerx_array=None, node=None,
            layout=None):
        """Creates a new :class:`Variable` without the validations for
        optimizing performance.
        """

        # Create a Variable without invoking __init__
        var = Variable.__new__(Variable)
        var._init_impl(
            data, device, name, grad, grad_valid, requires_grad,
            is_chainerx_array, node, layout)
        return var

    def _init_impl(self, data, device, name, grad, grad_valid, requires_grad,
                   is_chainerx_array, node, layout):
        # `device` must be of type chainer.backend.Device.
        # Check is skipped for performance.

        self._requires_grad = requires_grad  # type: bool
        self._loss_scale = None
        self._grad_var = None
        self._device = device
        # A flag to prevent grad from being used before calling cleargrad().
        # It becomes True when either
        # - cleargrad() is called, or
        # - zerograd() is called, or
        # - grad is set.
        # Note that it won't be True by merely initializing an uninitialized
        # Parameter.
        self._grad_valid = grad_valid
        self._layout = layout

        if is_chainerx_array is None:
            is_chainerx_array = isinstance(data, chainerx.ndarray)

        if is_chainerx_array:
            if not requires_grad and grad is not None:
                raise ValueError(
                    'Cannot initialize a variable with gradients if the '
                    'require_grad argument is False.')
            self._set_chainerx_array(data, grad)  # type: ignore

            # ChainerX itself has own node objects, but not exposed to python.
            self._node = None  # type: tp.Optional[VariableNode]
            self._chainerx_name = name
        else:
            # Use a list as a data structure to hold the data array indirectly
            # to abstract its initialized/uninitialized state.
            self._data = [data]  # type: tp.List[tp.Optional[types.NdArray]]
            if node is None:
                self._node = VariableNode(self, name)
            else:
                self._node = node
            self._grad = grad

    def __copy__(self):
        return self._copy_to(Variable())

    def _copy_to(self, target):
        target.__dict__ = copy.copy(self.__dict__)
        target._node = VariableNode(target, self.name)
        return target

    def __reduce__(self):
        args = (
            self.array, self.name, self.grad, self._requires_grad, self.device)
        return _create_variable, args

    def __repr__(self):
        return variable_repr(self)

    def __str__(self):
        return variable_str(self)

    def _clear_chainerx(self):
        self._chainerx_nobp_array_cache = None
        self._chainerx_grad_cache = None
        self._chainerx_fallback_array = None

    def _ensure_grad_var_up_to_date(self):
        # For non-ChainerX, this method creates _grad_var if it's not yet
        # created and _grad is set.
        # For ChainerX, this method checks consistency between
        # _grad_var._data[0] and self._data[0].grad and recreates _grad_var
        # as necessary. (chainerx.ndarray.grad can be altered independently
        # from chainer)
        if self._has_chainerx_array:
            self._grad = None
            # Update gradient variable if it has not yet been initialized or
            # it happens to be dirty w.r.t. the actual gradient of the
            # underlying chainerx.ndarray.
            arr = self._data[0]
            actual_grad = (
                arr.grad
                if arr is not None and arr.is_grad_required()
                else None)
            if actual_grad is None:
                self._grad_var = None
            else:
                grad_var = self._grad_var
                old_grad = None if grad_var is None else grad_var._data[0]
                if actual_grad is not old_grad:
                    self._grad_var = Variable(
                        actual_grad,
                        requires_grad=actual_grad.is_backprop_required(),
                        layout=self._layout)
            return

        if self._grad_var is None:
            if self._grad is not None:
                self._grad_var = Variable(self._grad, layout=self._layout)

    def _set_chainerx_array(
            self,
            array: tp.Optional['chainerx.ndarray'],
            grad: tp.Optional['chainerx.ndarray']
    ) -> None:

        # Sets chainerx array and grad.
        assert array is None or isinstance(array, chainerx.ndarray)
        requires_grad = self._requires_grad

        self._grad = None

        if (not requires_grad
                and array is not None
                and array.is_backprop_required()):
            raise ValueError(
                'Cannot initialize a variable to not require '
                'gradients if the ChainerX array already requires '
                'backprop.')

        # Create a view of the given data to hold internally and modify.
        if array is None:
            self._data = [None]
        else:
            # If the array `array` is not connected to a graph, a view of it is
            # created and kept, in order not to change the no-graph status of
            # it. If the array is connected, the graph status is kept track of.
            if not array.is_backprop_required():
                array = array.view()
            if requires_grad:
                array.require_grad()
                if grad is not None:
                    array.set_grad(grad)
            self._data = [array]

        self._has_chainerx_array = True  # even if data is None
        self._chainerx_nobp_array_cache = None
        self._chainerx_grad_cache = None
        self._chainerx_fallback_array = None

    @property
    def device(self):
        """Device on which the data array of this variable reside."""
        # lazy initialization for performance
        if self._device is None:
            if self._data[0] is None:
                self._device = backend.CpuDevice()
            else:
                self._device = backend.get_device_from_array(self._data[0])
        return self._device

    @property
    def xp(self) -> tp.Optional[types.Xp]:
        """Array module for the data array of this variable."""
        if self._has_chainerx_array:
            return chainerx
        else:
            device = self.device
            return None if device is None else device.xp

    @property
    def name(self):
        if self._has_chainerx_array:
            return self._chainerx_name
        return self._node.name

    @name.setter
    def name(self, n):
        if self._has_chainerx_array:
            self._chainerx_name = n
            return
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
- backend: {backend}
- shape: {shape}
- dtype: {dtype}
- statistics: {stats}
- grad: {grad}"""

        stats_msg = 'mean={0:.8f}, std={1:.8f}'

        array = self.array
        device = self.device
        with chainer.using_device(device):
            xp = device.xp

            if array is None:
                # `array` can be `None` if constructed without any arguments
                device = None
                backend = None
                stats = None
            else:
                device = getattr(array, 'device', 'CPU')
                backend = type(array)
                stats = stats_msg.format(float(xp.mean(array)),
                                         float(xp.std(array)))
            shape = getattr(array, 'shape', None)
            dtype = getattr(array, 'dtype', None)

            if self.grad is None:
                grad = None
            elif xp.all(self.grad == 0):
                grad = 0
            else:
                grad = stats_msg.format(float(xp.mean(self.grad)),
                                        float(xp.std(self.grad)))

        return msg.format(summary=self.summary(), device=device,
                          backend=backend, shape=shape, dtype=dtype,
                          stats=stats, grad=grad)

    def __pos__(self):
        return self

    def __len__(self):
        """Returns the first dimension of the data array.

        Returns:
            int: Number of the first dimension of the data array.

        """
        return len(self.array)

    @property
    def label(self):
        """Short text that represents the variable."""
        if self._has_chainerx_array:
            raise RuntimeError(
                'A variable of ChainerX does not provide a node label.')
        return self._node.label

    @property
    def creator(self):
        """Function implementation that created this variable.

        When this variable has been created by an old-style function (i.e., it
        is implemented as a subclass of :class:`Function`), this property
        returns that :class:`Function` object.

        When this variable has been created by a new-style function (i.e., it
        is implemented as a subclass of :class:`FunctionNode` class), this
        property returns that node object.

        """
        if self._has_chainerx_array:
            raise RuntimeError(
                'A variable of ChainerX does not provide a creator.')
        return self._node.creator

    @creator.setter
    def creator(self, func):
        if self._has_chainerx_array:
            raise RuntimeError(
                'A variable of ChainerX does not provide a creator.')
        self._node.creator = func

    @property
    def creator_node(self):
        """:class:`FunctionNode` object that created this variable.

        This property has a setter to which ``None`` can be set. Setting
        ``None`` to this property is equivalent to call :meth:`unchain`;
        it purges the variable from the function that created this variable.

        The setter also accepts the original :class:`FunctionNode` object that
        created this variable. For example, you can once set ``None`` to this
        property and then set the original value again.

        .. note::
           Setting an irrelevant :meth:`FunctionNode` object does not emit any
           error immediately, whereas the behavior is undefined. Do not set
           a :meth:`FunctionNode` object that did not create this variable
           object.

        """
        if self._has_chainerx_array:
            raise RuntimeError(
                'A variable of ChainerX does not provide a creator_node.')
        return self._node._creator_node

    @creator_node.setter
    def creator_node(self, func):
        if self._has_chainerx_array:
            raise RuntimeError(
                'A variable of ChainerX does not provide a creator_node.')
        self._node.creator_node = func

    @property
    def array(self) -> tp.Optional[types.NdArray]:
        """The underlying data array.

        It is either :class:`numpy.ndarray` or :class:`cupy.ndarray` object,
        or ``None`` if the variable in in an uninitialized state.

        """
        return self._get_array()

    def _get_array(self):
        if (self._layout is not None
                and not (
                    _allow_array_access_with_nonstandard_layout())):
            raise RuntimeError(
                'Cannot directly retrieve the underlying array from a '
                'variable with non-standard layout.')
        return self.raw_array

    @property
    def raw_array(self):
        """The underlying raw data array.

        Its shape does not have to be the semantic shape, if the memory layout
        is non-standard.
        """
        # For ChainerX, this property always returns a grad-stopped view.
        # The view is cached to reduce potential overhead.
        if self._has_chainerx_array:
            if (self._chainerx_nobp_array_cache is None
                    and self._data[0] is not None):
                self._chainerx_nobp_array_cache = (
                    self._data[0].as_grad_stopped())  # type: ignore
            return self._chainerx_nobp_array_cache

        return self._data[0]

    @array.setter
    def array(self, d: tp.Optional[types.NdArray]) -> None:
        self._set_array(d)

    def _set_array(self, d, *, layout_check=True):
        if (layout_check
                and self._layout is not None
                and not (
                    _allow_array_access_with_nonstandard_layout())):
            raise RuntimeError(
                'Cannot directly set the underlying array of a variable with '
                'non-standard layout.')
        if self._has_chainerx_array:
            d_old = self._data[0]
            if (d_old is not None
                    and (d_old.is_backprop_required()  # type: ignore
                         or d.is_backprop_required())):  # type: ignore
                raise ValueError(
                    'Cannot update the array of a Variable if either the '
                    'existing or the new array requires backprop.')

            self._set_chainerx_array(d, None)  # type: ignore
        else:
            self._node._update_data_info(d)  # type: ignore # _node doesn't have value when xp is chainerx # NOQA
            self._data[0] = d
            self._has_chainerx_array = False

    @property
    def chx_array(self):
        """A view of the raw ChainerX array.

        In contrary to :data:`Variable.array` which is always disconnected,
        the array represented by this attribute may be connected to the
        computational graph.

        It is a view, so it has a distinct gradient from the original array.

        If this attribute is queried on a :class:`Variable` with a non-ChainerX
        array, :class:`ValueError` will be raised.
        """
        if not self._has_chainerx_array:
            raise ValueError(
                'chx_array is not available for Variable with '
                'non-ChainerX array.')
        return self._data[0].view()

    @property
    def data(self) -> tp.Optional[types.NdArray]:
        """The underlying data array (equivalent to :attr:`array`).

        Note that using this attribute directly is discouraged; use
        :attr:`array` instead. Using :attr:`array`, you can find an error
        earlier when your code mixes up Variable and ndarray because
        ndarray does not have an attribute ``.array`` while it has
        ``.data``.

        """
        return self.array

    @data.setter
    def data(self, d: types.NdArray) -> None:
        self.array = d

    @property
    def layout(self):
        return self._layout

    def _set_chainerx_grad(self, g, from_grad_var):
        # Assigns chainerx.ndarray.grad.
        #
        # If the main array is connected to the graph, in order to enable
        # double-backprop, the grad will also be backprop-required
        # (a view is created not to affect the given grad).
        # If the given grad is from a grad_var, this operation is skipped,
        # as the status of the given grad reflects the necessity of
        # double-backprop.
        assert self.xp is chainerx
        if not self._requires_grad and g is not None:
            raise RuntimeError(
                'Cannot set the gradient of a variable that is flagged to not '
                'require one.')
        arr = self._data[0]
        if arr is None:
            if g is not None:
                raise RuntimeError(
                    'Cannot set a gradient to an empty variable')
        elif arr.is_backprop_required():
            # If g is grad-stopped, require grad on it.
            # Make a view in order not to affect the input.
            if (g is not None
                    and not from_grad_var
                    and not g.is_backprop_required()):
                g = g.view().require_grad()
            arr.set_grad(g)

    def _set_grad_without_check(self, g):
        if self._has_chainerx_array:
            self._set_chainerx_grad(g, False)
            self._grad_var = None
            self._grad_valid = True
            return

        self._grad = g
        self._grad_var = None
        self._grad_valid = True

    @property
    def grad(self) -> tp.Optional[types.NdArray]:
        """Gradient array of this variable.

        Note that this property returns the underlying array of the gradient
        variable instead of the gradient variable itself; to get/set
        gradient variable, use :attr:`grad_var` instead.

        If the underlying array is a :class:`chainerx.ndarray` and
        requires_grad is false, trying to access the gradient will results in
        and error.

        """
        return self._get_grad()

    def _get_grad(self):
        if (self._layout is not None
                and not (
                    _thread_local.allow_array_access_with_nonstandard_layout)):
            raise RuntimeError(
                'Cannot directly retrieve the gradient array of a '
                'variable with non-standard layout.')
        if not self._grad_valid:
            raise RuntimeError(
                'Cannot retrieve Variable.grad. '
                'Either it must be set manually or Variable.cleargrad() '
                'must be called beforehand.')

        if self._has_chainerx_array:
            arr = self._data[0]
            if arr is None or not arr.is_backprop_required():
                self._chainerx_grad_cache = None
                return None

            actual_grad = arr.grad

            if actual_grad is None:
                self._chainerx_grad_cache = None
                return None

            # If grad is cached and the actual grad has not changed, return
            # the cache.
            if self._chainerx_grad_cache is not None:
                orig_grad, grad_stopped_grad = self._chainerx_grad_cache
                if orig_grad is actual_grad:
                    return grad_stopped_grad

            # Update the cache
            grad_stopped_grad = actual_grad.as_grad_stopped()
            self._chainerx_grad_cache = (actual_grad, grad_stopped_grad)

            return grad_stopped_grad

        if self._grad_var is not None:
            return self._grad_var.array
        return self._grad

    @grad.setter
    def grad(self, g: tp.Optional[types.NdArray]) -> None:
        self._set_grad(g)

    def _set_grad(self, g, *, layout_check=True):
        if (layout_check
                and self._layout is not None
                and not (
                    _allow_array_access_with_nonstandard_layout())):
            raise RuntimeError(
                'Cannot directly set the gradient array of a '
                'variable with non-standard layout.')
        if g is not None:
            _check_grad_type(None, self, False, g)
        self._set_grad_without_check(g)

    def _set_grad_var_without_check(self, gv):
        if self._has_chainerx_array:
            self._set_chainerx_grad(
                None if gv is None else gv._data[0],
                True)
            self._grad_var = gv
            return

        self._grad_var = gv
        self._grad = None if gv is None else gv.array

    @property
    def grad_var(self) -> tp.Optional['Variable']:
        """Gradient variable."""
        self._ensure_grad_var_up_to_date()
        return self._grad_var

    @grad_var.setter
    def grad_var(self, g: tp.Optional['Variable']) -> None:
        if g is not None:
            _check_grad_type(None, self, False, g.array)
        self._set_grad_var_without_check(g)

    @property
    def shape(self):
        raw_shape = self._data[0].shape
        if self._layout is not None:
            # Convert to semantic shape
            return chainer.memory_layouts._transpose_shape(
                raw_shape, self._layout, None)
        return raw_shape

    @property
    def ndim(self):
        return self._data[0].ndim

    @property
    def size(self):
        return self._data[0].size

    @property
    def dtype(self):
        return self._data[0].dtype

    @property
    def rank(self):
        if self._has_chainerx_array:
            raise RuntimeError(
                'A variable of ChainerX does not provide a node rank.')
        return self._node.rank

    @property
    def node(self):
        if self._has_chainerx_array:
            raise RuntimeError(
                'A variable of ChainerX does not provide a node.')
        return self._node

    @property
    def requires_grad(self):
        """It indicates that ``grad`` will be set in backward calculation."""
        return self._requires_grad

    @property
    def T(self):
        """Transposition of this variable."""
        return chainer.functions.transpose(self)

    def to_cpu(self):
        """Copies the data and gradient arrays to CPU."""
        self.to_device(backend.CpuDevice())

    def to_gpu(self, device=None):
        """Copies the data and gradient arrays to specified GPU.

        Args:
            device: Target device specifier. If omitted, the current device is
                used.

        """
        cuda.check_cuda_available()
        self.to_device(cuda._get_device_or_current(device))

    def to_intel64(self):
        """Copies the data and gradient arrays to intel64 specific mdarray.

        If the array is not suited for intel64, it will be converted to
        :class:`numpy.ndarray`.
        """
        intel64.check_ideep_available()
        self.to_device(intel64.Intel64Device())

    def to_chx(self):
        """Converts the array and gradient to ChainerX arrays without copy.

        This method converts the underlying array and gradient to
        :class:`chainerx.ndarray` on the same physical device. It does nothing
        if the array held by the Variable object is already a ChainerX array.
        The new array is a view of the original one.

        """
        self._to_chx(allow_unchaining=False)

    def _to_chx(self, allow_unchaining):
        if not chainerx.is_available():
            raise RuntimeError('ChainerX is not available.')

        if self._has_chainerx_array:
            return

        if not allow_unchaining and self.creator is not None:
            raise RuntimeError(
                'A variable with a creator cannot be converted into ChainerX '
                'array')

        self._to_device(
            backend.ChainerxDevice.from_fallback_device(self.device),
            allow_unchaining)

    def from_chx(self):
        """Converts the array and gradient to non-ChainerX arrays without copy.

        This method converts the underlying ChainerX array and gradient
        residing in either a ``native`` or ``cuda`` device to NumPy or CuPy
        arrays respectively, on their same physical device. It does nothing
        if the array held by the Variable object is not a ChainerX array. The
        new array is a view of the original one.

        Raises an error if such a conversion is not supported for the device.

        """
        self._from_chx(allow_unchaining=False)

    def _from_chx(self, allow_unchaining):
        if not self._has_chainerx_array:
            return

        if not allow_unchaining and self._data[0].is_backprop_required():
            raise RuntimeError(
                'Cannot convert from a Variable with a ChainerX array that is '
                'connected to a graph.')

        self.to_device(self.device.fallback_device)

    def to_device(self, device):
        """Copies the data and gradient arrays to specified device.

        Args:
            device: Target device specifier. See
                :func:`~chainer.get_device` for available values.

        """
        self._to_device(device, allow_unchaining=False)

    def _to_device(self, device, allow_unchaining):
        device = chainer.get_device(device)

        was_chainerx = self._has_chainerx_array
        is_chainerx = device.xp is chainerx

        if not allow_unchaining:
            if was_chainerx and not is_chainerx:
                chx_arr = self._data[0]
                if chx_arr is not None and chx_arr.is_backprop_required():
                    raise RuntimeError(
                        'A variable of a ChainerX array which requires '
                        'gradients cannot be copied into non-chainerx device '
                        '({}).'.format(device))
            elif not was_chainerx and is_chainerx:
                arr = self._data[0]
                if arr is not None and self.creator is not None:
                    raise RuntimeError(
                        'A variable of a non-ChainerX array which is '
                        'connected to a graph cannot be copied to a ChainerX '
                        'device ({}).'.format(device))

        arr = self._data[0]
        grad_var = self.grad_var

        if was_chainerx and not is_chainerx:
            self._clear_chainerx()
            self._node = VariableNode(self, self._chainerx_name)
        elif not was_chainerx and is_chainerx:
            self._chainerx_name = self._node.name

        self._device = device
        self._has_chainerx_array = is_chainerx

        if arr is None:
            return

        if backend.get_device_from_array(arr) == device:
            return

        new_arr = device.send(arr)
        if is_chainerx:
            if grad_var is None:
                new_grad = None
            else:
                new_grad = device.send(grad_var._data[0])
            self._set_chainerx_array(new_arr, new_grad)
        else:
            self._data = [new_arr]
            if grad_var is not None:
                grad_var._to_device(device, allow_unchaining=allow_unchaining)
                # _grad has been invalidated by the line above.
                self._grad = grad_var.raw_array

        # ensure that the node tracks the device migration
        node = self._node
        if is_chainerx:
            # ChainerX itself has own node objects,
            # ensure that the node is disconnected with this variable.
            if node is not None:
                # Disconnect by replacing with an alternative of dead weakref
                node._variable = lambda: None
                self._node = None
        else:
            if node._data is not None:
                node.retain_data()

    def cleargrad(self):
        """Clears the gradient array."""
        self.grad_var = None
        self._grad_valid = True

    def zerograd(self):
        """Initializes the gradient array by zeros.


        Note that the gradient variable is unchained from the computational
        graph by this method, because this operation breaks the backprop
        validity.

        .. deprecated:: v1.15
           Use more efficient  :meth:`cleargrads` instead.

        """
        warnings.warn(
            'Variable.zerograd is deprecated. Use Variable.cleargrad instead.',
            DeprecationWarning)

        arr = self.array
        if arr is None:
            self._grad_valid = True
            return

        if self._has_chainerx_array:
            gv = self.grad_var
            if gv is None:
                self.grad = chainerx.zeros_like(
                    arr, device=self.device.device)
            else:
                gv._data[0].fill(0)
        else:
            with chainer.using_device(self.device):
                xp = self.device.xp
                if self._grad is None:
                    self._grad = xp.zeros_like(arr)
                    self._grad_var = None
                else:
                    gv = self._grad_var
                    if gv is not None:
                        gv.unchain()
                    self._grad.fill(0)
        self._grad_valid = True

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
            var (~chainer.Variable): Source variable.

        """
        src = var.array
        dst = self.array
        if src is None:
            if dst is None:
                return
            var.initialize(self.shape)
            src = var.array
        elif dst is None:
            self.initialize(src.shape)
            dst = self.array
        backend.copyto(dst, src)

    def addgrad(self, var):
        """Accumulates the gradient array from given source variable.

        This method adds the gradient of a given variable to the gradient of
        this variable. The accumulation is even done across the host and
        different devices. If this variable has uninitialized data/grad arrays,
        this method initializes it with the shape of the given variable and
        then accumulates the gradient.

        Args:
            var (~chainer.Variable): Source variable.

        """
        dst_device = self.device
        is_chainerx = dst_device.xp is chainerx

        if is_chainerx != (var.device.xp is chainerx):
            raise RuntimeError(
                'Variable.addgrad does not support addition between '
                'gradients on non-ChainerX and ChainerX devices.\n'
                'Adding gradient to: {}\n'
                'Adding gradient from: {}'.format(
                    dst_device, var.device))

        if var.grad is None:
            return

        src = var.grad_var

        if self.array is None:
            self.initialize(var.shape)

        dst = self.grad_var
        src_device = src.device
        if src_device != dst_device:
            src = chainer.functions.copy(src, dst_device)
        self.grad_var = src if dst is None else src + dst

    def set_creator(self, gen_func):
        """Notifies the variable that the given function is its creator.

        Args:
            gen_func (Function): Function object that creates this variable as
                one of its outputs.

        """
        if self._has_chainerx_array:
            raise RuntimeError(
                'A variable of ChainerX does not provide a creator.')
        self._node.set_creator(gen_func)

    def set_creator_node(self, fnode):
        """Notifies the variable that the given node is its creator.

        Args:
            fnode (FunctionNode): Function node that has this variable as an
                output.

        """
        if self._has_chainerx_array:
            raise RuntimeError(
                'A variable of ChainerX does not provide a creator node.')
        self._node.set_creator_node(fnode)

    def backward(self, retain_grad=False, enable_double_backprop=False,
                 loss_scale=None):
        """Runs error backpropagation (a.k.a.\\  backprop) from this variable.

        On backprop,
        :meth:`FunctionNode.backward() <chainer.FunctionNode.backward>`
        is called on each :class:`~chainer.FunctionNode` object appearing in
        the backward graph starting from this variable.
        The backward graph is represented by backward
        references from variable nodes to their creators, and from function
        nodes to their input variable nodes. The backprop stops at all root
        nodes. Some function nodes set ``None`` as gradients of some inputs,
        where further backprop does not take place at such inputs.

        This method uses :data:`grad` as the initial error array. User can
        manually set a gradient array before calling this method.
        If the shape of :data:`data` is ``()`` (i.e., it is scalar) and
        :data:`grad` is ``None``, then this method automatically complements
        1.0 as the initial error. This is useful on starting backprop from
        some scalar loss value.

        From v3, this method supports *differentiable backprop* (a.k.a. double
        backprop, grad of grads). To enable it, pass
        ``enable_double_backprop=True``.

        Args:
            retain_grad (bool): If ``True``, the gradient arrays of all
                intermediate variables are kept.
                Otherwise, :data:`~chainer.Variable.grad` of the
                intermediate variables are set to ``None`` on appropriate
                timing, which may reduce the maximum memory consumption.

                In most cases of training some models, the purpose of backprop
                is to compute gradients of parameters, not of all variables,
                and therefore it is recommended that this flag be set to
                ``False``.
            enable_double_backprop (bool): *(Added in v3.0)* If ``True``,
                computational trace of the whole backpropagation procedure is
                recorded to the computational graph so that one can further do
                backpropagation from the resulting gradients. Note that
                enabling it results in larger memory consumption needed to
                store the gradients w.r.t intermediate variables that are
                required for the second gradient computation.
            loss_scale (float): Loss scaling factor. Loss scaling is a useful
                technique to mitigate vanishing gradient issue that tends to
                happen when low precision data type like float16 is used during
                training. If you set loss scaling factor, gradients of loss
                values are to be multiplied by the factor before backprop
                starts. The factor is propagated to whole gradients in a
                computational graph along the backprop. The gradients of
                parameters are divided by the factor just before the parameters
                are to be updated.
        """
        if self._has_chainerx_array:
            if retain_grad:
                raise RuntimeError(
                    'retain_grad is not supported for ChainerX array.')
            arr = self._data[0]
            assert isinstance(arr, chainerx.ndarray)
            # pybind has issues when converting int -> opt<float>
            if loss_scale:
                loss_scale = float(loss_scale)
            chainerx.backward(
                arr, enable_double_backprop=enable_double_backprop,
                loss_scale=loss_scale)
            return

        # Initialize error by 1, if this is a loss variable
        if self.array.size == 1 and self.grad_var is None:
            if self.array.ndim != 0:
                warnings.warn(
                    'Treating a variable with only one element as a scalar'
                    ' in Variable.backward is deprecated. A scalar variable'
                    ' must be a 0-dimensional array. Apply'
                    ' chainer.functions.squeeze to obtain a scalar variable.'
                    ' If the size of this variable accidentally becomes one,'
                    ' set zero to grad.',
                    DeprecationWarning)
            with chainer.using_device(self.device):
                self.grad = self.device.xp.ones_like(self.array)
            if loss_scale is not None:
                self.grad *= loss_scale

        node = self.node
        grad_var = self.grad_var
        self.grad_var = None

        with chainer.using_config('enable_backprop', enable_double_backprop):
            # TODO(kataoka): The following line should not pass grad_var = None
            # to _backprop_to_all, but it is working because grad_var is
            # immediately popped away as None = _backprop_utils._reduce([None])
            _backprop._backprop_to_all(
                [(node, grad_var)], retain_grad, loss_scale)

    def item(self):
        """Converts the variable with one element to a Python scalar.

        This will incur host-device synchronization.

        Returns:
            int or float: The element of the array.

        """
        return self.array.item()

    def mean(self, axis=None, *, weights=None, keepdims=False):
        """Calculate weighted average of array elements over a given axis.

        .. seealso::
           :func:`chainer.functions.average` for full documentation,

        """
        return chainer.functions.average(self, axis, weights, keepdims)

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

        This method is equivalent to ``self.creator_node = None``.

        """
        if self._has_chainerx_array:
            raise RuntimeError(
                'A variable of ChainerX does not provide an unchain method.')
        self.creator_node = None

    def unchain_backward(self):
        """Deletes references between variable nodes and functions backward.

        After this method completes, intermediate variable nodes and functions
        that are not referenced from anywhere are deallocated by reference
        count GC. Also this variable itself deletes the reference to its
        creator function from the node, i.e. the node becomes root in the
        computation graph. It indicates that backprop after unchaining stops at
        this variable. This behavior is useful to implement truncated BPTT.

        """
        if self._has_chainerx_array:
            raise RuntimeError(
                'A variable of ChainerX does not provide an unchain_backward '
                'method.')
        cand_funcs = []
        seen_set = set()

        def add_cand(cand):
            if cand is not None and cand not in seen_set:
                cand_funcs.append(cand)
                seen_set.add(cand)

        add_cand(self.creator_node)

        while cand_funcs:
            func = cand_funcs.pop()
            for var in func.inputs:
                add_cand(var.creator_node)
            func.unchain()

    def retain_data(self):
        """Lets the corresponding variable node keep the underlying array."""
        if self._has_chainerx_array:
            raise RuntimeError(
                'A variable of ChainerX does not provide a retain_data '
                'method.')
        self._node.data = self._data[0]

    def _error_nobp_op(self, op):
        raise TypeError(
            'Variables do not support {} operator. '
            'You could use `array` attribute instead.'.format(op))

    def __lt__(self, other):
        """This operator is not supported in Variables."""
        self._error_nobp_op('<')

    def __le__(self, other):
        """This operator is not supported in Variables."""
        self._error_nobp_op('<=')

    def __eq__(self, other):
        """This operator is not supported in Variables."""
        self._error_nobp_op('==')

    def __ne__(self, other):
        """This operator is not supported in Variables."""
        self._error_nobp_op('!=')

    def __gt__(self, other):
        """This operator is not supported in Variables."""
        self._error_nobp_op('>')

    def __ge__(self, other):
        """This operator is not supported in Variables."""
        self._error_nobp_op('>=')

    def __nonzero__(self):
        """This operator is not supported in Variables."""
        # Python 2.x
        raise TypeError(
            'Variables cannot be evaluated as Python bool.')

    def __bool__(self):
        """This operator is not supported in Variables."""
        # Python 3.x
        raise TypeError(
            'Variables cannot be evaluated as Python bool.')

    __array_priority__ = 200  # type: int
    __hash__ = None  # type: tp.Callable[[object], int]


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
        initializer (~chainer.Initializer or :ref:`ndarray`):
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

    initializer = None  # type: tp.Optional[tp.Union[tp.Optional[types.AbstractInitializer], types.NdArray]] # NOQA
    # TODO(okapies): fix the behavior when shape is None and remove NdArray
    _grad_initializer = None  # type: tp.Optional[types.AbstractInitializer]

    def __init__(
            self,
            initializer: tp.Optional[types.InitializerSpec] = None,
            shape: tp.Optional[types.ShapeSpec] = None,
            name: tp.Optional[str] = None,
            *,
            layout=None
    ) -> None:
        if initializer is None:
            initializer = constant.NaN()
        elif numpy.isscalar(initializer):
            initializer = constant.Constant(initializer)
        if shape is None:
            if isinstance(initializer, chainer.get_array_types()):
                # parameter initialized by the initial array
                super(Parameter, self).__init__(
                    initializer, name=name, layout=layout)
            else:
                # uninitialized parameter
                super(Parameter, self).__init__(
                    name=name, _grad_valid=False, layout=layout)
                dtype = getattr(initializer, 'dtype', None)
                self._grad_initializer = constant.NaN(dtype)
        else:
            # parameter initialized with a given shape
            if isinstance(initializer, chainer.get_array_types()):
                xp = backend.get_array_module(initializer)
                initializer = constant.Constant(initializer)
            else:
                xp = numpy
            data = initializers.generate_array(initializer, shape, xp)  # type: ignore # NOQA
            grad = xp.full_like(data, numpy.nan)
            super(Parameter, self).__init__(
                data, name=name, grad=grad, layout=layout)

        self._initial_device = backend.CpuDevice()
        self.update_rule = None
        self.initializer = initializer

    def __copy__(self):
        return self._copy_to(Parameter())

    def __reduce__(self):
        args = (
            self.array, self.name, self._grad, self._grad_valid,
            self.initializer, self.update_rule, self.device)
        return _recover_parameter, args

    @property
    def is_initialized(self):
        return self._data[0] is not None

    @property
    def dtype(self):
        array = self._data[0]
        if array is not None:
            return array.dtype
        # uninitialized
        initializer = self.initializer
        if hasattr(initializer, 'dtype'):
            return numpy.dtype(initializer.dtype)
        raise RuntimeError(
            'Dtype of the parameter is not determined yet because it\'s '
            'uninitialized and dtype was not explicitly given.')

    def to_cpu(self):
        return self.to_device(backend.CpuDevice())

    def to_gpu(self, device=None):
        device = chainer.get_device(cuda._get_device_or_current(device))
        assert device.xp is cuda.cupy
        self.to_device(device)

    def to_intel64(self):
        self.to_device(intel64.Intel64Device())

    def to_chx(self):
        if not chainerx.is_available():
            raise RuntimeError('ChainerX is not available.')

        # Derive the target ChainerX device from the array if it is
        # initialized. Otherwise, from the current initial device.
        if self.array is not None:
            device = backend.get_device_from_array(self.array)
        else:
            device = self._initial_device

        if device.xp is numpy:
            self._initial_device = backend.ChainerxDevice(
                chainerx.get_device('native:0'))
        elif device.xp is cuda.cupy:
            self._initial_device = backend.ChainerxDevice(
                chainerx.get_device('cuda:{}'.format(device.device.id)))

        super(Parameter, self)._to_chx(allow_unchaining=True)

    def from_chx(self):
        if self.array is not None:
            device = backend.get_device_from_array(self.array)
        else:
            device = self._initial_device

        if device.xp is chainerx:
            backend_name = device.device.backend.name
            if backend_name == 'native':
                self._initial_device = backend.CpuDevice()
            elif backend_name == 'cuda':
                self._initial_device = backend.GpuDevice.from_device_id(
                    device.device.index)

        super(Parameter, self)._from_chx(allow_unchaining=True)

    def to_device(self, device):
        device = chainer.get_device(device)
        if self._data[0] is None and self._initial_device != device:
            self._data = [None]  # Renew placeholder to break sharing
            self._has_chainerx_array = False
        self._initial_device = device
        super(Parameter, self)._to_device(device, allow_unchaining=True)

    def cleargrad(self):
        super(Parameter, self).cleargrad()
        if not self.is_initialized:
            self._grad_initializer = None

    def zerograd(self):
        super(Parameter, self).zerograd()
        if not self.is_initialized:
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
        device = self._initial_device
        assert device is not None
        xp = device.xp

        data = initializers.generate_array(
            self.initializer, shape, xp, device=device)
        data = chainer.memory_layouts._transpose_array(data, None, self.layout)

        if self._grad_initializer is None:
            grad = None
        else:
            grad = initializers.generate_array(
                self._grad_initializer, shape, xp, device=device)
            grad = chainer.memory_layouts._transpose_array(
                grad, None, self.layout)

        self._set_array(data, layout_check=False)
        self._set_grad(grad, layout_check=False)

        # Convert the array for iDeep.
        # TODO(niboshi): This could be done in generate_array().
        if isinstance(self._initial_device, intel64.Intel64Device):
            self.to_intel64()

    def update(self):
        """Updates the data array using the gradient and the update rule.

        This method updates the parameter using the attached update rule.

        """
        if self.update_rule is not None:
            if not self.update_rule.is_elementwise:
                if self.layout is not None:
                    raise RuntimeError(
                        'Parameter with a non-standard layout cannot be '
                        'updated with a non-elementwise update rule '
                        '({}).'.format(self.update_rule))
            self.update_rule.update(self)


def as_variable(obj):
    """Converts an array or a variable into :class:`~chainer.Variable`.

    This is a convenient function to get a :class:`~chainer.Variable` object
    transparently from a raw array or a variable.

    Note that this function should only be used for type consistency (i.e., to
    enforce the return value of an API having type :class:`~chainer.Variable`).
    The :class:`~chainer.Variable.requires_grad` flag is kept as is; if ``obj``
    is a raw array, the newly created variable has ``requires_grad = False``.
    In order to make a variable w.r.t. which you want to compute the gradient,
    you should use :class:`~chainer.Variable` directly.

    Args:
        obj (:ref:`ndarray` or ~chainer.Variable): An array or
            a variable that you want to convert to :class:`~chainer.Variable`.

    Returns:
        ~chainer.Variable:
        A variable converted from ``obj``. If ``obj`` is a raw array, this is a
        new :class:`~chainer.Variable` object that wraps the array. If ``obj``
        is already a :class:`~chainer.Variable` object, this function returns
        ``obj`` as is.

    """
    if isinstance(obj, Variable):
        return obj

    if isinstance(obj, chainerx.ndarray):
        requires_grad = obj.is_backprop_required()
    else:
        requires_grad = False
    return Variable(obj, requires_grad=requires_grad)


def as_array(obj):
    """Returns the underlying array from a variable or an array.

    This is a convenient function to get the underlying array object
    transparently from an object that could be either a variable or an array.

    Args:
        obj (:ref:`ndarray` or ~chainer.Variable): An array or a variable.

    Returns:
        :ref:`ndarray` or ~chainer.Variable:
        The underlying array object of the argument.

    """
    if isinstance(obj, Variable):
        return obj.array
    return obj


def _recover_parameter(*args):
    if len(args) == 7:
        # latest
        data, name, grad, grad_valid, initializer, update_rule, device = args
    elif len(args) == 6:
        data, name, grad, initializer, update_rule, device = args
        grad_valid = True
    else:
        assert False, len(args)

    p = Parameter(initializer=initializer, name=name)
    p.array = data
    p._grad = grad
    p._grad_valid = grad_valid
    p.update_rule = update_rule
    p.to_device(device)
    return p


class _ChainerxVariableNodeProps(object):

    def __init__(self, x):
        self.shape = x.shape
        self.dtype = x.dtype


class _AllowArrayAccessWithNonstandardLayout:
    """Context manager within which access to Variable.array is allowed for \
variables with a non-standard layout."""

    def __enter__(self):
        self._old = _allow_array_access_with_nonstandard_layout()
        _thread_local.allow_array_access_with_nonstandard_layout = True

    def __exit__(self, typ, value, traceback):
        _thread_local.allow_array_access_with_nonstandard_layout = self._old


def _allow_array_access_with_nonstandard_layout():
    # Returns wether a thread-local variable
    # `allow_array_access_with_nonstandard_layout` is set to True.
    try:
        return _thread_local.allow_array_access_with_nonstandard_layout
    except AttributeError:
        return False
