import collections
import copy
import heapq
import traceback
import warnings
import weakref

import numpy
import six

import chainer
from chainer import _backprop_utils
from chainer import backend
from chainer.backends import cuda
from chainer.backends import intel64
from chainer import initializers
from chainer.initializers import constant
from chainer.utils import argument
import chainerx


def _check_grad_type(func, x, gx):
    if x.data is None or gx is None:
        # ``x.data is None`` implies that the data array is not retained
        return
    if not chainer.is_arrays_compatible((gx, x.data)):
        msg = ('Type of data and grad mismatch\ngrad: %s != data: %s' %
               (type(gx), type(x.data)))
        typ = TypeError
    elif gx.dtype != x.data.dtype:
        msg = ('Dtype of data and grad mismatch\ngrad: %s != data: %s' %
               (gx.dtype, x.data.dtype))
        typ = TypeError
    elif gx.shape != x.data.shape:
        msg = ('Shape of data and grad mismatch\ngrad: %s != data: %s' %
               (gx.shape, x.data.shape))
        typ = ValueError
    else:
        return

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
'''.format(type(func).__name__, func.label)

    raise typ(detail + msg)


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
        variable (Variable): The corresponding variable object.
        name (str): Name of the variable node.

    Attributes:
        dtype: Data type of the data array.
        shape: Shape of the data array.
        name (str): Name of the variable node.

    """

    _creator_node = None
    _data = None
    _rank = 0
    # Name of the Function is assigned if this variable is a gradient generated
    # by an old-style Function
    _old_style_grad_generator = None

    def __init__(self, variable, name, **kwargs):
        if kwargs:
            argument.check_unexpected_kwargs(
                kwargs,
                grad='unexpected keyword argument "grad": '
                     'pass the gradient to Variable instead'
            )
        self._variable = weakref.ref(variable)
        self.name = name
        self._requires_grad = variable.requires_grad

        vdata = variable.data
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
        return None if var is None else var._grad_var

    def _set_grad_var_if_available(self, g):
        var = self._variable()
        if var is not None:
            var._grad_var = g

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
            Variable: The variable object that refers this node.

        """
        var = self._variable()
        if var is not None:
            return var

        var = Variable(self.data, name=self.name,
                       requires_grad=self._requires_grad)
        var._node = self
        return var

    def get_variable_or_none(self):
        """Returns the holding :class:`~chainer.Variable` object or ``None``.

        VariableNode object holds a weak reference of the variable object.If
        the reference is alive, it is returned by this property. Otherwise,
        returns ``None``.

        Returns:
            Variable: The variable object that refers this node.

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
        if d is None:
            self.dtype = None
            self.shape = None
        else:
            self.dtype = d.dtype
            self.shape = d.shape

        # If the node has a reference to data, update it as well.
        if self._data is not None:
            self._data = d

    def _check_old_style_gradient(self):
        if self._old_style_grad_generator is not None:
            raise RuntimeError(
                'cannot twice-differentiate an old style Function "%s"' %
                self._old_style_grad_generator)


def _create_variable(data, name, grad, requires_grad):
    return Variable(
        data, name=name, grad=grad, requires_grad=requires_grad)


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

    .. warning::

       ``volatile`` argument is not supported anymore since v2.
       Instead, use :func:`chainer.no_backprop_mode`.

    Args:
        data (numpy.ndarray or cupy.ndarray): Initial data array.
        name (str): Name of the variable.
        grad (numpy.ndarray or cupy.ndarray): Initial gradient array.
        requires_grad (bool): Boolean indicating whether ``grad`` will be set
            in backward calculation.

    """  # NOQA

    def __init__(self, data=None, **kwargs):
        name, grad, requires_grad = argument.parse_kwargs(
            kwargs, ('name', None), ('grad', None), ('requires_grad', True),
            volatile='volatile argument is not supported anymore. '
            'Use chainer.using_config')
        if data is not None:
            array_types = chainer.get_array_types()
            if not isinstance(data, array_types):
                msg = '{} or {} are expected. Actual: {}'.format(
                    ', '.join([str(at) for at in array_types[:-1]]),
                    array_types[-1], type(data))
                raise TypeError(msg)

        # Use a list as a data structure to hold the data array indirectly to
        # abstract its initialized/uninitialized state.
        self._data = [data]
        self._loss_scale = None
        self._grad_var = None if grad is None else Variable(grad)

        # ChainerX itself has own node objects, but not exposed to python.
        if chainerx.is_available() and isinstance(data, chainerx.ndarray):
            if requires_grad:
                data.require_grad()
                data.set_grad(grad)
            elif grad is not None:
                raise ValueError(
                    'Cannot initialize variable with gradients if the'
                    ' require_grad argument is False')

            self._is_chainerx = True
            self._requires_grad = None
            self._node = None
        else:
            self._is_chainerx = False
            # self._requires_grad need to be set before creating the node.
            self._requires_grad = requires_grad
            self._node = VariableNode(self, name)

    def __copy__(self):
        return self._copy_to(Variable())

    def _copy_to(self, target):
        target.__dict__ = copy.copy(self.__dict__)
        target._node = VariableNode(target, self.name)
        return target

    def __reduce__(self):
        return _create_variable, (self.data, self.name, self.grad,
                                  self._requires_grad)

    def __repr__(self):
        return variable_repr(self)

    def __str__(self):
        return variable_str(self)

    @property
    def xp(self):
        """Array module for this variable.

        Depending on which of CPU/GPU this variable is on, this property
        returns :mod:`numpy` or :mod:`cupy`.

        """
        return cuda.get_array_module(self)

    @property
    def name(self):
        if self._is_chainerx:
            raise RuntimeError(
                'A variable of ChainerX does not provide a node name.')
        return self._node.name

    @name.setter
    def name(self, n):
        if self._is_chainerx:
            raise RuntimeError(
                'A variable of ChainerX does not provide a node name.')
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

        data = self.data
        with cuda.get_device_from_array(data) as dev:
            xp = numpy if int(dev) == -1 else cuda.cupy

            if data is None:
                # `data` can be `None` if constructed without any arguments
                device = None
                backend = None
                stats = None
            else:
                device = getattr(data, 'device', 'CPU')
                backend = type(data)
                stats = stats_msg.format(float(xp.mean(data)),
                                         float(xp.std(data)))
            shape = getattr(data, 'shape', None)
            dtype = getattr(data, 'dtype', None)

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
        return len(self.data)

    @property
    def label(self):
        """Short text that represents the variable."""
        if self._is_chainerx:
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
        if self._is_chainerx:
            raise RuntimeError(
                'A variable of ChainerX does not provide a creator.')
        return self._node.creator

    @creator.setter
    def creator(self, func):
        if self._is_chainerx:
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
        if self._is_chainerx:
            raise RuntimeError(
                'A variable of ChainerX does not provide a creator_node.')
        return self._node._creator_node

    @creator_node.setter
    def creator_node(self, func):
        if self._is_chainerx:
            raise RuntimeError(
                'A variable of ChainerX does not provide a creator_node.')
        self._node.creator_node = func

    @property
    def array(self):
        """The underlying data array.

        It is either :class:`numpy.ndarray` or :class:`cupy.ndarray` object,
        or ``None`` if the variable in in an uninitialized state.

        """
        return self._data[0]

    @array.setter
    def array(self, d):
        self._data[0] = d
        if not self._is_chainerx:
            self._node._update_data_info(d)

    @property
    def data(self):
        """The underlying data array (equivalent to :attr:`array`).

        Note that using this attribute directly is discouraged; use
        :attr:`array` instead. Using :attr:`array`, you can find an error
        earlier when your code mixes up Variable and ndarray because
        ndarray does not have an attribute ``.array`` while it has
        ``.data``.

        """
        return self._data[0]

    @data.setter
    def data(self, d):
        self._data[0] = d
        if not self._is_chainerx:
            self._node._update_data_info(d)

    @property
    def grad(self):
        """Gradient array of this variable.

        Note that this property returns the underlying array of the gradient
        variable instead of the gradient variable itself; to get/set
        gradient variable, use :attr:`grad_var` instead.

        If the underlying array is a :class:`chainerx.ndarray` and
        requires_grad is false, trying to access the gradient will results in
        and error.

        """
        gv = self.grad_var
        return None if gv is None else gv.data

    @grad.setter
    def grad(self, g):
        self.grad_var = None if g is None else Variable(g)

    @property
    def grad_var(self):
        """Gradient variable."""
        if self._is_chainerx and self._grad_var is not None and \
                self._grad_var.data is not self.data.grad:
            self._grad_var = Variable(self.data.grad)
        return self._grad_var

    @grad_var.setter
    def grad_var(self, g):
        if g is not None:
            _check_grad_type(None, self, g.data)
            if self._is_chainerx and self.data is not None:
                self.data.set_grad(g.data)
        self._grad_var = g

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
        if self._is_chainerx:
            raise RuntimeError(
                'A variable of ChainerX does not provide a node rank.')
        return self._node.rank

    @property
    def node(self):
        if self._is_chainerx:
            raise RuntimeError(
                'A variable of ChainerX does not provide a node.')
        return self._node

    @property
    def requires_grad(self):
        """It indicates that ``grad`` will be set in backward calculation."""
        if self._is_chainerx:
            return self.data.is_grad_required()
        return self._requires_grad

    @property
    def T(self):
        """Transposition of this variable."""
        return chainer.functions.transpose(self)

    def to_cpu(self):
        """Copies the data and gradient arrays to CPU."""
        if self._is_chainerx:
            raise RuntimeError(
                'A variable of ChainerX does not provide a to_cpu method.')

        data = self.data
        if data is None:
            return

        if isinstance(data, cuda.ndarray):
            # cupy.ndarray to numpy.ndarray
            self._data = [cuda.to_cpu(data)]
        elif isinstance(data, intel64.mdarray):
            # ideep.mdarray to numpy.ndarray
            self._data = [numpy.array(data)]

        if self._grad_var is not None:
            self._grad_var.to_cpu()
        # ensure that the node tracks the device migration
        node = self._node
        if node._data is not None:
            node.retain_data()

    def to_gpu(self, device=None):
        """Copies the data and gradient arrays to specified GPU.

        Args:
            device: Target device specifier. If omitted, the current device is
                used.

        """
        if self._is_chainerx:
            raise RuntimeError(
                'A variable of ChainerX does not provide a to_gpu method.')

        if self.data is None:
            self._data = [None]  # Renew placeholder to break sharing
        else:
            self._data = [cuda.to_gpu(self.data, device)]
            if self._grad_var is not None:
                self._grad_var.to_gpu(device)
            # ensure that the node tracks the device migration
            node = self._node
            if node._data is not None:
                node.retain_data()

    def to_intel64(self):
        """Copies the data and gradient arrays to intel64 specific mdarray.

        If the array is not suited for intel64, it will be converted to
        :class:`numpy.ndarray`.
        """
        if self._is_chainerx:
            raise RuntimeError(
                'A variable of ChainerX does not provide a to_intel64 method.')

        intel64.check_ideep_available()
        data = self.data
        if data is not None:
            if isinstance(data, cuda.ndarray):
                # cupy.ndarray to numpy.ndarray
                data = data.get()
            if (isinstance(data, numpy.ndarray) and data.ndim in (1, 2, 4)):
                # TODO(kmaehashi): Remove ndim validation once iDeep has fixed.
                # Currently iDeep only supports (1, 2, 4)-dim arrays.
                # Note that array returned from `ideep.array` may not be an
                # iDeep mdarray, e.g., when the dtype is not float32.
                data = intel64.ideep.array(
                    data, itype=intel64.ideep.wgt_array)
            self._data = [data]

        if self._grad_var is not None:
            self._grad_var.to_intel64()
        # ensure that the node tracks the device migration
        node = self._node
        if node._data is not None:
            node.retain_data()

    def cleargrad(self):
        """Clears the gradient array."""
        if self._is_chainerx:
            self.data.cleargrad()
        self._grad_var = None

    def zerograd(self):
        """Initializes the gradient array by zeros.

        Note that the gradient variable is unchained from the computational
        graph by this method because this operation breaks the backprop
        validity.

        .. deprecated:: v1.15
           Use :meth:`cleargrad` instead.

        """
        warnings.warn(
            'Variable.zerograd is deprecated. Use Variable.cleargrad instead.',
            DeprecationWarning)

        if self.data is None:
            return

        gv = self._grad_var

        if self._is_chainerx:
            if gv is None:
                self.grad = chainerx.zeros_like(self.data)
            else:
                self.grad.cleargrad()
                self.grad.fill(0)
        else:
            with cuda.get_device_from_array(self.data) as dev:
                if gv is None:
                    xp = numpy if dev.id == -1 else cuda.cupy
                    self.grad = xp.zeros_like(self.data)
                else:
                    gv.unchain()
                    gv.data.fill(0)

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
        backend.copyto(dst, src)

    def addgrad(self, var):
        """Accumulates the gradient array from given source variable.

        This method adds the gradient of a given variable to the gradient of
        this variable. The accumulation is even done across the host and
        different devices. If this variable has uninitialized data/grad arrays,
        this method initializes it with the shape of the given variable and
        then accumulates the gradient.

        Args:
            var (Variable): Source variable.

        """
        # TODO(sonots): Implement for ChainerX
        if self._is_chainerx:
            raise NotImplementedError()
        src = var._grad_var
        if src is None:
            return

        if self.data is None:
            self.initialize(var.shape)
        dst = self._grad_var

        src_dev = cuda.get_device_from_array(src.data)
        dst_dev = cuda.get_device_from_array(self.data)

        if src_dev.id != dst_dev.id:
            src = chainer.functions.copy(src, dst_dev.id)
        self._grad_var = src if dst is None else src + dst

    def set_creator(self, gen_func):
        """Notifies the variable that the given function is its creator.

        Args:
            gen_func (Function): Function object that creates this variable as
                one of its outputs.

        """
        if self._is_chainerx:
            raise RuntimeError(
                'A variable of ChainerX does not provide a creator.')
        self._node.set_creator(gen_func)

    def set_creator_node(self, fnode):
        """Notifies the variable that the given node is its creator.

        Args:
            fnode (FunctionNode): Function node that has this variable as an
                output.

        """
        if self._is_chainerx:
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
                and therefore it is recommended to set this flag ``False``.
            enable_double_backprop (bool): *(Added in v3.0)* If ``True``,
                computational trace of the whole backpropagation procedure is
                recorded to the computational graph so that one can further do
                backpropagation from the resulting gradients. Note that
                enabling it results in larger memory consumption needed to
                store the gradients w.r.t intermediate variables that are
                required for the second gradient computation.
            loss_scale (float): Loss scaling factor. Loss scaling is a usefull
                technique to mitigate vanishing gradient issue that tends to
                happen when low precision data type like float16 is used during
                training. If you set loss scaling factor, gradients of loss
                values are to be multiplied by the factor before backprop
                starts. The factor is propagated to whole gradients in a
                computational graph along the backprop. The gradients of
                parameters are divided by the factor just before the parameters
                are to be updated.
        """
        with chainer.using_config('enable_backprop', enable_double_backprop):
            self._backward_main(retain_grad, loss_scale)

    def _backward_main(self, retain_grad, loss_scale):
        # TODO(sonots): Implement for ChainerX
        if self._is_chainerx:
            raise NotImplementedError()
        self._node._check_old_style_gradient()
        if self.creator_node is None:
            return

        cand_funcs = []
        seen_set = set()
        grads = _backprop_utils.GradTable(load_if_new=True)

        # Initialize error by 1, if this is a loss variable
        if self.data.size == 1 and self._grad_var is None:
            if self.data.ndim != 0:
                warnings.warn(
                    'Treating a scalar as a variable with only one element'
                    ' in Variable.backward is deprecated. A scalar variable'
                    ' must be a 0-dimensional array. Apply'
                    ' chainer.functions.squeeze to obtain a scalar variable.'
                    ' If the size of this variable accidentally becomes one,'
                    ' set zero to grad.',
                    DeprecationWarning)
            with cuda.get_device_from_array(self.data) as device:
                if device is cuda.DummyDevice:
                    self.grad = numpy.ones_like(self.data)
                else:
                    self.grad = cuda.cupy.ones_like(self.data)
            if loss_scale is not None:
                self.grad *= loss_scale
        grads[self._node] = self._grad_var

        def add_cand(cand):
            if cand not in seen_set:
                # Negate since heapq is min-heap
                heapq.heappush(cand_funcs, (-cand.rank, len(seen_set), cand))
                seen_set.add(cand)

        add_cand(self.creator_node)
        leaf_nodes = set()

        while cand_funcs:
            _, _, func = heapq.heappop(cand_funcs)
            inputs = func.inputs
            target_input_indexes = tuple([
                i for i, x in enumerate(inputs) if x.requires_grad
            ])
            outputs = [y() for y in func.outputs]  # access via weak ref
            out_grad = tuple([grads.pop(y) for y in outputs])
            if not target_input_indexes:
                continue

            in_data = tuple([x.data for x in inputs])
            out_grad_data = tuple(
                [None if g is None else g.data for g in out_grad])
            hooks = chainer.get_function_hooks()
            if func._n_local_function_hooks != 0:
                hooks = collections.OrderedDict(hooks)
                hooks.update(func.local_function_hooks)
            hooks = hooks.values()  # avoid six for performance

            with cuda.get_device_from_array(*(in_data + out_grad_data)):
                for hook in hooks:
                    hook.backward_preprocess(func, in_data, out_grad_data)

                # Collect the current input gradients.
                target_inputs = [inputs[i] for i in target_input_indexes]
                # Keep the order for the portability, rather than
                # in_grad = {x: grads.get_as_list(x)
                #            for x in set(target_inputs)}
                in_grad = collections.OrderedDict()
                for x in target_inputs:
                    if x not in in_grad:
                        in_grad[x] = grads.get_as_list(x)
                        # to reduce memory usage
                        x._set_grad_var_if_available(None)

                _backprop_utils.backprop_step(
                    func, target_input_indexes, out_grad, in_grad)

                for hook in hooks:
                    hook.backward_postprocess(func, in_data, out_grad_data)

            for y, gy in six.moves.zip(outputs, out_grad):
                if y is not None and y is not self.node:
                    y._set_grad_var_if_available(
                        gy if retain_grad else None)
            del gy, out_grad  # to reduce memory usage

            for x, gx in in_grad.items():
                if not gx:  # gradient == None
                    continue

                for gx_elem in gx:
                    _check_grad_type(func, x, gx_elem.data)
                del gx_elem  # to reduce memory usage

                if x.creator_node is None:  # leaf
                    leaf_nodes.add(x)
                else:
                    add_cand(x.creator_node)
            del gx, in_grad  # to reduce memory usage

        for x in leaf_nodes:
            x_var = x.get_variable_or_none()
            gx = grads.pop(x)
            if x_var is not None:
                x_var._grad_var = gx
                x_var._loss_scale = loss_scale
        grads.assert_no_grads()

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
        if self._is_chainerx:
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
        if self._is_chainerx:
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
        if self._is_chainerx:
            raise RuntimeError(
                'A variable of ChainerX does not provide a retain_data '
                'method.')
        self._node.data = self._data[0]

    def __lt__(self, other):
        """This operator is not defined for Variable."""
        raise NotImplementedError()

    def __le__(self, other):
        """This operator is not defined for Variable."""
        raise NotImplementedError()

    def __eq__(self, other):
        """This operator is not defined for Variable."""
        raise NotImplementedError()

    def __ne__(self, other):
        """This operator is not defined for Variable."""
        raise NotImplementedError()

    def __gt__(self, other):
        """This operator is not defined for Variable."""
        raise NotImplementedError()

    def __ge__(self, other):
        """This operator is not defined for Variable."""
        raise NotImplementedError()

    def __nonzero__(self):
        """This operator is not defined for Variable."""
        raise NotImplementedError()

    def __bool__(self):
        """This operator is not defined for Variable."""
        raise NotImplementedError()

    __array_priority__ = 200
    __hash__ = None


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
    _initial_backend = None
    _initial_device = None

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
                dtype = getattr(initializer, 'dtype', None)
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
        self.initializer = initializer

    def __copy__(self):
        return self._copy_to(Parameter())

    def __reduce__(self):
        return _recover_parameter, (self.data, self.name, self.grad,
                                    self.initializer, self.update_rule)

    def to_cpu(self):
        super(Parameter, self).to_cpu()
        if self.data is None:
            self._initial_backend = None
            self._initial_device = None

    def to_gpu(self, device=None):
        super(Parameter, self).to_gpu(device)
        if self.data is None:
            if device is None:
                device = cuda.Device().id
            self._initial_backend = 'cuda'
            self._initial_device = device

    def to_intel64(self):
        super(Parameter, self).to_intel64()
        if self.data is None:
            self._initial_backend = 'intel64'
            self._initial_device = None

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
        xp = numpy if self._initial_backend != 'cuda' else cuda.cupy
        with cuda.get_device_from_id(self._initial_device):
            data = initializers.generate_array(self.initializer, shape, xp)

            ginit = self._grad_initializer
            grad = None if ginit is None else initializers.generate_array(
                ginit, shape, xp)

        self.data = data
        self.grad = grad

        # Convert the array for iDeep.
        if self._initial_backend == 'intel64':
            self.to_intel64()

    def update(self):
        """Updates the data array using the gradient and the update rule.

        This method updates the parameter using the attached update rule.

        """
        if self.update_rule is not None:
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
        obj (numpy.ndarray or cupy.ndarray or ~chainer.Variable): An array or
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
    return Variable(obj, requires_grad=False)


def as_array(obj):
    """Returns the underlying array from a variable or an array.

    This is a convenient function to get the underlying array object
    transparently from an object that could be either a variable or an array.

    Args:
        obj (chainerx.ndarray numpy.ndarray or cupy.ndarray or
            ~chainer.Variable): An array or a variable.

    Returns:
        chainerx.ndarray numpy.ndarray or cupy.ndarray or ~chainer.Variable:
        The underlying array object of the argument.

    """
    if isinstance(obj, Variable):
        return obj.array
    return obj


def _recover_parameter(data, name, grad, initializer, update_rule):
    p = Parameter(initializer=initializer, name=name)
    p.data = data
    p.grad = grad
    p.update_rule = update_rule
    return p
