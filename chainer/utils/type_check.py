import contextlib
import functools
import operator
import sys
import threading

import numpy
import six

import chainer
from chainer.backends import cuda


_thread_local = threading.local()


@contextlib.contextmanager
def get_function_check_context(f):
    try:
        default = _thread_local.current_function
    except AttributeError:
        default = None
    _thread_local.current_function = f
    try:
        yield
    finally:
        _thread_local.current_function = default


class TypeInfo(object):

    """Type information of an input/gradient array.

    It contains type information of an array, such as the shape of array and
    the number of dimensions.
    This information is independent of CPU or GPU array.
    """

    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype
        self.ndim = len(shape)

    @property
    def size(self):
        return functools.reduce(operator.mul, self.shape, 1)


class TypeInfoTuple(tuple):

    """Type information of input/gradient tuples.

    It is a sub-class of tuple containing :class:`TypeInfo`. The i-th element
    of this object contains type information of the i-th input/gradient data.
    As each element is :class:`Expr`, you can easily check its validity.
    """

    def size(self):
        """Returns an expression representing its length.

        Returns:
            Expr: An expression object representing length of the tuple.
        """
        return Variable(len(self), '{0}.size'.format(self.name))


class LightTypeInfoTuple(tuple):

    """Type information of input/gradient tuples for light-weight check.

    It is a sub-class of tuple containing :class:`TypeInfo`. The i-th element
    of this object contains type information of the i-th input/gradient data.
    """

    def size(self):
        """Returns its length.

        Returns:
            int: Length of the tuple.
        """
        return len(self)


def get_types(data, name, accept_none):
    assert isinstance(data, tuple)

    info = TypeInfoTuple(
        _get_type(name, i, x, accept_none) for i, x in enumerate(data))
    # I don't know a method to set an attribute in an initializer of tuple.
    info.name = name
    return info


def get_light_types(data):
    assert(isinstance(data, tuple))

    return LightTypeInfoTuple(data)


def _get_type(name, index, array, accept_none):
    var = '{0}[{1}]'.format(name, index)

    if accept_none and array is None:
        # case that gradient is not given
        return Variable(TypeInfo((), None), var)

    assert isinstance(array, chainer.get_array_types())
    return Variable(TypeInfo(array.shape, array.dtype), var)


def _make_un_operator(exp, priority, func):
    def f(x):
        return UnaryOperator(priority, x, exp, func)
    return f


def _make_bin_operator(exp, priority, func, right_associative=False):
    def f(x, y):
        return BinaryOperator(priority, x, y, exp, func, right_associative)
    return f


def _make_bool_operator(exp, inv, func):
    def f(x, y):
        return BoolBinaryOperator(x, y, exp, inv, func)
    return f


def _flip(f):
    return lambda x, y: f(y, x)


class Expr(object):

    """Abstract syntax tree of an expression.

    It represents an abstract syntax tree, and isn't a value. You can get its
    actual value with :meth:`eval` function, and get syntax representation with
    the :meth:`__str__` method.
    Each comparison operator (e.g. ``==``) generates a new :class:`Expr` object
    which represents the result of comparison between two expressions.

    .. admonition:: Example

       Let ``x`` and ``y`` be instances of :class:`Expr`, then ::

          >>> x = Variable(1, 'x')
          >>> y = Variable(1, 'y')
          >>> c = (x == y)

       is also an instance of :class:`Expr`. To evaluate and get its value,
       call :meth:`eval` method::

          >>> c.eval()
          True

       Call ``str`` function to get a representation of the original
       equation::

          >>> str(c)
          'x == y'

       You can actually compare an expression with a value::

          >>> (x == 1).eval()
          True

       Note that you can't use boolean operators such as ``and``, as they try
       to cast expressions to boolean values::

          >>> z = Variable(1, 'z')
          >>> x == y and y == z  # raises an error
          Traceback (most recent call last):
          RuntimeError: Don't convert Expr to bool. Please call Expr.eval \
method to evaluate expression.


    """

    def __init__(self, priority):
        self.priority = priority

    def eval(self):
        """Evaluates the tree to get actual value.

        Behavior of this function depends on an implementation class.
        For example, a binary operator ``+`` calls the ``__add__`` function
        with the two results of :meth:`eval` function.
        """
        raise NotImplementedError()

    def __getattr__(self, name):
        return GetAttr(self, name)

    def __getitem__(self, key):
        return GetItem(self, key)

    def __call__(self, *args):
        return Call(self, args)

    def __nonzero__(self):
        # When a user calls a boolean operator like `(x == y and z == w)`,
        # `and` operator evaluate the first expression.
        # If it returns `True` (and it's default behavior), the `and` operator
        # returns *the second expression*, not a boolean value.
        # So, `(x == y and z == w)` returns the result of `z == w`, and
        # `(x == y and z == w).expect()` raise no errors but only checks
        # `z == w`. It is confusing.
        # See also:
        # https://docs.python.org/3/library/stdtypes.html
        msg = ('Don\'t convert Expr to bool. '
               'Please call Expr.eval method to evaluate expression.')
        raise RuntimeError(msg)

    def __bool__(self):
        self.__nonzero__()

    __eq__ = _make_bool_operator('==', '!=', operator.__eq__)
    __ne__ = _make_bool_operator('!=', '==', operator.__ne__)
    __lt__ = _make_bool_operator('<', '>=', operator.__lt__)
    __le__ = _make_bool_operator('<=', '>', operator.__le__)
    __gt__ = _make_bool_operator('>', '<=', operator.__gt__)
    __ge__ = _make_bool_operator('>=', '<', operator.__ge__)

    # Please refer the Python documentation to know priority of operators.
    # https://docs.python.org/3/reference/expressions.html

    __add__ = _make_bin_operator('+', 4, operator.__add__)
    __radd__ = _flip(__add__)
    __sub__ = _make_bin_operator('-', 4, operator.__sub__)
    __rsub__ = _flip(__sub__)
    __mul__ = _make_bin_operator('*', 5, operator.__mul__)
    __rmul__ = _flip(__mul__)

    if sys.version_info < (3, 0, 0):
        __div__ = _make_bin_operator('/', 5, operator.__div__)  # type: ignore # NOQA
        __rdiv__ = _flip(__div__)
    else:
        __truediv__ = _make_bin_operator('/', 5, operator.__truediv__)
        __rtruediv__ = _flip(__truediv__)

    __floordiv__ = _make_bin_operator('//', 5, operator.__floordiv__)
    __rfloordiv__ = _flip(__floordiv__)
    __mod__ = _make_bin_operator('%', 5, operator.__mod__)
    __rmod__ = _flip(__mod__)
    # Only '**' operator is right-associative
    __pow__ = _make_bin_operator('**', 7, operator.__mod__,
                                 right_associative=True)

    __lshift__ = _make_bin_operator('<<', 3, operator.__lshift__)
    __rlshift__ = _flip(__lshift__)
    __rshift__ = _make_bin_operator('>>', 3, operator.__rshift__)
    __rrshift__ = _flip(__rshift__)

    __and__ = _make_bin_operator('&', 2, operator.__and__)
    __rand__ = _flip(__and__)
    __xor__ = _make_bin_operator('^', 1, operator.__xor__)
    __rxor__ = _flip(__xor__)
    __or__ = _make_bin_operator('|', 0, operator.__or__)
    __ror__ = _flip(__or__)

    __neg__ = _make_un_operator('-', 6, operator.__neg__)
    __pos__ = _make_un_operator('+', 6, operator.__pos__)
    __invert__ = _make_un_operator('~', 6, operator.__invert__)


def _eval_expr(v):
    if isinstance(v, Expr):
        return v.eval()
    elif isinstance(v, list):
        return list(map(_eval_expr, v))
    elif isinstance(v, tuple):
        return tuple(map(_eval_expr, v))
    else:
        return v


def _repr(v):
    if isinstance(v, Expr):
        return str(v)
    elif isinstance(v, list):
        return '[{0}]'.format(', '.join(map(_repr, v)))
    elif isinstance(v, tuple):
        if len(v) == 0:
            return '()'
        elif len(v) == 1:
            return '({0},)'.format(_repr(v[0]))
        else:
            return '({0})'.format(', '.join(map(_repr, v)))
    else:
        return repr(v)


class Atom(Expr):

    def __init__(self):
        super(Atom, self).__init__(8)


class Constant(Atom):

    def __init__(self, value):
        super(Constant, self).__init__()
        self.value = value

    def __str__(self):
        return _repr(self.value)

    def eval(self):
        return self.value


class Variable(Atom):

    def __init__(self, value, name):
        super(Variable, self).__init__()
        self.value = value
        self.name = name

    def __str__(self):
        return self.name

    def eval(self):
        return self.value


class GetAttr(Atom):

    def __init__(self, obj, name):
        super(GetAttr, self).__init__()
        self.obj = obj
        self.name = name

    def __str__(self):
        if isinstance(self.name, str):
            return '{0}.{1}'.format(_repr(self.obj), self.name)
        elif (isinstance(self.name, Constant) and
              isinstance(self.name.value, str)):
            return '{0}.{1}'.format(_repr(self.obj), self.name.value)
        else:
            return 'getattr({0}, {1})'.format(_repr(self.obj),
                                              _repr(self.name))

    def eval(self):
        return getattr(_eval_expr(self.obj), _eval_expr(self.name))


def _str_subscript(exp):
    if exp is Ellipsis:
        return '...'
    elif isinstance(exp, slice):
        def key_str(v):
            return '' if v is None else _repr(v)

        if exp.step is None:
            return '{0}:{1}'.format(key_str(exp.start),
                                    key_str(exp.stop))
        else:
            return '{0}:{1}:{2}'.format(key_str(exp.start),
                                        key_str(exp.stop),
                                        key_str(exp.step))
    elif isinstance(exp, tuple):
        return ', '.join(map(_str_subscript, exp))

    else:
        return _repr(exp)


class GetItem(Atom):

    def __init__(self, obj, key):
        super(GetItem, self).__init__()
        self.obj = obj
        self.key = key

    def __str__(self):
        key = _str_subscript(self.key)
        return '{0}[{1}]'.format(_repr(self.obj), key)

    def eval(self):
        return _eval_expr(self.obj)[_eval_expr(self.key)]


class Call(Atom):

    def __init__(self, obj, args):
        assert isinstance(args, tuple)
        super(Call, self).__init__()
        self.obj = obj
        self.args = args

    def __str__(self):
        return '{0}({1})'.format(_repr(self.obj),
                                 ', '.join(map(_repr, self.args)))

    def eval(self):
        args = map(_eval_expr, self.args)
        func = _eval_expr(self.obj)
        return func(*args)


class UnaryOperator(Expr):

    def __init__(self, priority, term, exp, func):
        super(UnaryOperator, self).__init__(priority)
        self.term = term
        self.exp = exp
        self.func = func

    def eval(self):
        return self.func(_eval_expr(self.term))

    def __str__(self):
        exp = _repr(self.term)
        if isinstance(self.term, Expr) and self.term.priority < self.priority:
            exp = '(' + exp + ')'

        return self.exp + exp


class BinaryOperator(Expr):

    def __init__(self, priority, lhs, rhs, exp, func, right_associative=False):
        super(BinaryOperator, self).__init__(priority)
        self.lhs = lhs
        self.rhs = rhs
        self.exp = exp
        self.func = func
        self.right_associative = right_associative

    def eval(self):
        left = self._eval_left()
        right = self._eval_right()
        return self.func(left, right)

    def _eval_left(self):
        return _eval_expr(self.lhs)

    def _eval_right(self):
        return _eval_expr(self.rhs)

    def __str__(self):
        # When an infix operator is left-associative, we need to append parens
        # when rhs has the same priority
        #  e.g. x << (y << z) != x << y << z

        left = _repr(self.lhs)
        if isinstance(self.lhs, Expr) and (
                self.priority > self.lhs.priority or
                (self.right_associative and
                 self.priority == self.lhs.priority)):
            left = '(' + left + ')'

        right = _repr(self.rhs)
        if isinstance(self.rhs, Expr) and (
                self.priority > self.rhs.priority or
                (not self.right_associative and
                 self.priority == self.rhs.priority)):
            right = '(' + right + ')'

        return '{0} {2} {1}'.format(left, right, self.exp)


class Testable(object):

    def expect(self):
        raise NotImplementedError()


class BoolBinaryOperator(BinaryOperator, Testable):

    def __init__(self, lhs, rhs, exp, inv, func):
        BinaryOperator.__init__(self, -1, lhs, rhs, exp, func)
        self.inv = inv

    def expect(self):
        left = self._eval_left()
        right = self._eval_right()

        if not self.func(left, right):
            raise InvalidType(
                '{0} {1} {2}'.format(self.lhs, self.exp, self.rhs),
                '{0} {1} {2}'.format(left, self.inv, right))


class InvalidType(Exception):
    """Raised when types of data for forward/backward are invalid.

    """

    def __init__(self, expect, actual, msg=None):
        if msg is None:
            msg = 'Expect: {0}\nActual: {1}'.format(expect, actual)
            if (hasattr(_thread_local, 'current_function')
                    and _thread_local.current_function is not None):
                msg = '''
Invalid operation is performed in: {0} (Forward)

{1}'''.format(_thread_local.current_function.label, msg)

        super(InvalidType, self).__init__(msg)

        self.expect = expect
        self.actual = actual

    def __reduce__(self):
        msg, = self.args
        return (InvalidType, (self.expect, self.actual, msg))


def _argname(in_types, names):
    """Assigns user friendly names for the input types.

    This function also asserts that lenghts of in_types and names are the
    same.

    Args:
        in_types (tuple of TypeInfoTuple): Tuple of type information to assign
            name to.
        names (tuple of str): Human-readabel names of ``in_types``.
    """
    if len(in_types) != len(names):
        raise InvalidType(
            '{} argument(s)'.format(str(len(names))),
            '{} argument(s)'.format(str(len(in_types))),
            'Invalid number of arguments')
    for in_type, name in zip(in_types, names):
        if isinstance(in_type, Variable):
            in_type.name = name


def expect(*bool_exprs):
    """Evaluates and tests all given expressions.

    This function evaluates given boolean expressions in order. When at least
    one expression is evaluated as ``False``, that means the given condition is
    not satisfied.
    You can check conditions with this function.

    Args:
        bool_exprs (tuple of Bool expressions): Bool expressions you want to
            evaluate.
    """
    if in_light_mode():
        if not all(bool_exprs):
            raise InvalidType('', '')
    else:
        for expr in bool_exprs:
            assert isinstance(expr, Testable)
            expr.expect()


def same_types(*arrays):
    for x in arrays:
        if not isinstance(x, numpy.ndarray):
            break
    else:
        return True
    for x in arrays:
        if not isinstance(x, cuda.ndarray):
            return False
    return True


def eval(exp):
    if in_light_mode():
        return exp
    else:
        return exp.eval()


def make_variable(value, name):
    if in_light_mode():
        return value
    else:
        return Variable(value, name)


class LightMode(object):

    def __enter__(self):
        _thread_local.light_mode = True

    def __exit__(self, exc_type, exc_value, traceback):
        _thread_local.light_mode = False


def _prod_impl(xs):
    result = 1
    for x in xs:
        result *= x
    return result


_prod = Variable(_prod_impl, 'prod')
light_mode = LightMode()


def in_light_mode():
    try:
        return _thread_local.light_mode
    except AttributeError:
        _thread_local.light_mode = False
    return False


def prod(xs):
    if in_light_mode():
        return _prod_impl(xs)
    else:
        return _prod(xs)


def expect_broadcast_shapes(*shape_types):
    """Checks if shapes can be broadcasted together.

    Args:
        shapes_types: Type-checked shapes of the arrays to broadcast.

    """
    shapes = [eval(s) for s in shape_types]
    error = None
    try:
        # simulate the shape calculation using zero-sized arrays
        numpy.broadcast(*[numpy.empty(s + (0,)) for s in shapes])
    except ValueError:
        msgs = ['cannot broadcast inputs of the following shapes:']
        for shape_type, shape in six.moves.zip(shape_types, shapes):
            msgs.append('{} = {}'.format(shape_type, shape))
        error = InvalidType('', '', msg='\n'.join(msgs))
    if error is not None:
        raise error
