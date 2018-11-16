import functools
import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer.functions.math import einsum
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer import utils


def _tuple_to_gpu(xs):
    return tuple(cuda.to_gpu(x) for x in xs)


def _from_str_subscript(subscript):
    # subscript should be lower case (a-z)
    return [
        (Ellipsis if char == '@' else ord(char) - ord('a'))
        for char in subscript.replace('...', '@')
    ]


def _skip_if(cond, reason):
    def decorator(impl):
        @functools.wraps(impl)
        def wrapper(self, *args, **kwargs):
            if cond(self):
                raise unittest.SkipTest(reason)
            else:
                impl(self, *args, **kwargs)
        return wrapper
    return decorator


_skip_if_float16 = _skip_if(
    lambda self: self.dtype == numpy.float16,
    'float16 is not supported. See numpy issue #10899.'
)


@testing.parameterize(*testing.product_dict(
    [
        {'subscripts': 'ij,jk->ik', 'shapes': ((2, 3), (3, 4))},
        {'subscripts': ',ij->i', 'shapes': ((), (3, 4),)},
        {'subscripts': 'kj,ji->ik', 'shapes': ((2, 3), (3, 4))},
        {'subscripts': 'ij,jk,kl->il', 'shapes': ((5, 2), (2, 3), (3, 4))},
        {'subscripts': 'ij,ij->i', 'shapes': ((2, 3), (2, 3))},
        {'subscripts': 'ij,jk', 'shapes': ((2, 3), (3, 4))},
        {'subscripts': 'i->', 'shapes': ((3,),)},
        {'subscripts': 'ii', 'shapes': ((2, 2),)},
        {'subscripts': 'ii->i', 'shapes': ((2, 2),)},
        {'subscripts': 'j,j', 'shapes': ((3,), (3))},
        {'subscripts': 'j,ij', 'shapes': ((3,), (2, 3))},
        {'subscripts': 'j,iij', 'shapes': ((3,), (2, 2, 3))},
        {'subscripts': 'iij,kkj', 'shapes': ((2, 2, 3), (4, 4, 3))},
        {'subscripts': '...ij,...jk->...ik',
         'shapes': ((2, 1, 2, 3), (2, 1, 3, 4))},
        {'subscripts': 'i...j,jk...->k...i', 'shapes': ((4, 2, 3), (3, 5, 2))},
        {'subscripts': 'ii...,...jj', 'shapes': ((2, 2, 4), (4, 3, 3))},
        {'subscripts': '...i,i', 'shapes': ((2, 2, 3), (3,))},
        {'subscripts': 'i...,i->...i', 'shapes': ((3, 2, 2), (3,))},
        {'subscripts': 'i,ji,i', 'shapes': ((3,), (2, 3), (3,))},
        {'subscripts': 'i,i,i->i', 'shapes': ((3,), (3,), (3,))},
    ],
    testing.product({
        'dtype': [numpy.float16, numpy.float32, numpy.float64],
        'subscript_type': ['str', 'int'],
    }),
))
class TestEinSum(unittest.TestCase):

    def setUp(self):
        self.inputs = tuple([
            self._setup_tensor(-1, 1, shape, self.dtype)
            for shape in self.shapes
        ])
        if self.dtype == numpy.float16:
            # Avoid numpy issue #10899
            self.forward_answer = numpy.einsum(
                *self._get_args(self.inputs),
                dtype=numpy.float64
            ).astype(self.dtype)
        else:
            self.forward_answer = numpy.einsum(*self._get_args(self.inputs))
        self.g = self._setup_tensor(
            -1, 1, self.forward_answer.shape, self.dtype)
        self.gg_inputs = tuple([
            self._setup_tensor(-1, 1, shape, self.dtype)
            for shape in self.shapes
        ])
        self.op = lambda *xs: einsum.einsum(*self._get_args(xs))

    def _get_args(self, xs):
        if self.subscript_type == 'str':
            return (self.subscripts,) + xs
        else:
            args = []
            subscripts = self.subscripts.split('->')
            for in_subscript, x in zip(subscripts[0].split(','), xs):
                args.extend([x, _from_str_subscript(in_subscript)])
            if len(subscripts) == 2:
                args.append(_from_str_subscript(subscripts[1]))
            return tuple(args)

    def _setup_tensor(self, _min, _max, shape, dtype):
        return numpy.random.uniform(_min, _max, shape).astype(dtype)

    def check_forward(self, inputs_data, atol=1e-4, rtol=1e-5):
        out = self.op(*[chainer.Variable(x) for x in inputs_data])
        testing.assert_allclose(self.forward_answer, out.data, atol, rtol)

    @_skip_if_float16
    def test_einsum_forward_cpu(self):
        if self.dtype == numpy.float16:
            self.check_forward(self.inputs, atol=1e-3, rtol=1e-3)
        else:
            self.check_forward(self.inputs)

    @attr.gpu
    def test_einsum_forward_gpu(self):
        inputs = _tuple_to_gpu(self.inputs)
        if self.dtype == numpy.float16:
            self.check_forward(inputs, atol=1e-3, rtol=1e-3)
        else:
            self.check_forward(inputs)

    def check_backward(self, inputs_data, output_grad, atol, rtol):
        gradient_check.check_backward(
            self.op, inputs_data, output_grad, atol=atol, rtol=rtol,
            dtype=numpy.float64)

    @_skip_if_float16
    def test_einsum_backward_cpu(self):
        self.check_backward(self.inputs, self.g, atol=1e-2, rtol=5e-2)

    @attr.gpu
    def test_einsum_backward_gpu(self):
        self.check_backward(
            _tuple_to_gpu(self.inputs),
            cuda.to_gpu(self.g), atol=1e-2, rtol=5e-2)

    def check_double_backward(
            self, inputs_data, y_grad, inputs_grad_grad,
            atol, rtol):
        gradient_check.check_double_backward(
            self.op, inputs_data, y_grad, inputs_grad_grad,
            atol=atol, rtol=rtol, dtype=numpy.float64)

    @_skip_if_float16
    def test_einsum_double_backward_cpu(self):
        self.check_double_backward(
            self.inputs, self.g, self.gg_inputs,
            atol=1e-2, rtol=5e-2)

    @attr.gpu
    def test_einsum_double_backward_gpu(self):
        self.check_double_backward(
            _tuple_to_gpu(self.inputs), cuda.to_gpu(self.g),
            _tuple_to_gpu(self.gg_inputs), atol=1e-2, rtol=1e-2)


@testing.parameterize(
    # mismatch: 'i'
    {'subscripts': 'i,i', 'shapes': ((2,), (3,))},
    {'subscripts': 'i,i->i', 'shapes': ((2,), (3,))},
    {'subscripts': 'ii', 'shapes': ((2, 3),)},

    # mismatch: '...'
    {'subscripts': '...i,...i', 'shapes': ((2, 2), (3, 2))},
    {'subscripts': '...i,...j', 'shapes': ((2, 3), (3, 2))},
    {'subscripts': '...i,j...', 'shapes': ((2, 3), (2, 3))},
    {'subscripts': 'i...,j...', 'shapes': ((2, 3), (3, 2))},

    # F.einsum does not allow broadcasting
    {'subscripts': '...i,...i', 'shapes': ((2, 2), (1, 2))},
    {'subscripts': '...i,...i', 'shapes': ((2,), (1, 2))},
)
class TestEinSumInvalid(unittest.TestCase):

    def setUp(self):
        self.inputs = tuple([
            numpy.zeros(shape, numpy.float32)
            for shape in self.shapes
        ])

    def test_raise_invalid_type(self):
        with self.assertRaises(utils.type_check.InvalidType):
            einsum.einsum(self.subscripts, *self.inputs)


@testing.parameterize(
    {'subscripts': 'i,i', 'shapes': ((2,), (2,), (2,))},
    {'subscripts': 'i,i', 'shapes': ((2,),)},
    {'subscripts': 'i,i->j', 'shapes': ((2,), (2,))},
    {'subscripts': 'i,i->...', 'shapes': ((2,), (2,))},
)
class TestEinSumParseError(unittest.TestCase):

    def setUp(self):
        self.inputs = tuple([
            numpy.zeros(shape, numpy.float32)
            for shape in self.shapes
        ])

    def test_raise_parse_error(self):
        with self.assertRaises(ValueError):
            einsum.einsum(self.subscripts, *self.inputs)


@testing.parameterize(
    {'subscripts': '...->', 'shapes': ((2,),)},
    {'subscripts': 'j...i->ij', 'shapes': ((2, 1, 3),)},
    {'subscripts': 'i,...i->', 'shapes': ((2,), (3, 2))},
)
class TestEinSumUndefinedSemantics(unittest.TestCase):

    def setUp(self):
        self.inputs = tuple([
            numpy.zeros(shape, numpy.float32)
            for shape in self.shapes
        ])

    def test_bad_ellipsis_sum(self):
        with self.assertRaises(ValueError):
            einsum.einsum(self.subscripts, *self.inputs)


def diag_einsum(
        input_subscripts, output_subscript, *ioperands, **kwargs):
    output_shape, = utils.argument.parse_kwargs(kwargs, ('output_shape', None))
    return einsum.DiagEinSum(
        in_subs=input_subscripts,
        out_sub=output_subscript,
        out_shape=output_shape,
    ).apply(ioperands)[0]


@testing.parameterize(*testing.product_dict(
    [
        {'subscripts': 'i->ij', 'i_shapes': ((3,),), 'o_shape': (3, 4)},
        {'subscripts': '->i', 'i_shapes': ((),), 'o_shape': (3,)},
        {'subscripts': ',i->ij', 'i_shapes': ((), (2,),), 'o_shape': (2, 3)},
        {'subscripts': ',ij->i', 'i_shapes': ((), (3, 4),), 'o_shape': (3,)},
    ],
    [
        {'dtype': numpy.float32},
        {'dtype': numpy.float64},
    ]
))
class TestDiagEinSum(unittest.TestCase):

    def setUp(self):
        self.inputs = [
            self._setup_tensor(-1, 1, shape, self.dtype)
            for shape in self.i_shapes
        ]
        self.g = self._setup_tensor(-1, 1, self.o_shape, self.dtype)
        self.gg_inputs = [
            self._setup_tensor(-1, 1, shape, self.dtype)
            for shape in self.i_shapes
        ]
        i_sub, o_sub = self.subscripts.split('->')
        self.op = lambda *xs: diag_einsum(
            i_sub, o_sub, *xs, output_shape=self.o_shape)

    def _setup_tensor(self, _min, _max, shape, dtype):
        return numpy.random.uniform(_min, _max, shape).astype(dtype)

    # TODO(kataoka): test forward

    def check_backward(self, inputs_data, output_grad, atol, rtol):
        gradient_check.check_backward(
            self.op, inputs_data, output_grad, atol=atol, rtol=rtol,
            dtype=numpy.float64)

    def test_einsum_backward_cpu(self):
        self.check_backward(self.inputs, self.g, atol=1e-2, rtol=5e-2)

    @attr.gpu
    def test_einsum_backward_gpu(self):
        self.check_backward(
            _tuple_to_gpu(self.inputs),
            cuda.to_gpu(self.g), atol=1e-2, rtol=5e-2)

    def check_double_backward(
            self, inputs_data, y_grad, inputs_grad_grad,
            atol, rtol):
        gradient_check.check_double_backward(
            self.op, inputs_data, y_grad, inputs_grad_grad,
            atol=atol, rtol=rtol, dtype=numpy.float64)

    def test_einsum_double_backward_cpu(self):
        self.check_double_backward(
            self.inputs, self.g, self.gg_inputs,
            atol=1e-2, rtol=5e-2)

    @attr.gpu
    def test_einsum_double_backward_gpu(self):
        self.check_double_backward(
            _tuple_to_gpu(self.inputs), cuda.to_gpu(self.g),
            _tuple_to_gpu(self.gg_inputs), atol=1e-2, rtol=1e-2)


testing.run_module(__name__, __file__)
