import math

import numpy

from chainer import cuda
from chainer import function
from chainer import model
from chainer.utils import array
from chainer.utils import type_check
from chainer import variable


class BilinearFunction(function.Function):

    def check_type_forward(self, in_types):
        n_in = in_types.size().eval()
        if n_in != 3 and n_in != 6:
            raise type_check.InvalidType(
                '%s or %s' % (in_types.size() == 3, in_types.size() == 6),
                '%s == %s' % (in_types.size(), n_in))

        e1_type, e2_type, W_type = in_types[:3]
        type_check_prod = type_check.Variable(numpy.prod, 'prod')
        type_check.expect(
            e1_type.dtype == numpy.float32,
            e1_type.ndim >= 2,
            e2_type.dtype == numpy.float32,
            e2_type.ndim >= 2,
            e1_type.shape[0] == e2_type.shape[0],
            W_type.dtype == numpy.float32,
            W_type.ndim == 3,
            type_check_prod(e1_type.shape[1:]) == W_type.shape[0],
            type_check_prod(e2_type.shape[1:]) == W_type.shape[1],
        )

        if n_in == 6:
            out_size = W_type.shape[2]
            V1_type, V2_type, b_type = in_types[3:]
            type_check.expect(
                V1_type.dtype == numpy.float32,
                V1_type.ndim == 2,
                V1_type.shape[0] == W_type.shape[0],
                V1_type.shape[1] == out_size,
                V2_type.dtype == numpy.float32,
                V2_type.ndim == 2,
                V2_type.shape[0] == W_type.shape[1],
                V2_type.shape[1] == out_size,
                b_type.dtype == numpy.float32,
                b_type.ndim == 1,
                b_type.shape[0] == out_size,
            )

    def forward(self, inputs):
        e1 = array.as_mat(inputs[0])
        e2 = array.as_mat(inputs[1])
        W = inputs[2]

        xp = cuda.get_array_module(*inputs)
        if xp is numpy:
            y = numpy.einsum('ij,ik,jkl->il', e1, e2, W)
        else:
            i_len, j_len = e1.shape
            k_len = e2.shape[1]
            # 'ij,ik->ijk'
            e1e2 = e1[:, :, None] * e2[:, None, :]
            # ijk->i[jk]
            e1e2 = e1e2.reshape(i_len, j_len * k_len)
            # jkl->[jk]l
            W_mat = W.reshape(-1, W.shape[2])
            # 'i[jk],[jk]l->il'
            y = e1e2.dot(W_mat)

        if len(inputs) == 6:
            V1, V2, b = inputs[3:]
            y += e1.dot(V1)
            y += e2.dot(V2)
            y += b
        return y,

    def backward(self, inputs, grad_outputs):
        e1 = array.as_mat(inputs[0])
        e2 = array.as_mat(inputs[1])
        W = inputs[2]
        gy = grad_outputs[0]

        xp = cuda.get_array_module(*inputs)
        if xp is numpy:
            gW = numpy.einsum('ij,ik,il->jkl', e1, e2, gy)
            ge1 = numpy.einsum('ik,jkl,il->ij', e2, W, gy)
            ge2 = numpy.einsum('ij,jkl,il->ik', e1, W, gy)
        else:
            kern = cuda.reduce('T in0, T in1, T in2', 'T out',
                               'in0 * in1 * in2', 'a + b', 'out = a', 0,
                               'bilinear_product')

            e1_b = e1[:, :, None, None]  # ij
            e2_b = e2[:, None, :, None]  # ik
            gy_b = gy[:, None, None, :]  # il
            W_b = W[None, :, :, :]  # jkl

            gW = kern(e1_b, e2_b, gy_b, axis=0)  # 'ij,ik,il->jkl'
            ge1 = kern(e2_b, W_b, gy_b, axis=(2, 3))  # 'ik,jkl,il->ij'
            ge2 = kern(e1_b, W_b, gy_b, axis=(1, 3))  # 'ij,jkl,il->ik'

        ret = ge1.reshape(inputs[0].shape), ge2.reshape(inputs[1].shape), gW
        if len(inputs) == 6:
            V1, V2, b = inputs[3:]
            gV1 = e1.T.dot(gy)
            gV2 = e2.T.dot(gy)
            gb = gy.sum(0)
            ge1 += gy.dot(V1.T)
            ge2 += gy.dot(V2.T)
            ret += gV1, gV2, gb
        return ret


def bilinear(e1, e2, W, V1=None, V2=None, b=None):
    """Applies a bilinear function based on given parameters.

    Note that V1, V2, b are optional, though they must not be None if at least
    one of them is not None.

    Args:
        e1 (~chainer.Variable): Left input variable.
        e2 (~chainer.Variable): Right input variable.
        W (~chainer.Variable): Quadratic weight variable.
        V1 (~chainer.Variable): Left coefficient variable.
        V2 (~chainer.Variable): Right coefficient variable.
        b (~chainer.Variable): Bias variable.

    Reutnrs:
        ~chainer.Variable: Output variable.

    """
    flags = [V1 is None, V2 is None, b is None]
    if any(flags):
        if not all(flags):
            raise ValueError('All coefficients and bias for bilinear() must '
                             'be None, if at least one of them is None.')
        return BilinearFunction()(e1, e2, W)
    else:
        return BilinearFunction()(e1, e2, W, V1, V2, b)


class Bilinear(model.Model):

    """Bilinear function, an extension of Linear function.

    ``Bilinear`` function takes two input vectors and outputs one vector.
    If one of the input vectors is fixed, this function works
    as an affine transform of the other input vector.

    ``Bilinear`` function is a building block of Neural Tensor Network
    (See the reference paper below).

    To be precise, ``Bilinear`` function has four parameters,
    :math:`W\in \mathbb{R}^{J \cdot K \cdot L}`,
    :math:`V^1\in \mathbb{R}^{J \cdot L}`,
    :math:`V^2\in \mathbb{R}^{K \cdot L}`, and :math:`b\in \mathbb{R}^{L}`.
    In this document, we call :math:`V^1`, :math:`V^2`,
    and :math:`b` linear parameters.

    Given two inputs (in a mini-batch manner)
    :math:`e^1\in \mathbb{R}^{I\cdot J}` and
    :math:`e^2\in \mathbb{R}^{I\cdot K}`
    where :math:`I` is mini-batch size, the output of forward propagation is
    calculated as

    .. math::

      y_{il} = \sum_{jk} e^1_{ij} e^2_{ik} W_{jkl} + \
        \sum_{j} e^1_{ij} V^1_{jl} + \sum_{k} e^2_{ik} V^2_{kl} + b_{l}.

    If ``nobias`` option is set ``True``, ``Bilinear`` function does
    not have linear parameters, that is, the last three term is omitted
    and only :math:`W` works as the parameter.

    .. note::

       ``Bilinear`` function accepts an input variable of a non-matrix array.
       In this case, the leading dimension is treated as the batch dimension,
       and the other dimensions are reduced to one dimension.

    .. note::

       In the original paper, :math:`J` and :math:`K`
       must be equal and the author denotes :math:`[V^1 V^2]`
       (concatenation of matrices) by :math:`V`.

    Args:
        left_size (int): Dimension of input vector :math:`e^1` (:math:`J`)
        right_size (int): Dimension of input vector :math:`e^2` (:math:`K`)
        out_size (int): Dimension of output vector :math:`y` (:math:`L`)
        nobias (bool): If ``True``, linear parameters are omitted.
        initialW (3-D Array): Initial value of :math:`W`.
            Shape of this argument must be
            ``(left_size, right_size, out_size)``. If ``None``,
            :math:`W` is initialized by centered Gaussian distribution properly
            scaled according to the dimension of inputs and outputs.
        initial_bias (tuple): Intial values of :math:`V^1`, :math:`V^2`
            and :math:`b`. The length this argument must be 3.
            Each element of this tuple must have the shapes of
            ``(left_size, output_size)``, ``(right_size, output_size)``,
            and ``(output_size,)``, respectively. If ``None``, :math:`V^1`
            and :math:`V^2` is initialized by scaled centered Gaussian
            distributions and :math:`b` is set to :math:`0`.

    See:
        `Reasoning With Neural Tensor Networks for Knowledge Base Completion
        <http://papers.nips.cc/paper/5028-reasoning-with-neural-tensor-
        networks-for-knowledge-base-completion>`_ [Socher+, NIPS2013].
    """

    def __init__(self, left_size, right_size, out_size, nobias=False,
                 initialW=None, initial_bias=None):
        super(Bilinear, self).__init__()
        self.in_sizes = (left_size, right_size)
        self.nobias = nobias

        if initialW is not None:
            assert initialW.shape == (left_size, right_size, out_size)
        else:
            # TODO(Kenta OONO): I do not know appropriate way of
            # initializing weights in tensor network.
            # This initialization is a modification of
            # that of Linear function.
            in_size = left_size * right_size * out_size
            initialW = numpy.random.normal(
                0, math.sqrt(1. / in_size), (left_size, right_size, out_size)
            ).astype(numpy.float32)
        self.params['W'] = variable.Variable(initialW)

        if not self.nobias:
            if initial_bias is not None:
                V1, V2, b = initial_bias
                assert V1.shape == (left_size, out_size)
                assert V2.shape == (right_size, out_size)
                assert b.shape == (out_size,)
            else:
                V1 = numpy.random.normal(
                    0, math.sqrt(1. / left_size), (left_size, out_size)
                ).astype(numpy.float32)
                V2 = numpy.random.normal(
                    0, math.sqrt(1. / right_size), (right_size, out_size)
                ).astype(numpy.float32)
                b = numpy.zeros(out_size, dtype=numpy.float32)
            self.params['V1'] = variable.Variable(V1)
            self.params['V2'] = variable.Variable(V2)
            self.params['b'] = variable.Variable(b)

    def __call__(self, e1, e2):
        if self.nobias:
            return bilinear(e1, e2, self.params['W'])
        else:
            return bilinear(e1, e2, self.params['W'], self.params['V1'],
                            self.params['V2'], self.params['b'])
