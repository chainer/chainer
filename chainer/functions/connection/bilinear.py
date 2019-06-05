import numpy

import chainer
from chainer import backend
from chainer import function_node
from chainer.utils import type_check


def _as_mat(x):
    if x.ndim == 2:
        return x
    return x.reshape(len(x), -1)


def _ij_ik_il_to_jkl(a, b, c):
    ab = chainer.functions.matmul(a[:, :, None], b[:, None, :])  # ijk
    return chainer.functions.matmul(_as_mat(ab).T, c).reshape(
        a.shape[1], b.shape[1], c.shape[1])


def _ij_ik_jkl_to_il(a, b, c):
    ab = chainer.functions.matmul(a[:, :, None], b[:, None, :])  # ijk
    c = c.reshape(-1, c.shape[-1])  # [jk]l
    return chainer.functions.matmul(_as_mat(ab), c)


def _ij_il_jkl_to_ik(a, b, c):
    return _ij_ik_jkl_to_il(a, b, chainer.functions.swapaxes(c, 1, 2))


def _ik_il_jkl_to_ij(a, b, c):
    return _ij_ik_jkl_to_il(a, b, chainer.functions.rollaxis(c, 0, c.ndim))


class BilinearFunction(function_node.FunctionNode):
    def check_type_forward(self, in_types):
        n_in = type_check.eval(in_types.size())
        if n_in != 3 and n_in != 6:
            raise type_check.InvalidType(
                '{0} or {1}'.format(
                    in_types.size() == 3, in_types.size() == 6),
                '{0} == {1}'.format(in_types.size(), n_in))

        e1_type, e2_type, W_type = in_types[:3]
        type_check_prod = type_check.make_variable(numpy.prod, 'prod')
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
        self.retain_inputs(tuple(range(len(inputs))))

        e1 = _as_mat(inputs[0])
        e2 = _as_mat(inputs[1])
        W = inputs[2]

        xp = backend.get_array_module(*inputs)
        # optimize: y = xp.einsum('ij,ik,jkl->il', e1, e2, W)
        y = xp.tensordot(xp.einsum('ij,ik->ijk', e1, e2), W, axes=2)

        if len(inputs) == 6:
            V1, V2, b = inputs[3:]
            y += e1.dot(V1)
            y += e2.dot(V2)
            y += b
        return y,

    def backward(self, indexes, grad_outputs):
        inputs = self.get_retained_inputs()
        e1, e2, W = inputs[:3]
        gy, = grad_outputs

        if len(inputs) == 6:
            V1, V2 = inputs[3], inputs[4]
            return BilinearFunctionGrad().apply((e1, e2, W, V1, V2, gy))
        return BilinearFunctionGrad().apply((e1, e2, W, gy))


class BilinearFunctionGrad(function_node.FunctionNode):

    def forward(self, inputs):
        self.retain_inputs(tuple(range(len(inputs))))

        e1 = _as_mat(inputs[0])
        e2 = _as_mat(inputs[1])
        W, gy = inputs[2], inputs[-1]

        xp = backend.get_array_module(*inputs)
        # optimize: gW = xp.einsum('ij,ik,il->jkl', e1, e2, gy)
        gW = xp.einsum('ij,ik->jki', e1, e2).dot(gy)

        gy_W = xp.tensordot(gy, W, axes=(1, 2))  # 'il,jkl->ijk'
        # optimize: ge1 = xp.einsum('ik,jkl,il->ij', e2, W, gy)
        ge1 = xp.einsum('ik,ijk->ij', e2, gy_W)
        # optimize: ge2 = xp.einsum('ij,jkl,il->ik', e1, W, gy)
        ge2 = xp.einsum('ij,ijk->ik', e1, gy_W)

        ret = ge1.reshape(inputs[0].shape), ge2.reshape(inputs[1].shape), gW

        if len(inputs) == 6:
            V1, V2 = inputs[3], inputs[4]
            gV1 = e1.T.dot(gy)
            gV2 = e2.T.dot(gy)
            gb = gy.sum(0)
            ge1 += gy.dot(V1.T)
            ge2 += gy.dot(V2.T)
            ret += gV1, gV2, gb

        return ret

    def backward(self, indexes, grad_outputs):
        inputs = self.get_retained_inputs()

        e1 = _as_mat(inputs[0])
        e2 = _as_mat(inputs[1])
        W, gy = inputs[2], inputs[-1]

        gge1 = _as_mat(grad_outputs[0])
        gge2 = _as_mat(grad_outputs[1])
        ggW = grad_outputs[2]

        dge1_de2 = _ij_il_jkl_to_ik(gge1, gy, W)
        dge1_dW = _ij_ik_il_to_jkl(gge1, e2, gy)
        dge1_dgy = _ij_ik_jkl_to_il(gge1, e2, W)

        dge2_de1 = _ik_il_jkl_to_ij(gge2, gy, W)
        dge2_dW = _ij_ik_il_to_jkl(e1, gge2, gy)
        dge2_dgy = _ij_ik_jkl_to_il(e1, gge2, W)

        dgW_de1 = _ik_il_jkl_to_ij(e2, gy, ggW)
        dgW_de2 = _ij_il_jkl_to_ik(e1, gy, ggW)
        dgW_dgy = _ij_ik_jkl_to_il(e1, e2, ggW)

        ge1 = dgW_de1 + dge2_de1
        ge2 = dgW_de2 + dge1_de2
        gW = dge1_dW + dge2_dW
        ggy = dgW_dgy + dge1_dgy + dge2_dgy

        if len(inputs) == 6:
            V1, V2 = inputs[3], inputs[4]
            ggV1, ggV2, ggb = grad_outputs[3:]

            gV1 = chainer.functions.matmul(gge1, gy, transa=True)
            gV2 = chainer.functions.matmul(gge2, gy, transa=True)

            ge1 += chainer.functions.matmul(gy, ggV1, transb=True)
            ge2 += chainer.functions.matmul(gy, ggV2, transb=True)
            ggy += chainer.functions.matmul(gge1, V1)
            ggy += chainer.functions.matmul(gge2, V2)
            ggy += chainer.functions.matmul(e1, ggV1)
            ggy += chainer.functions.matmul(e2, ggV2)
            ggy += chainer.functions.broadcast_to(ggb, ggy.shape)

        ge1 = ge1.reshape(inputs[0].shape)
        ge2 = ge2.reshape(inputs[1].shape)

        if len(inputs) == 6:
            return ge1, ge2, gW, gV1, gV2, ggy
        return ge1, ge2, gW, ggy


def bilinear(e1, e2, W, V1=None, V2=None, b=None):
    """Applies a bilinear function based on given parameters.

    This is a building block of Neural Tensor Network (see the reference paper
    below). It takes two input variables and one or four parameters, and
    outputs one variable.

    To be precise, denote six input arrays mathematically by
    :math:`e^1\\in \\mathbb{R}^{I\\cdot J}`,
    :math:`e^2\\in \\mathbb{R}^{I\\cdot K}`,
    :math:`W\\in \\mathbb{R}^{J \\cdot K \\cdot L}`,
    :math:`V^1\\in \\mathbb{R}^{J \\cdot L}`,
    :math:`V^2\\in \\mathbb{R}^{K \\cdot L}`, and
    :math:`b\\in \\mathbb{R}^{L}`,
    where :math:`I` is mini-batch size.
    In this document, we call :math:`V^1`, :math:`V^2`, and :math:`b` linear
    parameters.

    The output of forward propagation is calculated as

    .. math::

      y_{il} = \\sum_{jk} e^1_{ij} e^2_{ik} W_{jkl} + \\
        \\sum_{j} e^1_{ij} V^1_{jl} + \\sum_{k} e^2_{ik} V^2_{kl} + b_{l}.

    Note that V1, V2, b are optional. If these are not given, then this
    function omits the last three terms in the above equation.

    .. note::

       This function accepts an input variable ``e1`` or ``e2`` of a non-matrix
       array. In this case, the leading dimension is treated as the batch
       dimension, and the other dimensions are reduced to one dimension.

    .. note::

       In the original paper, :math:`J` and :math:`K`
       must be equal and the author denotes :math:`[V^1 V^2]`
       (concatenation of matrices) by :math:`V`.

    Args:
        e1 (:class:`~chainer.Variable` or :ref:`ndarray`):
            Left input variable.
        e2 (:class:`~chainer.Variable` or :ref:`ndarray`):
            Right input variable.
        W (:class:`~chainer.Variable` or :ref:`ndarray`):
            Quadratic weight variable.
        V1 (:class:`~chainer.Variable` or :ref:`ndarray`):
            Left coefficient variable.
        V2 (:class:`~chainer.Variable` or :ref:`ndarray`):
            Right coefficient variable.
        b (:class:`~chainer.Variable` or :ref:`ndarray`):
            Bias variable.

    Returns:
        ~chainer.Variable: Output variable.

    See:
        `Reasoning With Neural Tensor Networks for Knowledge Base Completion
        <https://papers.nips.cc/paper/5028-reasoning-with-neural-tensor-
        networks-for-knowledge-base-completion>`_ [Socher+, NIPS2013].

    .. seealso::

        :class:`~chainer.links.Bilinear` to manage the model parameters
        ``W``, ``V1``, ``V2``, and ``b``.

    """
    flags = [V1 is None, V2 is None, b is None]
    if any(flags):
        if not all(flags):
            raise ValueError('All coefficients and bias for bilinear() must '
                             'be None, if at least one of them is None.')
        return BilinearFunction().apply((e1, e2, W))[0]
    return BilinearFunction().apply((e1, e2, W, V1, V2, b))[0]
