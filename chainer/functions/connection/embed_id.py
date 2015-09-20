import numpy

from chainer import cuda
from chainer import function
from chainer import parameterized
from chainer.utils import type_check
from chainer import variable


class EmbedIDFunction(function.Function):

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        x_type, w_type = in_types
        type_check.expect(
            x_type.dtype == numpy.int32,
            x_type.ndim == 1,
        )
        type_check.expect(
            w_type.dtype == numpy.float32,
            w_type.ndim == 2
        )

    def forward(self, inputs):
        x, W = inputs
        return W.take(x, axis=0),

    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        x, W = inputs
        gy = grad_outputs[0]
        gW = xp.zeros_like(W)

        if xp is numpy:
            numpy.add.at(gW, x, gy)
        else:
            cuda.elementwise(
                'T gy, int32 x, int32 n_out', 'raw T gW',
                'int w_ind[] = {x, i % n_out}; atomicAdd(&gW[w_ind], gy)',
                'embed_id_bwd')(gy, x[:, None], gW.shape[1], gW)
        return None, gW


def embed_id(x, W):
    """Efficient linear function for one-hot input.

    Args:
        x (~chainer.Variable): Input variable with one-hot representation.
        W (~chainer.Variable): Representation of each ID (a.k.a.
            word embeddings).

    Returns:
        ~chainer.Variable: Output variable.

    .. seealso:: :class:`EmbedID`

    """
    return EmbedIDFunction()(x, W)


class EmbedID(parameterized.ParameterizedObject):

    """Efficient linear function for one-hot input.

    This is a parameterized function to embed the given discrete identifier
    (e.g. word) into a continuous vector space. This function just holds
    embedding vectors for all identifiers as one large matrix ``W``, which is
    learnable. The identifiers are directly used as indexes of the matrix
    ``W``.

    Args:
        in_size (int): Number of different identifiers (a.k.a. vocabulary
            size).
        out_size (int): Size of embedding vector.

    .. note::

       This function is non-differentiable with respect to the input
       identifiers.

    """
    def __init__(self, in_size, out_size):
        super(EmbedID, self).__init__()
        self.params['W'] = variable.Variable(numpy.random.randn(
            in_size, out_size).astype(numpy.float32))

    def __call__(self, x):
        return embed_id(x, self.params['W'])
