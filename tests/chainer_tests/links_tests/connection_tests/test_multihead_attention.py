import random
import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import links
from chainer import testing


def _scaled_dot_attn_ref(Q, K, V, dims, unseen_mask=False, src_lengths=None):
    """ Numpy-based reference implementation of scaled dot attention
    for testing"""
    QKT = _batchmatmul(
        Q,
        numpy.transpose(K, axes=[0, 1, 3, 2])
        / numpy.sqrt(dims[3], dtype=numpy.float32),  # divide by sqrt(d_head)
    )
    if unseen_mask or src_lengths is not None:
        b1, b2, s1, s2 = QKT.shape
        # assert s1 == s2
        for i in range(b1):
            for j in range(b2):
                for m in range(s1):
                    for n in range(s2):
                        if unseen_mask and n > m:
                            QKT[i, j, m, n] = -numpy.inf
                        if src_lengths is not None and n >= src_lengths[i]:
                            QKT[i, j, m, n] = -numpy.inf
    reference = _softmax(QKT)
    reference = _batchmatmul(reference, V)
    return reference


def _batchmatmul(a, b):  # batchmatmul over 4 dim matrix
    """ Numpy-based batch matrix multiply over 4 dim matrix"""
    with chainer.using_config('use_cuda', False), \
            chainer.no_backprop_mode():
        retval = functions.matmul(a, b).array
    return retval


def _softmax(x):  # softmax over 4 dim matrix
    """ Numpy-based reference softmax over 4 dim matrix"""
    output = numpy.zeros(x.shape, dtype=numpy.float32)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                x_curr = x[i, j, k, :]
                e_x = numpy.exp(x_curr - numpy.amax(x_curr))
                output[i, j, k, :] = e_x / numpy.sum(e_x)
    return output


def _generate_src_lengths(batch_size, seq_len):
    src_lengths = numpy.array([random.randint(1, seq_len)
                               for i in range(batch_size)])

    # max source length has to equal seq_len, so randomly choose
    # one example to have source length = seq_len
    max_len_example_i = random.randint(0, batch_size - 1)
    src_lengths[max_len_example_i] = seq_len

    src_lengths_tensor = src_lengths.astype(numpy.int32)
    return src_lengths, src_lengths_tensor


def _split_heads_ref(X, dims, nheads, d_head):
    X_split = numpy.reshape(X, dims[:2] + [nheads, d_head])
    X_split_transposed = numpy.transpose(X_split, [0, 2, 1, 3])
    reference = numpy.reshape(
        X_split_transposed, [dims[0], nheads, dims[1], d_head])
    return reference


def _combine_heads_ref(X, dims, nheads, d_head):
    X_transposed = numpy.transpose(X, [0, 2, 1, 3])
    reference = numpy.reshape(X_transposed, dims[:2] + [nheads * d_head])
    return reference


def _fc(X, X_name, module, start=None, end=None):
    X_fc_b = None
    X_fc_w = None
    for name, param in module.namedparams():
        if X_name + "W" in name:
            if X_fc_w is not None:
                raise Exception("Duplicate FC name found")
            X_fc_w = param[start:end, :].array
        elif X_name + "b" in name:
            if X_fc_b is not None:
                raise Exception("Duplicate FC name found")
            X_fc_b = param[start:end].array
    return numpy.matmul(X, numpy.transpose(X_fc_w)) + X_fc_b


def _create_src_lengths_mask(batch_size, src_lengths):
    """
    Generate boolean mask to prevent attention beyond the end of source
    Inumpyuts:
        batch_size : int
        src_lengths : [batch_size] of sentence lengths
    Outputs:
        [batch_size, max_src_len]
    """
    max_srclen = src_lengths.max()
    src_indices = numpy.arange(max_srclen)[
        None, Ellipsis].astype(src_lengths.dtype)
    src_indices = numpy.broadcast_to(src_indices, (batch_size, max_srclen))
    src_lengths = numpy.broadcast_to(
        numpy.expand_dims(src_lengths, 1), (batch_size, max_srclen))
    # returns [batch_size, max_seq_len]
    return (src_indices < src_lengths).astype(numpy.int32)


class TestMultiheadAttention(unittest.TestCase):
    # ref: https://github.com/pytorch/pytorch/blob/163f0e182c7e092d6b61600e3a5e20347abab848/test/test_nn.py#L3240  # NOQA

    def check_multihead_attention(self, use_src_lengths):
        for _ in range(100):
            batch_sz, seq_len = [random.randint(2, 10) for r in range(2)]
            d_head = random.randint(3, 10)
            nheads = random.randint(3, 10)
            d_model = d_head * nheads
            dims = [batch_sz, seq_len, d_model]

            src_lengths = None
            src_lengths_tensor = None
            if use_src_lengths:
                src_lengths, src_lengths_tensor = _generate_src_lengths(
                    batch_size=batch_sz, seq_len=seq_len
                )

            decoder_state = numpy.random.rand(
                batch_sz, d_model).astype(numpy.float32)
            K = numpy.random.rand(*dims).astype(numpy.float32)
            V = K
            Q = numpy.expand_dims(decoder_state, 1)

            decoder_state_tensor = decoder_state.copy()
            source_hid_tensor = numpy.transpose(
                K.copy(), (1, 0) + tuple(range(2, K.ndim)))

            multihead_attn_module = links.MultiHeadAttention(
                nheads, d_model, False,
                source_hid_tensor.shape[-1], source_hid_tensor.shape[-1])

            _batch_size = len(decoder_state_tensor)
            _Q = numpy.expand_dims(decoder_state_tensor, 1)
            _Q = numpy.transpose(_Q, (1, 0) + tuple(range(2, _Q.ndim)))
            _V = source_hid_tensor
            _K = source_hid_tensor
            src_len_mask = None
            if src_lengths is not None and use_src_lengths:
                # [batch_size, 1, seq_len]
                src_len_mask_int = _create_src_lengths_mask(
                    batch_size=_batch_size, src_lengths=src_lengths_tensor
                )
                src_len_mask = src_len_mask_int != 1

            result = multihead_attn_module(
                _Q, _K, _V,
                key_padding_mask=src_len_mask,
                return_weights=True)[0].array
            result = cuda.to_cpu(result).squeeze(0)

            Q_fc = _fc(Q, "/proj_in_", multihead_attn_module, end=d_model)
            K_fc = _fc(
                K, "/proj_in_", multihead_attn_module,
                start=d_model, end=2 * d_model
            )
            V_fc = _fc(V, "/proj_in_", multihead_attn_module,
                       start=2 * d_model)

            Q_split = _split_heads_ref(
                Q_fc, [batch_sz, 1, d_model], nheads, d_head
            )
            K_split = _split_heads_ref(K_fc, dims, nheads, d_head)
            V_split = _split_heads_ref(V_fc, dims, nheads, d_head)

            attn_heads = _scaled_dot_attn_ref(
                Q=Q_split,
                K=K_split,
                V=V_split,
                dims=Q_split.shape,
                src_lengths=src_lengths,
            )

            combined_attn_heads = _combine_heads_ref(
                X=attn_heads, dims=[batch_sz, 1], nheads=nheads, d_head=d_head
            )

            reference = _fc(
                combined_attn_heads, "/out_proj/", multihead_attn_module
            )
            reference = numpy.squeeze(reference, axis=1)

            # result = reference
            assert result.shape == (batch_sz, d_model)
            numpy.testing.assert_allclose(result, reference, atol=1e-5)

    def test_multihead_attn_no_masking(self):
        self.check_multihead_attention(False)

    def test_multihead_attn_with_src_lengths(self):
        self.check_multihead_attention(True)


testing.run_module(__name__, __file__)
