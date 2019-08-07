import typing as tp  # NOQA

import chainer
from chainer.functions.activation import softmax
from chainer.functions.array import concat
from chainer.functions.array import expand_dims
from chainer.functions.array import split_axis
from chainer.functions.array import repeat
from chainer.functions.array import reshape
from chainer.functions.array import transpose
from chainer.functions.array import where
from chainer.functions.math import average
from chainer.functions.math import matmul
from chainer.functions.noise import dropout
from chainer.functions.connection import linear
from chainer import types
from chainer import variable


InputType = tp.Union[variable.Variable, types.NdArray]


def generate_zeros(device, shape, dtype):
    with chainer.using_device(device):
        return device.xp.zeros(shape=shape, dtype=dtype)


def generate_ones(device, shape, dtype):
    with chainer.using_device(device):
        return device.xp.ones(shape=shape, dtype=dtype)


def multihead_attention(
    n_head,                   # type: int
    embedding_size,           # type: int
    query,                    # type: InputType
    key,                      # type: InputType
    value,                    # type: InputType
    proj_in_W,                # type: tp.Union[variable.Variable, tp.Tuple[variable.Variable, variable.Variable, variable.Variable]]  # NOQA
    proj_in_b,                # type: tp.Optional[variable.Variable]
    bias_k,                   # type: variable.Variable
    bias_v,                   # type: variable.Variable
    proj_out_W,               # type: variable.Variable
    proj_out_b,               # type: variable.Variable
    add_zero_attn=False,      # type: bool
    attention_dropout=0,      # type: float
    post_dropout=0,           # type: float
    key_padding_mask=None,    # type: tp.Optional[InputType]
    attn_mask=None,           # type: tp.Optional[InputType]
    dot_product_scaler=None,  # type: tp.Optional[float]
    softmax_scaler=1.0,       # type: float
    return_weights=True       # type: bool
):
    # type: (...) -> tp.Tuple[variable.Variable, variable.Variable]  # NOQA
    """Multi-head Attention forward function.

    Args:
        query (:class:`~chainer.Variable` or :ref:`ndarray`):
            A batch of query vectors whose shape is :math:`(L, B, E)` where
            :math:`L` is the target sequence length, :math:`B` is the
            batch size, and :math:`E` is the embedding size.
        key (:class:`~chainer.Variable` or :ref:`ndarray`)
            A batch of key vectors whose shape is :math:`(S, B, E)` where
            :math:`S` is the source sequence length, :math:`B` is the
            batch size, and :math:`E` is the embedding size.
        value (:class:`~chainer.Variable` or :ref:`ndarray`)
            A batch of value vectors whose shape is :math:`(S, B, E)` where
            :math:`S` is the source sequence length, :math:`B` is the
            batch size, and :math:`E` is the embedding size.
        expected_embedding_size (int): Total number of units of the model.
        n_head (int): The number of parallel attention heads.
        proj_in_W (:obj:`tuple`, :class:`~chainer.Variable` or :ref:`ndarray`):
            Weight(s) to project `query`, `key`, and `value` vectors.
            If the input sizes of `query`, `key`, and `value` are different,
            this should be the tuple of three weights, otherwise, one weight.
        proj_in_b (:class:`~chainer.Variable` or :ref:`ndarray`):
            Bias added to projected `query`, `key`, and `value` vectors.
        add_zero_attn (bool): If ``True``, add a new batch of zeros to
            the key and value sequences at axis=1.
        attention_dropout (float): Dropout ratio at the attention layer.
        post_dropout (float): Dropout ratio at the output.
        proj_out_W (:class:`~chainer.Variable` or :ref:`ndarray`):
            Weight to project attention.
        proj_out_b (:class:`~chainer.Variable` or :ref:`ndarray`):
            Bias for projected attention.
        key_padding_mask (:class:`~chainer.Variable` or :ref:`ndarray`):
            If not ``None``, specified padding elements in the key
            will be ignored by the attention.
            The shape is :math:`(B, S)` where :math:`B` is the batch size,
            and :math:`S` is the source sequence length.
        attn_mask (:class:`~chainer.Variable` or :ref:`ndarray`):
            Mask help attention ignores certain positions.
            The shape is :math:`(L, L)` where :math:`L` is
            the target sequence length.
        dot_product_scaler: (float): Scaler for dot product. If ``None``,
            :math:`1 / \\sqrt{embedding_size / n_head}` is used.
        softmax_scaler (float): Softmax smoothing, or sharpening, coefficient.
            This value is for cuDNN implementation.
        return_weights (bool): If ``True``, return averaged attention weights.

    Returns:
        tuple: This function returns a tuple containing ``attn_output`` and
        ``attn_output_weights``.

        - ``attn_output`` is the output of attention whose shape is
          :math:`(L, B, E)` where :math:`L` is the target sequence length,
          :math:`B` is the batch size, and :math:`E` is the embedding size.
        - ``attn_output_weights`` is the weights of attention whose shape is
          :math:`(B, L, S)` where :math:`B` is the batch size,
          :math:`L` is the target sequence length,
          and :math:`S` is the source sequence length. If ``return_weights`` is
          ``False``, this return value is ``None``.

    .. seealso:: :class:`~chainer.links.MultiHeadAttention`

    """

    are_different = isinstance(proj_in_W, tuple)

    def _in_proj(x, start=0, end=None, weight_idx=None):
        if are_different:
            W = proj_in_W[weight_idx]
        else:
            W = proj_in_W[start:end, :]
        if proj_in_b is not None:
            b = proj_in_b[start:end]
        return linear.linear(x, W, b, n_batch_axes=x.ndim-1)

    def _in_proj_qkv(query):
        return split_axis.split_axis(_in_proj(query), 3, axis=-1)

    def _in_proj_kv(key):
        return split_axis.split_axis(
            _in_proj(key, embedding_size), 2, axis=-1)

    def _in_proj_q(query):
        return _in_proj(query, end=embedding_size, weight_idx=0)

    def _in_proj_k(key):
        return _in_proj(
            value, embedding_size, 2 * embedding_size, weight_idx=1)

    def _in_proj_v(value):
        return _in_proj(value, start=2 * embedding_size, weight_idx=2)

    if embedding_size % n_head != 0:
        raise ValueError(
            '`embedding_size` ({}) need to be '.format(embedding_size) +
            'divisible by `n_head` ({})'.format(embedding_size, n_head))
    if (bias_k is None) != (bias_v is None):
        raise ValueError
    qkv_same = (query is key) and (query is value)
    kv_same = key is value
    target_length, batch_size, embedding_size = query.shape
    head_size = embedding_size // n_head
    if dot_product_scaler is None:
        dot_product_scaler = head_size ** -0.5

    device = chainer.backend.get_device_from_array(query)
    xp = device.xp
    dtype = query.dtype

    if qkv_same:
        # self-attention
        q, k, v = _in_proj_qkv(query)
    elif kv_same:
        q = _in_proj_q(query)
        if key is None:
            k, v = None, None
        else:
            k, v = _in_proj_kv(key)
    else:
        q = _in_proj_q(query)
        k = _in_proj_k(key)
        v = _in_proj_v(value)
    q *= dot_product_scaler

    if bias_k is not None:
        k = concat.concat((k, repeat.repeat(bias_k, batch_size, axis=1)))
        v = concat.concat((v, repeat.repeat(bias_v, batch_size, axis=1)))
        if attn_mask is not None:
            attn_mask = concat.concat(
                (
                    attn_mask,
                    generate_zeros(
                        device, (len(attn_mask), 1), dtype)
                )
            )
        if key_padding_mask is not None:
            key_padding_mask = concat.concat(
                (
                    key_padding_mask,
                    generate_zeros(
                        device, (len(key_padding_mask), 1),
                        key_padding_mask.dtype
                    )
                )
            )
    q = reshape.reshape(q, (target_length, batch_size * n_head, head_size))
    q = transpose.transpose(q, (1, 0, 2))
    if k is not None:
        k = reshape.reshape(k, (-1, batch_size * n_head, head_size))
        k = transpose.transpose(k, (1, 0, 2))
        v = reshape.reshape(v, (-1, batch_size * n_head, head_size))
        v = transpose.transpose(v, (1, 0, 2))

    # TODO(crcrpar): Investigate the possibility that
    # the input of `key` is `None` and `bias_k` is also `None`
    source_length = k.shape[1]

    if key_padding_mask is not None:
        if key_padding_mask.shape[:2] != (batch_size, source_length):
            raise ValueError(
                '`key_padding_mask` has wrong shape. '
                'Expected: ({}, {}), Actual: ({}, {})'.format(
                    batch_size, source_length,
                    key_padding_mask.shape[0], key_padding_mask.shape[1]))

    if add_zero_attn:
        source_length += 1
        k = concat.concat((
            k,
            generate_zeros(device, (len(k), 1) + k.shape[2:], dtype)
        ))
        v = concat.concat((
            v,
            generate_zeros(device, (len(v), 1) + v.shape[2:], dtype)
        ))
        if attn_mask is not None:
            attn_mask = concat.cocnat((
                attn_mask,
                generate_zeros(device, (len(attn_mask), 1), dtype)
            ))
        if key_padding_mask is not None:
            key_padding_mask = concat.concat(
                (
                    key_padding_mask,
                    generate_zeros(
                        device, (len(key_padding_mask), 1),
                        key_padding_mask.dtype)
                )
            )
    attn_output_weights = matmul.matmul(
        q, transpose.transpose(k, (0, 2, 1)))
    if (attn_output_weights.shape !=
            (batch_size * n_head, target_length, source_length)):
        raise ValueError('`attn_output_weights` is shaped wrongly')

    if attn_mask is not None:
        attn_mask = expand_dims.expand_dims(attn_mask, 0)
        attn_output_weights += attn_mask

    if key_padding_mask is not None:
        attn_output_weights = reshape.reshape(
            attn_output_weights,
            (batch_size, n_head, target_length, source_length)
        )
        expanded_mask = expand_dims.expand_dims(
            expand_dims.expand_dims(key_padding_mask, 1), 2)
        attn_output_weights = where.where(
            expanded_mask,
            -xp.inf * generate_ones(
                device, attn_output_weights.shape, dtype),
            attn_output_weights
        )
        attn_output_weights = reshape.reshape(
            attn_output_weights,
            (batch_size * n_head, target_length, source_length)
        )

    attn_output_weights = softmax.softmax(attn_output_weights, axis=-1)
    if attention_dropout > 0.0:
        attn_output_weights = dropout.dropout(
            attn_output_weights, attention_dropout)

    attn_output = matmul.matmul(attn_output_weights, v)
    attn_output = transpose.transpose(
        attn_output,
        (1, 0) + tuple(range(2, attn_output.ndim))
    )
    attn_output = reshape.reshape(
        attn_output, (target_length, batch_size, embedding_size))
    attn_output = linear.linear(
        attn_output, proj_out_W, proj_out_b,
        n_batch_axes=attn_output.ndim-1)
    if post_dropout > 0.0:
        attn_output = dropout.dropout(attn_output, post_dropout)

    if return_weights:
        attn_output_weights = reshape.reshape(
            attn_output_weights,
            (batch_size, n_head, target_length, source_length)
        )
        attn_output_weights = average.average(attn_output_weights, axis=1)
    else:
        attn_output_weights = None
    return attn_output, attn_output_weights
