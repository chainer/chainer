import typing as tp

import chainer
from chainer.functions.activation import softmax
from chainer.functions.array import concat
from chainer.functions.array import expand_dims
from chainer.functions.array import split_axis
from chainer.functions.array import reshape
from chainer.functions.array import transpose
from chainer.functions.array import tile
from chainer.functions.array import where
from chainer.functions.math import average
from chainer.functions.math import matmul
from chainer.functions.noise import dropout
from chainer.functions.connection import linear
from chainer import types
from chainer import variable


InputType = tp.Union[variable.Variable, types.NdArray]


def _generate_zeros(device, shape, dtype):
    with chainer.using_device(device):
        return device.xp.zeros(shape=shape, dtype=dtype)


def _generate_ones(device, shape, dtype):
    with chainer.using_device(device):
        return device.xp.ones(shape=shape, dtype=dtype)


def multi_head_attention(
        n_heads,                   # type: int
        embedding_size,            # type: int
        query,                     # type: InputType
        key,                       # type: InputType,
        value,                     # type: InputType,
        in_proj_W,                 # type: tp.Union[variable.Variable, tp.Tuple[variable.Variable, variable.Variable, variable.Variable]],  # NOQA
        in_proj_b,                 # type:  tp.Optional[variable.Variable]
        bias_k,                    # type: tp.Optional[variable.Variable]
        bias_v,                    # type: tp.Optional[variable.Variable]
        out_proj_W,                # type: variable.Variable
        out_proj_b,                # type: variable.Variable
        add_zero_attention=False,  # type:  bool
        attention_dropout=0.0,     # type:  float
        post_dropout=0.0,          # type: float
        key_padding_mask=None,     # type: tp.Optional[InputType]
        attention_mask=None,       # type: tp.Optional[InputType]
        dot_product_scaler=None,   # type: tp.Optional[float]
        softmax_scaler=1.0,        # type: float
        return_weights=True        # type: bool
):
    # type: (...) -> tp.Tuple[variable.Variable, variable.Variable]
    """Multi-head Attention forward function.

    Args:
        query (:class:`~chainer.Variable` or :ref:`ndarray`):
            A batch of query vectors.
        key (:class:`~chainer.Variable` or :ref:`ndarray`)
            A batch of key vectors.
        value (:class:`~chainer.Variable` or :ref:`ndarray`)
            A batch of value vectors.
        n_heads (int): The number of parallel attention heads.
        in_proj_W (:obj:`tuple`, :class:`~chainer.Variable` or :ref:`ndarray`):
            Weight(s) to project ``query``, ``key``, and ``value`` vectors.
            If three inputs have different size, this ``in_proj_W`` should be
            a tuple of three weights (:class:`~chainer.Variable`\\s).
        in_proj_b (:class:`~chainer.Variable` or :ref:`ndarray`):
            Bias added to projected `query`, `key`, and `value` vectors.
        bias_k (:class:`~chainer.Variable`, :ref:`ndarray`, or ``None``):
            A bias concatenated to input ``key``.
        bias_v (:class:`~chainer.Variable`, :ref:`ndarray`, or ``None``):
            A bias concatenated to input ``value``.
        add_zero_attention (bool): If ``True``, add a new batch of zeros to
            the key and value sequences at axis=1.
        attention_dropout (float): Dropout ratio at the attention layer.
        post_dropout (float): Dropout ratio at the output.
        out_proj_W (:class:`~chainer.Variable` or :ref:`ndarray`):
            Weight to project attention.
        out_proj_b (:class:`~chainer.Variable` or :ref:`ndarray`):
            Bias for projected attention.
        key_padding_mask (:class:`~chainer.Variable` or :ref:`ndarray`):
            If not ``None``, specified padding elements in the key
            will be ignored by the attention.
        attention_mask (:class:`~chainer.Variable` or :ref:`ndarray`):
            This is a mask that helps attention to ignore certain positions.
            Masking is done by adding ``-inf`` to the elements to be ignored
            (:math:`\\exp(-\\text{inf}) = 0`).
        dot_product_scaler: (float): Scaler for dot product. If ``None``,
            :math:`1 / \\sqrt{embedding_size / n_heads}` is used.
        softmax_scaler (float): Softmax smoothing, or sharpening, coefficient.
            This value is for cuDNN implementation.
        return_weights (bool): If ``True``, return averaged attention weights.

    Returns:
        tuple: This function returns a tuple containing ``attention_output``
        and ``attention_output_weights``.

        - ``attention_output`` is the output of attention.
        - ``attention_output_weights`` is the weights of attention.

    Shape of Inputs:
        - query: :math:`(L, B, E)` where :math:`L` is
          the target sequence length, :math:`B` is the batch size,
          :math:`E` is the embedding size (same as the ``embedding_size``
          argument of :class:`~chainer.links.MultiHeadAttention`).
        - key: :math:`(S, B, E_{\\rm key})`, where :math:`S` is
          the source sequence length, :math:`B` is the batch size,
          :math:`E_{\\rm key}` is the size of a vector of ``key``.
        - value: :math:`(S, B, E_{\\rm value})` where :math:`S` is
          the source sequence length, :math:`B` is the batch size,
          :math:`E_{\\rm value}` is the embedding size.
        - in_proj_weight: If this is a :class:`~chainer.Variable`,
          :math:`(3E, E)`. Otherwise, this is a tuple of weights and they have
          the shape of :math:`(E, E)`, :math:`(E, E_{\\rm key})`, and
          :math:`(E, E_{\\rm value})`, respectively.
        - in_proj_b: :math:`3E`. This is divided into three vectors and they
          are added to projected ``query``, ``key``, and ``value``,
          respectively.
        - bias_k: :math:`L`.
        - bias_v: :math:`L`.
        - out_proj_W: :math:`(E, E)`.
        - out_proj_b: :math:`E`.
        - key_padding_mask: :math:`(B, S)` where :math:`B` is the batch size,
          :math:`S` is the source sequence length.
        - attention_mask: :math:`(L, S)` where :math:`L` is the target sequence
          length, :math:`S` is the source sequence length.

    Shape of Outputs:
        - attention_output: :math:`(L, B, E)` where :math:`L` is the target
          sequence length, :math:`B` is the batch size, :math:`E` is
          the embedding dimension.
        - attention_output_weights: :math:`(B, L, S)` where :math:`B` is
          the batch size, :math:`L` is the target sequence length,
          :math:`S` is the source sequence length.

    .. seealso:: :class:`~chainer.links.MultiHeadAttention`

    """
    chainer.utils.experimental('chainer.functions.multi_head_attention')

    are_different = isinstance(in_proj_W, tuple)

    def _in_proj(x, start=0, end=None, weight_idx=None):
        if are_different:
            W = in_proj_W[weight_idx]
        else:
            W = in_proj_W[start:end, :]
        if in_proj_b is not None:
            b = in_proj_b[start:end]
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
            key, embedding_size, 2 * embedding_size, weight_idx=1)

    def _in_proj_v(value):
        return _in_proj(value, start=2 * embedding_size, weight_idx=2)

    if embedding_size % n_heads != 0:
        raise ValueError(
            '`embedding_size` ({}) need to be '.format(embedding_size) +
            'divisible by `n_heads` ({})'.format(embedding_size, n_heads))
    if (bias_k is None) != (bias_v is None):
        _msg_fmt = 'bias for {} is not `None` while that for {} is `None`'
        if bias_v is None:
            msg = _msg_fmt.format('`key`', '`value`')
        else:
            msg = _msg_fmt.format('`value`', '`key`')
        raise ValueError(msg)
    qkv_same = (query is key) and (query is value)
    kv_same = key is value
    target_length, batch_size, embedding_size = query.shape
    head_size = embedding_size // n_heads
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
        k = concat.concat(
            (k, tile.tile(bias_k, (1, batch_size, 1))), axis=0)
        v = concat.concat(
            (v, tile.tile(bias_v, (1, batch_size, 1))), axis=0)
        if attention_mask is not None:
            attention_mask = concat.concat(
                (
                    attention_mask,
                    _generate_zeros(
                        device, (len(attention_mask), 1), dtype)
                )
            )
        if key_padding_mask is not None:
            key_padding_mask = concat.concat(
                (
                    key_padding_mask,
                    _generate_zeros(
                        device, (len(key_padding_mask), 1),
                        key_padding_mask.dtype
                    )
                )
            )
    q = reshape.reshape(q, (target_length, batch_size * n_heads, head_size))
    q = transpose.transpose(q, (1, 0, 2))
    if k is not None:
        k = reshape.reshape(k, (-1, batch_size * n_heads, head_size))
        k = transpose.transpose(k, (1, 0, 2))
        v = reshape.reshape(v, (-1, batch_size * n_heads, head_size))
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

    if add_zero_attention:
        source_length += 1
        k = concat.concat((
            k,
            _generate_zeros(device, (len(k), 1) + k.shape[2:], dtype)
        ))
        v = concat.concat((
            v,
            _generate_zeros(device, (len(v), 1) + v.shape[2:], dtype)
        ))
        if attention_mask is not None:
            attention_mask = concat.cocnat((
                attention_mask,
                _generate_zeros(device, (len(attention_mask), 1), dtype)
            ))
        if key_padding_mask is not None:
            key_padding_mask = concat.concat(
                (
                    key_padding_mask,
                    _generate_zeros(
                        device, (len(key_padding_mask), 1),
                        key_padding_mask.dtype)
                )
            )
    attention_output_weights = matmul.matmul(
        q, transpose.transpose(k, (0, 2, 1)))
    if (attention_output_weights.shape !=
            (batch_size * n_heads, target_length, source_length)):
        raise ValueError('`attention_output_weights` is shaped wrongly')

    if attention_mask is not None:
        attention_mask = expand_dims.expand_dims(attention_mask, 0)
        attention_output_weights += attention_mask

    if key_padding_mask is not None:
        attention_output_weights = reshape.reshape(
            attention_output_weights,
            (batch_size, n_heads, target_length, source_length)
        )
        expanded_mask = expand_dims.expand_dims(
            expand_dims.expand_dims(key_padding_mask, 1), 2)
        attention_output_weights = where.where(
            expanded_mask,
            -xp.inf * _generate_ones(
                device, attention_output_weights.shape, dtype),
            attention_output_weights
        )
        attention_output_weights = reshape.reshape(
            attention_output_weights,
            (batch_size * n_heads, target_length, source_length)
        )

    attention_output_weights = softmax.softmax(
        attention_output_weights, axis=-1)
    if attention_dropout > 0.0:
        attention_output_weights = dropout.dropout(
            attention_output_weights, attention_dropout)

    attention_output = matmul.matmul(attention_output_weights, v)
    attention_output = transpose.transpose(
        attention_output,
        (1, 0) + tuple(range(2, attention_output.ndim))
    )
    attention_output = reshape.reshape(
        attention_output, (target_length, batch_size, embedding_size))
    attention_output = linear.linear(
        attention_output, out_proj_W, out_proj_b,
        n_batch_axes=attention_output.ndim-1)
    if post_dropout > 0.0:
        attention_output = dropout.dropout(attention_output, post_dropout)

    if return_weights:
        attention_output_weights = reshape.reshape(
            attention_output_weights,
            (batch_size, n_heads, target_length, source_length)
        )
        attention_output_weights = average.average(
            attention_output_weights, axis=1)
    else:
        attention_output_weights = None
    return attention_output, attention_output_weights
