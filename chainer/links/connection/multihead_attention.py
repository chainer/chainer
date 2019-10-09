import typing as tp

import chainer
from chainer import functions
from chainer import initializers
from chainer import link
from chainer import links
from chainer import types
from chainer import variable


InputType = tp.Union[variable.Variable, types.NdArray]


class MultiHeadAttention(link.Chain):

    """Multi-Head Attention.

    Args:
        n_head (int): The number of heads.
        embedding_size (int):
            The size of input query vectors
            and projected query, key, and value vectors.
        self_attention (bool): If ``True``, this becomes self-attention.
        ksize (int):
            The size of input key vectors.
        vsize (int):
            The size of input value vectors.
        attention_dropout (float):
            The dropout ratio applied to attention before softmax.
        post_dropout (float):
            The dropout ratio applied to attention after softmax.
        scaler (float): The scaler value that defaults to
            :math:`1/\\sqrt{n_{\\text{head}}}`.
        softmax_scaler (float): Softmax smoothing, or sharpening, coefficient.
            The default value is 1.0.
        initialW (:ref:`initializer <initializer>`): Initializer to initialize
            the weight.
        initial_bias (:ref:`initializer <initializer>`): Initializer to
            initialize the bias. If ``None``, the bias will be initialized to
            zero.
        nobias (bool): Whether to add bias to projected query, key, and value
            if the instance of this class is used as self-attention.
        nobias_kv (bool):
            If ``True``, no bias is added to projected key and value.

    See: `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_

    """

    def __init__(
            self,
            n_head: int,
            embedding_size: int,
            self_attention: bool = False,
            ksize: tp.Optional[int] = None,
            vsize: tp.Optional[int] = None,
            attention_dropout: float = 0.0,
            post_dropout: float = 0.0,
            scaler: tp.Optional[float] = None,
            softmax_scaler: tp.Optional[float] = None,
            initialW: tp.Optional[types.InitializerSpec] = None,
            initial_bias: tp.Optional[types.InitializerSpec] = None,
            nobias: bool = False,
            nobias_kv: bool = True
    ) -> None:
        super().__init__()

        if embedding_size % n_head != 0:
            raise ValueError(
                '`embedding_size` ({}) must be '.format(embedding_size) +
                'divisible by `n_head` ({})'.format(n_head))
        if not self_attention and (ksize is None or vsize is None):
            raise ValueError(
                '`ksize` and `vsize` are required '
                'if `self_attention` is `False`.')
        else:
            ksize = embedding_size
            vsize = embedding_size
        self.n_head = n_head
        self.embedding_size = embedding_size  # == qsize
        self.head_size = self.embedding_size // self.n_head
        self._self_attention = self_attention
        self.scaler = scaler
        self.softmax_scaler = softmax_scaler

        if self._self_attention:
            ksize = self.embedding_size
            vsize = self.embedding_size
        self.ksize = ksize
        self.vsize = vsize
        self.qkv_same_size = (
            self.embedding_size == self.ksize
            and self.embedding_size == self.vsize
        )

        self.attention_dropout = attention_dropout
        self.post_dropout = post_dropout

        with self.init_scope():
            if initialW is None:
                _initialW = initializers.GlorotNormal()
            if initial_bias is None:
                _initial_bias = initializers.Zero()
            if self.qkv_same_size:
                self.proj_in_W: variable.Variable = variable.Parameter(
                    _initialW, (3 * self.embedding_size, self.embedding_size))
            else:
                self.proj_q_weight: variable.Variable = variable.Parameter(
                    _initialW, (self.embedding_size, self.embedding_size))
                self.proj_k_weight: variable.Variable = variable.Parameter(
                    _initialW, (self.embedding_size, self.ksize))
                self.proj_v_weight: variable.Variable = variable.Parameter(
                    _initialW, (self.embedding_size, self.vsize))
            if not nobias:
                self.proj_in_b: variable.Variable = variable.Parameter(
                    _initial_bias, (3 * self.embedding_size,))
            else:
                self.proj_in_b = None

            self.out_proj = links.Linear(
                self.embedding_size, self.embedding_size,
                initialW=_initialW, initial_bias=_initial_bias, nobias=nobias)
            self.proj_out_W = self.out_proj.W
            self.proj_out_b = self.out_proj.b

            if not nobias_kv:
                self.bias_k: variable.Variable = variable.Parameter(
                    _initial_bias, (self.embedding_size,))
                self.bias_v: variable.Variable = variable.Parameter(
                    _initial_bias, (self.embedding_size,))
            else:
                self.bias_k, self.bias_v = None, None

    def forward(
            self,
            query: InputType,
            key: tp.Optional[InputType] = None,
            value: tp.Optional[InputType] = None,
            key_padding_mask: tp.Optional[InputType] = None,
            attention_mask: tp.Optional[InputType] = None,
            add_zero_attention: tp.Optional[InputType] = False,
            return_weights: tp.Optional[InputType] = False
    ) -> tp.Union[tp.Tuple[variable.Variable, variable.Variable], variable.Variable]:  # NOQA
        """Compute attention weight and context vector.

        This computes and returns ``attention``. If ``return_weights`` is
        ``True``, the return value is a :obj:`tuple` of ``attention`` and
        ``attention_weights``.

        Self-attention can be implemented by passing the same arguments for
        query, key, and value. Timesteps can be masked by
        giving a time x time mask in the ``attention_mask``. Padding elements
        can be excluded from the key by passing a binary mask with the shape of
        :math:`({\\text batch_size}, {\\text source_length})`
        where padding elements are indicated by 1.

        Args:
            query (:class:`~chainer.Variable` or :ref:`ndarray`):
                The query vectors with the shape of
                :math:`({\\text time}, {\\text batch_size}, {\\text query_in_size})`.  # NOQA
            key (:class:`~chainer.Variable` or :ref:`ndarray`):
                The key of the memory vectors with the shape of
                :math:`({\\text time}, {\\text batch_size}, {\\text key_in_size})`.  # NOQA
            value (:class:`~chainer.Variable` or :ref:`ndarray`):
                The value of the memory with the shape of
                :math:`({\\text time}, {\\text batch_size}, {\\text value_in_size})`.  # NOQA
            key_padding_mask (:class:`~chainer.Variable` or :ref:`ndarray`):
                If not ``None``, mask the memory slots.
                The shape is :math:`({\\text batch_size}, {\\text source_length})`.  # NOQA
                Each value is 0 (``False``) or 1 (``True``) where 1 means that
                the memory slot will not be used.
            attention_mask (:class:`~chainer.Variable` or :ref:`ndarray`):
                Mask help attention ignores certain positions.
                The shape is :math:`(L, S)` where :math:`L` is
                the target sequence length and :math:`S` is the source sequence
                length.
            add_zero_attention (bool): If ``True``, add a new batch of zeros to
                the key and value sequences at axis=1.
            return_weights (bool): If ``True``, return both ``attention``
                and ``attention_weights``. The default value is ``False``.

        Returns:
            :class:`~chainer.Variable` or
            :obj:`tuple` of :class:`~chainer.Variable`\\ s: This returns tuple
            of ``attention`` and ``attention_weights`` when ``return_weight``
            is ``True``.

            - ``attn_output`` is the output of attention whose shape is
              :math:`(L, B, E)` where :math:`L` is the target sequence length,
              :math:`B` is the batch size, and :math:`E` is the embedding size.

            - ``attn_output_weights`` is the weights of attention whose shape
              is :math:`(B, L, S)` where :math:`B` is the batch size,
              :math:`L` is the target sequence length,
              and :math:`S` is the source sequence length.

        .. seealso:: :func:`~chainer.functions.multihead_attention`

        """

        chainer.utils.experimental('chainer.links.MultiHeadAttention')

        if hasattr(self, 'proj_in_W'):
            proj_in_W = self.proj_in_W
        else:
            proj_in_W = (
                self.proj_q_weight, self.proj_k_weight, self.proj_v_weight)

        attention, attention_weights = functions.multihead_attention(
            self.n_head, self.embedding_size, query, key, value,
            proj_in_W, self.proj_in_b, self.bias_k, self.bias_v,
            self.proj_out_W, self.proj_out_b,
            add_zero_attention, self.attention_dropout, self.post_dropout,
            key_padding_mask, attention_mask, self.scaler, self.softmax_scaler,
            return_weights
        )
        if return_weights:
            return attention, attention_weights
        return attention
