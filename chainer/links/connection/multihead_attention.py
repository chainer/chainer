import typing as tp


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
            :math:`1/\\sqrt{n_{head}}`.
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
        n_head,                 # type: int
        embedding_size,         # type: int
        self_attention=False,   # type: bool
        ksize=None,             # type: tp.Optional[int]
        vsize=None,             # type: tp.Optional[int]
        attention_dropout=0.0,  # type: float
        post_dropout=0.0,       # type: float
        scaler=None,            # type: tp.Optional[float]
        softmax_scaler=None,    # type: tp.Optional[float]
        initialW=None,          # type: tp.Optional[types.InitializerSpec]
        initial_bias=None,      # type: tp.Optional[types.InitializerSpec]
        nobias=False,           # type: bool
        nobias_kv=True          # type: bool
    ):
        # type (...) -> None
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
                self.proj_in_W = variable.Parameter(
                    _initialW, (3 * self.embedding_size, self.embedding_size))  # type: variable.Variable  # NOQA
            else:
                self.proj_q_weight = variable.Parameter(
                    _initialW, (self.embedding_size, self.embedding_size))  # type: variable.Variable  # NOQA
                self.proj_k_weight = variable.Parameter(
                    _initialW, (self.embedding_size, self.ksize))  # type: variable.Variable  # NOQA
                self.proj_v_weight = variable.Parameter(
                    _initialW, (self.embedding_size, self.vsize))  # type: variable.Variable  # NOQA
            if not nobias:
                self.proj_in_b = variable.Parameter(
                    _initial_bias, (3 * self.embedding_size,))  # type: variable.Variable  # NOQA
            else:
                self.proj_in_b = None

            self.out_proj = links.Linear(
                self.embedding_size, self.embedding_size,
                initialW=_initialW, initial_bias=_initial_bias, nobias=nobias)
            self.proj_out_W = self.out_proj.W
            self.proj_out_b = self.out_proj.b

            if not nobias_kv:
                self.bias_k = variable.Parameter(
                    _initial_bias, (self.embedding_size,))  # type: variable.Variable  # NOQA
                self.bias_v = variable.Parameter(
                    _initial_bias, (self.embedding_size,))  # type: variable.Variable  # NOQA
            else:
                self.bias_k, self.bias_v = None, None

    def forward(
        self,
        query,                     # type: InputType
        key=None,                  # type: tp.Optional[InputType]
        value=None,                # type: tp.Optional[InputType]
        key_padding_mask=None,     # type: tp.Optional[InputType]
        attention_mask=None,       # type: tp.Optional[InputType]
        add_zero_attention=False,  # type: bool
        return_weights=False       # type: bool
    ):
        # type: (...) -> tp.Union[tp.Tuple[variable.Variable, variable.Variable], variable.Variable]  # NOQA
        """Compute attention weight and context vector.

        This computes and returns ``attention``. If ``return_weights`` is
        ``True``, the return value is a :obj:`tuple` of ``attention`` and
        ``attention_weights``

        Self-attention can be implemented by passing the same arguments for
        query, key, and value. Timesteps can be masked by
        giving a time x time mask in the `attention_mask`. Padding elements
        can be excluded from the key by passing a batch_size x source_length
        mask where padding elements are indicated by 1.

        Args:
            query (:class:`~chainer.Variable` or :ref:`ndarray`):
                The query vectors with the shape of
                (time, batch_size, query_in_size).
            key (:class:`~chainer.Variable` or :ref:`ndarray`):
                The key of the memory vectors with the shape of
                (time, batch_size, key_in_size).
            value (:class:`~chainer.Variable` or :ref:`ndarray`):
                The value of the memory with the shape of
                (time, batch_size, value_in_size).
            key_padding_mask (:class:`~chainer.Variable` or :ref:`ndarray`):
                If not ``None``, mask the memory slots.
                The shape is (batch_size, source_length).
                Each value is 0 (``False``) or 1 (``True``) where 1 means that
                the memory slot will not be used.
            attention_mask (:class:`~chainer.Variable` or :ref:`ndarray`):
                Mask help attention ignores certain positions.
                The shape is :math:`(L, L)` where :math:`L` is
                the target sequence length.
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
