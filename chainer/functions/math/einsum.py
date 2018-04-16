from chainer.backends import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check


class DiagEinSum(function_node.FunctionNode):

    def __init__(self, in_subs, out_sub, out_shape=None):
        self.in_subs = in_subs
        self.out_sub = out_sub
        self.out_shape = out_shape
        if '@' in out_sub:
            self.broadcast_dims = [
                slice(len(left_sub), -len(right_sub) or None)
                for left_sub, right_sub in \
                    [s.split('@') for s in in_subs.split(',')]
            ]
        else:
            self.broadcast_dims = None
        print(self.broadcast_dims)

    def check_type_forward(self, in_types):
        in_subs = self.in_subs.split(',')
        type_check.expect(in_types.size() == len(in_subs))

        for in_type in in_types:
            type_check.expect(in_type.dtype.kind == 'f')

        if self.broadcast_dims is not None:
            broadcast_shapes = [
                in_type.shape[dims]
                for in_type, dims in zip(in_types, self.broadcast_dims)
            ]
            for broadcast_shape_i, broadcast_shape_i_plus_one in zip(
                    broadcast_shapes, broadcast_shapes[1:]):
                type_check.expect(
                    broadcast_shape_i == broadcast_shape_i_plus_one)

    def forward(self, inputs):
        n_args = len(inputs)
        self.retain_inputs(tuple(range(n_args)))
        """
        if n_args >= 2:
            self.retain_inputs(tuple(range(n_args)))
        else:
            self.input_shape = inputs[0].shape
        self.n_args = n_args
        """

        xp = cuda.get_array_module(inputs[0])

        # out_sub = self.out_sub
        if self.broadcast_dims is None:
            broadcast_shape = None
        else:
            broadcast_shape = inputs[0].shape[self.broadcast_dims[0]]
            # out_sub = out_sub.replace('...', '?'*len(broadcast_shape))
        # print(broadcast_shape)

        ein_sub = []
        axis = 0
        subscript_dict = {}
        diag_map = []
        for left_or_right, out_sub in enumerate(self.out_sub.split('@')):
            if left_or_right == 1:  # process '...'
                ein_sub.append('@')
                axis += len(broadcast_shape)
            for s in out_sub:
                if s in subscript_dict:
                    diag_map.append((axis, subscript_dict[s]))
                else:
                    subscript_dict[s] = axis
                    if s in self.in_subs:
                        ein_sub.append(s)
                    else:
                        diag_map.append((axis, None))
                axis += 1

        subscript = '{}->{}'.format(
            self.in_subs,
            ''.join(ein_sub)
        )
        y = utils.force_array(
            xp.einsum(subscript.replace('@', '...'), *inputs))

        shape = list(y.shape)
        for i, i0 in diag_map:
            if i0 is None:
                # broadcast to new axis
                assert self.out_shape is not None, \
                    "Give out_shape to put new subscripts in the result"
                shape.insert(i, self.out_shape[i])
                y = xp.broadcast_to(xp.expand_dims(y, axis=i), shape)
            elif i0 != i:
                # make diagonal
                size = shape[i0]
                shape.insert(i, size)
                z = xp.zeros(shape, dtype=y.dtype)
                indexer = (
                    (slice(None),) * i0
                    + (xp.arange(size),)
                    + (slice(None),) * (i - i0 - 1))
                z[indexer + (xp.arange(size), Ellipsis,)] = \
                    y[indexer + (Ellipsis,)]
                y = z
        return y,

    def backward(self, indices, grad_outputs):
        inputs = self.get_retained_inputs()
        """
        if self.n_args >= 2:
            inputs = self.get_retained_inputs()
        else:
            # TODO(kataoka): fix
            xp = cuda.get_array_module(grad_outputs[0])
            inputs = [xp.zeros(self.input_shape, dtype=grad_outputs[0].dtype)]
        """

        g, = grad_outputs

        fwd_in_subs = self.in_subs.split(',')
        fwd_out_sub = self.out_sub
        return tuple(
            inputs[i] * 0. +  # it seems a bug of gradient_check
            DiagEinSum(
                in_subs=','.join([
                    (fwd_out_sub if j == i else s)
                    for j, s in enumerate(fwd_in_subs)
                ]),
                out_sub=fwd_in_subs[i],
                out_shape=inputs[i].shape,
            ).apply(tuple(
                (g if j == i else x)
                for j, x in enumerate(inputs)
            ))[0]
            for i in indices
        )


def _diag_einsum(
        input_subscripts, output_subscript, *ioperands, output_shape=None):
    return DiagEinSum(
        in_subs=input_subscripts,
        out_sub=output_subscript,
        out_shape=output_shape,
    ).apply(ioperands)[0]


def einsum(*operands):
    input_subscripts, output_subscript, ioperands = \
        _parse_einsum_input(operands)
    return DiagEinSum(
        in_subs=input_subscripts,
        out_sub=output_subscript,
    ).apply(ioperands)[0]


# #################### cupy.linalg.einsum ####################
# From cupy PR #873

einsum_symbols = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
einsum_symbols_set = set(einsum_symbols)


def _parse_einsum_input(operands):
    """Parses einsum operands.

    This function is based on `numpy.core.einsumfunc._parse_einsum_input`
    function in NumPy 1.14.

    Returns
    -------
    input_strings : str
        Parsed input strings
    output_string : str
        Parsed output string
    operands : list of array_like
        The operands to use in the numpy contraction

    Examples
    --------
    The operand list is simplified to reduce printing:

    >>> a = np.random.rand(4, 4)
    >>> b = np.random.rand(4, 4, 4)
    >>> _parse_einsum_input(('...a,...a->...', a, b))
    ('@a,@a', '@', [a, b])

    >>> _parse_einsum_input((a, [Ellipsis, 0], b, [Ellipsis, 0]))
    ('@a,@a', '@', [a, b])
    """

    if len(operands) == 0:
        raise ValueError("No input operands")

    if isinstance(operands[0], str):
        subscripts = operands[0].replace(" ", "")
        operands = operands[1:]

        # Ensure all characters are valid
        for s in subscripts:
            if s in '.,->':
                continue
            if s not in einsum_symbols:
                raise ValueError("Character %s is not a valid symbol." % s)

        # Check for proper "->"
        if ("-" in subscripts) or (">" in subscripts):
            invalid = (subscripts.count("-") > 1) or (subscripts.count(">") > 1)
            if invalid or (subscripts.count("->") != 1):
                raise ValueError("Subscripts can only contain one '->'.")

        # Parse "..."
        subscripts = subscripts.replace("...", "@")
        if "." in subscripts:
            raise ValueError("Invalid Ellipses.")

    else:
        tmp_operands = list(operands)
        operand_list = []
        subscript_list = []
        for p in range(len(operands) // 2):
            operand_list.append(tmp_operands.pop(0))
            subscript_list.append(tmp_operands.pop(0))

        output_list = tmp_operands[-1] if len(tmp_operands) else None
        operands = operand_list
        subscripts = ""
        last = len(subscript_list) - 1
        for num, sub in enumerate(subscript_list):
            for s in sub:
                if s is Ellipsis:
                    subscripts += "@"
                elif isinstance(s, int):
                    subscripts += einsum_symbols[s]
                else:
                    raise TypeError("For this input type lists must contain "
                                    "either int or Ellipsis")
            if num != last:
                subscripts += ","

        if output_list is not None:
            subscripts += "->"
            for s in output_list:
                if s is Ellipsis:
                    subscripts += "@"
                elif isinstance(s, int):
                    subscripts += einsum_symbols[s]
                else:
                    raise TypeError("For this input type lists must contain "
                                    "either int or Ellipsis")

    # Build output string if does not exist
    if "->" in subscripts:
        input_subscripts, output_subscript = subscripts.split("->")

        # Make sure output subscripts are in the input
        for char in output_subscript:
            if char not in input_subscripts:
                raise ValueError("Output character %s did not appear in the input"
                                 % ('...' if char == '@' else char))

    else:
        input_subscripts = subscripts
        # Build output subscripts
        tmp_subscripts = subscripts.replace(",", "")
        output_subscript = ""
        for s in sorted(set(tmp_subscripts)):
            if s == '@' or tmp_subscripts.count(s) == 1:
                output_subscript += s

    # Make sure number operands is equivalent to the number of terms
    if len(input_subscripts.split(',')) != len(operands):
        raise ValueError("Number of einsum subscripts must be equal to the "
                         "number of operands.")

    return (input_subscripts, output_subscript, operands)
