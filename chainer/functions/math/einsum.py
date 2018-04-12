from chainer.backends import cuda
from chainer import function_node
from chainer import utils


def _parse_subscripts(subscripts):
    subscripts = subscripts.split('->')
    assert 1 <= len(subscripts) <= 2
    subscripts_in = subscripts[0].split(',')
    if len(subscripts) == 1:
        raise "TODO(kataoka)"
    else:
        subscript_out = subscripts[1]
    return subscript_out, subscripts_in


class DiagEinSum(function_node.FunctionNode):

    def __init__(self, in_subs, out_sub, out_shape=None):
        self.in_subs = in_subs
        self.out_sub = out_sub
        self.out_shape = out_shape
        # self.shape_dict = shape_dict or {}

    def check_type_forward(self, in_types):
        pass

    def forward(self, inputs):
        self.retain_inputs(tuple(range(len(inputs))))
        xp = cuda.get_array_module(inputs[0])

        out_sub = self.out_sub

        in_all_subs = {s for x_sub in self.in_subs for s in x_sub}
        """
        first_indices = [
            out_sub.index(s)
            for i, s in enumerate(out_sub)
        ]
        ein_sub = [out_sub[i] for i, i0 in enumerate(first_indices) if i == i0]
        """
        diag_map = []
        ein_sub = []
        for i, s in enumerate(out_sub):
            i0 = out_sub.index(s)
            if i == i0:
                if s in in_all_subs:
                    ein_sub.append(s)
                else:
                    i0 = None
            diag_map.append(i0)

        args = []
        for x, x_sub in zip(inputs, self.in_subs):
            args.extend([x, x_sub])
        args.append(ein_sub)
        y = utils.force_array(xp.einsum(*args))

        for i, i0 in enumerate(diag_map):
            if i0 is None:
                # broadcast to new axis
                assert self.out_shape is not None, \
                    "Give out_shape to put new subscripts in the result"
                y = xp.broadcast_to(
                    xp.expand_dims(y, axis=i),
                    y.shape[:i] + (self.out_shape[i],) + y.shape[i:]
                )
            elif i0 != i:
                # make diagonal
                size = y.shape[i0]
                z = xp.zeros(
                    y.shape[:i] + (size,) + y.shape[i:],
                    dtype=y.dtype)
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
        g, = grad_outputs

        fwd_in_subs = self.in_subs
        fwd_out_sub = self.out_sub
        return tuple(
            DiagEinSum(
                in_subs=[
                    (fwd_out_sub if j == i else s)
                    for j, s in enumerate(fwd_in_subs)
                ],
                out_sub=fwd_in_subs[i],
                out_shape=inputs[i].shape,
            ).apply(tuple(
                (g if j == i else x)
                for j, x in enumerate(inputs)
            ))[0]
            for i in indices
        )


def einsum(*operands):
    input_subscripts, output_subscript, ioperands = \
        _parse_einsum_input(operands)
    return DiagEinSum(
        in_subs=[_to_ints(s) for s in input_subscripts.split(',')],
        out_sub=_to_ints(output_subscript),
    ).apply(ioperands)[0]


def _to_ints(subs):
    # TODO(kataoka): numpy Issue #7741
    return [einsum_symbols.index(s) for s in subs.upper()]


"""
class EinSum(function_node.FunctionNode):

    def __init__(self, subscripts):
        self.subscripts = subscripts

    def check_type_forward(self, in_types):
        pass

    def forward(self, inputs):
        self.retain_inputs(tuple(range(len(inputs))))

        xp = cuda.get_array_module(inputs[0])
        output = xp.einsum(self.subscripts, *inputs)

        return utils.force_array(output),

    def backward(self, indices, grad_outputs):
        inputs = self.get_retained_inputs()
        g, = grad_outputs

        subscript_out, subscripts_in = _parse_subscripts(self.subscripts)

        ret = []
        for i in indices:
            bwd_subscripts_in = ','.join([
                (subscript_out if j == i else s)
                for j, s in enumerate(subscripts_in)
            ])
            ret.append(_diag_and_einsum(
                subscripts_in[i],
                inputs[i].shape,
                bwd_subscripts_in,
                *[(g if j == i else x) for j, x in enumerate(inputs)]
            ))
        return tuple(ret)


def _diag(
        subscript_out,
        shape,
        subscript_in,
        x):
    assert subscript_in == subscript_out, "TODO(kataoka)\n" + str(
        (subscript_in, subscript_out))
    return x


def _diag_and_einsum(
        subscript_out,
        shape,
        subscripts_in,
        *operands):
    subscript_tmp = []
    for ch in subscript_out:
        if ch in subscripts_in and ch not in subscript_tmp:
            subscript_tmp.append(ch)
    subscript_tmp = ''.join(subscript_tmp)
    tmp = einsum(
        subscripts_in + '->' + subscript_tmp,
        *operands
    )

    return _diag(subscript_out, shape, subscript_tmp, tmp)


def einsum(subscripts, *operands):
    return EinSum(subscripts).apply(operands)[0]
"""


# #################### cupy.linalg.einsum ####################
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
    ('za,xza', 'xz', [a, b])

    >>> _parse_einsum_input((a, [Ellipsis, 0], b, [Ellipsis, 0]))
    ('za,xza', 'xz', [a, b])
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
                    subscripts += "..."
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
                    subscripts += "..."
                elif isinstance(s, int):
                    subscripts += einsum_symbols[s]
                else:
                    raise TypeError("For this input type lists must contain "
                                    "either int or Ellipsis")
    # Check for proper "->"
    if ("-" in subscripts) or (">" in subscripts):
        invalid = (subscripts.count("-") > 1) or (subscripts.count(">") > 1)
        if invalid or (subscripts.count("->") != 1):
            raise ValueError("Subscripts can only contain one '->'.")

    # Parse ellipses
    if "." in subscripts:
        used = subscripts.replace(".", "").replace(",", "").replace("->", "")
        unused = list(einsum_symbols_set - set(used))
        ellipse_inds = "".join(unused)
        longest = 0

        if "->" in subscripts:
            input_tmp, output_sub = subscripts.split("->")
            split_subscripts = input_tmp.split(",")
            out_sub = True
        else:
            split_subscripts = subscripts.split(',')
            out_sub = False

        for num, sub in enumerate(split_subscripts):
            if "." in sub:
                if (sub.count(".") != 3) or (sub.count("...") != 1):
                    raise ValueError("Invalid Ellipses.")

                # Take into account numerical values
                if operands[num].shape == ():
                    ellipse_count = 0
                else:
                    ellipse_count = max(operands[num].ndim, 1)
                    ellipse_count -= (len(sub) - 3)

                if ellipse_count > longest:
                    longest = ellipse_count

                if ellipse_count < 0:
                    raise ValueError("Ellipses lengths do not match.")
                elif ellipse_count == 0:
                    split_subscripts[num] = sub.replace('...', '')
                else:
                    rep_inds = ellipse_inds[-ellipse_count:]
                    split_subscripts[num] = sub.replace('...', rep_inds)

        subscripts = ",".join(split_subscripts)
        if longest == 0:
            out_ellipse = ""
        else:
            out_ellipse = ellipse_inds[-longest:]

        if out_sub:
            subscripts += "->" + output_sub.replace("...", out_ellipse)
        else:
            # Special care for outputless ellipses
            output_subscript = ""
            tmp_subscripts = subscripts.replace(",", "")
            for s in sorted(set(tmp_subscripts)):
                if s not in (einsum_symbols):
                    raise ValueError("Character %s is not a valid symbol." % s)
                if tmp_subscripts.count(s) == 1:
                    output_subscript += s
            normal_inds = ''.join(sorted(set(output_subscript) -
                                         set(out_ellipse)))

            subscripts += "->" + out_ellipse + normal_inds

    # Build output string if does not exist
    if "->" in subscripts:
        input_subscripts, output_subscript = subscripts.split("->")
    else:
        input_subscripts = subscripts
        # Build output subscripts
        tmp_subscripts = subscripts.replace(",", "")
        output_subscript = ""
        for s in sorted(set(tmp_subscripts)):
            if s not in einsum_symbols:
                raise ValueError("Character %s is not a valid symbol." % s)
            if tmp_subscripts.count(s) == 1:
                output_subscript += s

    # Make sure output subscripts are in the input
    for char in output_subscript:
        if char not in input_subscripts:
            raise ValueError("Output character %s did not appear in the input"
                             % char)

    # Make sure number operands is equivalent to the number of terms
    if len(input_subscripts.split(',')) != len(operands):
        raise ValueError("Number of einsum subscripts must be equal to the "
                         "number of operands.")

    return (input_subscripts, output_subscript, operands)
