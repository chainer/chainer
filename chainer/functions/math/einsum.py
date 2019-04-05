import warnings

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import argument
from chainer.utils import type_check


def _enumerate_axes(subscripts):
    if '@' in subscripts:
        left_sub, right_sub = subscripts.split('@')
        for i, s in enumerate(left_sub):
            yield i, s
        yield slice(len(left_sub), -len(right_sub) or None), '@'
        for i, s in enumerate(right_sub):
            yield i - len(right_sub), s
    else:
        for i, s in enumerate(subscripts):
            yield i, s


def _einsum(xp, dtype, in_subscripts, out_subscript, *inputs, **kwargs):
    check_undefined_ellipsis_sum, = argument.parse_kwargs(
        kwargs, ('check_undefined_ellipsis_sum', False))
    sum_ellipsis = '@' in in_subscripts and '@' not in out_subscript
    if sum_ellipsis:
        # einsum does not usually allow summing over '...'
        subscripts = '{}->...{}'.format(
            in_subscripts.replace('@', '...'),
            out_subscript
        )
    else:
        subscripts = '{}->{}'.format(
            in_subscripts,
            out_subscript
        ).replace('@', '...')

    # Use optimize option whenever it is critical in speed.
    # Otherwise avoid bugs in numpy>=1.12,<1.15.
    einsum_kwargs = {}
    if len(inputs) >= 3:
        einsum_kwargs['optimize'] = True
    try:
        y = xp.einsum(subscripts, *inputs, **einsum_kwargs)
    except TypeError:
        warnings.warn(
            '{xp}.einsum does not support optimize option. '
            'Use newer version of {xp} to speed up.'
            .format(xp=xp.__name__),
            chainer.warnings.PerformanceWarning,
        )
        y = xp.einsum(subscripts, *inputs)

    if sum_ellipsis:
        sum_ndim = y.ndim - len(out_subscript)
        if check_undefined_ellipsis_sum and sum_ndim > 0:
            raise ValueError(
                'einsum should not support summing over Ellipsis, '
                'while NumPy 1.14 sometimes accidentally supports it. '
                'This feature is no longer supported by Chainer. '
                'See also NumPy issues #10926, #9984.',
            )
        y = xp.sum(y, axis=tuple(range(sum_ndim)))

    return utils.force_array(y, dtype)


class EinSum(function_node.FunctionNode):

    def __init__(self, in_subs, out_sub):
        self.in_subs = in_subs
        self.out_sub = out_sub

    def check_type_forward(self, in_types):
        for i, in_type in enumerate(in_types):
            type_check._argname((in_type,), ('x{}'.format(i),))
            type_check.expect(in_type.dtype.kind == 'f')

        in_subs = self.in_subs.split(',')
        type_check.expect(in_types.size() == len(in_subs))

        shape_dict = {}
        for in_sub, in_type in zip(in_subs, in_types):
            for axis, char in _enumerate_axes(in_sub):
                shape = in_type.shape[axis]
                if char in shape_dict:
                    type_check.expect(shape_dict[char] == shape)
                else:
                    shape_dict[char] = shape

    def forward(self, inputs):
        n_args = len(inputs)
        # TODO(kataoka): Do not retain inputs if n_args == 1
        self.retain_inputs(tuple(range(n_args)))

        xp = backend.get_array_module(inputs[0])
        dtype = xp.result_type(*[x.dtype for x in inputs])
        y = _einsum(xp, dtype, self.in_subs, self.out_sub, *inputs,
                    check_undefined_ellipsis_sum=True)
        return y,

    def backward(self, indices, grad_outputs):
        inputs = self.get_retained_inputs()
        g, = grad_outputs

        fwd_in_subs = self.in_subs.split(',')
        fwd_out_sub = self.out_sub
        return tuple(
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


class DiagEinSum(EinSum):

    def __init__(self, in_subs, out_sub, out_shape):
        self.in_subs = in_subs
        self.out_sub = out_sub
        self.out_shape = out_shape

    def forward(self, inputs):
        n_args = len(inputs)
        # TODO(kataoka): Do not retain inputs if n_args == 1
        self.retain_inputs(tuple(range(n_args)))

        xp = backend.get_array_module(inputs[0])
        dtype = xp.result_type(*[x.dtype for x in inputs])

        out_set = set(self.out_sub)

        # '@' is a single char, ',' is excluded.
        io_set = out_set.intersection(set(self.in_subs))

        if len(io_set) == len(self.out_sub):
            y = _einsum(xp, dtype, self.in_subs, self.out_sub, *inputs)
        else:
            direct_sub = []
            inverse_sub = []
            expander = []
            for c in sorted(out_set):
                if c in io_set:
                    direct_sub.append(c)
                    expander.append(slice(None))
                else:
                    expander.append(None)
                inverse_sub.append(c)

            y = xp.zeros(self.out_shape, dtype)
            diag_y = _einsum(
                xp, dtype, self.out_sub, ''.join(inverse_sub), y)
            if diag_y.base is not y:
                raise ValueError('Update CuPy to close CuPy Issue #1199')
            # Make the view writeable as numpy PR #5410 for numpy<1.10.
            if xp is not cuda.cupy:  # no setflags in cupy
                diag_y.setflags(write=True)
            diag_y[...] = _einsum(
                xp, dtype, self.in_subs, ''.join(direct_sub), *inputs
            )[tuple(expander)]
        return y,


def einsum(*operands):
    """Einstein summation

    This function supports two formats of inputs:

    - ``einsum(subscripts, op0, op1, ...)``
    - ``einsum(op0, sublist0, op1, sublist1, ..., [sublistout])``

    See also :func:`numpy.einsum`

    .. admonition:: Example

        The following example computes a batched application of a bilinear
        function with weight ``w``.

        >>> x1 = np.arange(12).reshape(3, 4).astype(np.float32)
        >>> x2 = np.arange(15).reshape(3, 5).astype(np.float32)
        >>> w = np.arange(120).reshape(4, 5, 6).astype(np.float32)
        >>> y = F.einsum('ij,ik,jkl->il', x1, x2, w)
        >>> y.shape
        (3, 6)

        The batch axes can be denoted by ``...``. If the string of output
        subscripts is omitted, the summation is taken over the subscript
        alphabets with two (or more) occurrences.

        >>> np.allclose(y.array, F.einsum('...j,...k,jkl', x1, x2, w).array)
        True

        In the other format:

        >>> y = F.einsum(x1, [0, 1], x2, [0, 2], w, [1, 2, 3], [0, 3])
        >>> y.shape
        (3, 6)
        >>> y = F.einsum(x1, [Ellipsis, 1], x2, [Ellipsis, 2], w, [1, 2, 3])
        >>> y.shape
        (3, 6)

    """
    input_subscripts, output_subscript, ioperands = \
        _parse_einsum_input(operands)
    return EinSum(
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
        raise ValueError('No input operands')

    if isinstance(operands[0], str):
        subscripts = operands[0].replace(' ', '')
        operands = operands[1:]

        # Ensure all characters are valid
        for s in subscripts:
            if s in '.,->':
                continue
            if s not in einsum_symbols:
                raise ValueError('Character %s is not a valid symbol.' % s)

        # Check for proper "->"
        if ('-' in subscripts) or ('>' in subscripts):
            if any((
                    subscripts.count('-') > 1,
                    subscripts.count('>') > 1,
                    subscripts.count('->') != 1,
            )):
                raise ValueError('Subscripts can only contain one \'->\'.')

        # Parse "..."
        subscripts = subscripts.replace('...', '@')
        if '.' in subscripts:
            raise ValueError('Invalid Ellipses.')

    else:
        tmp_operands = list(operands)
        operand_list = []
        subscript_list = []
        for p in range(len(operands) // 2):
            operand_list.append(tmp_operands.pop(0))
            subscript_list.append(tmp_operands.pop(0))

        output_list = tmp_operands[-1] if len(tmp_operands) else None
        operands = operand_list
        subscripts = ''
        last = len(subscript_list) - 1
        for num, sub in enumerate(subscript_list):
            for s in sub:
                if s is Ellipsis:
                    subscripts += '@'
                elif isinstance(s, int):
                    subscripts += einsum_symbols[s]
                else:
                    raise TypeError('For this input type lists must contain '
                                    'either int or Ellipsis')
            if num != last:
                subscripts += ','

        if output_list is not None:
            subscripts += '->'
            for s in output_list:
                if s is Ellipsis:
                    subscripts += '@'
                elif isinstance(s, int):
                    subscripts += einsum_symbols[s]
                else:
                    raise TypeError('For this input type lists must contain '
                                    'either int or Ellipsis')

    # Build output string if does not exist
    if '->' in subscripts:
        input_subscripts, output_subscript = subscripts.split('->')

        # Make sure output subscripts are in the input
        for char in output_subscript:
            if char not in input_subscripts:
                raise ValueError(
                    'Output character %s did not appear in the input'
                    % ('...' if char == '@' else char))

    else:
        input_subscripts = subscripts
        # Build output subscripts
        tmp_subscripts = subscripts.replace(',', '')
        output_subscript = ''
        for s in sorted(set(tmp_subscripts)):
            if s == '@' or tmp_subscripts.count(s) == 1:
                output_subscript += s

    # Make sure number operands is equivalent to the number of terms
    if len(input_subscripts.split(',')) != len(operands):
        raise ValueError('Number of einsum subscripts must be equal to the '
                         'number of operands.')

    return input_subscripts, output_subscript, operands
