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
