from chainer.functions.activation.elu import elu


def selu(x,
         alpha=1.6732632423543772848170429916717,
         scale=1.0507009873554804934193349852946):
    return scale * elu(x, alpha=alpha)
