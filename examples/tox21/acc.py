import chainer
import numpy


def multitask_acc(x, t):
    x = chainer.cuda.to_cpu(x.data)
    t = chainer.cuda.to_cpu(t.data)

    pred = x > 0.5
    valid = t != -1
    count = ((pred == t) & valid).sum().astype(numpy.float32)
    acc = chainer.Variable(chainer.utils.force_array(count / valid.size))
    return acc
