import chainer
import numpy


def multitask_acc(x, t):
    x = chainer.cuda.to_cpu(x.data)
    t = chainer.cuda.to_cpu(t.data)

    pred = x > 0
    valid = t != -1
    denom = ((pred == t) & valid).sum(axis=0).astype(numpy.float32)
    norm = valid.sum(axis=0).astype(numpy.float32)
    accs = denom / norm
    acc = chainer.utils.force_array(accs.mean())
    return chainer.Variable(acc)
