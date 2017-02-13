import chainer



def multitask_sce(x, t):
    xp = chainer.cuda.get_array_module(x)
    return chainer.Variable(xp.array(0., dtype=xp.float32))
