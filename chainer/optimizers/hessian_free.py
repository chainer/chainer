import six

import chainer
from chainer import cuda
from chainer import optimizer
from chainer.functions.math import identity


def params_dot(xs, ys):
    return sum([x.ravel().dot(y.ravel()) for x, y in six.moves.zip(xs, ys)])

def conjugate_gradient(hessian_vector_product, bs, xs):
    hxs = hessian_vector_product(xs)
    rs = [b - hx for b, hx in six.moves.zip(bs, hxs)]

    ps = [r.copy() for r in rs]
    for _ in range(3):
        rr = params_dot(rs, rs)
        if rr < 0.00001:
            break
        hps = hessian_vector_product(ps)
        hpp = params_dot(ps, hps)
        if hpp < 0.0001:
            break
        alpha = rr / hpp
        for x, p, r, hp in six.moves.zip(xs, ps, rs, hps):
            x += alpha * p
            r -= alpha * hp
        beta = params_dot(rs, rs) / rr
        for p, r in six.moves.zip(ps, rs):
            p *= beta
            p += r
    return xs


class HessianFree(optimizer.Optimizer):

    def update(self, lossfun=None, *args, **kwargs):
        loss = lossfun(*args, **kwargs)
        self.target.cleargrads()
        loss.backward(enable_double_backprop=True)
        valid_params = [x for x in self.target.params() if x.grad is not None]
        grad_vars = [x.grad_var for x in valid_params]
        
        def hessian_vector_product(gs):
            gxs = identity.Identity().apply(grad_vars)
            for gx, g, x in six.moves.zip(gxs, gs, valid_params):
                gx.grad = cuda.get_array_module(x).asarray(g, x.dtype)
            self.target.cleargrads()
            gxs[0].backward()
            dumping_strength = 0.01
            return [x.grad + dumping_strength * g
                    for x, g in six.moves.zip(valid_params, gs)]

        gs = [-x.grad for x in valid_params]
        xs0 = [cuda.get_array_module(x).zeros_like(x.data)
               for x in valid_params]
        dxs = conjugate_gradient(hessian_vector_product, gs, xs0)
        def func():
            return lossfun(*args, **kwargs).data
        grads = [x.grad for x in valid_params]
        self.line_search(valid_params, loss.data, func, grads, dxs)

    def line_search(self, valid_params, y, func, grads, dxs, beta=0.5, c=0.01):
        alpha = 1
        for param, dx in six.moves.zip(valid_params, dxs):
            param.data += dx
        while True:
            y_new = func()
            if y_new <= y + c * alpha * params_dot(grads, dxs):
                return
            for param, dx in six.moves.zip(valid_params, dxs):
                param.data -= (1 - beta) * alpha * dx
            alpha *= beta




if __name__ == '__main__':
    import numpy
    class Model(chainer.Chain):
        def __init__(self):
            super(Model, self).__init__()
            with self.init_scope():
                self.w1 = chainer.links.Linear(4, 3, initialW=0, initial_bias=0)

        def __call__(self, x, y):
            d = self.w1(x) - y
            return chainer.functions.sum(d * d)

    m = Model()
    o = HessianFree()
    o.setup(m)
    w = numpy.arange(12).reshape((4, 3)).astype('f')
    for i in range(10):
        x = numpy.random.uniform(-1, 1, (5, 4)).astype('f')
        y = x.dot(w)
        o.update(m, x, y)

    print(m.w1.W.data.ravel())
