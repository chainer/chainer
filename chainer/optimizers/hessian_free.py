import chainer
from chainer import cuda
from chainer import optimizer
from chainer.functions.math import identity


def params_dot(xs, ys):
    return sum([(x * y).sum() for x, y in zip(xs, ys)])

def conjugate_gradient(hessian_vector_product, bs, xs):
    hxs = hessian_vector_product(xs)
    rs = [b - hx for b, hx in zip(bs, hxs)]

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
        for x, p, r, hp in zip(xs, ps, rs, hps):
            x += alpha * p
            r -= alpha * hp
        beta = params_dot(rs, rs) / rr
        for p, r in zip(ps, rs):
            p *= beta
            p += r
    return xs


class HessianFree(optimizer.Optimizer):

    def update(self, lossfun=None, *args, **kwargs):
        loss = lossfun(*args, **kwargs)
        self.target.cleargrads()
        loss.backward(enable_double_backprop=True)
        grads = [x.grad_var for x in self.target.params()]
        
        def hessian_vector_product(gs):
            gxs = identity.Identity().apply(grads)
            for gx, g in zip(gxs, gs):
                gx.grad = g
            self.target.cleargrads()
            gxs[0].backward()
            return [x.grad + 0.01 * g for x, g in zip(self.target.params(), gs)]

        gs = [-x.grad for x in self.target.params()]
        xs0 = [cuda.get_array_module(x).zeros_like(x.data)
               for x in self.target.params()]
        dxs = conjugate_gradient(hessian_vector_product, gs, xs0)
        for param, dx in zip(self.target.params(), dxs):
            param.data += dx


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
