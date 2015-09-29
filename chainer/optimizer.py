import math
import warnings

import numpy
import six

from chainer import cuda
from chainer import link
from chainer import variable


class TupleLink(link.Link):

    def __init__(self, params_grads):
        super(TupleLink, self).__init__()
        for i, (param, grad) in enumerate(six.moves.zip(*params_grads)):
            var = variable.Variable(param)
            var.grad = grad
            self.params[str(i)] = var


class Optimizer(object):

    def __init__(self):
        self.t = 0
        self.target = None
        self.states = {}

    def init_state(self, param, state):
        with cuda.get_device(param) as d:
            if int(d) < 0:
                self.init_state_cpu(param, state)
            else:
                self.init_state_gpu(param, state)

    def init_state_cpu(self, param, state):
        pass

    def init_state_gpu(self, param, state):
        pass

    def setup(self, target):
        if isinstance(target, tuple):
            target = TupleLink(target)
        self.target = target
        for path, param in target.visitparams():
            state = {}
            self.init_state(param.data, state)
            self.states[path] = state

    def update(self, loss_func=None, *args, **kwds):
        raise NotImplementedError

    def zero_grads(self):
        warnings.warn('Optimizer.zero_grads is deprecated. '
                      'Use Model.zerograds instead.', DeprecationWarning)
        self.target.zerograds()

    def compute_grads_norm(self):
        warnings.warn('Optimizer.compute_grads_norm is deprecated.',
                      DeprecationWarning)
        sqnorm = 0
        for _, param in self.target.visitparams():
            grad = param.grad.ravel()
            sqnorm += float(grad.dot(grad))
        return math.sqrt(sqnorm)

    def clip_grads(self, maxnorm):
        warnings.warn('Optimizer.clip_grads is deprecated. '
                      'Use GradientClipping instead.', DeprecationWarning)
        GradientClipping(maxnorm)(self.target)

    def weight_decay(self, decay):
        warnings.warn('Optimizer.weight_decay is deprecated. '
                      'Use WeightDecay instead.', DeprecationWarning)
        WeightDecay(decay)(self.target)

    def accumulate_grads(self, grads):
        warnings.warn('Optimizer.accumulate_grads is deprecated. '
                      'Use Link.addgrads instead.',
                      DeprecationWarning)
        paths = []
        for path, _ in self.target.visitparams():
            paths.append(path)
        paths.sort()
        d = dict(six.moves.zip(paths, grads))

        for path, param in self.target.visitparams():
            dst = param.grad
            src = d[path]
            if isinstance(dst, numpy.ndarray):
                dst += cuda.to_cpu(src)
            elif isinstance(src, numpy.ndarray):
                dst += cuda.to_gpu(src, device=dst)
            elif src.device == dst.device:
                dst += src
            else:
                dst += cuda.copy(src, out_device=dst)

    def serialize(self, serializer):
        self.t = serializer('t', self.t)
        serializer = serializer['state']
        for path, state in six.iteritems(self.states):
            s = serializer[path[1:]]  # omit '/' at the head
            for key in state:
                state[key] = s(key, state[key])


class GradientMethod(Optimizer):

    def __init__(self):
        Optimizer.__init__(self)
        self.hooks = []

    def add_hook(self, name, hook):
        if not callable(hook):
            raise TypeError('Cannot set non-callable object as a hook')
        self.hooks.append((name, hook))

    def remove_hook(self, name):
        dels = []
        for i, (n, h) in enumerate(self.hooks):
            if n == name:
                dels.append(i)
        dels.reverse()
        for i in dels:
            del self.hooks[i]

    def update(self, loss_func=None, *args, **kwds):
        if loss_func is not None:
            self.target.zerograds()
            loss = loss_func(*args, **kwds)
            loss.backward()

        for _, hook in self.hooks:
            hook(self.target)

        self.t += 1
        for path, param in self.target.visitparams():
            state = self.states[path]
            self.update_param(param.data, param.grad, state)

    def update_param(self, param, grad, state):
        if isinstance(param, cuda.ndarray):
            self.update_param_gpu(param, grad, state)
        else:
            self.update_param_cpu(param, grad, state)

    def update_param_cpu(self, param, grad, state):
        raise NotImplementedError

    def update_param_gpu(self, param, grad, state):
        raise NotImplementedError


class WeightDecay(object):

    def __init__(self, decay):
        self.decay = decay

    def __call__(self, target):
        for _, param in target.visitparams():
            data = param.data
            with cuda.get_device(data):
                param.grad += self.decay * data


class GradientClipping(object):

    def __init__(self, maxnorm):
        self.maxnorm = maxnorm

    def __call__(self, target):
        # TODO(beam2d): Make it fast on GPU
        grads = []
        for _, param in target.visitparams():
            grads.append(param.grad)

        sqnorm = 0
        for grad in grads:
            grad = grad.ravel()
            sqnorm += float(grad.dot(grad))

        ratio = self.maxnorm / math.sqrt(sqnorm)
        if ratio < 1:
            for grad in grads:
                grad *= ratio
