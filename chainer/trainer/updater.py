import six

from chainer import variable


class Updater(object):

    """Base class for updating callback.

    TODO(beam2d): document it.

    """
    def __call__(self, inputs, target, optimizer):
        raise NotImplementedError

    def serialize(self, serializer):
        pass


class StandardUpdater(Updater):

    """Standard implementation of Updater.

    TODO(beam2d): document it.

    """
    def __init__(self, lossfun=None):
        self._lossfun = lossfun

    def __call__(self, inputs, target, optimizer):
        if not isinstance(inputs, tuple):
            inputs = inputs,
        in_vars = [variable.Variable(a) for a in inputs]
        lossfun = target if self._lossfun is None else self._lossfun

        target.zerograds()
        loss = lossfun(*in_vars)
        loss.backward()
        optimizer.update()

        result = {'loss': loss.data}
        for name, value in six.iteritems(target.__dict__):
            if isinstance(value, variable.Variable):
                x = value.data
                if x.size == 1:
                    result[name] = x
        return result
