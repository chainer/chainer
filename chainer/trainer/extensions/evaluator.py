from __future__ import print_function

import six

from chainer import cuda
from chainer.trainer import extension
from chainer import variable


class Evaluator(extension.Extension):

    """Trainer extension for evaluation on the validation set.

    TODO(beam2d): document it.

    """
    default_trigger = 1, 'epoch'
    default_name = 'validation'
    priority = extension.PRIORITY_READER

    def __init__(self, dataset, target, lossfun=None, batchsize=1,
                 prepare=None, device=None):
        self._dataset = dataset
        self._target = target
        self._lossfun = lossfun
        self._batchsize = batchsize
        self._prepare = prepare
        self._device = device

    def __call__(self, epoch, t, trainer, **kwargs):
        target = self._target.copy()  # evaluate model with distinct states
        lossfun = target if self._lossfun is None else self._lossfun

        if self._prepare is not None:
            self._prepare(target)

        accum = None
        for inputs in self._dataset.get_batch_iterator(
                self._batchsize, repeat=False, device=self._device):
            if not isinstance(inputs, tuple):
                inputs = inputs,
            n = len(inputs[0])
            # TODO(beam2d): better device handling
            in_vars = tuple(variable.Variable(a, volatile='on')
                            for a in inputs)
            loss = lossfun(*in_vars)
            result = {'loss': loss.data * n}
            for key, value in six.iteritems(target.__dict__):
                if isinstance(value, variable.Variable):
                    v = value.data
                    if v.size == 1:
                        result[key] = v * n
            if accum is None:
                accum = result
            else:
                for key in result:
                    accum[key] += result[key]

        N = len(self._dataset)
        return {key: value / N for key, value in six.iteritems(accum)}
