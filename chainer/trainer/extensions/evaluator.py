from __future__ import print_function

import six

from chainer import cuda
from chainer.trainer import extension
from chainer import variable


class Evaluator(extension.Extension):

    """Trainer extension for evaluation on the validation set.

    This extension evaluates the current parameters by the given loss function.
    The result is extracted from the target link by scanning its attributes of
    scalar variables (which is also done by the :class:`StandardUpdater`).
    The evaluator computes the mean values of extracted scalars over all
    minibatches, and it returns a result dictionary of these mean values.

    The evaluator copies the target link before the evaluation. This prevents
    the corruption of internal states of the link (e.g. the states of recurrent
    networks begin trained on infinite-length sequences).

    This extension is called once for each epoch by default.

    Args:
        dataset (Dataset): Validation dataset.
        target (Link): The target link.
        lossfun: Loss function. The returned loss value is added to the result
            dictionary with the key ``'loss'``. If it is None, then ``target``
            is used as the loss function.
        batchsize (int): Number of data points in each minibatch. This value is
            purely for the tradeoff between computational speed and memory
            consumption.
        prepare: Callback to preprocess the target link. The evaluator gives
            the (copied) target link to this callback before the evaluation.
        device: Device specifier. Minibatches are sent to this device. Negative
            values indicate CPU. If this is None, arrays are not copied across
            CPU/GPUs (i.e. each array given by the dataset is used as is).

    """
    trigger = 1, 'epoch'
    name = 'validation'
    priority = extension.PRIORITY_WRITER

    def __init__(self, dataset, target, lossfun=None, batchsize=1,
                 prepare=None, device=None):
        self._dataset = dataset
        self._target = target
        self._lossfun = lossfun
        self._batchsize = batchsize
        self._prepare = prepare
        self._device = device

    def __call__(self, trainer):
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
            in_vars = tuple(variable.Variable(a, volatile='on')
                            for a in inputs)
            loss = lossfun(*in_vars).data
            with cuda.get_device(loss):
                result = {'loss': loss * n}
            for key, value in six.iteritems(target.__dict__):
                if isinstance(value, variable.Variable):
                    v = value.data
                    if v.size == 1:
                        with cuda.get_device(v):
                            result[key] = v * n
            if accum is None:
                accum = result
            else:
                for key in result:
                    with cuda.get_device(result[key]):
                        accum[key] += result[key]

        N = len(self._dataset)
        ret = {}
        for key, value in six.iteritems(accum):
            with cuda.get_device(value):
                ret[key] = value / N

        return ret
