import copy

import six

from chainer.backends import cuda
from chainer.dataset import convert
from chainer import function
from chainer.training.updaters import standard_updater


class ParallelUpdater(standard_updater.StandardUpdater):

    """Implementation of a parallel GPU Updater.

    This is an implementation of :class:`Updater` that uses multiple GPUs.
    It behaves similarly to
    :class:`~chainer.training.updaters.StandardUpdater`.
    The update routine is modified to support data-parallel computation
    on multiple GPUs in one machine.
    It is based on synchronous parallel SGD: it
    parallelizes the gradient computation over a mini-batch, and updates the
    parameters only in the main device.

    Args:
        iterator: Dataset iterator for the training dataset. It can also be a
            dictionary that maps strings to iterators.
            If this is just an iterator, then the
            iterator is registered by the name ``'main'``.
        optimizer: Optimizer to update parameters. It can also be a dictionary
            that maps strings to optimizers.
            If this is just an optimizer, then the optimizer is
            registered by the name ``'main'``.
        converter: Converter function to build input arrays. Each batch
            extracted by the main iterator is split equally between the
            devices and then passed with corresponding ``device`` option to
            this function. :func:`~chainer.dataset.concat_examples` is used by
            default.
        models: Dictionary of models. The main model should be the same model
            attached to the ``'main'`` optimizer.
        devices: Dictionary of devices to which the training data is sent. The
            devices should be arranged in a dictionary with the same structure
            as ``models``.
        loss_func: Loss function. The model is used as a loss function by
            default.
        loss_scale (float): Loss scaling factor. Loss scaling is a usefull
            technique to mitigate vanishing gradient issue that tends to happen
            when low precision data type like float16 is used during training.
            If you set loss scaling factor, gradients of loss values are to be
            multiplied by the factor before backprop starts. The factor is
            propagated to whole gradients in a computational graph along the
            backprop. The gradients of parameters are divided by the factor
            just before the parameters are to be updated.
        auto_new_epoch (bool):  If ``True``,
            :meth:`~chainer.Optimizer.new_epoch` of the main optimizer is
            automatically called when the ``is_new_epoch`` attribute of the
            main iterator is ``True``.

    """

    def __init__(self, iterator, optimizer, converter=convert.concat_examples,
                 models=None, devices=None, loss_func=None, loss_scale=None,
                 auto_new_epoch=True):
        super(ParallelUpdater, self).__init__(
            iterator=iterator,
            optimizer=optimizer,
            converter=converter,
            loss_func=loss_func,
            loss_scale=loss_scale,
            auto_new_epoch=auto_new_epoch,
        )

        if models is None:
            if devices is None:
                raise ValueError('either models or devices must be specified')
            names = list(six.iterkeys(devices))

            try:
                names.remove('main')
            except ValueError:
                raise KeyError('\'devices\' must contain a \'main\' key.')

            models = {'main': optimizer.target}
            for name in names:
                model = copy.deepcopy(optimizer.target)
                if devices[name] >= 0:
                    model.to_gpu(devices[name])
                models[name] = model
            if devices['main'] >= 0:
                optimizer.target.to_gpu(devices['main'])

        self._devices = devices
        self._models = models

    def connect_trainer(self, trainer):
        # Add observers for all (other) models.
        model_main = self.get_optimizer('main').target
        models_others = {
            k: v for k, v in self._models.items() if v != model_main
        }
        for name, model in models_others.items():
            trainer.reporter.add_observer(name, model)

    def update_core(self):
        optimizer = self.get_optimizer('main')
        model_main = optimizer.target
        models_others = {k: v for k, v in self._models.items()
                         if v is not model_main}

        iterator = self.get_iterator('main')
        batch = iterator.next()

        #
        # Split the batch to sub-batches.
        #
        n = len(self._models)
        in_arrays_list = {}
        for i, key in enumerate(six.iterkeys(self._models)):
            in_arrays_list[key] = self.converter(
                batch[i::n], self._devices[key])

        # For reducing memory
        for model in six.itervalues(self._models):
            model.cleargrads()

        losses = []
        for model_key, model in six.iteritems(self._models):
            in_arrays = in_arrays_list[model_key]
            loss_func = self.loss_func or model

            with function.force_backprop_mode():
                dev_id = self._devices[model_key]
                dev_id = dev_id if 0 <= dev_id else None
                with cuda.get_device_from_id(dev_id):
                    if isinstance(in_arrays, tuple):
                        loss = loss_func(*in_arrays)
                    elif isinstance(in_arrays, dict):
                        loss = loss_func(**in_arrays)
                    else:
                        loss = loss_func(in_arrays)

            losses.append(loss)

        # For _uninitialized_params
        for model in six.itervalues(self._models):
            model.cleargrads()

        for loss in losses:
            loss.backward(loss_scale=self.loss_scale)

        for model in six.itervalues(models_others):
            model_main.addgrads(model)

        optimizer.update()

        for model in six.itervalues(models_others):
            model.copyparams(model_main)

        if self.auto_new_epoch and iterator.is_new_epoch:
            optimizer.new_epoch(auto=True)
