import warnings

import six

import chainer
from chainer.backends import cuda
from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
from chainer import device_resident
from chainer.training import _updater
from chainer.utils import argument


class StandardUpdater(_updater.Updater):

    """StandardUpdater(\
iterator, optimizer, converter=convert.concat_examples, device=None, \
loss_func=None, loss_scale=None, auto_new_epoch=True, *, input_device=None)

    Standard implementation of Updater.

    This is the standard implementation of :class:`~chainer.training.Updater`.
    It accepts one or more training datasets and one or more optimizers.
    The default update routine assumes that there is only one training dataset
    and one optimizer. Users can override this update routine by inheriting
    this class and overriding the :meth:`update_core` method. Each batch is
    converted to input arrays by :func:`chainer.dataset.concat_examples` by
    default, which can also be manually set by ``converter`` argument.

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
            extracted by the main iterator and the ``device`` option are passed
            to this function. :func:`chainer.dataset.concat_examples` is used
            by default.
        device(device specifier): Device to which the model is sent.
            If ``None``, the device of the model will stay unchanged.
        loss_func: Loss function. The target link of the main optimizer is used
            by default.
        loss_scale (float): Loss scaling factor. Loss scaling is a usefull
            technique to mitigate vanishing gradient issue that tends to happen
            when low precision data type like float16 is used during training.
            If you set loss scaling factor, gradients of loss values are to be
            multiplied by the factor before backprop starts. The factor is
            propagated to whole gradients in a computational graph along the
            backprop. The gradients of parameters are divided by the factor
            just before the parameters are to be updated.
        auto_new_epoch (bool): If ``True``,
            :meth:`~chainer.Optimizer.new_epoch` of the main optimizer is
            automatically called when the ``is_new_epoch`` attribute of the
            main iterator is ``True``.
        input_device (device specifier):
            Device to which the training data is sent.
            If ``input_device`` is omitted, it will match the ``device``
            argument.

    Attributes:
        converter: Converter function.
        loss_func: Loss function. If it is ``None``, the target link of the
                   main optimizer is used instead.
        device: Device to which the model is sent.
        input_device: Device to which the training data is sent.
        iteration: Current number of completed updates.
        auto_new_epoch: If ``True``, :meth:`~chainer.Optimizer.new_epoch` is
            automatically called by :meth:`update_core`. In this case, the
            :attr:`~chainer.Optimizer.use_auto_new_epoch` attribute of each
            optimizer is also set to ``True``. If :meth:`update_core` is
            overridden, the implementation should correctly call
            :meth:`~chainer.Optimizer.new_epoch` of each optimizer.

    """

    def __init__(self, iterator, optimizer, converter=convert.concat_examples,
                 device=None, loss_func=None, loss_scale=None,
                 auto_new_epoch=True, **kwargs):
        input_device, = argument.parse_kwargs(
            kwargs, ('input_device', None))

        if device is not None:
            device = chainer.get_device(device)

        # input_device falls back to device
        if input_device is None:
            input_device = device
        else:
            input_device = chainer.get_device(input_device)

        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main': iterator}
        self._iterators = iterator

        if not isinstance(optimizer, dict):
            optimizer = {'main': optimizer}
        self._optimizers = optimizer

        # Transfer the model
        if device is not None:
            for optimizer in six.itervalues(self._optimizers):
                if device.xp is cuda.cupy:
                    # Do not transfer between different cupy devices.
                    # Detect GPU-to-GPU transfer and raise FutureWarning.
                    # TODO(niboshi): Eventually replace it with to_device.

                    thread_local = device_resident._thread_local
                    has_gpu_to_gpu = False
                    try:
                        # Turn on GPU-to-GPU detection
                        thread_local.flag_gpu_to_gpu = False
                        with warnings.catch_warnings():
                            warnings.filterwarnings(
                                'ignore',
                                message='to_gpu is deprecated.',
                                category=DeprecationWarning)
                            optimizer.target.to_gpu(device.device.id)
                        has_gpu_to_gpu = thread_local.flag_gpu_to_gpu
                    finally:
                        # Turn off GPU-to-GPU detection
                        thread_local.flag_gpu_to_gpu = None

                    if has_gpu_to_gpu:
                        warnings.warn(
                            '''\
Transfer between @cupy devices was detected and skipped. \
StandardUpdater normally transfers the model to the specified device, but \
except for between @cupy devices. \
That is, if a part of the model is on @cupy:n device and the specified \
device is @cupy:m device, that part of the model will be left in @cupy:n \
device. This behavior is planned to be changed in near future. \
After that, the model will be transferred to the specified device regardless \
of device combination. \
If you want to keep the model device but only want to transfer the input data \
to a given device, specify the 'input_device' argument instead and leave the \
'device' argument unspecified.
''',
                            FutureWarning)
                else:
                    optimizer.target.to_device(device)

        self.converter = converter
        self.loss_func = loss_func
        self.iteration = 0
        self._device = device
        self._input_device = input_device

        self.loss_scale = loss_scale
        if loss_scale is not None:
            for optimizer in six.itervalues(self._optimizers):
                optimizer.set_loss_scale(loss_scale)

        self.auto_new_epoch = auto_new_epoch
        if auto_new_epoch:
            for o in six.itervalues(self._optimizers):
                o.use_auto_new_epoch = True

    @property
    def device(self):
        return self._device

    @property
    def input_device(self):
        return self._input_device

    @property
    def epoch(self):
        return self._iterators['main'].epoch

    @property
    def epoch_detail(self):
        return self._iterators['main'].epoch_detail

    @property
    def previous_epoch_detail(self):
        return self._iterators['main'].previous_epoch_detail

    @property
    def is_new_epoch(self):
        return self._iterators['main'].is_new_epoch

    def finalize(self):
        """Finalizes the updater object.

        This method calls the `finalize` method of each iterator that
        this updater has.
        It is called at the end of training loops.

        """
        for iterator in six.itervalues(self._iterators):
            iterator.finalize()

    def get_optimizer(self, name):
        """Gets the optimizer of given name.

        Args:
            name (str): Name of the optimizer.

        Returns:
            ~chainer.Optimizer: Corresponding optimizer.

        """
        return self._optimizers[name]

    def get_all_optimizers(self):
        """Gets a dictionary of all optimizers for this updater.

        Returns:
            dict: Dictionary that maps names to optimizers.

        """
        return dict(self._optimizers)

    def get_iterator(self, name):
        """Gets the dataset iterator of given name.

        Args:
            name (str): Name of the dataset iterator.

        Returns:
            ~chainer.dataset.Iterator: Corresponding dataset iterator.

        """
        return self._iterators[name]

    def update(self):
        """Updates the parameters of the target model.

        This method implements an update formula for the training task,
        including data loading, forward/backward computations, and actual
        updates of parameters.

        This method is called once at each iteration of the training loop.

        """
        self.update_core()
        self.iteration += 1

    def update_core(self):
        iterator = self._iterators['main']
        batch = iterator.next()
        in_arrays = convert._call_converter(
            self.converter, batch, self.input_device)

        optimizer = self._optimizers['main']
        loss_func = self.loss_func or optimizer.target

        if isinstance(in_arrays, tuple):
            optimizer.update(loss_func, *in_arrays)
        elif isinstance(in_arrays, dict):
            optimizer.update(loss_func, **in_arrays)
        else:
            optimizer.update(loss_func, in_arrays)

        if self.auto_new_epoch and iterator.is_new_epoch:
            optimizer.new_epoch(auto=True)

    def serialize(self, serializer):
        """Serializes the current state of the updater object."""
        for name, iterator in six.iteritems(self._iterators):
            iterator.serialize(serializer['iterator:' + name])

        for name, optimizer in six.iteritems(self._optimizers):
            optimizer.serialize(serializer['optimizer:' + name])
            optimizer.target.serialize(serializer['model:' + name])

        self.iteration = serializer('iteration', self.iteration)
