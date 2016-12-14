import copy
import math
from multiprocessing import Pipe
from multiprocessing import Process
import six
import time

from chainer import cuda
from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
from chainer import optimizer as optimizer_module
from chainer import reporter
from chainer import variable


class Updater(object):

    """Interface of updater objects for trainers.

    TODO(beam2d): document it.

    """

    def connect_trainer(self, trainer):
        """Connects the updater to the trainer that will call it.

        The typical usage of this method is to register additional links to the
        reporter of the trainer. This method is called at the end of the
        initialization of :class:`~chainer.training.Trainer`. The default
        implementation does nothing.

        Args:
            trainer (~chainer.training.Trainer): Trainer object to which the
                updater is registered.

        """
        pass

    def finalize(self):
        """Finalizes the updater object.

        This method is called at the end of training loops. It should finalize
        each dataset iterator used in this updater.

        """
        raise NotImplementedError

    def get_optimizer(self, name):
        """Gets the optimizer of given name.

        Updater holds one or more optimizers with names. They can be retrieved
        by this method.

        Args:
            name (str): Name of the optimizer.

        Returns:
            ~chainer.Optimizer: Optimizer of the name.

        """
        raise NotImplementedError

    def get_all_optimizers(self):
        """Gets a dictionary of all optimizers for this updater.

        Returns:
            dict: Dictionary that maps names to optimizers.

        """
        raise NotImplementedError

    def update(self):
        """Updates the parameters of the target model.

        This method implements an update formula for the training task,
        including data loading, forward/backward computations, and actual
        updates of parameters.

        This method is called once at each iteration of the training loop.

        """
        raise NotImplementedError

    def serialize(self, serializer):
        """Serializes the current state of the updater object."""
        raise NotImplementedError


class StandardUpdater(Updater):

    """Standard implementation of Updater.

    This is the standard implementation of :class:`Updater`. It accepts one or
    more training datasets and one or more optimizers. The default update
    routine assumes that there is only one training dataset and one optimizer.
    Users can override this update routine by inheriting this class and
    overriding the :meth:`update_core` method. Each batch is converted to input
    arrays by :func:`~chainer.datasets.concat_examples` by default, which can
    also be manually set by ``converter`` argument.

    Args:
        iterator: Dataset iterator for the training dataset. It can also be a
            dictionary of iterators. If this is just an iterator, then the
            iterator is registered by the name ``'main'``.
        optimizer: Optimizer to update parameters. It can also be a dictionary
            of optimizers. If this is just an optimizer, then the optimizer is
            registered by the name ``'main'``.
        converter: Converter function to build input arrays. Each batch
            extracted by the main iterator and the ``device`` option are passed
            to this function. :func:`~chainer.dataset.concat_examples` is used
            by default.
        device: Device to which the training data is sent. Negative value
            indicates the host memory (CPU).
        loss_func: Loss function. The target link of the main optimizer is used
            by default.

    Attributes:
        converter: Converter function.
        loss_func: Loss function. If it is ``None``, the target link of the
                   main optimizer is used instead.
        device: Device to which the training data is sent.
        iteration: Current number of completed updates.

    """

    def __init__(self, iterator, optimizer, converter=convert.concat_examples,
                 device=None, loss_func=None):
        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main': iterator}
        self._iterators = iterator

        if isinstance(optimizer, optimizer_module.Optimizer):
            optimizer = {'main': optimizer}
        self._optimizers = optimizer

        self.converter = converter
        self.loss_func = loss_func
        self.device = device
        self.iteration = 0

    @property
    def epoch(self):
        return self._iterators['main'].epoch

    @property
    def epoch_detail(self):
        return self._iterators['main'].epoch_detail

    @property
    def is_new_epoch(self):
        return self._iterators['main'].is_new_epoch

    def finalize(self):
        for iterator in six.itervalues(self._iterators):
            iterator.finalize()

    def get_optimizer(self, name):
        return self._optimizers[name]

    def get_all_optimizers(self):
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
        self.update_core()
        self.iteration += 1

    def update_core(self):
        batch = self._iterators['main'].next()
        in_arrays = self.converter(batch, self.device)

        optimizer = self._optimizers['main']
        loss_func = self.loss_func or optimizer.target

        if isinstance(in_arrays, tuple):
            in_vars = tuple(variable.Variable(x) for x in in_arrays)
            optimizer.update(loss_func, *in_vars)
        elif isinstance(in_arrays, dict):
            in_vars = {key: variable.Variable(x)
                       for key, x in six.iteritems(in_arrays)}
            optimizer.update(loss_func, **in_vars)
        else:
            in_var = variable.Variable(in_arrays)
            optimizer.update(loss_func, in_var)

    def serialize(self, serializer):
        for name, iterator in six.iteritems(self._iterators):
            iterator.serialize(serializer['iterator:' + name])

        for name, optimizer in six.iteritems(self._optimizers):
            optimizer.serialize(serializer['optimizer:' + name])
            optimizer.target.serialize(serializer['model:' + name])

        self.iteration = serializer('iteration', self.iteration)


class ParallelUpdater(StandardUpdater):

    """Implementation of a parallel GPU Updater.

    This is an implementation of :class:`Updater` that uses multiple GPUs.
    It behaves similarly to :class:`~chainer.training.StandardUpdater`. The
    update routine is modified to support data-parallel computation on multiple
    GPUs in one machine. It is based on synchronous parallel SGD: it
    parallelizes the gradient computation over a mini-batch, and updates the
    parameters only in the main device.

    Args:
        iterator: Dataset iterator for the training dataset. It can also be a
            dictionary of iterators. If this is just an iterator, then the
            iterator is registered by the name ``'main'``.
        optimizer: Optimizer to update parameters. It can also be a dictionary
            of optimizers. If this is just an optimizer, then the optimizer is
            registered by the name ``'main'``.
        converter: Converter function to build input arrays. Each batch
            extracted by the main iterator is split equally between the
            devices and then passed with corresponding ``device`` option to
            this function. :func:`~chainer.dataset.concat_examples` is used by
            default.
        models: Dictionary of models. The main model should be the same model
            attached to the ``'main'`` optimizer.
        devices: Dictionary or list of devices to which the training data is
            sent. The master device will be the first one in the list or the
            value attached to the key ``'main'``.
        loss_func: Loss function. The model is used as a loss function by
            default.

    """

    def __init__(self, iterator, optimizer, converter=convert.concat_examples,
                 models=None, devices=None, loss_func=None):
        super(ParallelUpdater, self).__init__(
            iterator=iterator,
            optimizer=optimizer,
            converter=converter,
            loss_func=loss_func,
        )

        if models is None:
            if devices is None:
                raise ValueError('either models or devices must be specified')

            if isinstance(devices, list):
                devices_dict = {'main': devices[0]}
                for i, device in enumerate(devices[1:]):
                    devices_dict[i] = device
                devices = devices_dict

            names = list(six.iterkeys(devices))

            try:
                names.remove('main')
            except ValueError:
                raise KeyError("'devices' must contain a 'main' key.")

            models = {'main': optimizer.target}
            for name in names:
                model = copy.deepcopy(optimizer.target)
                if devices[name] >= 0:
                    model.to_gpu(devices[name])
                models[name] = model
            if devices['main'] >= 0:
                optimizer.target.to_gpu(devices['main'])

        # Correct optimizer parameters for new minibatch size
        optim = optimizer.__class__.__name__
        if optim in ('Adam', 'AdaGrad', 'RMSprop'):
            optimizer.eps /= len(devices)
        elif optim in ('RMSpropGraves', 'AdaDelta'):
            optimizer.eps /= len(devices) ** 2
        else:
            optimizer.lr /= len(devices)

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

        batch = self.get_iterator('main').next()

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

            if isinstance(in_arrays, tuple):
                in_vars = tuple(variable.Variable(x) for x in in_arrays)
                losses.append(loss_func(*in_vars))
            elif isinstance(in_arrays, dict):
                in_vars = {key: variable.Variable(x)
                           for key, x in six.iteritems(in_arrays)}
                losses.append(loss_func(**in_vars))
            else:
                in_vars = variable.Variable(in_arrays)
                losses.append(loss_func(in_vars))

        # For _uninitialized_params
        for model in six.itervalues(self._models):
            model.cleargrads()

        for loss in losses:
            loss.backward()

        for model in six.itervalues(models_others):
            model_main.addgrads(model)

        optimizer.update()

        for model in six.itervalues(models_others):
            model.copyparams(model_main)


class Worker(Process):

    def __init__(self, pipe, device, model, converter):

        super().__init__()
        self.pipe = pipe
        self.device = device
        self.converter = converter
        self.model = model  # .copy()

    def setup(self):

        cuda.get_device(self.device).use()
        self.model.to_gpu(self.device)
        self.model.zerograds()
        self.model.set_parent(self.pipe.recv())
        self.pipe.send(list(self.model.get_handles()))

    def run(self):

        self.setup()
        while True:
            job, data = self.pipe.recv()
            if job == 'gathergrads':
                self.pipe.send(self.model.gathergrads())
            elif job == 'scatterparams':
                self.pipe.send(self.model.scatterparams())
            elif job == 'finalize':
                break
            else:
                batch = self.converter(data, self.device)
                observation = {}
                with reporter.report_scope(observation):
                    if isinstance(batch, tuple):
                        in_vars = tuple(variable.Variable(x) for x in batch)
                        loss = self.model(*in_vars)
                    elif isinstance(batch, dict):
                        in_vars = {key: variable.Variable(x)
                                   for key, x in six.iteritems(batch)}
                        loss = self.model(**in_vars)
                    else:
                        in_vars = variable.Variable(batch)
                        loss = self.model(in_vars)
                self.model.zerograds()
                loss.backward()
                self.pipe.send(observation)


class MultiprocessParallelUpdater(StandardUpdater):

    """Implementation of a multiprocess parallel GPU Updater.

    This is an implementation of :class:`Updater` that uses multiple GPUs
    with multi-process data parallelism. The GPUs provided are placed in
    a tree structure of masters and workers, each with its own process.

    It behaves similarly to :class:`~chainer.training.StandardUpdater`. The
    update routine is modified to support data-parallel computation on multiple
    GPUs in one machine. It is based on synchronous parallel SGD: it
    parallelizes the gradient computation over a mini-batch, and updates the
    parameters only in the main device.

    Unlike other built-in Updater classes, the model (attached to the
    optimizer) must be the loss function.

    Args:
        iterator: Dataset iterator for the training dataset.
        optimizer: Optimizer to update parameters. The model should be attached
            to the optimizer.
        converter: Converter function to build input arrays. Each batch
            extracted by the iterator is split equally between the devices and
            then passed with corresponding ``device`` option to this function.
            :func:`~chainer.dataset.concat_examples` is used by default.
        devices: Dictionary or list of devices to which the training data is
            sent. The master device will be the first one in the list or the
            value attached to the key ``'main'``.

    """

    def __init__(self, iterator, optimizer, converter=convert.concat_examples,
                 devices=None):

        # Correct optimizer parameters for new minibatch size
        optim = optimizer.__class__.__name__
        if optim in ('Adam', 'AdaGrad', 'RMSprop'):
            optimizer.eps /= len(devices)
        elif optim in ('RMSpropGraves', 'AdaDelta'):
            optimizer.eps /= len(devices) ** 2
        else:
            optimizer.lr /= len(devices)

        super(MultiprocessParallelUpdater, self).__init__(
            iterator=iterator,
            optimizer=optimizer,
            converter=converter
        )

        if isinstance(devices, dict):
            main = devices.pop('main')
            devices = list(six.itervalues(devices))
            devices = [main] + devices
        if devices is None or any(device is None for device in devices):
            raise ValueError('must specify GPU devices')

        self._master = optimizer.target
        self._devices = devices
        self._depth = math.ceil(math.log(len(devices)) / math.log(2))

        self._ipc_setup = False
        self.setup_workers()

    def setup_workers(self):

        devices = self._devices[1:]
        self._pipes, self._workers, worker_ends = [], [], []
        if len(devices) > 0:
            self._pipes, worker_ends = zip(*[Pipe() for _ in devices])
        self._master.zerograds()
        for pipe, (i, device) in zip(worker_ends, enumerate(devices)):
            worker = Worker(pipe, device, self._master, self.converter)
            worker.start()
            self._workers.append(worker)

        self._master.to_gpu(self._devices[0])

    def setup_ipc(self):

        ipc_parameters = [list(self._master.get_handles())]
        for i, pipe in enumerate(self._pipes):
            pipe.send(ipc_parameters[self._parent(i + 1)])
            ipc_parameters.append(pipe.recv())

        time.sleep(0.1)

        self._ipc_setup = True

    @staticmethod
    def _parent(index):
        if index == 0:
            return None
        return -2 ** format(index, '8b')[::-1].index('1')

    def _nodes_at(self, depth):
        return range(len(self._devices) - 1)[2 ** depth - 1::2 ** (depth + 1)]

    def update_core(self):
        optimizer = self.get_optimizer('main')
        batch = self.get_iterator('main').next()

        n = len(self._devices)
        master_device = self._devices[0] if self._started else -1
        master_batch = self.converter(batch[0::n], master_device)
        device_batches = [batch[i::n] for i in range(1, n)]

        if self._ipc_setup:
            for pipe, batch in zip(self._pipes, device_batches):
                pipe.send(('train', batch))

        if isinstance(master_batch, tuple):
            in_vars = tuple(variable.Variable(x) for x in master_batch)
            loss = self._master(*in_vars)
        elif isinstance(master_batch, dict):
            in_vars = {key: variable.Variable(x)
                       for key, x in six.iteritems(master_batch)}
            loss = self._master(**in_vars)
        else:
            in_vars = variable.Variable(master_batch)
            loss = self._master(in_vars)

        self._master.zerograds()
        loss.backward()

        if not self._ipc_setup:
            self.setup_ipc()
            for pipe, batch in zip(self._pipes, device_batches):
                pipe.send(('train', batch))

        [reporter.report(pipe.recv()) for pipe in self._pipes]

        for depth in range(self._depth):
            pipes = [self._pipes[i] for i in self._nodes_at(depth)]
            for pipe in pipes:
                pipe.send(('gathergrads', None))
            for pipe in pipes:
                pipe.recv()

        optimizer.update()

        for depth in range(self._depth, -1, -1):
            pipes = [self._pipes[i] for i in self._nodes_at(depth)]
            for pipe in pipes:
                pipe.send(('scatterparams', None))
            for pipe in pipes:
                pipe.recv()

    def finalize(self):
        for pipe in self._pipes:
            pipe.send(('finalize', None))
